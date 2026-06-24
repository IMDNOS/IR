import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from services.disk_index_models import BM25Params
from services.preprocessing_service import PreprocessingService


class IndexingService:
    """
    Memory-safe index builder.

    It never loads the whole processed_docs.jsonl file into RAM.
    It builds:
      - SQLite docs table
      - SQLite inverted index postings table
      - SQLite document frequency table
      - TF-IDF document norms on disk
      - embedding .npy chunks on disk

    Expected input fields:
      doc_id, raw_text, lexical_tokens, lexical_text, embedding_text
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_batch_size: int = 32,
        embedding_chunk_docs: int = 2048,
        commit_every_docs: int = 1000,
        bm25_params: BM25Params | None = None,
    ):
        self.model_name = model_name
        self.embedding_batch_size = embedding_batch_size
        self.embedding_chunk_docs = embedding_chunk_docs
        self.commit_every_docs = commit_every_docs
        self.bm25_params = bm25_params or BM25Params()

    def _iter_docs(self, processed_docs_path: Path) -> Iterable[dict]:
        if not processed_docs_path.exists():
            raise FileNotFoundError(
                f"Missing processed docs file: {processed_docs_path}. "
                "Run scripts/preprocess_datasets.py first."
            )

        required = {"doc_id", "raw_text", "lexical_tokens", "embedding_text"}
        with processed_docs_path.open("r", encoding="utf-8") as infile:
            for line_number, line in enumerate(infile, start=1):
                if not line.strip():
                    continue
                doc = json.loads(line)
                missing = required - set(doc.keys())
                if missing:
                    raise ValueError(
                        f"Missing fields {sorted(missing)} in {processed_docs_path}, line {line_number}"
                    )
                yield doc

    def _connect_for_build(self, sqlite_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(sqlite_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -200000")  # about 200 MB cache
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TABLE IF EXISTS metadata;
            DROP TABLE IF EXISTS docs;
            DROP TABLE IF EXISTS postings;
            DROP TABLE IF EXISTS df;

            CREATE TABLE metadata
            (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE docs
            (
                internal_id    INTEGER PRIMARY KEY,
                doc_id         TEXT    NOT NULL UNIQUE,
                raw_text       TEXT    NOT NULL,
                embedding_text TEXT    NOT NULL,
                doc_len        INTEGER NOT NULL
            );

            CREATE TABLE postings
            (
                term        TEXT    NOT NULL,
                internal_id INTEGER NOT NULL,
                tf          INTEGER NOT NULL,
                PRIMARY KEY (term, internal_id)
            );

            CREATE TABLE df
            (
                term TEXT PRIMARY KEY,
                df   INTEGER NOT NULL
            );
            """
        )
        conn.commit()

    def _upsert_df_counts(self, conn: sqlite3.Connection, df_counts: Counter[str]) -> None:
        conn.executemany(
            """
            INSERT INTO df(term, df)
            VALUES (?, ?)
            ON CONFLICT(term) DO UPDATE SET df = df + excluded.df
            """,
            df_counts.items(),
        )

    def _flush_sql_batches(
        self,
        conn: sqlite3.Connection,
        docs_rows: list[tuple[int, str, str, str, int]],
        postings_rows: list[tuple[str, int, int]],
        df_counts: Counter[str],
    ) -> None:
        if docs_rows:
            conn.executemany(
                """
                INSERT INTO docs(internal_id, doc_id, raw_text, embedding_text, doc_len)
                VALUES (?, ?, ?, ?, ?)
                """,
                docs_rows,
            )
        if postings_rows:
            conn.executemany(
                """
                INSERT INTO postings(term, internal_id, tf)
                VALUES (?, ?, ?)
                """,
                postings_rows,
            )
        if df_counts:
            self._upsert_df_counts(conn, df_counts)
        conn.commit()
        docs_rows.clear()
        postings_rows.clear()
        df_counts.clear()

    def _save_embedding_chunk(
        self,
        output_dir: Path,
        model: SentenceTransformer,
        chunk_index: int,
        start_internal_id: int,
        texts: list[str],
    ) -> dict:
        embeddings_dir = output_dir / "embedding_chunks"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        embeddings = model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        filename = f"chunk_{chunk_index:06d}.npy"
        np.save(embeddings_dir / filename, embeddings)

        return {
            "file": f"embedding_chunks/{filename}",
            "start_internal_id": start_internal_id,
            "end_internal_id": start_internal_id + len(texts) - 1,
            "num_documents": len(texts),
            "dimension": int(embeddings.shape[1]),
        }

    def build(self, processed_docs_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path = output_dir / "index.sqlite3"
        metadata_path = output_dir / "metadata.json"
        chunks_path = output_dir / "embedding_chunks.json"
        doc_norms_path = output_dir / "tfidf_doc_norms.float32.npy"

        if sqlite_path.exists():
            sqlite_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        if chunks_path.exists():
            chunks_path.unlink()
        if doc_norms_path.exists():
            doc_norms_path.unlink()

        embeddings_dir = output_dir / "embedding_chunks"
        if embeddings_dir.exists():
            for old_file in embeddings_dir.glob("*.npy"):
                old_file.unlink()

        conn = self._connect_for_build(sqlite_path)
        self._create_tables(conn)

        embedding_model = SentenceTransformer(self.model_name)

        docs_rows: list[tuple[int, str, str, str, int]] = []
        postings_rows: list[tuple[str, int, int]] = []
        df_counts: Counter[str] = Counter()

        embedding_texts: list[str] = []
        embedding_start_id = 0
        embedding_chunk_index = 0
        embedding_chunks: list[dict] = []

        num_documents = 0
        total_doc_len = 0

        # Track seen document IDs to skip duplicates
        seen_doc_ids: set[str] = set()
        next_internal_id = 0

        for doc in self._iter_docs(processed_docs_path):
            # Extract doc_id and skip duplicates
            doc_id = str(doc["doc_id"])
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            internal_id = next_internal_id
            next_internal_id += 1

            lexical_tokens = doc.get("lexical_tokens") or []
            if not isinstance(lexical_tokens, list):
                lexical_tokens = str(doc.get("lexical_text", "")).split()

            token_counts = Counter(str(t) for t in lexical_tokens if str(t).strip())
            doc_len = int(sum(token_counts.values()))

            docs_rows.append(
                (
                    internal_id,
                    doc_id,  # Use the pre-converted doc_id variable
                    str(doc["raw_text"]),
                    str(doc["embedding_text"]),
                    doc_len,
                )
            )
            postings_rows.extend(
                (term, internal_id, int(tf))
                for term, tf in token_counts.items()
            )
            df_counts.update(token_counts.keys())

            embedding_texts.append(str(doc["embedding_text"]))
            if len(embedding_texts) >= self.embedding_chunk_docs:
                embedding_chunks.append(
                    self._save_embedding_chunk(
                        output_dir=output_dir,
                        model=embedding_model,
                        chunk_index=embedding_chunk_index,
                        start_internal_id=embedding_start_id,
                        texts=embedding_texts,
                    )
                )
                embedding_chunk_index += 1
                embedding_start_id += len(embedding_texts)
                embedding_texts = []

            num_documents += 1
            total_doc_len += doc_len

            if num_documents % self.commit_every_docs == 0:
                self._flush_sql_batches(conn, docs_rows, postings_rows, df_counts)
                print(f"Indexed {num_documents:,} documents...")

        self._flush_sql_batches(conn, docs_rows, postings_rows, df_counts)

        if embedding_texts:
            embedding_chunks.append(
                self._save_embedding_chunk(
                    output_dir=output_dir,
                    model=embedding_model,
                    chunk_index=embedding_chunk_index,
                    start_internal_id=embedding_start_id,
                    texts=embedding_texts,
                )
            )

        if num_documents == 0:
            raise ValueError(f"No documents found in {processed_docs_path}")

        avg_doc_len = total_doc_len / num_documents
        embedding_dimension = int(embedding_chunks[0]["dimension"]) if embedding_chunks else 0

        print("Creating SQLite indexes...")
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
            CREATE INDEX IF NOT EXISTS idx_docs_doc_id ON docs(doc_id);
            """
        )
        conn.commit()

        print("Building TF-IDF document norms on disk...")
        self._build_tfidf_doc_norms(
            conn=conn,
            num_documents=num_documents,
            output_path=doc_norms_path,
        )

        metadata = {
            "index_type": "disk_chunked_v1",
            "num_documents": num_documents,
            "avg_doc_len": avg_doc_len,
            "embedding_model": self.model_name,
            "embedding_dimension": embedding_dimension,
            "bm25_k1": self.bm25_params.k1,
            "bm25_b": self.bm25_params.b,
            "files": {
                "sqlite": "index.sqlite3",
                "embedding_chunks": "embedding_chunks.json",
                "tfidf_doc_norms": "tfidf_doc_norms.float32.npy",
            },
        }

        with metadata_path.open("w", encoding="utf-8") as outfile:
            json.dump(metadata, outfile, ensure_ascii=False, indent=2)
        with chunks_path.open("w", encoding="utf-8") as outfile:
            json.dump(embedding_chunks, outfile, ensure_ascii=False, indent=2)

        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in metadata.items()],
        )
        conn.commit()
        conn.close()

        print(f"Finished disk index: {output_dir}")
        return output_dir

    def _build_tfidf_doc_norms(
        self,
        conn: sqlite3.Connection,
        num_documents: int,
        output_path: Path,
    ) -> None:
        doc_norms = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype="float32",
            shape=(num_documents,),
        )
        doc_norms[:] = 0.0

        cursor = conn.execute(
            """
            SELECT p.internal_id, p.tf, df.df
            FROM postings p
            JOIN df ON p.term = df.term
            """
        )

        for internal_id, tf, df_value in cursor:
            idf = math.log((num_documents + 1.0) / (df_value + 1.0)) + 1.0
            weight = (1.0 + math.log(float(tf))) * idf
            doc_norms[int(internal_id)] += weight * weight

        np.sqrt(doc_norms, out=doc_norms)
        doc_norms.flush()
