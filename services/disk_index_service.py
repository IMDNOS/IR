import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import PreprocessingService


RetrievalMode = Literal[
    "tfidf",
    "bm25",
    "embedding",
    "hybrid_serial",
    "hybrid_parallel",
]


@dataclass
class BM25Params:
    k1: float = 1.5
    b: float = 0.75


@dataclass
class HybridWeights:
    tfidf: float = 0.30
    bm25: float = 0.35
    embedding: float = 0.35


@dataclass
class SearchResult:
    doc_id: str
    score: float
    raw_text: str
    rank: int
    source_scores: dict[str, float]


class DiskIndexBuilder:
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


class DiskRetrievalService:
    """
    Memory-safe retrieval service.

    It loads metadata and model only. The SQLite index and embedding chunks stay on disk.
    Embedding similarity is computed chunk-by-chunk.
    """

    def __init__(
        self,
        index_dir: Path,
        preprocessor: PreprocessingService | None = None,
    ):
        self.index_dir = index_dir
        self.preprocessor = preprocessor or PreprocessingService()

        metadata_path = index_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing index metadata: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as infile:
            self.metadata = json.load(infile)

        chunks_path = index_dir / self.metadata["files"]["embedding_chunks"]
        with chunks_path.open("r", encoding="utf-8") as infile:
            self.embedding_chunks = json.load(infile)

        self.conn = sqlite3.connect(index_dir / self.metadata["files"]["sqlite"])
        self.conn.row_factory = sqlite3.Row

        self.doc_norms = np.load(
            index_dir / self.metadata["files"]["tfidf_doc_norms"],
            mmap_mode="r",
        )
        self.embedding_model = SentenceTransformer(self.metadata["embedding_model"])

    def close(self) -> None:
        self.conn.close()

    def _preprocess_query(self, query_text: str) -> dict:
        return self.preprocessor.preprocess_query("interactive_query", query_text)

    def _top_scores(self, scores: dict[int, float], top_k: int) -> dict[int, float]:
        if not scores:
            return {}
        items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return dict(items)

    def _fetch_df(self, term: str) -> int:
        row = self.conn.execute("SELECT df FROM df WHERE term = ?", (term,)).fetchone()
        return int(row["df"]) if row else 0

    def _tfidf_scores(self, query_tokens: list[str], top_k: int) -> dict[int, float]:
        num_docs = int(self.metadata["num_documents"])
        query_counts = Counter(query_tokens)
        dot_scores: dict[int, float] = {}
        query_norm_sq = 0.0

        for term, qtf in query_counts.items():
            df_value = self._fetch_df(term)
            if df_value == 0:
                continue

            idf = math.log((num_docs + 1.0) / (df_value + 1.0)) + 1.0
            q_weight = (1.0 + math.log(float(qtf))) * idf
            query_norm_sq += q_weight * q_weight

            rows = self.conn.execute(
                "SELECT internal_id, tf FROM postings WHERE term = ?",
                (term,),
            )
            for row in rows:
                internal_id = int(row["internal_id"])
                tf = float(row["tf"])
                d_weight = (1.0 + math.log(tf)) * idf
                dot_scores[internal_id] = dot_scores.get(internal_id, 0.0) + q_weight * d_weight

        if query_norm_sq <= 0.0:
            return {}

        query_norm = math.sqrt(query_norm_sq)
        cosine_scores: dict[int, float] = {}
        for internal_id, dot_score in dot_scores.items():
            doc_norm = float(self.doc_norms[internal_id])
            if doc_norm > 0.0:
                cosine_scores[internal_id] = dot_score / (query_norm * doc_norm)

        return self._top_scores(cosine_scores, top_k)

    def _bm25_scores(
        self,
        query_tokens: list[str],
        top_k: int,
        k1: float | None = None,
        b: float | None = None,
    ) -> dict[int, float]:
        num_docs = int(self.metadata["num_documents"])
        avg_doc_len = float(self.metadata["avg_doc_len"])
        k1 = float(k1 if k1 is not None else self.metadata["bm25_k1"])
        b = float(b if b is not None else self.metadata["bm25_b"])

        scores: dict[int, float] = {}
        for term, qtf in Counter(query_tokens).items():
            df_value = self._fetch_df(term)
            if df_value == 0:
                continue

            idf = math.log(1.0 + (num_docs - df_value + 0.5) / (df_value + 0.5))
            rows = self.conn.execute(
                """
                SELECT p.internal_id, p.tf, d.doc_len
                FROM postings p
                JOIN docs d ON d.internal_id = p.internal_id
                WHERE p.term = ?
                """,
                (term,),
            )
            for row in rows:
                internal_id = int(row["internal_id"])
                tf = float(row["tf"])
                doc_len = float(row["doc_len"])
                denom = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
                score = idf * ((tf * (k1 + 1.0)) / denom) * float(qtf)
                scores[internal_id] = scores.get(internal_id, 0.0) + score

        return self._top_scores(scores, top_k)

    def _query_embedding(self, embedding_text: str) -> np.ndarray:
        embedding = self.embedding_model.encode(
            [embedding_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]
        return embedding

    def _embedding_scores(self, embedding_text: str, top_k: int) -> dict[int, float]:
        query_embedding = self._query_embedding(embedding_text)
        best: dict[int, float] = {}

        for chunk in self.embedding_chunks:
            chunk_path = self.index_dir / chunk["file"]
            matrix = np.load(chunk_path, mmap_mode="r")
            scores = matrix @ query_embedding
            local_k = min(top_k, len(scores))
            if local_k <= 0:
                continue
            local_indices = np.argpartition(-scores, local_k - 1)[:local_k]
            start_id = int(chunk["start_internal_id"])
            for local_idx in local_indices:
                internal_id = start_id + int(local_idx)
                score = float(scores[local_idx])
                if len(best) < top_k or score > min(best.values()):
                    best[internal_id] = score
                    if len(best) > top_k * 2:
                        best = self._top_scores(best, top_k)

        return self._top_scores(best, top_k)

    def _embedding_scores_for_candidates(
        self,
        embedding_text: str,
        candidate_ids: Iterable[int],
    ) -> dict[int, float]:
        query_embedding = self._query_embedding(embedding_text)
        candidate_set = set(int(i) for i in candidate_ids)
        scores_out: dict[int, float] = {}

        for chunk in self.embedding_chunks:
            start_id = int(chunk["start_internal_id"])
            end_id = int(chunk["end_internal_id"])
            ids_in_chunk = [i for i in candidate_set if start_id <= i <= end_id]
            if not ids_in_chunk:
                continue
            local_indices = [i - start_id for i in ids_in_chunk]
            matrix = np.load(self.index_dir / chunk["file"], mmap_mode="r")
            local_scores = matrix[local_indices] @ query_embedding
            for internal_id, score in zip(ids_in_chunk, local_scores):
                scores_out[internal_id] = float(score)

        return scores_out

    def _normalize_dict_scores(self, scores: dict[int, float], ids: Iterable[int]) -> dict[int, float]:
        ids = list(ids)
        if not ids:
            return {}
        values = [float(scores.get(i, 0.0)) for i in ids]
        min_value = min(values)
        max_value = max(values)
        if max_value == min_value:
            return {i: 0.0 for i in ids}
        return {i: (float(scores.get(i, 0.0)) - min_value) / (max_value - min_value) for i in ids}

    def _fetch_docs(self, internal_ids: list[int]) -> dict[int, sqlite3.Row]:
        if not internal_ids:
            return {}
        placeholders = ",".join("?" for _ in internal_ids)
        rows = self.conn.execute(
            f"SELECT internal_id, doc_id, raw_text FROM docs WHERE internal_id IN ({placeholders})",
            internal_ids,
        ).fetchall()
        return {int(row["internal_id"]): row for row in rows}

    def _format_results(
        self,
        final_scores: dict[int, float],
        top_k: int,
        source_scores: dict[str, dict[int, float]],
    ) -> list[SearchResult]:
        ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        internal_ids = [internal_id for internal_id, _ in ranked]
        docs = self._fetch_docs(internal_ids)

        results: list[SearchResult] = []
        for rank, (internal_id, score) in enumerate(ranked, start=1):
            doc = docs[internal_id]
            results.append(
                SearchResult(
                    doc_id=str(doc["doc_id"]),
                    score=float(score),
                    raw_text=str(doc["raw_text"]),
                    rank=rank,
                    source_scores={
                        name: float(values.get(internal_id, 0.0))
                        for name, values in source_scores.items()
                    },
                )
            )
        return results

    def search(
        self,
        query_text: str,
        mode: RetrievalMode = "bm25",
        top_k: int = 10,
        serial_candidate_k: int = 100,
        fusion_pool_k: int = 1000,
        hybrid_weights: HybridWeights | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ) -> list[SearchResult]:
        processed_query = self._preprocess_query(query_text)
        query_tokens = processed_query["lexical_tokens"]
        embedding_text = processed_query["embedding_text"]
        weights = hybrid_weights or HybridWeights()

        if mode == "tfidf":
            tfidf = self._tfidf_scores(query_tokens, top_k=top_k)
            return self._format_results(tfidf, top_k, {"tfidf": tfidf})

        if mode == "bm25":
            bm25 = self._bm25_scores(query_tokens, top_k=top_k, k1=bm25_k1, b=bm25_b)
            return self._format_results(bm25, top_k, {"bm25": bm25})

        if mode == "embedding":
            emb = self._embedding_scores(embedding_text, top_k=top_k)
            return self._format_results(emb, top_k, {"embedding": emb})

        if mode == "hybrid_serial":
            bm25 = self._bm25_scores(
                query_tokens,
                top_k=max(serial_candidate_k, top_k),
                k1=bm25_k1,
                b=bm25_b,
            )
            emb = self._embedding_scores_for_candidates(embedding_text, bm25.keys())
            ids = list(bm25.keys())
            nb = self._normalize_dict_scores(bm25, ids)
            ne = self._normalize_dict_scores(emb, ids)
            final = {i: 0.30 * nb.get(i, 0.0) + 0.70 * ne.get(i, 0.0) for i in ids}
            return self._format_results(final, top_k, {"bm25": bm25, "embedding": emb})

        if mode == "hybrid_parallel":
            pool_k = max(fusion_pool_k, top_k)
            tfidf = self._tfidf_scores(query_tokens, top_k=pool_k)
            bm25 = self._bm25_scores(query_tokens, top_k=pool_k, k1=bm25_k1, b=bm25_b)
            emb = self._embedding_scores(embedding_text, top_k=pool_k)

            ids = sorted(set(tfidf) | set(bm25) | set(emb))
            nt = self._normalize_dict_scores(tfidf, ids)
            nb = self._normalize_dict_scores(bm25, ids)
            ne = self._normalize_dict_scores(emb, ids)

            final = {
                i: weights.tfidf * nt.get(i, 0.0)
                + weights.bm25 * nb.get(i, 0.0)
                + weights.embedding * ne.get(i, 0.0)
                for i in ids
            }
            return self._format_results(
                final,
                top_k,
                {"tfidf": tfidf, "bm25": bm25, "embedding": emb},
            )

        raise ValueError(f"Unsupported retrieval mode: {mode}")