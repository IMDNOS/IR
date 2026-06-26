import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from services.disk_index_models import RetrievalMode, HybridWeights, SearchResult
from services.preprocessing_service import PreprocessingService


class SearchService:
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
        serial_bm25_weight: float = 0.30,
        serial_embedding_weight: float = 0.70,
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
            total = serial_bm25_weight + serial_embedding_weight
            if total <= 0.0:
                bm25_w = 0.30
                emb_w = 0.70
            else:
                bm25_w = serial_bm25_weight / total
                emb_w = serial_embedding_weight / total
            final = {i: bm25_w * nb.get(i, 0.0) + emb_w * ne.get(i, 0.0) for i in ids}
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
