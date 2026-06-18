# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable, Literal
#
# import joblib
# import numpy as np
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# from services.preprocessing_service import PreprocessingService
#
# def split_on_space(text: str) -> list[str]:
#     return text.split()
#
#
# RetrievalMode = Literal[
#     "tfidf",
#     "bm25",
#     "embedding",
#     "hybrid_serial",
#     "hybrid_parallel",
# ]
#
#
# @dataclass
# class SearchResult:
#     doc_id: str
#     score: float
#     raw_text: str
#     rank: int
#     source_scores: dict
#
#
# @dataclass
# class BM25Params:
#     """
#     These are the classic BM25 parameters.
#
#     k1 controls term-frequency saturation.
#     b controls document-length normalization.
#
#     We expose them because the IR project asks either for BM25 parameter control
#     in the UI or for a report explanation of why fixed values were selected.
#     """
#
#     k1: float = 1.5
#     b: float = 0.75
#
#
# @dataclass
# class HybridWeights:
#     tfidf: float = 0.30
#     bm25: float = 0.35
#     embedding: float = 0.35
#
#
# class RepresentationIndexBuilder:
#     """
#     Builds TF-IDF, BM25, and embedding representations for one processed dataset.
#
#     Expected input file:
#         processed/<dataset_name>/processed_docs.jsonl
#
#     Expected document fields:
#         doc_id, raw_text, lexical_tokens, lexical_text, embedding_text
#     """
#
#     def __init__(
#         self,
#         model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
#         embedding_batch_size: int = 64,
#         bm25_params: BM25Params | None = None,
#     ):
#         self.model_name = model_name
#         self.embedding_batch_size = embedding_batch_size
#         self.bm25_params = bm25_params or BM25Params()
#
#     def load_processed_docs(self, processed_docs_path: Path) -> list[dict]:
#         if not processed_docs_path.exists():
#             raise FileNotFoundError(
#                 f"Missing processed docs file: {processed_docs_path}. "
#                 "Run scripts/preprocess_datasets.py first."
#             )
#
#         docs: list[dict] = []
#         with processed_docs_path.open("r", encoding="utf-8") as infile:
#             for line_number, line in enumerate(infile, start=1):
#                 if not line.strip():
#                     continue
#                 doc = json.loads(line)
#                 required = ["doc_id", "raw_text", "lexical_tokens", "lexical_text", "embedding_text"]
#                 missing = [field for field in required if field not in doc]
#                 if missing:
#                     raise ValueError(
#                         f"Missing fields {missing} in {processed_docs_path}, line {line_number}"
#                     )
#                 docs.append(doc)
#
#         if not docs:
#             raise ValueError(f"No processed documents found in: {processed_docs_path}")
#
#         return docs
#
#     def build(self, processed_docs_path: Path, output_dir: Path) -> Path:
#         output_dir.mkdir(parents=True, exist_ok=True)
#         docs = self.load_processed_docs(processed_docs_path)
#
#         doc_ids = [str(doc["doc_id"]) for doc in docs]
#         raw_texts = [doc["raw_text"] for doc in docs]
#         lexical_texts = [doc["lexical_text"] for doc in docs]
#         lexical_tokens = [doc["lexical_tokens"] for doc in docs]
#         embedding_texts = [doc["embedding_text"] for doc in docs]
#
#         # 1) VSM TF-IDF representation.
#         tfidf_vectorizer = TfidfVectorizer(
#             tokenizer=split_on_space,
#             preprocessor=None,
#             token_pattern=None,
#             lowercase=False,
#             norm="l2",
#         )
#         tfidf_matrix = tfidf_vectorizer.fit_transform(lexical_texts)
#
#         # 2) BM25 representation.
#         bm25 = BM25Okapi(
#             lexical_tokens,
#             k1=self.bm25_params.k1,
#             b=self.bm25_params.b,
#         )
#
#         # 3) Embedding representation.
#         embedding_model = SentenceTransformer(self.model_name)
#         embeddings = embedding_model.encode(
#             embedding_texts,
#             batch_size=self.embedding_batch_size,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#         ).astype("float32")
#
#         docs_store_path = output_dir / "docs_store.jsonl"
#         with docs_store_path.open("w", encoding="utf-8") as outfile:
#             for doc_id, raw_text in zip(doc_ids, raw_texts):
#                 outfile.write(
#                     json.dumps(
#                         {"doc_id": doc_id, "raw_text": raw_text},
#                         ensure_ascii=False,
#                     )
#                     + "\n"
#                 )
#
#         joblib.dump(tfidf_vectorizer, output_dir / "tfidf_vectorizer.joblib")
#         joblib.dump(tfidf_matrix, output_dir / "tfidf_matrix.joblib")
#         joblib.dump(bm25, output_dir / "bm25.joblib")
#         joblib.dump(lexical_tokens, output_dir / "tokenized_corpus.joblib")
#         np.save(output_dir / "embeddings.npy", embeddings)
#
#         metadata = {
#             "num_documents": len(docs),
#             "embedding_model": self.model_name,
#             "embedding_dimension": int(embeddings.shape[1]),
#             "bm25_k1": self.bm25_params.k1,
#             "bm25_b": self.bm25_params.b,
#             "files": {
#                 "docs_store": "docs_store.jsonl",
#                 "tfidf_vectorizer": "tfidf_vectorizer.joblib",
#                 "tfidf_matrix": "tfidf_matrix.joblib",
#                 "bm25": "bm25.joblib",
#                 "tokenized_corpus": "tokenized_corpus.joblib",
#                 "embeddings": "embeddings.npy",
#             },
#         }
#         with (output_dir / "metadata.json").open("w", encoding="utf-8") as outfile:
#             json.dump(metadata, outfile, ensure_ascii=False, indent=2)
#
#         return output_dir
#
#
# class RetrievalService:
#     """
#     Loads a dataset index and retrieves ranked results using:
#         - TF-IDF cosine similarity
#         - BM25
#         - Embedding cosine similarity
#         - Hybrid serial retrieval
#         - Hybrid parallel fusion
#     """
#
#     def __init__(
#         self,
#         index_dir: Path,
#         preprocessor: PreprocessingService | None = None,
#     ):
#         self.index_dir = index_dir
#         self.preprocessor = preprocessor or PreprocessingService()
#
#         metadata_path = index_dir / "metadata.json"
#         if not metadata_path.exists():
#             raise FileNotFoundError(f"Missing index metadata: {metadata_path}")
#
#         with metadata_path.open("r", encoding="utf-8") as infile:
#             self.metadata = json.load(infile)
#
#         self.docs = self._load_docs(index_dir / self.metadata["files"]["docs_store"])
#         self.doc_ids = [doc["doc_id"] for doc in self.docs]
#         self.raw_texts = [doc["raw_text"] for doc in self.docs]
#
#         self.tfidf_vectorizer = joblib.load(index_dir / self.metadata["files"]["tfidf_vectorizer"])
#         self.tfidf_matrix = joblib.load(index_dir / self.metadata["files"]["tfidf_matrix"])
#         self.bm25 = joblib.load(index_dir / self.metadata["files"]["bm25"])
#         self.tokenized_corpus = joblib.load(index_dir / self.metadata["files"]["tokenized_corpus"])
#         self.embeddings = np.load(index_dir / self.metadata["files"]["embeddings"])
#
#         self.embedding_model = SentenceTransformer(self.metadata["embedding_model"])
#
#     def _load_docs(self, docs_store_path: Path) -> list[dict]:
#         docs = []
#         with docs_store_path.open("r", encoding="utf-8") as infile:
#             for line in infile:
#                 if line.strip():
#                     docs.append(json.loads(line))
#         return docs
#
#     def _preprocess_query(self, query_text: str) -> dict:
#         return self.preprocessor.preprocess_query("interactive_query", query_text)
#
#     def _tfidf_scores(self, processed_query: dict) -> np.ndarray:
#         query_vector = self.tfidf_vectorizer.transform([processed_query["lexical_text"]])
#         scores = self.tfidf_matrix @ query_vector.T
#         return np.asarray(scores.toarray()).ravel()
#
#     def _bm25_scores(self, processed_query: dict, k1: float | None = None, b: float | None = None) -> np.ndarray:
#         # If no custom BM25 parameters are passed, use the saved BM25 object.
#         if k1 is None and b is None:
#             return np.asarray(self.bm25.get_scores(processed_query["lexical_tokens"]), dtype="float32")
#
#         # If custom parameters are passed, rebuild BM25 from the stored corpus tokens.
#         # rank_bm25 does not expose changing k1/b after object creation.
#         custom_bm25 = BM25Okapi(
#             self.tokenized_corpus,
#             k1=float(k1 if k1 is not None else self.metadata["bm25_k1"]),
#             b=float(b if b is not None else self.metadata["bm25_b"]),
#         )
#         return np.asarray(custom_bm25.get_scores(processed_query["lexical_tokens"]), dtype="float32")
#
#     def _embedding_scores(self, processed_query: dict) -> np.ndarray:
#         query_embedding = self.embedding_model.encode(
#             [processed_query["embedding_text"]],
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#         ).astype("float32")[0]
#         return self.embeddings @ query_embedding
#
#     def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
#         scores = np.asarray(scores, dtype="float32")
#         min_score = float(scores.min())
#         max_score = float(scores.max())
#         if max_score == min_score:
#             return np.zeros_like(scores, dtype="float32")
#         return (scores - min_score) / (max_score - min_score)
#
#     def _top_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
#         if top_k >= len(scores):
#             return np.argsort(-scores)
#         candidate_indices = np.argpartition(-scores, top_k)[:top_k]
#         return candidate_indices[np.argsort(-scores[candidate_indices])]
#
#     def _format_results(
#         self,
#         final_scores: np.ndarray,
#         top_k: int,
#         source_scores: dict[str, np.ndarray] | None = None,
#     ) -> list[SearchResult]:
#         source_scores = source_scores or {}
#         ranked_indices = self._top_indices(final_scores, top_k)
#
#         results = []
#         for rank, idx in enumerate(ranked_indices, start=1):
#             per_doc_scores = {
#                 name: float(scores[idx])
#                 for name, scores in source_scores.items()
#             }
#             results.append(
#                 SearchResult(
#                     doc_id=self.doc_ids[idx],
#                     score=float(final_scores[idx]),
#                     raw_text=self.raw_texts[idx],
#                     rank=rank,
#                     source_scores=per_doc_scores,
#                 )
#             )
#         return results
#
#     def search(
#         self,
#         query_text: str,
#         mode: RetrievalMode = "bm25",
#         top_k: int = 10,
#         serial_candidate_k: int = 100,
#         hybrid_weights: HybridWeights | None = None,
#         bm25_k1: float | None = None,
#         bm25_b: float | None = None,
#     ) -> list[SearchResult]:
#         processed_query = self._preprocess_query(query_text)
#         weights = hybrid_weights or HybridWeights()
#
#         if mode == "tfidf":
#             tfidf_scores = self._tfidf_scores(processed_query)
#             return self._format_results(tfidf_scores, top_k, {"tfidf": tfidf_scores})
#
#         if mode == "bm25":
#             bm25_scores = self._bm25_scores(processed_query, k1=bm25_k1, b=bm25_b)
#             return self._format_results(bm25_scores, top_k, {"bm25": bm25_scores})
#
#         if mode == "embedding":
#             embedding_scores = self._embedding_scores(processed_query)
#             return self._format_results(embedding_scores, top_k, {"embedding": embedding_scores})
#
#         if mode == "hybrid_serial":
#             # Serial hybrid: retrieve candidate documents with BM25 first,
#             # then rerank only those candidates with embedding similarity.
#             bm25_scores = self._bm25_scores(processed_query, k1=bm25_k1, b=bm25_b)
#             embedding_scores = self._embedding_scores(processed_query)
#
#             candidate_k = min(max(serial_candidate_k, top_k), len(self.docs))
#             candidate_indices = self._top_indices(bm25_scores, candidate_k)
#
#             final_scores = np.full(len(self.docs), -np.inf, dtype="float32")
#             normalized_bm25 = self._minmax_normalize(bm25_scores)
#             normalized_embedding = self._minmax_normalize(embedding_scores)
#
#             # Rerank candidates using mostly semantic similarity, while keeping some BM25 signal.
#             final_scores[candidate_indices] = (
#                 0.30 * normalized_bm25[candidate_indices]
#                 + 0.70 * normalized_embedding[candidate_indices]
#             )
#
#             return self._format_results(
#                 final_scores,
#                 top_k,
#                 {"bm25": bm25_scores, "embedding": embedding_scores},
#             )
#
#         if mode == "hybrid_parallel":
#             # Parallel hybrid: run all retrieval models independently,
#             # normalize their scores, then fuse them using weighted sum.
#             tfidf_scores = self._tfidf_scores(processed_query)
#             bm25_scores = self._bm25_scores(processed_query, k1=bm25_k1, b=bm25_b)
#             embedding_scores = self._embedding_scores(processed_query)
#
#             final_scores = (
#                 weights.tfidf * self._minmax_normalize(tfidf_scores)
#                 + weights.bm25 * self._minmax_normalize(bm25_scores)
#                 + weights.embedding * self._minmax_normalize(embedding_scores)
#             )
#
#             return self._format_results(
#                 final_scores,
#                 top_k,
#                 {
#                     "tfidf": tfidf_scores,
#                     "bm25": bm25_scores,
#                     "embedding": embedding_scores,
#                 },
#             )
#
#         raise ValueError(f"Unsupported retrieval mode: {mode}")
