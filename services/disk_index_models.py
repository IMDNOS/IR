from dataclasses import dataclass
from typing import Literal


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
