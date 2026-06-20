from typing import Literal

from pydantic import BaseModel, Field


DatasetName = Literal["argsme_touche2022", "clinicaltrials_2021"]
RetrievalMode = Literal["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]


class HybridWeightsRequest(BaseModel):
    tfidf: float = Field(default=0.30, ge=0.0, le=1.0)
    bm25: float = Field(default=0.35, ge=0.0, le=1.0)
    embedding: float = Field(default=0.35, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    dataset: DatasetName
    query: str = Field(..., min_length=1, examples=["climate change policy"])
    mode: RetrievalMode = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)

    # Used only by hybrid_serial.
    serial_candidate_k: int = Field(default=100, ge=1, le=10000)

    # Used only by hybrid_parallel.
    fusion_pool_k: int = Field(default=1000, ge=1, le=50000)
    weights: HybridWeightsRequest = Field(default_factory=HybridWeightsRequest)

    # Optional BM25 parameters. If omitted, index defaults are used.
    bm25_k1: float | None = Field(default=None, gt=0.0, le=5.0)
    bm25_b: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceScoresResponse(BaseModel):
    tfidf: float | None = None
    bm25: float | None = None
    embedding: float | None = None


class SearchResultResponse(BaseModel):
    rank: int
    doc_id: str
    score: float
    raw_text: str
    source_scores: dict[str, float]


class SearchResponse(BaseModel):
    dataset: str
    mode: str
    query: str
    top_k: int
    count: int
    results: list[SearchResultResponse]


class DatasetStatus(BaseModel):
    name: str
    ready: bool
    index_dir: str
    metadata_exists: bool
    sqlite_exists: bool
    embeddings_manifest_exists: bool
    tfidf_norms_exists: bool
    num_documents: int | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None


class HealthResponse(BaseModel):
    status: str
    project_root: str
    available_datasets: list[str]
