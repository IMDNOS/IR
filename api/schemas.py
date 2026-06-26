from typing import Literal

from pydantic import BaseModel, Field, model_validator


DatasetName = Literal["argsme_touche2022", "clinicaltrials_2021"]
RetrievalMode = Literal["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]


class HybridWeightsRequest(BaseModel):
    tfidf: float = Field(default=0.30, ge=0.0, le=1.0)
    bm25: float = Field(default=0.35, ge=0.0, le=1.0)
    embedding: float = Field(default=0.35, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "HybridWeightsRequest":
        total = self.tfidf + self.bm25 + self.embedding
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Hybrid weights must sum to 1.00. Current sum: {total:.2f}")
        return self


class SearchRequest(BaseModel):
    dataset: DatasetName
    query: str = Field(..., min_length=1, examples=["climate change policy"])
    mode: RetrievalMode = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)

    # Used only by hybrid_serial.
    serial_candidate_k: int = Field(default=100, ge=1, le=10000)
    serial_bm25_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    serial_embedding_weight: float = Field(default=0.70, ge=0.0, le=1.0)

    # Used only by hybrid_parallel.
    fusion_pool_k: int = Field(default=1000, ge=1, le=50000)
    weights: HybridWeightsRequest = Field(default_factory=HybridWeightsRequest)

    # Optional BM25 parameters. If omitted, index defaults are used.
    bm25_k1: float | None = Field(default=None, gt=0.0, le=5.0)
    bm25_b: float | None = Field(default=None, ge=0.0, le=1.0)

    # Query refinement toggles
    enable_spelling_correction: bool = Field(default=False)
    enable_synonym_expansion: bool = Field(default=False)
    enable_search_history: bool = Field(default=False)
    show_original_results: bool = Field(default=False)

    @model_validator(mode="after")
    def serial_weights_must_not_both_be_zero(self) -> "SearchRequest":
        if self.serial_bm25_weight + self.serial_embedding_weight <= 0.0:
            raise ValueError("Hybrid serial weights must not both be zero.")
        return self


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


class RefinementInfo(BaseModel):
    original_query: str
    corrected_query: str | None = None
    expanded_query: str | None = None
    history_boosted_query: str | None = None
    final_query: str
    refinement_log: list[str]
    applied_refinements: list[str]


class SearchResponse(BaseModel):
    dataset: str
    mode: str
    query: str
    top_k: int
    count: int
    results: list[SearchResultResponse]
    refinement_info: RefinementInfo | None = None
    original_results: list[SearchResultResponse] | None = None


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


class EvaluationRunRequest(BaseModel):
    dataset: DatasetName
    mode: RetrievalMode | str = "all"
    top_k: int = Field(default=10, ge=10, le=100)
    serial_candidate_k: int = Field(default=100, ge=1, le=10000)
    serial_bm25_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    serial_embedding_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    fusion_pool_k: int = Field(default=1000, ge=1, le=50000)
    weights: HybridWeightsRequest = Field(default_factory=HybridWeightsRequest)
    bm25_k1: float | None = Field(default=None, gt=0.0, le=5.0)
    bm25_b: float | None = Field(default=None, ge=0.0, le=1.0)
    enable_spelling_correction: bool = Field(default=False)
    enable_synonym_expansion: bool = Field(default=False)
    enable_search_history: bool = Field(default=False)

    @model_validator(mode="after")
    def serial_weights_must_not_both_be_zero(self) -> "EvaluationRunRequest":
        if self.serial_bm25_weight + self.serial_embedding_weight <= 0.0:
            raise ValueError("Hybrid serial weights must not both be zero.")
        return self

    def has_refinements(self) -> bool:
        return (
            self.enable_spelling_correction
            or self.enable_synonym_expansion
            or self.enable_search_history
        )


class EvaluationRecordResponse(BaseModel):
    evaluation_id: str
    dataset: str
    mode: str
    created_at: str
    file_path: str
    top_k: int
    metrics: dict[str, float]
    num_queries: int
    avg_relevant_docs: float
    refinements_enabled: bool
    enabled_refinements: list[str] = Field(default_factory=list)
    bm25_k1: float | None = None
    bm25_b: float | None = None
    serial_bm25_weight: float | None = None
    serial_embedding_weight: float | None = None
    query_source: str = "title_description"
