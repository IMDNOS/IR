import sys
from pathlib import Path

# Allow running from the project root using:
# uvicorn api.main:app --reload
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.retrieval_manager import DATASETS, MODES, PROJECT_ROOT, RetrievalServiceManager
from api.schemas import (
    EvaluationRecordResponse,
    EvaluationRunRequest,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    RefinementInfo,
    SearchResultResponse,
)
from services.evaluation_service import EvaluationService

app = FastAPI(
    title="IR Project Search API",
    description="REST API for TF-IDF, BM25, embedding, hybrid serial, and hybrid parallel retrieval.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = RetrievalServiceManager()
evaluation_service = EvaluationService(
    data_root=PROJECT_ROOT / "data",
    evaluations_root=PROJECT_ROOT / "evaluations",
    manager=manager,
)


@app.on_event("shutdown")
def shutdown_event() -> None:
    manager.close_all()


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "message": "IR Project Search API is running.",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        project_root=str(PROJECT_ROOT),
        available_datasets=DATASETS,
    )


@app.get("/api/v1/datasets", tags=["metadata"])
def datasets() -> dict:
    return {
        "datasets": [status.dict() for status in manager.all_dataset_statuses()]
    }


@app.get("/api/v1/modes", tags=["metadata"])
def modes() -> dict:
    return {
        "modes": MODES,
        "notes": {
            "tfidf": "Classical vector-space model using TF-IDF cosine similarity.",
            "bm25": "Probabilistic lexical ranking. Optional bm25_k1 and bm25_b can be changed at search time.",
            "embedding": "Dense semantic search using SentenceTransformer embeddings.",
            "hybrid_serial": "BM25 retrieves candidates, then embeddings rerank them.",
            "hybrid_parallel": "TF-IDF, BM25, and embeddings run independently, then scores are fused.",
        },
    }


@app.post("/api/v1/search", response_model=SearchResponse, tags=["search"])
def search(request: SearchRequest) -> SearchResponse:
    refinement_info = None
    original_results = None
    query_to_search = request.query

    # Check if any refinement is enabled
    has_refinement = (
        request.enable_spelling_correction
        or request.enable_synonym_expansion
        or request.enable_search_history
    )

    # Apply query refinement if enabled
    if has_refinement:
        try:
            refined = manager.query_refinement_service.refine(
                query=request.query,
                dataset=request.dataset,
                enable_spelling_correction=request.enable_spelling_correction,
                enable_synonym_expansion=request.enable_synonym_expansion,
                enable_search_history=request.enable_search_history,
            )
            refinement_info = RefinementInfo(
                original_query=refined.original_query,
                corrected_query=refined.corrected_query,
                expanded_query=refined.expanded_query,
                history_boosted_query=refined.history_boosted_query,
                final_query=refined.final_query,
                refinement_log=refined.refinement_log,
                applied_refinements=refined.applied_refinements,
            )
            query_to_search = refined.final_query

            # If show_original_results is enabled, search with original query too
            if request.show_original_results:
                try:
                    original_request = SearchRequest(
                        dataset=request.dataset,
                        query=request.query,
                        mode=request.mode,
                        top_k=request.top_k,
                        serial_candidate_k=request.serial_candidate_k,
                        fusion_pool_k=request.fusion_pool_k,
                        weights=request.weights,
                        bm25_k1=request.bm25_k1,
                        bm25_b=request.bm25_b,
                    )
                    original_search_results = manager.search(original_request)
                    original_results = [
                        SearchResultResponse(
                            rank=result.rank,
                            doc_id=result.doc_id,
                            score=result.score,
                            raw_text=result.raw_text,
                            source_scores=result.source_scores,
                        )
                        for result in original_search_results
                    ]
                except Exception:
                    original_results = None

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Query refinement failed: {exc}") from exc

    # Search with refined query (or original if no refinement)
    modified_request = SearchRequest(
        dataset=request.dataset,
        query=query_to_search,
        mode=request.mode,
        top_k=request.top_k,
        serial_candidate_k=request.serial_candidate_k,
        fusion_pool_k=request.fusion_pool_k,
        weights=request.weights,
        bm25_k1=request.bm25_k1,
        bm25_b=request.bm25_b,
    )

    try:
        results = manager.search(modified_request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    # Record search in history if refinement was applied
    if has_refinement and results:
        try:
            top_doc_ids = [result.doc_id for result in results[:10]]
            manager.query_refinement_service.record_search(query_to_search, request.dataset, top_doc_ids)
        except Exception:
            pass  # Don't fail the search if history recording fails

    return SearchResponse(
        dataset=request.dataset,
        mode=request.mode,
        query=query_to_search,
        top_k=request.top_k,
        count=len(results),
        results=[
            SearchResultResponse(
                rank=result.rank,
                doc_id=result.doc_id,
                score=result.score,
                raw_text=result.raw_text,
                source_scores=result.source_scores,
            )
            for result in results
        ],
        refinement_info=refinement_info,
        original_results=original_results,
    )


@app.post("/api/v1/evaluations", tags=["evaluation"])
def run_evaluation(request: EvaluationRunRequest) -> dict:
    try:
        record = evaluation_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    if isinstance(record, list):
        return {"evaluations": [EvaluationRecordResponse(**item.__dict__).model_dump() for item in record]}
    return {"evaluations": [EvaluationRecordResponse(**record.__dict__).model_dump()]}


@app.get("/api/v1/evaluations", tags=["evaluation"])
def list_evaluations(dataset: str | None = None) -> dict:
    return {"evaluations": evaluation_service.list_runs(dataset=dataset)}

# @app.get("/api/v1/search", response_model=SearchResponse, tags=["search"])
# def search_get(
#     dataset: str = Query(..., examples=["argsme_touche2022"]),
#     query: str = Query(..., min_length=1, examples=["climate change policy"]),
#     mode: str = Query("bm25", examples=["bm25"]),
#     top_k: int = Query(10, ge=1, le=100),
#     serial_candidate_k: int = Query(100, ge=1, le=10000),
#     fusion_pool_k: int = Query(1000, ge=1, le=50000),
#     bm25_k1: float | None = Query(None, gt=0.0, le=5.0),
#     bm25_b: float | None = Query(None, ge=0.0, le=1.0),
#     tfidf_weight: float = Query(0.30, ge=0.0, le=1.0),
#     bm25_weight: float = Query(0.35, ge=0.0, le=1.0),
#     embedding_weight: float = Query(0.35, ge=0.0, le=1.0),
# ) -> SearchResponse:
#     if dataset not in DATASETS:
#         raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
#     if mode not in MODES:
#         raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
#
#     request = SearchRequest(
#         dataset=dataset,  # type: ignore[arg-type]
#         query=query,
#         mode=mode,  # type: ignore[arg-type]
#         top_k=top_k,
#         serial_candidate_k=serial_candidate_k,
#         fusion_pool_k=fusion_pool_k,
#         bm25_k1=bm25_k1,
#         bm25_b=bm25_b,
#         weights={
#             "tfidf": tfidf_weight,
#             "bm25": bm25_weight,
#             "embedding": embedding_weight,
#         },
#     )
#     return search(request)
