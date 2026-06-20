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
from api.schemas import HealthResponse, SearchRequest, SearchResponse


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
    try:
        results = manager.search(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return SearchResponse(
        dataset=request.dataset,
        mode=request.mode,
        query=request.query,
        top_k=request.top_k,
        count=len(results),
        results=[
            {
                "rank": result.rank,
                "doc_id": result.doc_id,
                "score": result.score,
                "raw_text": result.raw_text,
                "source_scores": result.source_scores,
            }
            for result in results
        ],
    )


@app.get("/api/v1/search", response_model=SearchResponse, tags=["search"])
def search_get(
    dataset: str = Query(..., examples=["argsme_touche2022"]),
    query: str = Query(..., min_length=1, examples=["climate change policy"]),
    mode: str = Query("bm25", examples=["bm25"]),
    top_k: int = Query(10, ge=1, le=100),
    serial_candidate_k: int = Query(100, ge=1, le=10000),
    fusion_pool_k: int = Query(1000, ge=1, le=50000),
    bm25_k1: float | None = Query(None, gt=0.0, le=5.0),
    bm25_b: float | None = Query(None, ge=0.0, le=1.0),
    tfidf_weight: float = Query(0.30, ge=0.0, le=1.0),
    bm25_weight: float = Query(0.35, ge=0.0, le=1.0),
    embedding_weight: float = Query(0.35, ge=0.0, le=1.0),
) -> SearchResponse:
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    request = SearchRequest(
        dataset=dataset,  # type: ignore[arg-type]
        query=query,
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
        serial_candidate_k=serial_candidate_k,
        fusion_pool_k=fusion_pool_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        weights={
            "tfidf": tfidf_weight,
            "bm25": bm25_weight,
            "embedding": embedding_weight,
        },
    )
    return search(request)
