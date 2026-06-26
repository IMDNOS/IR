import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.disk_index_models import HybridWeights
from services.search_service import SearchService
from services.query_refinement_service import QueryRefinementService

from api.schemas import DatasetStatus, SearchRequest


INDEXES_ROOT = PROJECT_ROOT / "indexes"
DATASETS = ["argsme_touche2022", "clinicaltrials_2021"]
MODES = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]


class RetrievalServiceManager:
    """
    Keeps one SearchService instance per dataset.

    This avoids loading spaCy + SentenceTransformer on every request.
    The first search for a dataset may be slower because the model is loaded.
    Later searches reuse the same service instance.
    """

    def __init__(self) -> None:
        self._services: dict[str, SearchService] = {}
        self.query_refinement_service = QueryRefinementService(PROJECT_ROOT / "data")

    def get_service(self, dataset: str) -> SearchService:
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if dataset not in self._services:
            index_dir = INDEXES_ROOT / dataset
            if not self.is_dataset_ready(dataset):
                raise FileNotFoundError(
                    f"Index for dataset '{dataset}' is not ready. Expected files under: {index_dir}"
                )
            self._services[dataset] = SearchService(index_dir=index_dir)

        return self._services[dataset]

    def search(self, request: SearchRequest):
        service = self.get_service(request.dataset)
        return service.search(
            query_text=request.query,
            mode=request.mode,
            top_k=request.top_k,
            serial_candidate_k=request.serial_candidate_k,
            serial_bm25_weight=request.serial_bm25_weight,
            serial_embedding_weight=request.serial_embedding_weight,
            fusion_pool_k=request.fusion_pool_k,
            hybrid_weights=HybridWeights(
                tfidf=request.weights.tfidf,
                bm25=request.weights.bm25,
                embedding=request.weights.embedding,
            ),
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
        )

    def close_all(self) -> None:
        for service in self._services.values():
            service.close()
        self._services.clear()

    def is_dataset_ready(self, dataset: str) -> bool:
        status = self.dataset_status(dataset)
        return status.ready

    def dataset_status(self, dataset: str) -> DatasetStatus:
        index_dir = INDEXES_ROOT / dataset
        metadata_path = index_dir / "metadata.json"
        metadata_exists = metadata_path.exists()

        sqlite_exists = False
        embeddings_manifest_exists = False
        tfidf_norms_exists = False
        num_documents = None
        embedding_model = None
        embedding_dimension = None

        if metadata_exists:
            try:
                with metadata_path.open("r", encoding="utf-8") as infile:
                    metadata = json.load(infile)
                files = metadata.get("files", {})
                sqlite_exists = (index_dir / files.get("sqlite", "index.sqlite3")).exists()
                embeddings_manifest_exists = (
                    index_dir / files.get("embedding_chunks", "embedding_chunks.json")
                ).exists()
                tfidf_norms_exists = (
                    index_dir / files.get("tfidf_doc_norms", "tfidf_doc_norms.float32.npy")
                ).exists()
                num_documents = metadata.get("num_documents")
                embedding_model = metadata.get("embedding_model")
                embedding_dimension = metadata.get("embedding_dimension")
            except Exception:
                sqlite_exists = (index_dir / "index.sqlite3").exists()
                embeddings_manifest_exists = (index_dir / "embedding_chunks.json").exists()
                tfidf_norms_exists = (index_dir / "tfidf_doc_norms.float32.npy").exists()
        else:
            sqlite_exists = (index_dir / "index.sqlite3").exists()
            embeddings_manifest_exists = (index_dir / "embedding_chunks.json").exists()
            tfidf_norms_exists = (index_dir / "tfidf_doc_norms.float32.npy").exists()

        ready = metadata_exists and sqlite_exists and embeddings_manifest_exists and tfidf_norms_exists

        return DatasetStatus(
            name=dataset,
            ready=ready,
            index_dir=str(index_dir),
            metadata_exists=metadata_exists,
            sqlite_exists=sqlite_exists,
            embeddings_manifest_exists=embeddings_manifest_exists,
            tfidf_norms_exists=tfidf_norms_exists,
            num_documents=num_documents,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

    def all_dataset_statuses(self) -> list[DatasetStatus]:
        return [self.dataset_status(dataset) for dataset in DATASETS]
