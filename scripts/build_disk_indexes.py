import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.disk_index_models import BM25Params
from services.indexing_service import IndexingService

DATASETS = [
    "argsme_touche2022",
    "clinicaltrials_2021",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build memory-safe disk indexes for TF-IDF, BM25, and embeddings."
    )
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used for embedding representation.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size inside SentenceTransformer.encode. Lower this if RAM is tight.",
    )
    parser.add_argument(
        "--embedding-chunk-docs",
        type=int,
        default=2048,
        help="How many document embeddings to write per .npy chunk.",
    )
    parser.add_argument(
        "--commit-every-docs",
        type=int,
        default=1000,
        help="How often to flush SQLite insert batches to disk.",
    )
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = DATASETS if args.dataset == "all" else [args.dataset]

    builder = IndexingService(
        model_name=args.model_name,
        embedding_batch_size=args.embedding_batch_size,
        embedding_chunk_docs=args.embedding_chunk_docs,
        commit_every_docs=args.commit_every_docs,
        bm25_params=BM25Params(k1=args.bm25_k1, b=args.bm25_b),
    )

    for dataset_name in selected:
        processed_docs_path = PROJECT_ROOT / "processed" / dataset_name / "processed_docs.jsonl"
        output_dir = PROJECT_ROOT / "indexes" / dataset_name
        builder.build(processed_docs_path=processed_docs_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
