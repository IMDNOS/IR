# import argparse
# import sys
# from pathlib import Path
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
#
# from services.representation_service import HybridWeights, RetrievalService
#
# DATASETS = [
#     "argsme_touche2022",
#     "clinicaltrials_2021",
# ]
# MODES = [
#     "tfidf",
#     "bm25",
#     "embedding",
#     "hybrid_serial",
#     "hybrid_parallel",
# ]
#
#
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Search demo for representation models.")
#     parser.add_argument("--dataset", required=True, choices=DATASETS)
#     parser.add_argument("--query", required=True)
#     parser.add_argument("--mode", default="bm25", choices=MODES)
#     parser.add_argument("--top-k", type=int, default=10)
#     parser.add_argument("--serial-candidate-k", type=int, default=100)
#
#     # BM25 parameters are exposed for the UI/report requirement.
#     parser.add_argument("--bm25-k1", type=float, default=None)
#     parser.add_argument("--bm25-b", type=float, default=None)
#
#     # Parallel hybrid fusion weights.
#     parser.add_argument("--tfidf-weight", type=float, default=0.30)
#     parser.add_argument("--bm25-weight", type=float, default=0.35)
#     parser.add_argument("--embedding-weight", type=float, default=0.35)
#     return parser.parse_args()
#
#
# def main() -> None:
#     args = parse_args()
#     index_dir = PROJECT_ROOT / "indexes" / args.dataset
#     service = RetrievalService(index_dir=index_dir)
#
#     weights = HybridWeights(
#         tfidf=args.tfidf_weight,
#         bm25=args.bm25_weight,
#         embedding=args.embedding_weight,
#     )
#
#     results = service.search(
#         query_text=args.query,
#         mode=args.mode,
#         top_k=args.top_k,
#         serial_candidate_k=args.serial_candidate_k,
#         hybrid_weights=weights,
#         bm25_k1=args.bm25_k1,
#         bm25_b=args.bm25_b,
#     )
#
#     print(f"\nDataset: {args.dataset}")
#     print(f"Mode: {args.mode}")
#     print(f"Query: {args.query}\n")
#
#     for result in results:
#         preview = result.raw_text.replace("\n", " ")[:250]
#         print(f"#{result.rank} | doc_id={result.doc_id} | score={result.score:.4f}")
#         print(f"source_scores={result.source_scores}")
#         print(preview)
#         print("-" * 100)
#
#
# if __name__ == "__main__":
#     main()
