# import argparse
# import sys
# from pathlib import Path
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
#
# from services.representation_service import BM25Params, RepresentationIndexBuilder
#
# DATASETS = [
#     "argsme_touche2022",
#     "clinicaltrials_2021",
# ]
#
#
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Build TF-IDF, BM25, and embedding representations for processed datasets."
#     )
#     parser.add_argument(
#         "--dataset",
#         choices=DATASETS + ["all"],
#         default="all",
#         help="Dataset to index, or all datasets.",
#     )
#     parser.add_argument(
#         "--model-name",
#         default="sentence-transformers/all-MiniLM-L6-v2",
#         help="SentenceTransformer model used for embedding representation.",
#     )
#     parser.add_argument(
#         "--embedding-batch-size",
#         type=int,
#         default=64,
#         help="Batch size for embedding encoding.",
#     )
#     parser.add_argument(
#         "--bm25-k1",
#         type=float,
#         default=1.5,
#         help="BM25 k1 parameter. Default 1.5.",
#     )
#     parser.add_argument(
#         "--bm25-b",
#         type=float,
#         default=0.75,
#         help="BM25 b parameter. Default 0.75.",
#     )
#     return parser.parse_args()
#
#
# def main() -> None:
#     args = parse_args()
#     selected_datasets = DATASETS if args.dataset == "all" else [args.dataset]
#
#     builder = RepresentationIndexBuilder(
#         model_name=args.model_name,
#         embedding_batch_size=args.embedding_batch_size,
#         bm25_params=BM25Params(k1=args.bm25_k1, b=args.bm25_b),
#     )
#
#     for dataset_name in selected_datasets:
#         processed_docs_path = (
#             PROJECT_ROOT / "processed" / dataset_name / "processed_docs.jsonl"
#         )
#         output_dir = PROJECT_ROOT / "indexes" / dataset_name
#
#         built_dir = builder.build(
#             processed_docs_path=processed_docs_path,
#             output_dir=output_dir,
#         )
#         print(f"Built representations for {dataset_name}: {built_dir}")
#
#
# if __name__ == "__main__":
#     main()
