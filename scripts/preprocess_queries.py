import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.preprocessing_service import PreprocessingService


DATASETS = [
    PROJECT_ROOT / "data" / "argsme_touche2022",
    PROJECT_ROOT / "data" / "clinicaltrials_2021",
]
PROCESSED_ROOT = PROJECT_ROOT / "processed"

QUERY_ID_FIELDS = ("query_id", "qid", "id", "_id")
QUERY_TEXT_FIELDS = ("text", "query", "title", "contents", "description")


def get_query_id(query: dict) -> str:
    for key in QUERY_ID_FIELDS:
        value = query.get(key)
        if value is not None and str(value).strip():
            return str(value)

    raise ValueError(f"Could not find query id in fields: {sorted(query.keys())}")


def extract_query_text(query: dict) -> str:
    parts = []

    for field in QUERY_TEXT_FIELDS:
        value = query.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if parts:
        return " ".join(parts)

    raise ValueError(f"Could not find query text in fields: {sorted(query.keys())}")


def iter_query_batches(input_path: Path, batch_size: int, limit: int | None = None):
    batch = []
    seen = 0

    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if limit is not None and seen >= limit:
                break
            if not line.strip():
                continue

            try:
                query = json.loads(line)
                batch.append((get_query_id(query), extract_query_text(query)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {input_path} line {line_number}") from exc

            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def preprocess_queries(
    dataset_dir: Path,
    preprocessor: PreprocessingService,
    batch_size: int = 64,
    limit: int | None = None,
) -> Path:
    input_path = dataset_dir / "queries.jsonl"
    output_dir = PROCESSED_ROOT / dataset_dir.name
    output_path = output_dir / "processed_queries.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as outfile:
        batches = iter_query_batches(input_path, batch_size=batch_size, limit=limit)

        for batch in tqdm(batches, desc=f"Preprocessing queries for {dataset_dir.name}"):
            for processed_query in preprocessor.preprocess_queries_batch(
                batch,
                batch_size=batch_size,
            ):
                outfile.write(json.dumps(processed_query, ensure_ascii=False) + "\n")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset queries.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of queries per dataset for smoke testing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of records to process per spaCy batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor = PreprocessingService()

    for dataset_dir in DATASETS:
        output_path = preprocess_queries(
            dataset_dir,
            preprocessor,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(f"Saved processed queries to: {output_path}")


if __name__ == "__main__":
    main()
