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

DOCUMENT_ID_FIELDS = ("doc_id", "id", "_id", "document_id")
DOCUMENT_TEXT_FIELDS = (
    "title",
    "text",
    "contents",
    "body",
    "abstract",
    "summary",
    "detailed_description",
    "condition",
    "eligibility",
    "description",
    "source_title",
    "topic",
    "premises_texts",
    "conclusion",
    "aspects_names",
)


def get_doc_id(doc: dict) -> str:
    for key in DOCUMENT_ID_FIELDS:
        value = doc.get(key)
        if value is not None and str(value).strip():
            return str(value)

    raise ValueError(f"Could not find document id in fields: {sorted(doc.keys())}")


def extract_text_from_doc(doc: dict) -> str:
    parts = []

    for field in DOCUMENT_TEXT_FIELDS:
        value = doc.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if parts:
        return " ".join(parts)

    raise ValueError(f"Could not find document text in fields: {sorted(doc.keys())}")


def iter_document_batches(input_path: Path, batch_size: int, limit: int | None = None):
    batch = []
    seen = 0

    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if limit is not None and seen >= limit:
                break
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                batch.append((get_doc_id(doc), extract_text_from_doc(doc)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {input_path} line {line_number}") from exc

            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def preprocess_jsonl_dataset(
    dataset_dir: Path,
    preprocessor: PreprocessingService,
    batch_size: int = 64,
    limit: int | None = None,
) -> Path:
    input_path = dataset_dir / "docs.jsonl"
    output_dir = PROCESSED_ROOT / dataset_dir.name
    output_path = output_dir / "processed_docs.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as outfile:
        batches = iter_document_batches(input_path, batch_size=batch_size, limit=limit)

        for batch in tqdm(batches, desc=f"Preprocessing docs for {dataset_dir.name}"):
            for processed_doc in preprocessor.preprocess_documents_batch(
                batch,
                batch_size=batch_size,
            ):
                outfile.write(json.dumps(processed_doc, ensure_ascii=False) + "\n")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset documents.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of documents per dataset for smoke testing.",
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
        output_path = preprocess_jsonl_dataset(
            dataset_dir,
            preprocessor,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(f"Saved processed documents to: {output_path}")


if __name__ == "__main__":
    main()