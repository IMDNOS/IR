# IR Project

Information retrieval project with preprocessing, disk-backed indexing, multiple retrieval modes, and a FastAPI search API.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the required NLP resources:

```bash
python -m spacy download en_core_web_sm
python setup_nltk.py
```

Optional environment check:

```bash
python check_env.py
```

## Preprocess Data

Generate processed documents and queries:

```bash
python scripts/preprocess_datasets.py
python scripts/preprocess_queries.py
```

For a quick smoke test:

```bash
python scripts/preprocess_datasets.py --limit 10
python scripts/preprocess_queries.py --limit 10
```

Processed files are written under:

```text
processed/<dataset>/
```

Supported datasets:

```text
argsme_touche2022
clinicaltrials_2021
```

## Build Indexes

Build disk-backed indexes one dataset at a time:

```bash
python scripts/build_disk_indexes.py \
  --dataset argsme_touche2022 \
  --embedding-batch-size 16 \
  --embedding-chunk-docs 1024 \
  --commit-every-docs 500
```

```bash
python scripts/build_disk_indexes.py \
  --dataset clinicaltrials_2021 \
  --embedding-batch-size 16 \
  --embedding-chunk-docs 1024 \
  --commit-every-docs 500
```

Or build all datasets:

```bash
python scripts/build_disk_indexes.py --dataset all --embedding-batch-size 16
```

Index files are written under:

```text
indexes/<dataset>/
```

The disk-backed index uses SQLite for documents, postings, and document frequencies; `.npy` files for TF-IDF norms; and chunked `.npy` files for embeddings.

## Search From CLI

Supported retrieval modes:

```text
tfidf
bm25
embedding
hybrid_serial
hybrid_parallel
```

Examples:

```bash
python scripts/search_disk_demo.py \
  --dataset argsme_touche2022 \
  --mode bm25 \
  --query "climate change policy"
```

```bash
python scripts/search_disk_demo.py \
  --dataset clinicaltrials_2021 \
  --mode hybrid_serial \
  --query "diabetes treatment" \
  --serial-candidate-k 100
```

BM25 parameters can be changed at query time:

```bash
python scripts/search_disk_demo.py \
  --dataset clinicaltrials_2021 \
  --mode bm25 \
  --query "diabetes treatment" \
  --bm25-k1 1.2 \
  --bm25-b 0.70
```

## API

Run the FastAPI service from the project root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

```http
GET  /
GET  /api/v1/health
GET  /api/v1/datasets
GET  /api/v1/modes
GET  /api/v1/search
POST /api/v1/search
```

POST search example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "argsme_touche2022",
    "query": "climate change policy",
    "mode": "bm25",
    "top_k": 10
  }'
```

Hybrid parallel example:

```json
{
  "dataset": "argsme_touche2022",
  "query": "climate change policy",
  "mode": "hybrid_parallel",
  "top_k": 10,
  "fusion_pool_k": 1000,
  "weights": {
    "tfidf": 0.2,
    "bm25": 0.55,
    "embedding": 0.25
  }
}
```

The first API request can be slower because models and index metadata are loaded lazily, then cached per dataset.

## Web UI

Start the API first, then run the Streamlit UI:

```bash
streamlit run ui/app.py
```

The UI uses the API base URL `http://127.0.0.1:8000` by default.

## Bruno

The `bruno/` folder contains API request examples. Use the `Local` environment after starting the API on `127.0.0.1:8000`.
