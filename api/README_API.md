# IR Project Search API

This folder adds a separated REST API layer on top of the existing disk-based IR engine.

It does **not** replace your indexing scripts. It only exposes search through HTTP.

## Folder location

Copy the whole `api/` folder into the project root:

```text
~/IR/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── retrieval_manager.py
│   ├── schemas.py
│   ├── requirements_api.txt
│   └── README_API.md
├── indexes/
├── scripts/
├── services/
└── ...
```

## Install API dependencies

From the project root:

```bash
pip install -r api/requirements_api.txt
```

Your main project dependencies must already be installed too, especially:

```text
spacy
nltk
numpy
sentence-transformers
fastapi
uvicorn
```

## Run the API

From the project root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

FastAPI will show an interactive Swagger UI.

## Endpoints

### Health

```http
GET /api/v1/health
```

### Dataset status

```http
GET /api/v1/datasets
```

This checks whether each index has:

```text
metadata.json
index.sqlite3
embedding_chunks.json
tfidf_doc_norms.float32.npy
```

### Retrieval modes

```http
GET /api/v1/modes
```

Supported modes:

```text
tfidf
bm25
embedding
hybrid_serial
hybrid_parallel
```

### Search using POST

```http
POST /api/v1/search
```

Example body:

```json
{
  "dataset": "argsme_touche2022",
  "query": "climate change policy",
  "mode": "bm25",
  "top_k": 10
}
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

BM25 parameter example:

```json
{
  "dataset": "clinicaltrials_2021",
  "query": "diabetes treatment",
  "mode": "bm25",
  "top_k": 10,
  "bm25_k1": 1.2,
  "bm25_b": 0.7
}
```

### Search using GET

Useful for quick browser testing:

```http
GET /api/v1/search?dataset=argsme_touche2022&mode=bm25&query=climate%20change%20policy&top_k=10
```

## Example curl commands

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

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "clinicaltrials_2021",
    "query": "diabetes treatment",
    "mode": "hybrid_serial",
    "top_k": 10,
    "serial_candidate_k": 100
  }'
```

## Notes

The first request can be slower because the API loads:

```text
spaCy preprocessing model
SentenceTransformer embedding model
SQLite index metadata
```

After the first request, the service is cached per dataset.

BM25 `k1` and `b` are accepted at search time, so changing them does not require rebuilding the index.
