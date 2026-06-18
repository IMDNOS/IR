# Memory-safe Step 2 indexing

Your old builder loads all processed documents into RAM, then builds several large Python lists, a BM25 object, a TF-IDF matrix, and one huge embedding matrix. For a `processed/` directory around 7.5 GB, that can easily kill the OS.

This replacement builds a disk-backed index:

- SQLite `docs` table
- SQLite inverted index `postings(term, internal_id, tf)`
- SQLite document-frequency table `df(term, df)`
- TF-IDF document norms stored as a memory-mapped `.npy`
- embeddings stored as many `.npy` chunks under `embedding_chunks/`

## Copy files

Copy these files into your project:

```text
services/disk_index_service.py
scripts/build_disk_indexes.py
scripts/search_disk_demo.py
```

Your tree should keep the same structure:

```text
~/IR/
├── processed/
├── indexes/
├── scripts/
└── services/
```

## Build one dataset first

Do not build both datasets first. Start with the smaller or safer one:

```bash
python scripts/build_disk_indexes.py \
  --dataset argsme_touche2022 \
  --embedding-batch-size 16 \
  --embedding-chunk-docs 1024 \
  --commit-every-docs 500
```

Then build the second dataset:

```bash
python scripts/build_disk_indexes.py \
  --dataset clinicaltrials_2021 \
  --embedding-batch-size 16 \
  --embedding-chunk-docs 1024 \
  --commit-every-docs 500
```

After both work, you may run:

```bash
python scripts/build_disk_indexes.py --dataset all --embedding-batch-size 16
```

## Test search

```bash
python scripts/search_disk_demo.py \
  --dataset argsme_touche2022 \
  --mode bm25 \
  --query "climate change policy"
```

```bash
python scripts/search_disk_demo.py \
  --dataset argsme_touche2022 \
  --mode hybrid_parallel \
  --query "climate change policy"
```

```bash
python scripts/search_disk_demo.py \
  --dataset clinicaltrials_2021 \
  --mode hybrid_serial \
  --query "diabetes treatment" \
  --serial-candidate-k 100
```

## BM25 parameters

BM25 parameters can now be changed at query time without rebuilding the whole BM25 object:

```bash
python scripts/search_disk_demo.py \
  --dataset clinicaltrials_2021 \
  --mode bm25 \
  --query "diabetes treatment" \
  --bm25-k1 1.2 \
  --bm25-b 0.70
```

## What files should appear?

Inside each dataset index folder:

```text
indexes/<dataset>/
├── index.sqlite3
├── index.sqlite3-wal
├── index.sqlite3-shm
├── metadata.json
├── embedding_chunks.json
├── tfidf_doc_norms.float32.npy
└── embedding_chunks/
    ├── chunk_000000.npy
    ├── chunk_000001.npy
    └── ...
```

The `-wal` and `-shm` files are normal SQLite files.

## Important

The first build can take time because it writes many postings to SQLite and computes embeddings. But it should not consume all RAM because documents, postings, and embeddings are flushed to disk continuously.
