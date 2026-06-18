# IR Project - Step 2: Document Representations

This step implements the required document representations:

- VSM TF-IDF
- Embedding representation using SentenceTransformer/BERT-style embeddings
- BM25
- Hybrid Serial Representation
- Hybrid Parallel Representation using score fusion

## Files to add

```text
services/representation_service.py
scripts/build_representations.py
scripts/search_demo.py
```

Keep your existing preprocessing files:

```text
services/preprocessing_service.py
scripts/preprocess_datasets.py
scripts/preprocess_queries.py
```

## Required order

First run preprocessing:

```bash
python scripts/preprocess_datasets.py
python scripts/preprocess_queries.py
```

Then build the representations:

```bash
python scripts/build_representations.py --dataset all
```

This creates:

```text
indexes/argsme_touche2022/
indexes/clinicaltrials_2021/
```

Each index folder contains:

```text
docs_store.jsonl
tfidf_vectorizer.joblib
tfidf_matrix.joblib
bm25.joblib
tokenized_corpus.joblib
embeddings.npy
metadata.json
```

## Test each retrieval mode

```bash
python scripts/search_demo.py --dataset argsme_touche2022 --mode tfidf --query "climate change policy"
python scripts/search_demo.py --dataset argsme_touche2022 --mode bm25 --query "climate change policy"
python scripts/search_demo.py --dataset argsme_touche2022 --mode embedding --query "climate change policy"
python scripts/search_demo.py --dataset argsme_touche2022 --mode hybrid_serial --query "climate change policy"
python scripts/search_demo.py --dataset argsme_touche2022 --mode hybrid_parallel --query "climate change policy"
```

For clinical trials:

```bash
python scripts/search_demo.py --dataset clinicaltrials_2021 --mode bm25 --query "diabetes treatment"
python scripts/search_demo.py --dataset clinicaltrials_2021 --mode hybrid_parallel --query "diabetes treatment"
```

## BM25 parameter control

The project asks for a way to see/change BM25 parameters during execution or explain them in the report.
This implementation exposes them from the CLI:

```bash
python scripts/search_demo.py \
  --dataset clinicaltrials_2021 \
  --mode bm25 \
  --query "diabetes treatment" \
  --bm25-k1 1.2 \
  --bm25-b 0.70
```

In the UI later, expose these two values as sliders:

```text
k1: 0.5 to 2.0, default 1.5
b:  0.0 to 1.0, default 0.75
```

## Hybrid Serial

Serial hybrid means two-stage retrieval:

1. BM25 retrieves the first candidate documents.
2. Embeddings rerank those candidates semantically.

Implemented as:

```text
BM25 top N candidates -> embedding reranking -> final ranked results
```

## Hybrid Parallel

Parallel hybrid means all models run independently:

```text
TF-IDF scores
BM25 scores
Embedding scores
```

Then scores are normalized and fused:

```text
final_score = 0.30 * TFIDF + 0.35 * BM25 + 0.35 * Embedding
```

These weights can later be changed in the UI.

## Report text

Use this in your Arabic report later:

```text
تم تمثيل الوثائق باستخدام أربع طرق رئيسية. أولاً، تم استخدام نموذج VSM-TF-IDF بالاعتماد على النص المعالج بعد إزالة الضجيج وتطبيق Lemmatization. ثانياً، تم استخدام BM25 اعتماداً على قائمة الكلمات المعالجة، مع اعتماد القيم k1=1.5 و b=0.75 كقيم افتراضية شائعة، مع إتاحة تعديلها أثناء التنفيذ. ثالثاً، تم استخدام تمثيل Embedding بالاعتماد على نموذج SentenceTransformer لتحويل النصوص إلى متجهات دلالية كثيفة. رابعاً، تم تنفيذ التمثيل الهجين بطريقتين: الطريقة التسلسلية Serial حيث يتم استرجاع المرشحين أولاً باستخدام BM25 ثم إعادة ترتيبهم باستخدام Embeddings، والطريقة المتوازية Parallel حيث يتم حساب درجات TF-IDF و BM25 و Embedding بشكل مستقل ثم دمجها باستخدام Weighted Score Fusion بعد التطبيع.
```
