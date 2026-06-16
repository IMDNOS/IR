# Project Setup Guide

Use this guide after cloning the project from Git.

## 1. Clone the repository

```bash
git clone <repository-url>
cd IR
```

Replace `<repository-url>` with the Git URL of this repository.

## 2. Create a Python virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After activation, your terminal should show `(.venv)`.

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs the main packages used by the project, including:

- `numpy`, `pandas`, `scipy`
- `nltk`, `spacy`, `vaderSentiment`
- `ir_datasets`
- `scikit-learn`, `rank-bm25`
- `torch`, `transformers`, `sentence-transformers`
- `ranx`
- `streamlit`
- `jupyter`, `ipykernel`

## 5. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

The environment check script loads this model, so this step is required.

## 6. Download NLTK resources

```bash
python setup_nltk.py
```

This downloads the NLTK resources used by the project:

- `punkt`
- `punkt_tab`
- `stopwords`
- `wordnet`
- `omw-1.4`

## 7. Verify the environment

Run:

```bash
python check_env.py
```

If everything is installed correctly, the script should print:

```text
Python environment is working.
spaCy model loaded.
All imports succeeded.
```

It will also print your installed PyTorch version and whether CUDA is available.

## 8. Preprocess the datasets

After installing dependencies and downloading the spaCy/NLTK resources, generate the
processed document and query files:

```bash
python scripts/preprocess_datasets.py
python scripts/preprocess_queries.py
```

The preprocessing scripts write:

- `processed/argsme_touche2022/processed_docs.jsonl`
- `processed/argsme_touche2022/processed_queries.jsonl`
- `processed/clinicaltrials_2021/processed_docs.jsonl`
- `processed/clinicaltrials_2021/processed_queries.jsonl`

For a quick smoke test, limit the number of processed records per dataset:

```bash
python scripts/preprocess_datasets.py --limit 10
python scripts/preprocess_queries.py --limit 10
```
