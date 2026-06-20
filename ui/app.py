import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DATASETS = ["argsme_touche2022", "clinicaltrials_2021"]
DEFAULT_MODES = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]


st.set_page_config(page_title="IR Search", layout="wide")


def api_get(base_url: str, path: str, timeout: int = 20) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    request = Request(url, method="GET")
    return _send_request(request, timeout)


def api_post(base_url: str, path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send_request(request, timeout)


def _send_request(request: Request, timeout: int) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        try:
            parsed = json.loads(detail)
            message = parsed.get("detail", detail)
        except json.JSONDecodeError:
            message = detail
        raise RuntimeError(f"API returned {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach API: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("API request timed out.") from exc


@st.cache_data(ttl=10)
def load_health(base_url: str) -> dict[str, Any]:
    return api_get(base_url, "/api/v1/health")


@st.cache_data(ttl=10)
def load_datasets(base_url: str) -> dict[str, Any]:
    return api_get(base_url, "/api/v1/datasets")


@st.cache_data(ttl=60)
def load_modes(base_url: str) -> dict[str, Any]:
    return api_get(base_url, "/api/v1/modes")


def format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return ""


def render_dataset_status(datasets_response: dict[str, Any]) -> None:
    datasets = datasets_response.get("datasets", [])
    if not datasets:
        st.info("No dataset metadata was returned by the API.")
        return

    rows = []
    for item in datasets:
        rows.append(
            {
                "dataset": item.get("name"),
                "ready": item.get("ready"),
                "documents": item.get("num_documents"),
                "sqlite": item.get("sqlite_exists"),
                "embeddings": item.get("embeddings_manifest_exists"),
                "tfidf_norms": item.get("tfidf_norms_exists"),
                "embedding_model": item.get("embedding_model"),
                "embedding_dimension": item.get("embedding_dimension"),
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_results(response: dict[str, Any]) -> None:
    results = response.get("results", [])
    st.caption(
        f"{response.get('count', 0)} result(s) for "
        f"`{response.get('query', '')}` using `{response.get('mode', '')}` "
        f"on `{response.get('dataset', '')}`"
    )

    if not results:
        st.info("No results returned.")
        return

    summary_rows = []
    for result in results:
        source_scores = result.get("source_scores") or {}
        summary_rows.append(
            {
                "rank": result.get("rank"),
                "doc_id": result.get("doc_id"),
                "score": result.get("score"),
                "tfidf": source_scores.get("tfidf"),
                "bm25": source_scores.get("bm25"),
                "embedding": source_scores.get("embedding"),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    for result in results:
        title = f"#{result.get('rank')} | {result.get('doc_id')} | score {format_score(result.get('score'))}"
        with st.expander(title, expanded=result.get("rank") == 1):
            source_scores = result.get("source_scores") or {}
            if source_scores:
                st.json(source_scores)
            st.write(result.get("raw_text") or "")


with st.sidebar:
    st.header("API")
    api_base_url = st.text_input("Base URL", DEFAULT_API_BASE_URL)

    if st.button("Refresh metadata", use_container_width=True):
        load_health.clear()
        load_datasets.clear()
        load_modes.clear()

    health_response: dict[str, Any] | None = None
    datasets_response: dict[str, Any] | None = None
    modes_response: dict[str, Any] | None = None

    try:
        health_response = load_health(api_base_url)
        datasets_response = load_datasets(api_base_url)
        modes_response = load_modes(api_base_url)
        st.success("API connected")
    except RuntimeError as exc:
        st.error(str(exc))


st.title("IR Search")

if health_response:
    with st.expander("API status", expanded=False):
        st.json(health_response)

if datasets_response:
    with st.expander("Dataset status", expanded=False):
        render_dataset_status(datasets_response)

available_datasets = (
    health_response.get("available_datasets", DEFAULT_DATASETS)
    if health_response
    else DEFAULT_DATASETS
)
available_modes = modes_response.get("modes", DEFAULT_MODES) if modes_response else DEFAULT_MODES

with st.form("search_form"):
    left, right = st.columns([2, 1])

    with left:
        query = st.text_input("Query", value="climate change policy")
        dataset = st.selectbox("Dataset", available_datasets)
        mode = st.selectbox("Retrieval mode", available_modes, index=available_modes.index("bm25") if "bm25" in available_modes else 0)

    with right:
        top_k = st.slider("Top K", min_value=1, max_value=100, value=10)
        tune_bm25 = st.checkbox("Tune BM25", value=False)
        bm25_k1 = st.slider("BM25 k1", min_value=0.1, max_value=5.0, value=1.5, step=0.1, disabled=not tune_bm25)
        bm25_b = st.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05, disabled=not tune_bm25)

    if mode == "hybrid_serial":
        serial_candidate_k = st.slider("Serial candidate K", min_value=1, max_value=10000, value=100)
    else:
        serial_candidate_k = 100

    if mode == "hybrid_parallel":
        st.subheader("Hybrid weights")
        w1, w2, w3 = st.columns(3)
        with w1:
            tfidf_weight = st.slider("TF-IDF weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        with w2:
            bm25_weight = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        with w3:
            embedding_weight = st.slider("Embedding weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        fusion_pool_k = st.slider("Fusion pool K", min_value=1, max_value=50000, value=1000)
    else:
        tfidf_weight = 0.30
        bm25_weight = 0.35
        embedding_weight = 0.35
        fusion_pool_k = 1000

    submitted = st.form_submit_button("Search", type="primary", use_container_width=True)


if submitted:
    if not query.strip():
        st.warning("Enter a query before searching.")
    else:
        payload: dict[str, Any] = {
            "dataset": dataset,
            "query": query.strip(),
            "mode": mode,
            "top_k": top_k,
            "serial_candidate_k": serial_candidate_k,
            "fusion_pool_k": fusion_pool_k,
            "weights": {
                "tfidf": tfidf_weight,
                "bm25": bm25_weight,
                "embedding": embedding_weight,
            },
        }

        if tune_bm25:
            payload["bm25_k1"] = bm25_k1
            payload["bm25_b"] = bm25_b

        with st.spinner("Searching..."):
            try:
                search_response = api_post(api_base_url, "/api/v1/search", payload)
                render_results(search_response)
            except RuntimeError as exc:
                st.error(str(exc))
