import json
from collections import defaultdict
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

try:
    from evaluation_charts import render_evaluation_charts
except ModuleNotFoundError:
    from ui.evaluation_charts import render_evaluation_charts


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DATASETS = ["argsme_touche2022", "clinicaltrials_2021"]
DEFAULT_MODES = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
DEFAULT_EVALUATION_MODES = ["all", "tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]


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


@st.cache_data(ttl=10)
def load_evaluations(base_url: str, dataset: str | None = None) -> dict[str, Any]:
    path = "/api/v1/evaluations"
    if dataset:
        path = f"{path}?dataset={dataset}"
    return api_get(base_url, path)


def format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return ""


def display_value(value: Any) -> Any:
    return value if value is not None else "N/A"


def mode_uses_param(mode: str, param_name: str) -> bool:
    serial_params = {
        "serial_candidate_k",
        "serial_bm25_weight",
        "serial_embedding_weight",
    }
    parallel_params = {
        "tfidf_weight",
        "bm25_weight",
        "embedding_weight",
        "fusion_pool_k",
    }
    bm25_params = {
        "bm25_k1",
        "bm25_b",
    }

    if param_name in serial_params:
        return mode == "hybrid_serial"
    if param_name in parallel_params:
        return mode == "hybrid_parallel"
    if param_name in bm25_params:
        return mode in {"bm25", "hybrid_serial", "hybrid_parallel"}
    return True


def display_param(item: dict[str, Any], param_name: str) -> Any:
    mode = str(item.get("mode") or "")
    if not mode_uses_param(mode, param_name):
        return "N/A"
    return display_value(item.get(param_name))


def render_hybrid_weight_sum(tfidf_weight: float, bm25_weight: float, embedding_weight: float) -> bool:
    total = tfidf_weight + bm25_weight + embedding_weight
    is_valid = abs(total - 1.0) < 1e-9
    if is_valid:
        st.success(f"Hybrid weight sum: {total:.2f}")
    else:
        st.error(f"Hybrid weight sum must be 1.00. Current sum: {total:.2f}")
    return is_valid


def render_serial_weight_sum(serial_bm25_weight: float, serial_embedding_weight: float) -> bool:
    total = serial_bm25_weight + serial_embedding_weight
    is_valid = abs(total - 1.0) < 1e-9
    if is_valid:
        st.success(f"Hybrid serial weight sum: {total:.2f}")
    else:
        st.error(f"Hybrid serial weight sum must be 1.00. Current sum: {total:.2f}")
    return is_valid


def mode_supports_bm25_tuning(mode: str) -> bool:
    return mode in {"all", "bm25", "hybrid_serial", "hybrid_parallel"}


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


def render_refinement_info(refinement_info: dict[str, Any]) -> None:
    with st.expander("Refinement Details", expanded=True):
        st.write(f"**Original query:** `{refinement_info.get('original_query')}`")
        st.write(f"**Final query:** `{refinement_info.get('final_query')}`")

        if refinement_info.get("corrected_query"):
            st.write(f"  → Corrected: `{refinement_info.get('corrected_query')}`")

        if refinement_info.get("expanded_query"):
            st.write(f"  → Expanded: `{refinement_info.get('expanded_query')}`")

        if refinement_info.get("history_boosted_query"):
            st.write(f"  → History boosted: `{refinement_info.get('history_boosted_query')}`")

        if refinement_info.get("refinement_log"):
            st.write("**Changes made:**")
            for log_entry in refinement_info.get("refinement_log", []):
                st.write(f"  - {log_entry}")


def render_results(response: dict[str, Any], label: str = "") -> None:
    results = response.get("results", [])
    label_text = f" {label}" if label else ""
    st.caption(
        f"{response.get('count', 0)} result(s){label_text} for "
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


def render_evaluation_runs(evaluations_response: dict[str, Any]) -> None:
    evaluations = evaluations_response.get("evaluations", [])
    if not evaluations:
        st.info("No evaluations have been run yet.")
        return

    mode_order = {
        "tfidf": 0,
        "bm25": 1,
        "embedding": 2,
        "hybrid_serial": 3,
        "hybrid_parallel": 4,
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evaluations:
        evaluation_id = str(item.get("evaluation_id") or "unknown")
        groups[evaluation_id].append(item)

    for evaluation_id in sorted(groups.keys(), reverse=True):
        group_items = groups[evaluation_id]
        first_item = group_items[0]
        dataset = first_item.get("dataset") or "N/A"
        top_k = first_item.get("top_k") if first_item.get("top_k") is not None else "N/A"
        refinements_enabled = any(item.get("refinements_enabled") for item in group_items)
        refinements_label = "enabled" if refinements_enabled else "disabled"
        enabled_refinements = first_item.get("enabled_refinements") or []
        enabled_refinements_label = ", ".join(enabled_refinements) if enabled_refinements else "none"
        title = (
            f"Evaluation {evaluation_id} | {dataset} | top_k={top_k} | "
            f"refinements={refinements_label} | {len(group_items)} modes"
        )

        rows = []
        sorted_items = sorted(
            group_items,
            key=lambda item: (mode_order.get(item.get("mode"), 999), str(item.get("mode") or "")),
        )
        for item in sorted_items:
            metrics = item.get("metrics") or {}
            rows.append(
                {
                    "dataset": item.get("dataset"),
                    "mode": item.get("mode"),
                    "created_at": item.get("created_at"),
                    "top_k": display_value(item.get("top_k")),
                    "serial_candidate_k": display_param(item, "serial_candidate_k"),
                    "serial_bm25_weight": display_param(item, "serial_bm25_weight"),
                    "serial_embedding_weight": display_param(item, "serial_embedding_weight"),
                    "tfidf_weight": display_param(item, "tfidf_weight"),
                    "bm25_weight": display_param(item, "bm25_weight"),
                    "embedding_weight": display_param(item, "embedding_weight"),
                    "fusion_pool_k": display_param(item, "fusion_pool_k"),
                    "bm25_k1": display_param(item, "bm25_k1"),
                    "bm25_b": display_param(item, "bm25_b"),
                    "num_queries": item.get("num_queries"),
                    "MAP": metrics.get("MAP"),
                    "Recall": metrics.get("Recall"),
                    "Precision@10": metrics.get("Precision@10"),
                    "nDCG": metrics.get("nDCG"),
                    "refinements_enabled": item.get("refinements_enabled"),
                    "enabled_refinements": ", ".join(item.get("enabled_refinements") or []),
                }
            )

        with st.expander(title, expanded=False):
            st.caption(f"Enabled refinements: {enabled_refinements_label}")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_eval_table(evaluations: list[dict[str, Any]]) -> None:
    if not evaluations:
        st.info("No evaluation data returned.")
        return
    df = pd.DataFrame(
        [
            {
                "Dataset": item.get("dataset"),
                "Mode": item.get("mode"),
                "Top K": display_value(item.get("top_k")),
                "serial_candidate_k": display_param(item, "serial_candidate_k"),
                "serial_bm25_weight": display_param(item, "serial_bm25_weight"),
                "serial_embedding_weight": display_param(item, "serial_embedding_weight"),
                "tfidf_weight": display_param(item, "tfidf_weight"),
                "bm25_weight": display_param(item, "bm25_weight"),
                "embedding_weight": display_param(item, "embedding_weight"),
                "fusion_pool_k": display_param(item, "fusion_pool_k"),
                "bm25_k1": display_param(item, "bm25_k1"),
                "bm25_b": display_param(item, "bm25_b"),
                "MAP": item.get("metrics", {}).get("MAP"),
                "Recall": item.get("metrics", {}).get("Recall"),
                "Precision@10": item.get("metrics", {}).get("Precision@10"),
                "nDCG": item.get("metrics", {}).get("nDCG"),
            }
            for item in evaluations
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)


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
available_datasets = (
    health_response.get("available_datasets", DEFAULT_DATASETS)
    if health_response
    else DEFAULT_DATASETS
)
available_modes = modes_response.get("modes", DEFAULT_MODES) if modes_response else DEFAULT_MODES

search_tab, evaluation_tab, runs_tab, charts_tab = st.tabs(["Search", "Run Evaluation", "Evaluation History", "Evaluation Charts"])

with search_tab:
    if health_response:
        with st.expander("API status", expanded=False):
            st.json(health_response)

    if datasets_response:
        with st.expander("Dataset status", expanded=False):
            render_dataset_status(datasets_response)

    left, right = st.columns([2, 1])

    with left:
        query = st.text_input("Query", value="climate change policy")
        dataset = st.selectbox("Dataset", available_datasets)
        mode = st.selectbox("Retrieval mode", available_modes, index=available_modes.index("bm25") if "bm25" in available_modes else 0)

    with right:
        top_k = st.slider("Top K", min_value=1, max_value=100, value=10)
        bm25_tuning_available = mode_supports_bm25_tuning(mode)
        tune_bm25_requested = st.checkbox("Tune BM25", value=False, disabled=not bm25_tuning_available)
        tune_bm25 = bm25_tuning_available and tune_bm25_requested
        if not bm25_tuning_available:
            st.caption("BM25 tuning applies only to BM25 and hybrid modes.")

    if tune_bm25:
        bm25_col1, bm25_col2 = st.columns(2)
        with bm25_col1:
            bm25_k1 = st.slider("BM25 k1", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        with bm25_col2:
            bm25_b = st.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    else:
        bm25_k1 = None
        bm25_b = None

    if mode == "hybrid_serial":
        serial_candidate_k = st.slider("Serial candidate K", min_value=1, max_value=10000, value=100)
        st.subheader("Hybrid Serial weights")
        serial_weight_col1, serial_weight_col2 = st.columns(2)
        with serial_weight_col1:
            serial_bm25_weight = st.slider("Serial BM25 weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        with serial_weight_col2:
            serial_embedding_weight = st.slider("Serial Embedding weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
        serial_weights_valid = render_serial_weight_sum(serial_bm25_weight, serial_embedding_weight)
    else:
        serial_candidate_k = 100
        serial_bm25_weight = 0.30
        serial_embedding_weight = 0.70
        serial_weights_valid = True

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
        hybrid_weights_valid = render_hybrid_weight_sum(tfidf_weight, bm25_weight, embedding_weight)
    else:
        tfidf_weight = 0.30
        bm25_weight = 0.35
        embedding_weight = 0.35
        fusion_pool_k = 1000
        hybrid_weights_valid = True

    st.subheader("Query Refinement")
    refine_col1, refine_col2 = st.columns(2)
    with refine_col1:
        enable_spelling_correction = st.checkbox("Spelling Correction", value=False)
        enable_synonym_expansion = st.checkbox("Synonym Expansion", value=False)
    with refine_col2:
        enable_search_history = st.checkbox("Search History", value=False)
        show_original_results = st.checkbox("Show Original Results", value=False)

    submitted = st.button("Search", type="primary", use_container_width=True, disabled=not (hybrid_weights_valid and serial_weights_valid))

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
                "serial_bm25_weight": serial_bm25_weight,
                "serial_embedding_weight": serial_embedding_weight,
                "fusion_pool_k": fusion_pool_k,
                "weights": {
                    "tfidf": tfidf_weight,
                    "bm25": bm25_weight,
                    "embedding": embedding_weight,
                },
                "enable_spelling_correction": enable_spelling_correction,
                "enable_synonym_expansion": enable_synonym_expansion,
                "enable_search_history": enable_search_history,
                "show_original_results": show_original_results,
            }

            if tune_bm25:
                payload["bm25_k1"] = bm25_k1
                payload["bm25_b"] = bm25_b

            with st.spinner("Searching..."):
                try:
                    search_response = api_post(api_base_url, "/api/v1/search", payload)

                    refinement_info = search_response.get("refinement_info")
                    if refinement_info:
                        render_refinement_info(refinement_info)

                    original_results = search_response.get("original_results")
                    if original_results and show_original_results:
                        left_col, right_col = st.columns(2)
                        with left_col:
                            st.subheader("Original Query Results")
                            original_response = {
                                **search_response,
                                "results": original_results,
                                "count": len(original_results),
                                "query": query.strip(),
                            }
                            render_results(original_response, "(Original)")
                        with right_col:
                            st.subheader("Refined Query Results")
                            render_results(search_response, "(Refined)")
                    else:
                        render_results(search_response)

                except RuntimeError as exc:
                    st.error(str(exc))

with evaluation_tab:
    st.subheader("Run Evaluation")

    eval_dataset = st.selectbox("Dataset", available_datasets, key="eval_dataset")
    evaluation_name = st.text_input(
        "Evaluation name",
        value="",
        placeholder="baseline_bm25_v1",
        help="Leave empty to auto-generate a timestamp-based name.",
    )
    eval_modes = ["all"] + available_modes if "all" not in available_modes else available_modes
    eval_mode = st.selectbox("Retrieval mode", eval_modes, index=eval_modes.index("all") if "all" in eval_modes else 0, key="eval_mode")
    eval_top_k = st.slider("Top K", min_value=10, max_value=100, value=10, key="eval_top_k")

    eval_left, eval_right = st.columns(2)
    with eval_left:
        eval_bm25_tuning_available = mode_supports_bm25_tuning(eval_mode)
        eval_tune_bm25_requested = st.checkbox("Tune BM25", value=False, disabled=not eval_bm25_tuning_available, key="eval_tune_bm25")
        eval_tune_bm25 = eval_bm25_tuning_available and eval_tune_bm25_requested
        if not eval_bm25_tuning_available:
            st.caption("BM25 tuning applies only to BM25 and hybrid modes.")
        eval_enable_spell = st.checkbox("Spelling Correction", value=False, key="eval_enable_spell")
        eval_enable_synonyms = st.checkbox("Synonym Expansion", value=False, key="eval_enable_synonyms")
    with eval_right:
        eval_enable_history = st.checkbox("Search History", value=False, key="eval_enable_history")
        st.checkbox("Include original results flag", value=False, disabled=True, key="eval_show_original")

    if eval_mode in {"hybrid_serial", "all"}:
        eval_serial_candidate_k = st.slider("Serial candidate K", min_value=1, max_value=10000, value=100, key="eval_serial_candidate_k")
    else:
        eval_serial_candidate_k = 100

    if eval_mode in {"hybrid_serial", "all"}:
        st.subheader("Hybrid Serial weights")
        eval_serial_weight_col1, eval_serial_weight_col2 = st.columns(2)
        with eval_serial_weight_col1:
            eval_serial_bm25_weight = st.slider("Serial BM25 weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key="eval_serial_bm25_weight")
        with eval_serial_weight_col2:
            eval_serial_embedding_weight = st.slider("Serial Embedding weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05, key="eval_serial_embedding_weight")
        eval_serial_weights_valid = render_serial_weight_sum(eval_serial_bm25_weight, eval_serial_embedding_weight)
    else:
        eval_serial_bm25_weight = 0.30
        eval_serial_embedding_weight = 0.70
        eval_serial_weights_valid = True

    if eval_mode in {"hybrid_parallel", "all"}:
        st.subheader("Hybrid Parallel weights")
        c1, c2, c3 = st.columns(3)
        with c1:
            eval_tfidf_weight = st.slider("TF-IDF weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key="eval_tfidf_weight")
        with c2:
            eval_bm25_weight = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="eval_bm25_weight")
        with c3:
            eval_embedding_weight = st.slider("Embedding weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="eval_embedding_weight")
        eval_fusion_pool_k = st.slider("Fusion pool K", min_value=eval_top_k, max_value=50000, value=max(1000, eval_top_k), key="eval_fusion_pool_k")
        eval_hybrid_weights_valid = render_hybrid_weight_sum(eval_tfidf_weight, eval_bm25_weight, eval_embedding_weight)
    else:
        eval_tfidf_weight = 0.30
        eval_bm25_weight = 0.35
        eval_embedding_weight = 0.35
        eval_fusion_pool_k = 1000
        eval_hybrid_weights_valid = True

    if eval_tune_bm25:
        eval_bm25_k1 = st.slider("BM25 k1", min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="eval_bm25_k1")
        eval_bm25_b = st.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key="eval_bm25_b")
    else:
        eval_bm25_k1 = None
        eval_bm25_b = None

    run_eval = st.button("Run evaluation", type="primary", use_container_width=True, disabled=not (eval_hybrid_weights_valid and eval_serial_weights_valid))

    if run_eval:
        payload: dict[str, Any] = {
            "dataset": eval_dataset,
            "mode": eval_mode,
            "top_k": eval_top_k,
            "serial_candidate_k": eval_serial_candidate_k,
            "serial_bm25_weight": eval_serial_bm25_weight,
            "serial_embedding_weight": eval_serial_embedding_weight,
            "fusion_pool_k": eval_fusion_pool_k,
            "weights": {
                "tfidf": eval_tfidf_weight,
                "bm25": eval_bm25_weight,
                "embedding": eval_embedding_weight,
            },
            "enable_spelling_correction": eval_enable_spell,
            "enable_synonym_expansion": eval_enable_synonyms,
            "enable_search_history": eval_enable_history,
        }
        if eval_bm25_k1 is not None:
            payload["bm25_k1"] = eval_bm25_k1
            payload["bm25_b"] = eval_bm25_b
        custom_evaluation_name = evaluation_name.strip()
        if custom_evaluation_name:
            payload["evaluation_name"] = custom_evaluation_name

        with st.spinner("Running evaluation..."):
            try:
                evaluation_response = api_post(api_base_url, "/api/v1/evaluations", payload, timeout=1800)
                st.success("Evaluation completed")
                load_evaluations.clear()
                render_eval_table(evaluation_response.get("evaluations", []))
            except RuntimeError as exc:
                st.error(str(exc))

with runs_tab:
    selected_dataset = st.selectbox("Dataset filter", ["All"] + available_datasets)
    dataset_filter = None if selected_dataset == "All" else selected_dataset
    if st.button("Refresh evaluations", use_container_width=True):
        load_evaluations.clear()
    try:
        evaluations_response = load_evaluations(api_base_url, dataset_filter)
        render_evaluation_runs(evaluations_response)
    except RuntimeError as exc:
        st.error(str(exc))

with charts_tab:
    selected_dataset = st.selectbox(
        "Dataset filter",
        ["All"] + available_datasets,
        key="charts_dataset_filter",
    )
    dataset_filter = None if selected_dataset == "All" else selected_dataset

    if st.button("Refresh chart data", use_container_width=True):
        load_evaluations.clear()

    try:
        evaluations_response = load_evaluations(api_base_url, dataset_filter)
        render_evaluation_charts(evaluations_response)
    except RuntimeError as exc:
        st.error(str(exc))
