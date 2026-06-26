from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


MODE_ORDER = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
METRIC_COLUMNS = ["MAP", "Recall", "Precision@10", "nDCG"]


def format_mode_name(mode: str) -> str:
    return {
        "tfidf": "TF-IDF",
        "bm25": "BM25",
        "embedding": "Embedding",
        "hybrid_serial": "Hybrid Serial",
        "hybrid_parallel": "Hybrid Parallel",
    }.get(mode, mode)


def evaluations_to_dataframe(evaluations_response: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in evaluations_response.get("evaluations", []):
        metrics = item.get("metrics") or {}
        rows.append(
            {
                "evaluation_id": item.get("evaluation_id"),
                "dataset": item.get("dataset"),
                "mode": item.get("mode"),
                "created_at": item.get("created_at"),
                "top_k": item.get("top_k"),
                "num_queries": item.get("num_queries"),
                "refinements_enabled": item.get("refinements_enabled"),
                "enabled_refinements": ", ".join(item.get("enabled_refinements") or []),
                "MAP": metrics.get("MAP"),
                "Recall": metrics.get("Recall"),
                "Precision@10": metrics.get("Precision@10"),
                "nDCG": metrics.get("nDCG"),
                "serial_candidate_k": item.get("serial_candidate_k"),
                "serial_bm25_weight": item.get("serial_bm25_weight"),
                "serial_embedding_weight": item.get("serial_embedding_weight"),
                "tfidf_weight": item.get("tfidf_weight"),
                "bm25_weight": item.get("bm25_weight"),
                "embedding_weight": item.get("embedding_weight"),
                "fusion_pool_k": item.get("fusion_pool_k"),
                "bm25_k1": item.get("bm25_k1"),
                "bm25_b": item.get("bm25_b"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    df["mode_label"] = df["mode"].astype(str).apply(format_mode_name)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df.sort_values(["evaluation_id", "mode"], ascending=[False, True])


def _selected_evaluation_ids(df: pd.DataFrame) -> list[str]:
    ids = [str(evaluation_id) for evaluation_id in df["evaluation_id"].dropna().unique()]
    return sorted(ids, reverse=True)


def _render_metadata(selected_df: pd.DataFrame) -> None:
    first = selected_df.iloc[0]
    enabled_refinements = first.get("enabled_refinements") or "none"
    cols = st.columns(5)
    cols[0].metric("Dataset", str(first.get("dataset") or "N/A"))
    cols[1].metric("Top K", str(first.get("top_k") or "N/A"))
    cols[2].metric("Refinements", str(bool(first.get("refinements_enabled"))))
    cols[3].metric("Enabled refinements", str(enabled_refinements))
    cols[4].metric("Modes", str(len(selected_df)))


def _render_single_metric_chart(selected_df: pd.DataFrame) -> str:
    metric = st.selectbox("Metric", METRIC_COLUMNS, index=METRIC_COLUMNS.index("nDCG"))
    chart_df = selected_df.dropna(subset=[metric]).copy()
    if chart_df.empty:
        st.info(f"No values available for {metric}.")
        return metric

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("mode_label:N", title="Retrieval mode", sort=[format_mode_name(mode) for mode in MODE_ORDER]),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color("mode_label:N", title="Mode", legend=None),
            tooltip=[
                alt.Tooltip("mode_label:N", title="Mode"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".4f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    return metric


def _render_grouped_metric_chart(selected_df: pd.DataFrame) -> None:
    selected_metrics = st.multiselect("Metrics to compare", METRIC_COLUMNS, default=METRIC_COLUMNS)
    if not selected_metrics:
        st.info("Select at least one metric to show the comparison chart.")
        return

    chart_df = selected_df.melt(
        id_vars=["mode", "mode_label"],
        value_vars=selected_metrics,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])
    if chart_df.empty:
        st.info("No metric values available for the selected metrics.")
        return

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("mode_label:N", title="Retrieval mode", sort=[format_mode_name(mode) for mode in MODE_ORDER]),
            y=alt.Y("score:Q", title="Score"),
            color=alt.Color("metric:N", title="Metric"),
            column=alt.Column("metric:N", title=None),
            tooltip=[
                alt.Tooltip("mode_label:N", title="Mode"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_best_mode_summary(selected_df: pd.DataFrame) -> None:
    st.subheader("Best Mode Summary")
    lines: list[str] = []
    for metric in ["nDCG", "Precision@10", "MAP", "Recall"]:
        metric_df = selected_df.dropna(subset=[metric])
        if metric_df.empty:
            lines.append(f"Best mode by {metric}: N/A")
            continue
        best_row = metric_df.loc[metric_df[metric].idxmax()]
        lines.append(f"Best mode by {metric}: {format_mode_name(str(best_row['mode']))}")
    st.write("\n".join(lines))


def _render_trend_chart(df: pd.DataFrame) -> None:
    if df["evaluation_id"].nunique() <= 1:
        return

    st.subheader("Metric Trend Across Evaluations")
    col1, col2 = st.columns(2)
    available_modes = [mode for mode in MODE_ORDER if mode in set(df["mode"].astype(str))]
    with col1:
        selected_mode = st.selectbox(
            "Trend mode",
            available_modes,
            format_func=format_mode_name,
            key="charts_trend_mode",
        )
    with col2:
        selected_metric = st.selectbox("Trend metric", METRIC_COLUMNS, index=METRIC_COLUMNS.index("nDCG"), key="charts_trend_metric")

    trend_df = df[df["mode"].astype(str) == selected_mode].dropna(subset=[selected_metric]).copy()
    if trend_df.empty:
        st.info("No trend data available for this mode and metric.")
        return

    trend_df = trend_df.sort_values(["created_at_dt", "evaluation_id"], na_position="last")
    trend_df["evaluation_label"] = trend_df["evaluation_id"].astype(str)
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("evaluation_label:N", title="Evaluation", sort=None),
            y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
            tooltip=[
                alt.Tooltip("evaluation_id:N", title="Evaluation ID"),
                alt.Tooltip("created_at:N", title="Created at"),
                alt.Tooltip(f"{selected_metric}:Q", title=selected_metric, format=".4f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_evaluation_charts(evaluations_response: dict[str, Any]) -> None:
    df = evaluations_to_dataframe(evaluations_response)
    if df.empty:
        st.info("No evaluations available for charts yet.")
        return

    evaluation_ids = _selected_evaluation_ids(df)
    if not evaluation_ids:
        st.info("No evaluations available for charts yet.")
        return

    selected_evaluation_id = st.selectbox("Evaluation ID", evaluation_ids)
    selected_df = df[df["evaluation_id"].astype(str) == selected_evaluation_id].copy()
    selected_df = selected_df.sort_values("mode")
    if selected_df.empty:
        st.info("No rows found for the selected evaluation.")
        return

    _render_metadata(selected_df)

    st.subheader("Single Metric Comparison")
    _render_single_metric_chart(selected_df)

    st.subheader("Grouped Metric Comparison")
    _render_grouped_metric_chart(selected_df)

    _render_best_mode_summary(selected_df)
    _render_trend_chart(df)
