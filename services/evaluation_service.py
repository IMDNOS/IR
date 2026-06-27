import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.schemas import EvaluationRunRequest, SearchRequest


@dataclass
class EvaluationRecord:
    evaluation_id: str
    dataset: str
    mode: str
    created_at: str
    file_path: str
    top_k: int
    metrics: dict[str, float]
    num_queries: int
    avg_relevant_docs: float
    refinements_enabled: bool
    enabled_refinements: list[str]
    serial_candidate_k: int
    serial_bm25_weight: float
    serial_embedding_weight: float
    tfidf_weight: float
    bm25_weight: float
    embedding_weight: float
    fusion_pool_k: int
    bm25_k1: float | None
    bm25_b: float | None
    query_source: str


@dataclass
class EvaluationSummary:
    dataset: str
    mode: str
    top_k: int
    metrics: dict[str, float]
    num_queries: int
    avg_relevant_docs: float
    refinements_enabled: bool
    enabled_refinements: list[str]
    serial_candidate_k: int
    serial_bm25_weight: float
    serial_embedding_weight: float
    tfidf_weight: float
    bm25_weight: float
    embedding_weight: float
    fusion_pool_k: int
    bm25_k1: float | None
    bm25_b: float | None
    query_source: str
    results_file: str


class EvaluationService:
    def __init__(self, data_root: Path, evaluations_root: Path, manager: Any) -> None:
        self.data_root = data_root
        self.evaluations_root = evaluations_root
        self.manager = manager
        self.evaluations_root.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset: str) -> Path:
        return self.data_root / dataset

    def _queries_path(self, dataset: str) -> Path:
        return self._dataset_dir(dataset) / "queries.jsonl"

    def _qrels_path(self, dataset: str) -> Path:
        return self._dataset_dir(dataset) / "qrels.tsv"

    def _evaluation_dir(self, dataset: str) -> Path:
        path = self.evaluations_root / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _default_evaluation_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _evaluation_id_exists(self, evaluation_id: str) -> bool:
        prefix = f"{evaluation_id}_"

        if not self.evaluations_root.exists():
            return False

        for dataset_dir in self.evaluations_root.iterdir():
            if not dataset_dir.is_dir():
                continue

            for path in dataset_dir.iterdir():
                if not path.is_file():
                    continue
                if not path.name.startswith(prefix):
                    continue
                if path.name.endswith(".tsv") or path.name.endswith(".summary.json"):
                    return True

        return False

    def _load_queries(self, dataset: str) -> dict[str, str]:
        queries_path = self._queries_path(dataset)
        if not queries_path.exists():
            raise FileNotFoundError(f"Missing queries file: {queries_path}")

        queries: dict[str, str] = {}
        with queries_path.open("r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                query_id = str(payload.get("query_id"))
                title = payload.get("title") or ""
                description = payload.get("description") or ""
                text = f"{title} {description}".strip() or payload.get("text") or ""
                queries[query_id] = text.strip()
        return queries

    def _load_qrels(self, dataset: str) -> dict[str, dict[str, int]]:
        qrels_path = self._qrels_path(dataset)
        if not qrels_path.exists():
            raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

        qrels: dict[str, dict[str, int]] = {}
        with qrels_path.open("r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                query_id = str(row[0]).strip()
                raw_target = str(row[1]).strip()
                relevance_scores = [int(value) for value in row[2:] if str(value).strip()]
                if not relevance_scores or max(relevance_scores) <= 0:
                    continue
                relevance = max(relevance_scores)
                for doc_id in self._normalize_qrel_target(raw_target):
                    current = qrels.setdefault(query_id, {}).get(doc_id, 0)
                    if relevance > current:
                        qrels[query_id][doc_id] = relevance
        return qrels

    @staticmethod
    def _normalize_qrel_target(raw_target: str) -> list[str]:
        base_ids: list[str] = []
        for part in raw_target.split(","):
            part = part.strip()
            if not part:
                continue
            if "__" in part:
                part = part.split("__", 1)[0]
            if part and part not in base_ids:
                base_ids.append(part)
        return base_ids

    @staticmethod
    def _average_precision(ranked_doc_ids: list[str], relevant: dict[str, int]) -> float:
        if not relevant:
            return 0.0
        hits = 0
        sum_precisions = 0.0
        for idx, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in relevant:
                hits += 1
                sum_precisions += hits / idx
        return sum_precisions / len(relevant)

    @staticmethod
    def _recall(ranked_doc_ids: list[str], relevant: dict[str, int]) -> float:
        if not relevant:
            return 0.0
        retrieved_relevant = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant)
        return retrieved_relevant / len(relevant)

    @staticmethod
    def _precision_at_k(ranked_doc_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
        if k <= 0:
            return 0.0
        top_k = ranked_doc_ids[:k]
        if not top_k:
            return 0.0
        return sum(1 for doc_id in top_k if doc_id in relevant) / k

    @staticmethod
    def _ndcg(ranked_doc_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
        if not relevant:
            return 0.0

        def dcg(doc_ids: list[str]) -> float:
            score = 0.0
            for idx, doc_id in enumerate(doc_ids[:k], start=1):
                rel = relevant.get(doc_id, 0)
                if rel <= 0:
                    continue
                score += (2**rel - 1) / math.log2(idx + 1)
            return score

        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        ideal_dcg = 0.0
        for idx, rel in enumerate(ideal_rels, start=1):
            ideal_dcg += (2**rel - 1) / math.log2(idx + 1)
        if ideal_dcg <= 0.0:
            return 0.0
        return dcg(ranked_doc_ids) / ideal_dcg

    @staticmethod
    def _enabled_refinements(request: EvaluationRunRequest) -> list[str]:
        enabled: list[str] = []
        if request.enable_spelling_correction:
            enabled.append("spelling_correction")
        if request.enable_synonym_expansion:
            enabled.append("synonym_expansion")
        if request.enable_search_history:
            enabled.append("search_history")
        return enabled

    def _apply_query_refinements(
        self,
        request: EvaluationRunRequest,
        query_text: str,
    ) -> tuple[str, list[str], list[str]]:
        if not request.has_refinements():
            return query_text, [], []

        refined = self.manager.query_refinement_service.refine(
            query=query_text,
            dataset=request.dataset,
            enable_spelling_correction=request.enable_spelling_correction,
            enable_synonym_expansion=request.enable_synonym_expansion,
            enable_search_history=request.enable_search_history,
        )
        return refined.final_query, refined.applied_refinements, refined.refinement_log

    def _evaluate_mode(
        self,
        request: EvaluationRunRequest,
        query_texts: dict[str, str],
        qrels: dict[str, dict[str, int]],
        mode: str,
        evaluation_id: str,
    ) -> tuple[EvaluationSummary, list[dict[str, Any]]]:
        per_query_rows: list[dict[str, Any]] = []
        ap_values: list[float] = []
        recall_values: list[float] = []
        p10_values: list[float] = []
        ndcg_values: list[float] = []
        rel_counts: list[int] = []

        search_top_k = request.top_k
        for query_id, query_text in query_texts.items():
            final_query, applied_refinements, refinement_log = self._apply_query_refinements(request, query_text)
            relevant = qrels.get(query_id, {})
            rel_counts.append(len(relevant))
            if relevant:
                search_results = self.manager.search(
                    SearchRequest(
                        dataset=request.dataset,
                        query=final_query,
                        mode=mode,  # type: ignore[arg-type]
                        top_k=search_top_k,
                        serial_candidate_k=request.serial_candidate_k,
                        serial_bm25_weight=request.serial_bm25_weight,
                        serial_embedding_weight=request.serial_embedding_weight,
                        fusion_pool_k=request.fusion_pool_k,
                        weights=request.weights,
                        bm25_k1=request.bm25_k1,
                        bm25_b=request.bm25_b,
                    )
                )
                ranked_doc_ids = [result.doc_id for result in search_results]
            else:
                ranked_doc_ids = []

            ap = self._average_precision(ranked_doc_ids, relevant)
            recall = self._recall(ranked_doc_ids, relevant)
            p10 = self._precision_at_k(ranked_doc_ids, relevant, k=10)
            ndcg = self._ndcg(ranked_doc_ids, relevant, k=10)

            ap_values.append(ap)
            recall_values.append(recall)
            p10_values.append(p10)
            ndcg_values.append(ndcg)
            per_query_rows.append(
                {
                    "evaluation_id": evaluation_id,
                    "dataset": request.dataset,
                    "mode": mode,
                    "query_id": query_id,
                    "original_query": query_text,
                    "final_query": final_query,
                    "applied_refinements": json.dumps(applied_refinements, ensure_ascii=False),
                    "refinement_log": json.dumps(refinement_log, ensure_ascii=False),
                    "relevant_docs": len(relevant),
                    "average_precision": ap,
                    "recall": recall,
                    "precision_at_10": p10,
                    "ndcg": ndcg,
                }
            )

        metrics = {
            "MAP": sum(ap_values) / len(ap_values),
            "Recall": sum(recall_values) / len(recall_values),
            "Precision@10": sum(p10_values) / len(p10_values),
            "nDCG": sum(ndcg_values) / len(ndcg_values),
        }
        summary = EvaluationSummary(
            dataset=request.dataset,
            mode=mode,
            top_k=request.top_k,
            metrics=metrics,
            num_queries=len(query_texts),
            avg_relevant_docs=sum(rel_counts) / len(rel_counts),
            refinements_enabled=request.has_refinements(),
            enabled_refinements=self._enabled_refinements(request),
            serial_candidate_k=request.serial_candidate_k,
            serial_bm25_weight=request.serial_bm25_weight,
            serial_embedding_weight=request.serial_embedding_weight,
            tfidf_weight=request.weights.tfidf,
            bm25_weight=request.weights.bm25,
            embedding_weight=request.weights.embedding,
            fusion_pool_k=request.fusion_pool_k,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            query_source="title_description",
            results_file="",
        )
        return summary, per_query_rows

    def run(self, request: EvaluationRunRequest) -> list[EvaluationRecord] | EvaluationRecord:
        queries = self._load_queries(request.dataset)
        qrels = self._load_qrels(request.dataset)
        if not queries:
            raise ValueError(f"No queries found for dataset: {request.dataset}")

        results_dir = self._evaluation_dir(request.dataset)
        modes = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"] if request.mode == "all" else [request.mode]
        evaluation_id = request.evaluation_name or self._default_evaluation_id()
        if self._evaluation_id_exists(evaluation_id):
            raise ValueError(f"Evaluation name already exists: {evaluation_id}")
        records: list[EvaluationRecord] = []
        for mode in modes:
            summary, per_query_rows = self._evaluate_mode(request, queries, qrels, mode, evaluation_id)
            output_path = results_dir / f"{evaluation_id}_{mode}.tsv"
            with output_path.open("w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(
                    outfile,
                    delimiter="\t",
                    fieldnames=[
                        "evaluation_id",
                        "dataset",
                        "mode",
                        "query_id",
                        "original_query",
                        "final_query",
                        "applied_refinements",
                        "refinement_log",
                        "relevant_docs",
                        "average_precision",
                        "recall",
                        "precision_at_10",
                        "ndcg",
                    ],
                )
                writer.writeheader()
                writer.writerows(per_query_rows)
            summary_path = results_dir / f"{evaluation_id}_{mode}.summary.json"
            summary.results_file = str(output_path)
            payload = {
                "evaluation_id": evaluation_id,
                "dataset": request.dataset,
                "mode": mode,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "top_k": summary.top_k,
                "metrics": summary.metrics,
                "num_queries": summary.num_queries,
                "avg_relevant_docs": summary.avg_relevant_docs,
                "refinements_enabled": summary.refinements_enabled,
                "enabled_refinements": summary.enabled_refinements,
                "serial_candidate_k": summary.serial_candidate_k,
                "serial_bm25_weight": summary.serial_bm25_weight,
                "serial_embedding_weight": summary.serial_embedding_weight,
                "tfidf_weight": summary.tfidf_weight,
                "bm25_weight": summary.bm25_weight,
                "embedding_weight": summary.embedding_weight,
                "fusion_pool_k": summary.fusion_pool_k,
                "bm25_k1": summary.bm25_k1,
                "bm25_b": summary.bm25_b,
                "query_source": summary.query_source,
                "results_file": summary.results_file,
            }
            with summary_path.open("w", encoding="utf-8") as outfile:
                json.dump(payload, outfile, ensure_ascii=False, indent=2)
            records.append(
                EvaluationRecord(
                    evaluation_id=evaluation_id,
                    dataset=request.dataset,
                    mode=mode,
                    created_at=payload["created_at"],
                    file_path=str(output_path),
                    top_k=summary.top_k,
                    metrics=summary.metrics,
                    num_queries=summary.num_queries,
                    avg_relevant_docs=summary.avg_relevant_docs,
                    refinements_enabled=summary.refinements_enabled,
                    enabled_refinements=summary.enabled_refinements,
                    serial_candidate_k=summary.serial_candidate_k,
                    serial_bm25_weight=summary.serial_bm25_weight,
                    serial_embedding_weight=summary.serial_embedding_weight,
                    tfidf_weight=summary.tfidf_weight,
                    bm25_weight=summary.bm25_weight,
                    embedding_weight=summary.embedding_weight,
                    fusion_pool_k=summary.fusion_pool_k,
                    bm25_k1=summary.bm25_k1,
                    bm25_b=summary.bm25_b,
                    query_source=summary.query_source,
                )
            )
        return records if request.mode == "all" else records[0]

    def list_runs(self, dataset: str | None = None) -> list[dict[str, Any]]:
        datasets = [dataset] if dataset else [p.name for p in self.evaluations_root.iterdir() if p.is_dir()]
        records: list[dict[str, Any]] = []
        for ds in datasets:
            ds_dir = self.evaluations_root / ds
            if not ds_dir.exists():
                continue
            for summary_file in sorted(ds_dir.glob("*.summary.json"), reverse=True):
                with summary_file.open("r", encoding="utf-8") as infile:
                    records.append(json.load(infile))
        return records
