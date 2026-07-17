"""Audit and summarize ``code_graph`` benchmark results.

Usage::

    uv run python benchmarks/code_graph_eval/analyze.py

The task (not individual stochastic run) is the bootstrap resampling unit.
"""

import json
import random
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ARMS = ("baseline", "treatment")
METRICS = ("accuracy", "tokens", "cost_usd", "seconds", "steps")
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 20_260_713


def mean(rows: Iterable[dict[str, Any]], metric: str) -> float:
    """Return the arithmetic mean of *metric* in *rows*."""
    return statistics.mean(float(row[metric]) for row in rows)


def selected(
    rows: list[dict[str, Any]], arm: str, category: str | None = None
) -> list[dict[str, Any]]:
    """Select one arm, optionally restricted to a task category."""
    return [
        row
        for row in rows
        if row["arm"] == arm
        and (category is None or row["category"] == category)
    ]


def relative_delta(baseline: float, treatment: float) -> float:
    """Return treatment-vs-baseline percentage change."""
    return 100.0 * (treatment / baseline - 1.0)


def bootstrap_interval(
    rows: list[dict[str, Any]], task_ids: list[str], metric: str
) -> tuple[float, float]:
    """Return a deterministic task-cluster bootstrap 95% interval."""
    rng = random.Random(BOOTSTRAP_SEED)
    deltas: list[float] = []
    rows_by_task_arm = {
        (task_id, arm): [
            row
            for row in rows
            if row["task_id"] == task_id and row["arm"] == arm
        ]
        for task_id in task_ids
        for arm in ARMS
    }
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.choices(task_ids, k=len(task_ids))
        baseline = [
            row
            for task_id in sample
            for row in rows_by_task_arm[task_id, "baseline"]
        ]
        treatment = [
            row
            for task_id in sample
            for row in rows_by_task_arm[task_id, "treatment"]
        ]
        deltas.append(relative_delta(mean(baseline, metric), mean(treatment, metric)))
    deltas.sort()
    return deltas[int(0.025 * BOOTSTRAP_SAMPLES)], deltas[
        int(0.975 * BOOTSTRAP_SAMPLES) - 1
    ]


def audit(
    rows: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> tuple[list[str], list[int]]:
    """Validate matrix completeness, grading, errors, and baseline purity."""
    task_ids = [str(task["id"]) for task in tasks]
    trials = sorted({int(row["trial"]) for row in rows})
    expected = {
        (arm, task_id, trial)
        for arm in ARMS
        for task_id in task_ids
        for trial in trials
    }
    actual = {
        (str(row["arm"]), str(row["task_id"]), int(row["trial"]))
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"incomplete/duplicate matrix: rows={len(rows)}, "
            f"expected={len(expected)}, missing={expected - actual}"
        )
    gold = {str(task["id"]): list(task["gold_facts"]) for task in tasks}
    for row in rows:
        answer = str(row["answer"]).casefold()
        score = sum(fact.casefold() in answer for fact in gold[row["task_id"]]) / len(
            gold[row["task_id"]]
        )
        if score != row["accuracy"]:
            raise ValueError(f"stale grade: {row['arm']} {row['task_id']}")
        if row["error"]:
            raise ValueError(f"recorded run error: {row['arm']} {row['task_id']}")
        if row["arm"] == "baseline" and row["code_graph_hint_hits"]:
            raise ValueError(f"contaminated baseline: {row['task_id']}")
    return task_ids, trials


def main() -> None:
    """Audit results and print machine-readable aggregate statistics."""
    rows: list[dict[str, Any]] = json.loads((EVAL_DIR / "results.json").read_text())
    tasks: list[dict[str, Any]] = json.loads((EVAL_DIR / "tasks.json").read_text())[
        "tasks"
    ]
    task_ids, trials = audit(rows, tasks)
    summary: dict[str, Any] = {
        "runs": len(rows),
        "tasks": len(task_ids),
        "trials": trials,
        "total_cost_usd": sum(float(row["cost_usd"]) for row in rows),
        "arms": {},
        "deltas_percent": {},
        "bootstrap_95_percent": {},
    }
    for arm in ARMS:
        arm_rows = selected(rows, arm)
        summary["arms"][arm] = {
            metric: mean(arm_rows, metric) for metric in METRICS
        } | {
            "hint_hits": sum(int(row["code_graph_hint_hits"]) for row in arm_rows),
            "tool_calls": sum(int(row["code_graph_tool_calls"]) for row in arm_rows),
            "errors": sum(bool(row["error"]) for row in arm_rows),
        }
    for metric in METRICS:
        baseline = summary["arms"]["baseline"][metric]
        treatment = summary["arms"]["treatment"][metric]
        summary["deltas_percent"][metric] = (
            0.0 if baseline == 0 else relative_delta(baseline, treatment)
        )
        if metric != "accuracy":
            summary["bootstrap_95_percent"][metric] = bootstrap_interval(
                rows, task_ids, metric
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
