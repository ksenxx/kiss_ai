# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Score a finished Harbor job: per-task pass rates, the overall mean reward, cost.

Terminal-Bench reports the mean reward over every attempt of every task.
Tasks of the dataset that have no trial in the job (GPU tasks on a host
without GPUs) count as failed attempts when ``--dataset-dir`` names the
downloaded dataset, so the headline number is comparable with the
leaderboard's.

Usage::

    uv run python -m kiss.benchmarks.terminal_bench.summarize jobs/opus-5-5 \\
        [jobs/opus-5-5-rerun ...] --dataset-dir /data/datasets/terminal-bench --attempts 5

Writes ``summary.json`` and ``summary.md`` into the first job directory and
prints the Markdown.  Further job directories hold re-runs of single tasks
(after a harness fix); their trials replace the first job's for those tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_trials(job_dir: Path) -> list[dict[str, Any]]:
    """The ``result.json`` of every finished trial under *job_dir*, sorted by trial name.

    Args:
        job_dir: A Harbor job directory.
    """
    trials = [json.loads(p.read_text()) for p in sorted(job_dir.glob("*/result.json"))]
    return [t for t in trials if "trial_name" in t]


def trial_reward(trial: dict[str, Any]) -> float:
    """The verifier's reward of *trial* (0 when the verifier did not run)."""
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    return float(rewards.get("reward") or 0.0)


def trial_cost(trial: dict[str, Any]) -> tuple[float, int]:
    """``(cost_usd, tokens)`` the agent reported for *trial* (zeros when absent).

    The token total comes from the agent's metadata (the KISS daemon reports
    one total; Harbor's own fields split input and output tokens).
    """
    agent = trial.get("agent_result") or {}
    metadata = agent.get("metadata") or {}
    return float(agent.get("cost_usd") or 0.0), int(metadata.get("tokens") or 0)


def summarize(job_dirs: list[Path], dataset_dir: Path | None, attempts: int) -> dict[str, Any]:
    """Compute the scores of one job, or of a job plus the re-runs of some of its tasks.

    Args:
        job_dirs: Harbor job directories; a task that appears in a later
            directory takes its trials from that directory alone (a task
            re-run after a harness fix replaces its earlier trials).
        dataset_dir: The downloaded dataset (one sub-directory per task) whose
            tasks without trials count as *attempts* failures each; ``None``
            scores only the tasks that have trials.
        attempts: Attempts per task the job was launched with (``-k``).

    Returns:
        ``{"tasks": {name: {...}}, "overall": {...}}``.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for job_dir in job_dirs:
        job_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in load_trials(job_dir):
            job_tasks[trial["task_name"].split("/")[-1]].append(trial)
        by_task.update(job_tasks)
    names = sorted(by_task)
    if dataset_dir is not None:
        names = sorted(set(names) | {p.name for p in dataset_dir.iterdir()
                                     if (p / "task.toml").is_file()})
    tasks: dict[str, Any] = {}
    for name in names:
        trials = by_task.get(name, [])
        rewards = [trial_reward(t) for t in trials]
        costs = [trial_cost(t) for t in trials]
        errors = [(t.get("exception_info") or {}).get("exception_type") for t in trials]
        tasks[name] = {
            "attempts": len(trials),
            "passed": sum(1 for r in rewards if r >= 1.0),
            "mean_reward": (sum(rewards) / attempts) if attempts else 0.0,
            "cost_usd": round(sum(c for c, _ in costs), 4),
            "tokens": sum(n for _, n in costs),
            "errors": [e for e in errors if e],
            "seconds": [round(_seconds(t), 1) for t in trials],
        }
    n_tasks = len(tasks)
    total_attempts = n_tasks * attempts
    passed = sum(t["passed"] for t in tasks.values())
    overall = {
        "job": " + ".join(d.name for d in job_dirs),
        "tasks": n_tasks,
        "tasks_with_trials": sum(1 for t in tasks.values() if t["attempts"]),
        "attempts_per_task": attempts,
        "attempts_run": sum(t["attempts"] for t in tasks.values()),
        "attempts_passed": passed,
        "mean_reward": (passed / total_attempts) if total_attempts else 0.0,
        "tasks_solved_at_least_once": sum(1 for t in tasks.values() if t["passed"]),
        "cost_usd": round(sum(t["cost_usd"] for t in tasks.values()), 2),
        "tokens": sum(t["tokens"] for t in tasks.values()),
        "errors": sum(len(t["errors"]) for t in tasks.values()),
    }
    return {"tasks": tasks, "overall": overall}


def _seconds(trial: dict[str, Any]) -> float:
    """Wall-clock seconds of the agent phase of *trial* (0 when not recorded)."""
    from datetime import datetime

    span = trial.get("agent_execution") or {}
    if not span.get("started_at") or not span.get("finished_at"):
        return 0.0
    start = datetime.fromisoformat(span["started_at"])
    end = datetime.fromisoformat(span["finished_at"])
    return (end - start).total_seconds()


def to_markdown(summary: dict[str, Any]) -> str:
    """Render *summary* (from :func:`summarize`) as a Markdown report."""
    o = summary["overall"]
    lines = [
        f"# {o['job']}",
        "",
        f"- Mean reward (Terminal-Bench score): **{100 * o['mean_reward']:.1f}%** "
        f"({o['attempts_passed']} of {o['tasks'] * o['attempts_per_task']} attempts, "
        f"{o['tasks']} tasks x {o['attempts_per_task']} attempts)",
        f"- Tasks solved at least once: {o['tasks_solved_at_least_once']} of {o['tasks']}",
        f"- Attempts run: {o['attempts_run']} ({o['tasks_with_trials']} tasks with trials); "
        f"trials with a Harbor exception: {o['errors']}",
        f"- Agent cost: ${o['cost_usd']:.2f}; tokens: {o['tokens']:,}",
        "",
        "| Task | Passed | Attempts | Cost (USD) | Agent minutes | Errors |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, t in summary["tasks"].items():
        minutes = ", ".join(f"{s / 60:.0f}" for s in t["seconds"]) or "-"
        errors = ", ".join(t["errors"]) or ""
        lines.append(f"| {name} | {t['passed']} | {t['attempts']} | {t['cost_usd']:.2f} "
                     f"| {minutes} | {errors} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write ``summary.json`` / ``summary.md`` into the job directory."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job_dirs", type=Path, nargs="+",
                        help="job directory, then the job directories of task re-runs")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="downloaded dataset; its tasks without trials count as failures")
    parser.add_argument("--attempts", type=int, default=1, help="attempts per task (-k)")
    args = parser.parse_args(argv)
    summary = summarize(args.job_dirs, args.dataset_dir, args.attempts)
    (args.job_dirs[0] / "summary.json").write_text(json.dumps(summary, indent=1))
    markdown = to_markdown(summary)
    (args.job_dirs[0] / "summary.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
