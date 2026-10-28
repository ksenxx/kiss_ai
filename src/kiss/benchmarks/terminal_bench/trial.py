# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""One Terminal-Bench trial of the coding SEA on the benchmark daemon.

Kept apart from :mod:`.harbor_agent` so that it imports without Harbor
installed (the tests and the daemon side use it).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


def run_trial(
    directory: Path, config: dict[str, Any], endpoint_file: Path, cancel: threading.Event,
) -> dict[str, Any]:
    """Run one trial of the coding SEA on the benchmark daemon and return its metrics.

    The client waits for the daemon indefinitely: the run ends when the agent
    finishes, its USD budget is spent, or *cancel* is set (Harbor's timeout).

    Args:
        directory: Trial directory (the SEA, its config and trajectory are written here).
        config: ``container``, ``workdir``, ``prompt``, ``model`` and ``max_budget``.
        endpoint_file: The benchmark daemon's ``sorcar-local.json``.
        cancel: Set by the caller to stop the task.

    Returns:
        ``model, container, agent_success, cost_usd, tokens, steps, task_id,
        final_text, error, seconds, review_cost_usd, submitted``; ``cost_usd``
        and ``tokens`` include the spend of the gate's side agents (the
        reviewer, the blind verifier's author:
        :func:`kiss.agents.seas.coding.coding_sea.review_usage`).
        ``agent_success`` is false for a run that ended on a text reply
        instead of a ``finish`` call (a content-filter refusal, a question)
        or on a finish let through after the verifier cap: nothing it
        delivered was checked.  ``submitted`` says whether the
        work tree was restored to the last verified checkpoint
        (:func:`kiss.agents.seas.coding.coding_sea.submit_best_checkpoint`).
    """
    from kiss.agents.seas.coding.coding_sea import (
        ended_unchecked,
        review_usage,
        submit_best_checkpoint,
        write_trial_sea,
    )
    from kiss.agents.sorcar import daemon_client

    sea_path = write_trial_sea(directory, config)
    trajectory = Path(config.get("trajectory") or directory.resolve() / "trajectory.jsonl")
    log_offset = trajectory.stat().st_size if trajectory.exists() else 0
    started = time.time()
    metrics: dict[str, Any] = {"model": config["model"], "container": config["container"]}
    try:
        result = daemon_client.run(
            config["prompt"],
            sea_path=str(sea_path),
            model=config["model"],
            max_budget=float(config["max_budget"]),
            use_web_tools=False,
            use_memory=False,
            auto_classify=False,
            timeout=None,
            endpoint_file=endpoint_file,
            cancel=cancel,
        )
        metrics.update({
            "agent_success": bool(result.success) and not ended_unchecked(trajectory, log_offset),
            "cost_usd": float(result.cost),
            "tokens": int(result.tokens),
            "steps": int(result.steps),
            "task_id": result.task_id,
            "final_text": (result.text or "")[:4000],
            "error": "",
        })
    except daemon_client.CancelledError as exc:  # Harbor's timeout: keep the spend so far
        metrics.update({
            "agent_success": False, "cost_usd": float(exc.result.cost),
            "tokens": int(exc.result.tokens), "steps": int(exc.result.steps),
            "task_id": exc.result.task_id, "final_text": "",
            "error": f"CancelledError: {exc}"[:2000],
        })
    except Exception as exc:  # daemon failure: recorded, not fatal
        metrics.update({
            "agent_success": False, "cost_usd": 0.0, "tokens": 0, "steps": 0,
            "task_id": "", "final_text": "", "error": f"{type(exc).__name__}: {exc}"[:2000],
        })
    try:
        metrics["submitted"] = ("accepted" if metrics["agent_success"]
                                else submit_best_checkpoint(directory / "config.json", log_offset))
    except Exception as exc:  # the restore is best effort; the work on disk stands
        metrics["submitted"] = f"error: {type(exc).__name__}: {exc}"[:500]
    review_cost, review_tokens = review_usage(trajectory, log_offset)
    metrics["review_cost_usd"] = review_cost
    metrics["cost_usd"] = float(metrics["cost_usd"]) + review_cost
    metrics["tokens"] = int(metrics["tokens"]) + review_tokens
    metrics["seconds"] = round(time.time() - started, 1)
    return metrics
