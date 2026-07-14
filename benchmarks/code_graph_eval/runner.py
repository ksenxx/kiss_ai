"""Controlled A/B benchmark for KISS Sorcar's ``code_graph`` feature.

This is a feature ablation, not a checkout-level benchmark.  Both arms run the
same current ``KISSAgent`` and ``UsefulTools`` code over isolated, identical
corpus copies.  The baseline has no graph and receives only Bash/Read, which
exercises the pre-feature path; treatment has a pre-built graph and also
receives the code_graph tool.  This isolates the feature from unrelated commit
differences while retaining the real production interception wiring.

Both arms use the same model, prompt, step limit, budget, and deterministic
gold-fact grading.  Results stream incrementally into ``results.json`` and the
full trajectories are retained under ``transcripts/``.

Prepare fresh sealed corpora (copies the repo without Git history or generated
state), then run or resume the matrix::

    uv run python benchmarks/code_graph_eval/runner.py --prepare
    uv run python benchmarks/code_graph_eval/runner.py

An individual arm can be resumed with ``--arm baseline`` or ``--arm treatment``.
"""

import argparse
import json
import re
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.code_graph import build_graph, make_code_graph_tool
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_agent import KISSAgent

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
CORPUS = {
    "baseline": REPO_ROOT / "tmp" / "corpus_baseline",
    "treatment": REPO_ROOT / "tmp" / "corpus_treatment",
}
MODEL = "claude-fable-5"
MAX_STEPS = 15
MAX_BUDGET = 1.0

PROMPT = (
    "You are answering a question about the codebase located at {work_dir}. "
    "Use the available tools to inspect the code. Do not guess. "
    "When you know the answer, call finish with success=True and put the "
    "complete answer in the summary.\n\nQuestion: {question}"
)

_COPY_EXCLUDES = {
    ".git",
    ".kiss",
    ".venv",
    "__pycache__",
    "benchmarks",
    "node_modules",
    "tmp",
}
_HINT_MARKER = "[code_graph] The code graph already knows about this pattern"
_TOOL_CALL_RE = re.compile(r"(?<![\w])code_graph\s*\(")


def grade(answer: str, gold_facts: list[str]) -> float:
    """Return case-insensitive gold-fact coverage of a final answer."""
    folded_answer = answer.casefold()
    hits = sum(1 for fact in gold_facts if fact.casefold() in folded_answer)
    return hits / len(gold_facts) if gold_facts else 0.0


def transcript_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Count grep interceptions and explicit code_graph calls separately."""
    hint_hits = 0
    tool_calls = 0
    for message in messages:
        content = str(message.get("content", ""))
        hint_hits += content.count(_HINT_MARKER)
        if message.get("role") == "model":
            tool_calls += len(_TOOL_CALL_RE.findall(content))
    return {
        "code_graph_hint_hits": hint_hits,
        "code_graph_tool_calls": tool_calls,
    }


def prepare_corpora() -> None:
    """Create history-sealed arm copies and build only treatment's graph."""
    REPO_ROOT.joinpath("tmp").mkdir(exist_ok=True)
    for corpus in CORPUS.values():
        shutil.rmtree(corpus, ignore_errors=True)
        shutil.copytree(
            REPO_ROOT,
            corpus,
            ignore=lambda _directory, names: sorted(_COPY_EXCLUDES & set(names)),
        )

    started = time.perf_counter()
    graph = build_graph(str(CORPUS["treatment"]), incremental=False)
    build_seconds = time.perf_counter() - started
    metadata = {
        "model": MODEL,
        "max_steps": MAX_STEPS,
        "max_budget_usd": MAX_BUDGET,
        "graph_build_seconds": round(build_seconds, 4),
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "graph_files": graph.stats.get("files"),
        "baseline_graph_absent": not (
            CORPUS["baseline"] / ".kiss" / "code_graph"
        ).exists(),
    }
    (EVAL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def validate_corpora() -> None:
    """Reject contaminated or unprepared benchmark arms."""
    for name, corpus in CORPUS.items():
        if not corpus.is_dir():
            raise RuntimeError(
                f"{name} corpus missing: run this script with --prepare first"
            )
        if (corpus / ".git").exists():
            raise RuntimeError(f"{name} corpus leaks Git history: {corpus / '.git'}")
    baseline_graph = CORPUS["baseline"] / ".kiss" / "code_graph"
    if baseline_graph.exists():
        raise RuntimeError(f"baseline corpus is contaminated: {baseline_graph} exists")
    treatment_graph = CORPUS["treatment"] / ".kiss" / "code_graph" / "graph.json"
    if not treatment_graph.exists():
        raise RuntimeError(f"treatment corpus has no graph: {treatment_graph} missing")


def run_one(arm: str, task: dict[str, Any], trial: int) -> dict[str, Any]:
    """Run one arm/task/trial cell and return its complete measurement."""
    corpus = str(CORPUS[arm])
    useful = UsefulTools(work_dir=corpus)
    tools: list[Callable[..., Any]] = [useful.Bash, useful.Read]
    if arm == "treatment":
        code_graph_tool = make_code_graph_tool(corpus)
        if code_graph_tool is None:
            raise RuntimeError("code_graph tool unavailable in treatment arm")
        tools.append(code_graph_tool)

    agent = KISSAgent(f"bench-{arm}-{task['id']}-t{trial}")
    started = time.perf_counter()
    error = ""
    try:
        answer = agent.run(
            MODEL,
            PROMPT,
            arguments={"work_dir": corpus, "question": task["prompt"]},
            tools=tools,
            max_steps=MAX_STEPS,
            max_budget=MAX_BUDGET,
            verbose=False,
            print_prompts=False,
        )
    except Exception:
        answer = ""
        error = traceback.format_exc()
    seconds = time.perf_counter() - started

    record: dict[str, Any] = {
        "arm": arm,
        "task_id": task["id"],
        "category": task["category"],
        "trial": trial,
        "answer": answer,
        "accuracy": grade(answer, task["gold_facts"]),
        "steps": agent.step_count,
        "tokens": agent.total_tokens_used,
        "cost_usd": agent.budget_used,
        "seconds": round(seconds, 2),
        "error": error,
    }
    record.update(transcript_stats(agent.messages))

    transcript_dir = EVAL_DIR / "transcripts"
    transcript_dir.mkdir(exist_ok=True)
    transcript_path = transcript_dir / f"{arm}_{task['id']}_t{trial}.json"
    transcript_path.write_text(json.dumps(agent.messages, indent=1, default=str) + "\n")
    return record


def run_matrix(selected_arm: str, trial: int, results_file: str) -> None:
    """Run or resume the selected matrix cells."""
    validate_corpora()
    tasks: list[dict[str, Any]] = json.loads(
        (EVAL_DIR / "tasks.json").read_text()
    )["tasks"]
    results_path = EVAL_DIR / results_file
    if results_path.parent != EVAL_DIR or results_path.suffix != ".json":
        raise ValueError("--results-file must be a JSON filename, not a path")
    results: list[dict[str, Any]] = []
    if results_path.exists():
        results = json.loads(results_path.read_text())
    done = {(r["arm"], r["task_id"], r["trial"]) for r in results}
    arms = ("baseline", "treatment") if selected_arm == "all" else (selected_arm,)

    for task in tasks:
        for arm in arms:
            key = (arm, task["id"], trial)
            if key in done:
                print(f"SKIP {key} (already done)", flush=True)
                continue
            print(f"RUN {key} ...", flush=True)
            record = run_one(arm, task, trial)
            results.append(record)
            results_path.write_text(json.dumps(results, indent=1) + "\n")
            print(
                f"DONE {key}: acc={record['accuracy']:.2f} steps={record['steps']} "
                f"tokens={record['tokens']} cost=${record['cost_usd']:.4f} "
                f"sec={record['seconds']} hints={record['code_graph_hint_hits']} "
                f"calls={record['code_graph_tool_calls']} "
                f"err={'yes' if record['error'] else 'no'}",
                flush=True,
            )
    print("ALL DONE", flush=True)


def main() -> None:
    """Parse CLI arguments and prepare or run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("all", "baseline", "treatment"), default="all"
    )
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--results-file", default="results.json")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_corpora()
        return
    run_matrix(args.arm, args.trial, args.results_file)


if __name__ == "__main__":
    main()
