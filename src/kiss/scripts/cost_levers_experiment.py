# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A/B cost experiments for the token-cost levers, with real model calls.

Each experiment runs the same task twice against a real model: once with
a lever switched off (the behaviour before
``projects/cost-levers-implementation-plan.md`` landed) and once with it
on, and reports tokens, cost and steps from the agents' own usage
counters.  Runs use an isolated ``KISS_HOME`` so nothing lands in the
production ``sorcar.db``; the tasks only read the repository.

Usage::

    uv run python -m kiss.scripts.cost_levers_experiment \\
        [--model claude-haiku-4-5] [--repeats 2] [--only E1,E3] [--out PATH]

Experiments:

* **E1 read-hygiene** — a Read-heavy exploration task with the Read
  outline/dedupe and tool-output compaction off vs on.
* **E2 review-profile** — one reviewer sub-agent spawned through the fan-out
  engine with the ``full`` toolset vs the ``review`` profile.
* **E3 system-prompt** — a trivial task with the previous ``SYSTEM.md``
  (mandatory first ``Read("./SORCAR.md")``) vs the current one.
* **E4 chat-digest** — a trivial follow-up in a chat whose earlier tasks have
  long HTML results, with the chat-history digest off vs on.

Caveats: the lever arms in E2 and E3 change what the model *does* (its
step count varies between runs), so total cost there is noisy; the
first-step context column is the controlled measure of the fixed
per-step overhead.  E1 and E4 change only what the tools return.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import sorcar_agent as sa
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task, _save_task_result
from kiss.core.base import SYSTEM_PROMPT
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.printer import Printer


def isolate_kiss_home() -> Path:
    """Point this process at a fresh ``KISS_HOME`` so runs never touch ``~/.kiss``.

    Called from :func:`main` only; importing the module has no side effects.

    Returns:
        The temporary home directory.
    """
    home = Path(tempfile.mkdtemp(prefix="kiss_cost_experiment_"))
    os.environ["KISS_HOME"] = str(home)
    os.environ["KISS_MUSE_AUTH"] = "0"
    th._db_conn = None
    th._KISS_DIR = home
    th._DB_PATH = home / "sorcar.db"
    return home


REPO = Path(__file__).resolve().parents[3]
PREVIOUS_SYSTEM_MD_COMMIT = "1e2e7676e"

READ_TASK = (
    "Using only the Read tool (no Bash), describe the module structure of these "
    "three files in at most 12 bullet points total, then call finish: "
    "src/kiss/agents/sorcar/sorcar_agent.py, src/kiss/agents/sorcar/persistence.py, "
    "src/kiss/agents/vscode/media/main.js. Read each file at most twice."
)
REVIEW_TASK = (
    "Review src/kiss/core/context_compaction.py for bugs. Read the file with the "
    "Read tool, then report only demonstrated issues with line numbers, or say it "
    "is correct. Finish within 4 steps."
)
TRIVIAL_TASK = "What is 17 * 23? Reply with just the number via finish."

LEVERS_OFF = {
    "read_dedupe": False, "read_outline_lines": 0, "tool_output_compaction": False,
}
LEVERS_ON = {
    "read_dedupe": True, "read_outline_lines": 2000, "tool_output_compaction": True,
}


@dataclass
class Run:
    """Usage of one agent run."""

    experiment: str
    variant: str
    repeat: int
    tokens: int
    cost: float
    steps: int
    seconds: float
    first_context: int = 0
    """Context tokens of the first model step (system prompt + tools + task)."""
    note: str = ""


class _UsagePrinter(Printer):
    """Printer that keeps the per-step context sizes and tool names."""

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[int] = []
        self.calls: list[str] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:  # noqa: A002
        if type == "usage_info":
            match = re.search(r"Context: ([\d,]+)", str(content))
            if match:
                self.contexts.append(int(match.group(1).replace(",", "")))
        elif type == "tool_call":
            self.calls.append(str(content))
        return ""

    def token_callback(self, token: str) -> None:
        return None

    def reset(self) -> None:
        return None


def _set_config(values: dict[str, Any]) -> None:
    for key, value in values.items():
        setattr(DEFAULT_CONFIG, key, value)


def _run_agent(
    prompt: str, model: str, **kwargs: Any,
) -> tuple[ChatSorcarAgent, float, _UsagePrinter]:
    agent = ChatSorcarAgent("cost-experiment")
    printer = _UsagePrinter()
    start = time.time()
    agent.run(
        prompt_template=prompt,
        model_name=model,
        work_dir=str(REPO),
        max_steps=25,
        max_budget=3.0,
        web_tools=False,
        use_memory=False,
        verbose=False,
        printer=printer,
        **kwargs,
    )
    return agent, time.time() - start, printer


def _first(contexts: list[int]) -> int:
    return contexts[0] if contexts else 0


def _usage(agent: Any) -> tuple[int, float, int]:
    return (
        int(getattr(agent, "total_tokens_used", 0) or 0),
        float(getattr(agent, "budget_used", 0.0) or 0.0),
        int(getattr(agent, "total_steps", 0) or 0),
    )


def e1_read_hygiene(model: str, repeat: int) -> list[Run]:
    """Read-heavy task: outline + dedupe + compaction off vs on."""
    runs = []
    for variant, values in (("off", LEVERS_OFF), ("on", LEVERS_ON)):
        _set_config(values)
        agent, seconds, printer = _run_agent(READ_TASK, model)
        tokens, cost, steps = _usage(agent)
        runs.append(Run(
            "E1 read-hygiene", variant, repeat, tokens, cost, steps, seconds,
            _first(printer.contexts), note=f"max ctx {max(printer.contexts, default=0):,}",
        ))
    _set_config(LEVERS_ON)
    return runs


def e2_review_profile(model: str, repeat: int) -> list[Run]:
    """One reviewer child through the fan-out engine: full vs review toolset."""
    runs = []
    for variant in ("full", "review"):
        totals: dict[str, float] = {}
        printer = _UsagePrinter()
        start = time.time()
        sa.run_tasks_parallel(
            [REVIEW_TASK], model_name=model, work_dir=str(REPO), max_budget=2.0,
            totals_out=totals, web_tools=False, use_memory=False, tool_profile=variant,
            printer=printer,
        )
        runs.append(Run(
            "E2 review-profile", variant, repeat, int(totals["total_tokens_used"]),
            float(totals["budget_used"]), int(totals["total_steps"]), time.time() - start,
            _first(printer.contexts), note=" ".join(printer.calls),
        ))
    return runs


def _previous_system_prompt() -> str:
    text = subprocess.run(
        ["git", "show", f"{PREVIOUS_SYSTEM_MD_COMMIT}:src/kiss/SYSTEM.md"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    if "Mandatory First Actions" not in text:  # pragma: no cover — repo history
        raise RuntimeError("previous SYSTEM.md does not contain the mandate block")
    return text


def e3_system_prompt(model: str, repeat: int) -> list[Run]:
    """Trivial task: previous SYSTEM.md (with the SORCAR.md mandate) vs current."""
    runs = []
    for variant, prompt in (("previous", _previous_system_prompt()), ("current", SYSTEM_PROMPT)):
        agent, seconds, printer = _run_agent(TRIVIAL_TASK, model, base_system_prompt=prompt)
        tokens, cost, steps = _usage(agent)
        runs.append(Run(
            "E3 system-prompt", variant, repeat, tokens, cost, steps, seconds,
            _first(printer.contexts), note=" ".join(printer.calls),
        ))
    return runs


def _seed_chat_with_long_results(n_tasks: int, result_chars: int) -> str:
    chat_id = ""
    for i in range(1, n_tasks + 1):
        task_id, chat_id = _add_task(f"Earlier task {i}: analyse module {i}", chat_id=chat_id)
        body = " ".join(f"finding {j} of task {i} with details" for j in range(result_chars // 40))
        _save_task_result(
            result=f"<h3>Result {i}</h3><p>{body}</p><pre><code>x = {i}</code></pre>",
            task_id=task_id,
        )
    return chat_id


def e4_chat_digest(model: str, repeat: int) -> list[Run]:
    """Follow-up task in a chat with 8 long prior results: digest off vs on.

    Each arm gets its own, identically seeded chat (a run appends its own
    task to the chat it resumes), and the arm order alternates per repeat.
    """
    runs = []
    arms = [("off", False), ("on", True)]
    if repeat % 2:
        arms.reverse()
    for variant, enabled in arms:
        chat_id = _seed_chat_with_long_results(8, 6000)
        DEFAULT_CONFIG.chat_history_digest = enabled
        agent = ChatSorcarAgent("cost-experiment")
        agent.resume_chat_by_id(chat_id)
        prefix_chars = len(agent.build_chat_prompt(TRIVIAL_TASK))
        agent.resume_chat_by_id(chat_id)
        printer = _UsagePrinter()
        start = time.time()
        agent.run(
            prompt_template=TRIVIAL_TASK, model_name=model, work_dir=str(REPO),
            max_steps=5, max_budget=1.0, web_tools=False, use_memory=False, verbose=False,
            printer=printer,
        )
        tokens, cost, steps = _usage(agent)
        runs.append(Run(
            "E4 chat-digest", variant, repeat, tokens, cost, steps, time.time() - start,
            _first(printer.contexts), note=f"prompt {prefix_chars:,} chars",
        ))
    DEFAULT_CONFIG.chat_history_digest = True
    return runs


_BASELINE_FIRST = ["off", "full", "previous", "on", "review", "current"]
"""Variant names in baseline-then-lever order for :func:`summarize`."""

EXPERIMENTS = {
    "E1": e1_read_hygiene, "E2": e2_review_profile,
    "E3": e3_system_prompt, "E4": e4_chat_digest,
}


def summarize(runs: list[Run]) -> list[dict[str, Any]]:
    """Average each (experiment, variant) over repeats and compute savings.

    Args:
        runs: Every run of the session.

    Returns:
        One row per experiment with the off/on averages and the relative
        cost and token change (negative = cheaper with the lever on).
    """
    rows = []
    for name in sorted({r.experiment for r in runs}):
        # Baseline arm first, as declared by the experiment (arm order in
        # *runs* may alternate between repeats).
        variants = list(dict.fromkeys(r.variant for r in runs if r.experiment == name))
        first, second = sorted(variants, key=_BASELINE_FIRST.index)
        avg: dict[str, dict[str, float]] = {}
        for variant in (first, second):
            group = [r for r in runs if r.experiment == name and r.variant == variant]
            avg[variant] = {
                "tokens": sum(r.tokens for r in group) / len(group),
                "cost": sum(r.cost for r in group) / len(group),
                "steps": sum(r.steps for r in group) / len(group),
                "seconds": sum(r.seconds for r in group) / len(group),
                "first_context": sum(r.first_context for r in group) / len(group),
            }
        base, lever = avg[first], avg[second]
        rows.append({
            "experiment": name, "baseline": first, "lever": second,
            "baseline_avg": base, "lever_avg": lever,
            "cost_change": (lever["cost"] - base["cost"]) / base["cost"] if base["cost"] else 0.0,
            "token_change": (
                (lever["tokens"] - base["tokens"]) / base["tokens"] if base["tokens"] else 0.0
            ),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the selected experiments and print/write the results."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--only", default="", help="comma-separated experiment ids")
    parser.add_argument("--out", default="tmp/experiments/results.json")
    args = parser.parse_args(argv)
    isolate_kiss_home()
    selected = [e.strip() for e in args.only.split(",") if e.strip()] or list(EXPERIMENTS)
    runs: list[Run] = []
    for repeat in range(args.repeats):
        for key in selected:
            batch = EXPERIMENTS[key](args.model, repeat)
            runs.extend(batch)
            for run in batch:
                print(
                    f"{run.experiment:<18} {run.variant:<9} r{run.repeat} "
                    f"tokens={run.tokens:>9,} cost=${run.cost:.4f} steps={run.steps:>3} "
                    f"step1-ctx={run.first_context:>6,} {run.seconds:6.1f}s {run.note}",
                    flush=True,
                )
    rows = summarize(runs)
    print()
    for row in rows:
        print(
            f"{row['experiment']:<18} {row['baseline']}→{row['lever']}: "
            f"cost {row['cost_change']:+.1%}, tokens {row['token_change']:+.1%}, "
            f"steps {row['baseline_avg']['steps']:.1f}→{row['lever_avg']['steps']:.1f}, "
            f"step-1 context {row['baseline_avg']['first_context']:,.0f}→"
            f"{row['lever_avg']['first_context']:,.0f}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "runs": [asdict(r) for r in runs], "summary": rows}, indent=2,
    ))
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
