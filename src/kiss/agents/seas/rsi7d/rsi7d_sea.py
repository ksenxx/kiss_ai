# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""rsi7d agent — 7-day recursive self-improvement of the indexed SEAs.

Two ways to run it (a slash command needs some task text, so ``/rsi7d``
alone is not dispatched)::

    /rsi7d all

    run_agent(agent="src/kiss/agents/seas/rsi7d/rsi7d_sea.py", task="all")

The task text starts with the *scope* (:func:`parse_scope`)::

    /rsi7d all                               every indexed SEA
    /rsi7d review_paper write_paper          just those SEAs
    /rsi7d --seas-dir ~/s10s/src/s10s/seas   the SEAs of that folder, now editable
    /rsi7d --seas-dir ~/s10s/src/s10s/seas triage    one SEA of that folder

Free-form instructions may follow the scope (``/rsi7d review_paper.
Replay the costliest run.``).  The tools read the scope from the running
task's text (``current_agent().last_user_prompt``) and enforce it:
``indexed_seas`` lists only the SEAs in scope, ``sea_runs`` and
``sea_findings`` mine only their runs, and ``sea_prompt`` /
``patch_sea_prompt`` / ``replay_in_clone`` refuse any other SEA.

Either makes the agent go over the runs of every SEA in scope of the last 7
days in ``~/.kiss/sorcar.db`` and improve each SEA by AI discovery: it
mines the trajectories for agentic mistakes, speed and cost sinks and
quality problems, proposes concrete instructions, judges them pairwise,
applies the winners to the SEA's ``SYSTEM_PROMPT`` constant, evaluates
the change (a real replay of the past task that best exercises the new
instructions; any run that cost below $500 is eligible) and keeps or
reverts it.  It also refreshes the observed model evidence the
autorouter SEA routes on.  rsi7d is itself one of the indexed SEAs: its
own finished sweeps are mined and its prompt patched the same way.

The deterministic tools live in this file (a SEA runs under the
installed kiss package, so it imports no sibling module of the
checkout); the reasoning is the agent's.

Trajectory mining
-----------------
The mining functions read the persisted task history and return plain
dicts:

* :func:`_mine_sea_runs` — the runs of every SEA in the last *days*
  days.  The database does not store which agent script a task ran, so
  a run is recognised through its parent's ``run_agent`` tool call
  (the ``agent`` argument names the SEA: a path ending in ``_sea.py`` or
  a channel name such as ``slack``; the ``task`` argument is the child
  row's prompt, verbatim or wrapped in a channel preamble) or, for runs
  the server dispatched itself, through the SEA's prompt text found in
  the run's ``system_prompt`` event (:func:`_runs_by_signature`).
* :func:`_run_findings` — deterministic mistake signals of one run
  (tool errors, rejected edits, repeated identical calls, timeouts,
  stalls, oversized tool results, missing summaries, ...), indexed by
  the digest entry numbers ``task_digest.transcript_page`` shows so
  each signal can be drilled into with ``task_digest.entry_detail``.
* :func:`_model_scorecard` — per-model speed, cost and reliability
  over every task of the window, the evidence the autorouter SEA's
  model notes are refreshed from.

The database holds 7 days of ~1.7M events: every query filters by
``task_id IN (...)`` (indexed) in chunks; nothing joins the events
table on the history timestamp.

Prompt editing
--------------
Edits are confined to ``SYSTEM_PROMPT``-style string constants (plain
literals or f-strings) of SEA files inside the editable folders: the
``--seas-dir`` folder when the scope names one, else the
``src/kiss/agents/seas`` and ``src/kiss/agents/third_party_agents``
directories of the task's work dir (the SEA's own directory when the
task does not run inside a KISS checkout).  The gate rejects any
candidate whose AST differs outside the constant's text, so code and
f-string placeholders are never changed.  SEAs registered from other
folders (channel agents, user folders) are analysed and reported on,
never edited.

Replays in clones
-----------------
A past task that modified files cannot be replayed in the checkout the
sweep edits.  :func:`replay_in_clone` clones the task's repository
(resolved from the run's ``work_dir``; a removed worktree path resolves
to its parent repository) at the commit the task started from — the
first parent of the task's auto-commit, found through the ``User
prompt:`` block auto-commits carry, else the repository ``HEAD`` at the
task's start time — into ``tmp/rsi7d/replays/<sea>-<id>``, rewrites
repository paths in the task text to the clone and dispatches this
checkout's SEA file (patched prompt included) there without a worktree
or auto-commit.  The replay's task id comes back for ``run_findings``.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import statistics
import string
import subprocess
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kiss.agents.sorcar import agent_dispatch, persistence, sea_commands, task_digest
from kiss.agents.sorcar.git_worktree import (
    TASK_RESULT_HEADING,
    USER_PROMPT_HEADING,
    strip_worktree_suffix,
)
from kiss.server.agent_state import current_agent
from kiss.server.tools_file import execute_python_file

DEFAULT_DAYS = 7
MAX_RUNS_LISTED = 40
WRAP_COLUMNS = 92
"""Prose written into a SEA prompt is wrapped here; the repo lints lines over 100."""
SIGNATURE_CHARS = 200
"""Prompt prefix length that identifies a SEA run's ``system_prompt`` event."""
PROMPT_GETTERS = ("system_prompt", "append_to_system_prompt", "add_to_system_prompt")
EVIDENCE_START = "<!-- rsi7d:model-evidence -->"
EVIDENCE_END = "<!-- /rsi7d:model-evidence -->"
"""Markers delimiting the observed-model-evidence block in the autorouter SEA's prompt."""
STAMP_PREFIX = "_Observed in the task history"
"""First words of the stamp line ``write_autorouter_evidence`` puts above the evidence."""
REPLAY_DIR = Path("tmp") / "rsi7d" / "replays"
"""Clones made by ``replay_in_clone`` live here under the task's work dir (gitignored)."""
SEAS_DIR_OPTION = "--seas-dir"
"""Task-text option naming the folder whose SEAs the sweep is restricted to (and may edit)."""

SYSTEM_PROMPT = """\
You are rsi7d, the KISS Sorcar agent that improves the other agents. Every indexed SEA
(Sorcar Extension Agent, a `<name>/<name>_sea.py` file registered as the slash command
`/<name>`) has a `SYSTEM_PROMPT` constant. Your job is to read the last 7 days of the
trajectories of every SEA in scope in the task history and make each of them finish with
higher quality (most important), fewer agentic mistakes, lower cost and higher speed, mostly
by adding precise instructions to its system prompt. You also refresh the observed model
evidence the autorouter SEA routes on.

rsi7d itself is one of those SEAs: its `src/kiss/agents/seas/rsi7d/rsi7d_sea.py` prompt is
editable through the same tools and its own finished sweeps (every run except the current
one) are trajectories to mine like any other's. Typical rsi7d failure modes worth checking:
verification claimed without a replay id, costs quoted from the running estimate instead of
`run_findings`, replays skipped for eligible runs, a report that does not list the changed
files. Patch its prompt with `patch_sea_prompt("rsi7d", ...)` exactly as for the others.

## Scope
Your task text starts with the scope: `all` (every indexed SEA), one or more SEA names
(`review_paper write_paper`), or `--seas-dir <folder>` (the SEAs of that folder, which then
are the editable ones instead of this checkout's, optionally followed by names). Free-form
instructions may follow the scope. The tools enforce the scope: `indexed_seas()` returns it
as `scope` and lists only the SEAs in it, `sea_runs` / `sea_findings` mine only their runs,
and `sea_prompt` / `patch_sea_prompt` / `replay_in_clone` refuse any other SEA. A SEA the
task text names but `scope.names` lacks was not recognised (a typo or an unregistered
name): stop and tell the user the recognised names. When `autorouter` is out of scope, skip
step 6 (`write_autorouter_evidence` refuses).

## Hard rules
- Change SEA files only through `patch_sea_prompt` and `write_autorouter_evidence`. They edit
  one string constant and reject anything that changes code. Never edit a SEA with
  Edit/Write, never touch files outside the editable folders (`src/kiss/agents/seas/` and
  `src/kiss/agents/third_party_agents/`, or the `--seas-dir` folder), never edit a SEA whose
  `editable_path` or `prompt_constant` is empty in `indexed_seas()` (user SEAs, and the
  channel SEAs such as `slack` or `gmail`, whose prompts are assembled at run time): analyse
  those and put recommendations in the report instead. `/ask` (`seas/ask/ask_sea.py`) does
  have a prompt constant and is optimized like the SEAs in `seas/`.
- Every instruction you add must be grounded in evidence from the trajectories: cite the
  task id and digest entry index (from `run_findings` / `run_transcript`) in your notes. No
  instruction for a failure mode that did not occur.
- Instructions must be specific, imperative and checkable ("Batch independent greps into one
  Bash call" beats "be efficient"). Do not restate what the prompt already says; read it
  first with `sea_prompt`. Prefer fixing the cause (e.g. "read `tmp/PROGRESS.md` before
  re-deriving the plan") over telling the agent to try harder.
- Keep each SEA's added text short: one section `## Lessons from recent runs (rsi7d)` of at
  most about 12 bullets. When that section already exists, replace it with a merged,
  deduplicated version (`patch_sea_prompt(name, old=<the whole existing section>,
  new=<merged section>)`) instead of appending a second one. Drop a bullet only when the
  evidence that motivated it is gone or a newer bullet supersedes it.
- A change that makes a SEA slower or costlier is acceptable only when it demonstrably
  improves the quality of the result; never trade quality for cost.
- Model names: use the names exactly as they appear in `model_scorecard` and
  `~/.kiss/MODEL_INFO.json`. Never invent a model name.
- Work in `./tmp/rsi7d/` for notes; the final report goes to
  `./reports/rsi7d-<YYYY-MM-DD>.md` and is `git add`ed.

## Procedure (AI discovery loop)
1. Baseline. Call `indexed_seas()` (check its `scope` against your task text), `sea_runs()`
   and `model_scorecard()`. Write `./tmp/rsi7d/baseline.md`: per SEA the number of runs,
   success/unsuccessful/failed counts, median cost, steps, seconds per step, models used; per
   model the scorecard row. Skip SEAs with zero runs in the window (say so in the report).
2. Mine mistakes per editable SEA with runs. Call `sea_findings(name)` for the aggregated
   signals, then drill into the runs that carry the most signal: every failed or
   unsuccessful run, the costliest, the slowest (seconds per step) and one typical
   successful run. Use `run_overview`, `run_transcript` (with `contains` to jump to errors,
   USER messages, summaries) and `run_entry` for the exact text. Look beyond the
   deterministic signals: read the user's follow-up messages and the final result to judge
   quality, look for re-derived plans, re-read files, serial one-command Bash calls,
   oversized tool outputs, unneeded sub-agents, wrong tool choices, missed requirements and
   hallucinated facts. Record each observation with its evidence in
   `./tmp/rsi7d/findings-<name>.md`.
3. Ideas. For each SEA write `./tmp/rsi7d/ideas.md`: candidate instructions with rationale,
   the evidence they rest on, and the aspect they improve (quality / mistakes / cost /
   speed). Use `decide` with pairwise "choice" questions to rank candidates; keep the
   winners (at most about 6 new bullets per SEA per run).
4. Implement. Call `sea_prompt(name)`, then `patch_sea_prompt(name, old, new)` (empty `old`
   appends the section). Re-read the result and make sure the section is coherent with the
   rest of the prompt.
5. Evaluate for real. Pick the past run of the SEA that best exercises the instructions you
   added (the failed or unsuccessful run whose mistake a new bullet targets, else the
   costliest successful run) among the runs that have no external side effects (messaging,
   payments, publishing). Do not restrict yourself to cheap runs: any run whose original
   cost was below $500 is eligible, and a cheaper run is preferred only when it carries the
   same signal. Cap every replay at twice the original run's cost, at most 500, so a
   regression cannot run away (a SEA that defines its own `max_budget()` getter overrides
   that cap and limits the replay itself; check with `grep -n "def max_budget" <sea file>`).
   Two ways to replay:
   - A task that modifies files (a paper, code, a report, a past rsi7d sweep) is replayed
     with `replay_in_clone(task_id, max_budget=<cap>)`. It clones the task's repository at
     the commit the task started from into `tmp/rsi7d/replays/<name>-<id>`, rewrites
     repository paths in the task text to the clone, and runs this checkout's patched SEA
     file there, so this checkout is never touched. Use the `replay_task_id` it returns and
     judge quality from `git -C <clone> status --short` / `diff` against the original run's
     result. Delete `tmp/rsi7d/replays` before you finish. Never replay such a task with
     `run_agent` in this checkout.
   - A task that changes nothing on disk is replayed with
     `run_agent(agent="src/kiss/agents/seas/<name>/<name>_sea.py", task=<the verbatim past
     task>, max_budget=<cap>)`.
   Then compare `run_findings(<new task id>)` with the original run (status, cost, steps,
   signal counts). Keep the change when the replay is not worse on status and signals and
   not clearly worse on cost/steps; otherwise revert with `git checkout --
   src/kiss/agents/seas/<name>/<name>_sea.py` and record why in `./tmp/rsi7d/explored-ideas.md` so
   the idea is not retried. Spend at most 60% of your remaining budget on replays and check
   `run_findings` of the sweep so far before each one; when no eligible run exists (every
   run cost $500 or more, or all have side effects) or the budget rule forbids the replay
   (a past rsi7d sweep is a full sweep and rarely fits), keep the change only if it is
   small, evidence-backed and passes `uv run pytest -q
   src/kiss/tests/agents/seas/test_<name>_sea.py` (when that test exists), and mark it "not
   replay-verified" in the report.
6. Autorouter evidence. From `model_scorecard()` and the per-SEA models, write a compact
   evidence block for the router with `write_autorouter_evidence(text)`: a Markdown table
   (model, tasks, role mix, failed/unsuccessful, median $ per step, median s per step,
   tool-error rate; keep every row under 92 characters by abbreviating headers or dropping a
   column, the tool rejects longer rows) followed by at most 8 bullets naming what each
   model is observed to be good or bad at, with the counts that support the claim. Only
   claim what at least 10 tasks support; say "insufficient data" otherwise. Include the
   window (`window_start`) so a reader can tell how fresh the evidence is.
7. Report. Write `./reports/rsi7d-<YYYY-MM-DD>.md`: baseline table, per SEA the findings,
   the added or changed bullets, the evaluation result (replay ids and metrics, or why not
   replayed), the autorouter evidence update, recommendations for non-editable SEAs, and
   ideas rejected with reasons. `git add` the report. Maintain `./tmp/PROGRESS.md` while you
   work.
8. Finish with a summary that lists every changed file, every added instruction, and the
   evaluation evidence."""


DAY_S = 86_400
_CHUNK = 400
"""Ids per ``IN (...)`` query; well under SQLite's 999-variable limit."""

_ERROR_PREFIX = re.compile(r"\s*(Error\b|error:|Traceback|KISSError|Denied by)")
_TIMEOUT = re.compile(r"(?i)\b(timed out|timeout|did not finish within)\b")
_STALL = re.compile(
    r"(?i)(stream stalled|consecutive errors|rate.?limit|overloaded|"
    r"\b529\b|retrying in|connection reset|budget exceeded|context (window|length))"
)
_EDIT_REJECTED = re.compile(
    r"(?i)(has not been (read|shown)|String not found in file|appears \d+ times|"
    r"not unique)"
)
_SHELL_FETCH = re.compile(r"\b(curl|wget)\b[^\n|]*https?://")
_REVIEWER_MISUSE = re.compile(
    r"(reviewer sub-agent and may not spawn|Review-round cap reached|"
    r"references the parent-repo path)"
)
_RESULT_SUCCESS = re.compile(r"^success:\s*(true|false)", re.MULTILINE)
_TASK_ERROR = re.compile(
    r"^(<p>|<h3>Partial result: )?(Task failed:|KISSError:|ModelRefusalError:|KISS Error:)"
    r"\s*(KISS Error:)?\s*"
)
"""In-process task errors (budget, consecutive model errors, stalls) as persisted in ``result``."""
_TOOL_ERROR_PATTERNS = (
    "%Error%",
    "%error:%",
    "%Traceback%",
    "%KISSError%",
    '%"content": "Denied by%',
    '%"is_error": true%',
)
"""LIKE prefilters for errored ``tool_result`` events (:data:`_ERROR_PREFIX` decides)."""
_HUGE_RESULT_CHARS = 30_000
_LONG_RUNNING_TOOLS = ("Bash", "run_commands_parallel", "run_agent", "run_parallel", "bash_job")
"""Tools whose results mention a timeout because the tool itself timed out."""
_ERROR_KIND_WORDS = ("ERROR", "FAIL", "EXCEPTION", "TRACEBACK")
"""Words that mark an uppercased event type (``TASK_ERROR``, ...) as an error event."""


def description() -> str:
    """Return the one-sentence help text shown by ``/rsi7d help``."""
    return (
        "Mines the last 7 days of the indexed SEAs' runs in ~/.kiss/sorcar.db for agentic "
        "mistakes, cost sinks and quality problems, applies and evaluates improvements to each "
        "SEA's SYSTEM_PROMPT (its own included; file-modifying tasks are replayed in a clone "
        "at the task's commit) and refreshes the autorouter SEA's model evidence. The task "
        "text starts with the scope: `/rsi7d all` (every SEA), `/rsi7d review_paper "
        "write_paper` (those SEAs), `/rsi7d --seas-dir <folder> [<name> ...]` (the SEAs of "
        "that folder, which become the editable ones); instructions may follow. Or "
        '`run_agent(agent="rsi7d", task="all")`.'
    )


def _is_error_kind(kind: str) -> bool:
    """Return whether digest entry *kind* is an error event rather than progress or UI noise."""
    return not kind.startswith("RESULT") and any(word in kind for word in _ERROR_KIND_WORDS)


_REVIEWER_TASK_PREFIX = "Verify the listed changes"


def sea_name_of(agent: object) -> str:
    """Return the SEA name of a ``run_agent`` ``agent`` argument.

    ``/x/y/review_paper_sea.py`` -> ``review_paper``; ``slack`` -> ``slack``;
    anything empty (a plain sub-agent) -> ``""``.
    """
    text = str(agent or "").strip()
    if not text:
        return ""
    stem = PurePosixPath(text.replace("\\", "/")).name
    stem = stem.removesuffix(".py")
    return stem.removesuffix("_sea") if stem else text


def _rows_since(days: float) -> list[dict[str, Any]]:
    """Return every task_history row inserted in the last *days* days, oldest first."""
    persistence._flush_chat_events()
    since = time.time() - days * DAY_S
    with persistence._rw_lock.read_lock():
        rows = (
            persistence._get_db()
            .execute(
                persistence._HISTORY_SELECT
                + "WHERE timestamp > ? ORDER BY timestamp ASC, rowid ASC",
                (since,),
            )
            .fetchall()
        )
    return [persistence._history_row_to_dict(r) for r in rows]


def _events_like(
    task_ids: list[str], *patterns: str, any_of: tuple[str, ...] = ()
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(task_id, event)`` for events of *task_ids* whose JSON matches every LIKE
    pattern in *patterns* and at least one in *any_of* (when given)."""
    found: list[tuple[str, dict[str, Any]]] = []
    clauses = ["event_json LIKE ?" for _ in patterns]
    if any_of:
        clauses.append("(" + " OR ".join("event_json LIKE ?" for _ in any_of) + ")")
    where = " AND ".join(clauses)
    with persistence._rw_lock.read_lock():
        db = persistence._get_db()
        for i in range(0, len(task_ids), _CHUNK):
            chunk = task_ids[i : i + _CHUNK]
            marks = ",".join("?" * len(chunk))
            sql = (
                f"SELECT task_id, event_json FROM events WHERE task_id IN ({marks})"
                + (f" AND {where}" if where else "")
                + " ORDER BY task_id, seq"
            )
            for tid, raw in db.execute(sql, [*chunk, *patterns, *any_of]):
                try:
                    found.append((tid, json.loads(raw)))
                except json.JSONDecodeError:
                    continue
    return found


def _final_success(task_ids: list[str]) -> dict[str, bool | None]:
    """Return the ``success`` flag of each task's final ``result`` event (``None`` when absent)."""
    flags: dict[str, bool | None] = {}
    for tid, ev in _events_like(task_ids, '%"type": "result"%'):
        if ev.get("type") != "result":
            continue
        match = _RESULT_SUCCESS.search(str(ev.get("text") or ""))
        flags[tid] = (match.group(1) == "true") if match else None
    return flags


def _status(row: dict[str, Any], final_success: bool | None) -> str:
    """``running`` / ``failed`` / ``unsuccessful`` / ``success`` of a history row."""
    if not int(row.get("end_ts") or 0):
        return "running"
    result = str(row.get("result") or "")
    if (
        persistence._is_failed_result(result)
        or result.startswith("Task stopped by user")
        or _TASK_ERROR.match(result)
    ):
        return "failed"
    if final_success is False:
        return "unsuccessful"
    return "success"


def _duration_s(row: dict[str, Any]) -> float:
    start, end = int(row.get("start_ts") or 0), int(row.get("end_ts") or 0)
    return round((end - start) / 1000, 1) if start and end > start else 0.0


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _run_record(
    row: dict[str, Any],
    children: list[dict[str, Any]],
    final_success: bool | None,
) -> dict[str, Any]:
    """Return the metrics dict of one task row.

    A parent row's ``cost`` / ``steps`` / ``tokens`` include its
    sub-agents' (the agent folds them in); the ``own_*`` fields subtract
    the children's rows so per-step figures describe the model itself.
    """
    cost = float(row.get("cost") or 0)
    steps = int(row.get("steps") or 0)
    tokens = int(row.get("tokens") or 0)
    return {
        "task_id": row["id"],
        "model": str(row.get("model") or ""),
        "task": task_digest.clip(row.get("task"), 200),
        "status": _status(row, final_success),
        "result": task_digest.clip(row.get("result"), 160),
        "cost": round(cost, 4),
        "own_cost": round(cost - sum(float(c.get("cost") or 0) for c in children), 4),
        "steps": steps,
        "own_steps": steps - sum(int(c.get("steps") or 0) for c in children),
        "tokens": tokens,
        "own_tokens": tokens - sum(int(c.get("tokens") or 0) for c in children),
        "duration_s": _duration_s(row),
        "children": len(children),
        "started": task_digest._fmt_ts(task_digest._start_ms(row)),
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary statistics of a list of run records (finished runs only for medians)."""
    done = [r for r in runs if r["status"] != "running"]
    steps = [r["steps"] for r in done if r["steps"] > 0]
    return {
        "runs": len(runs),
        "finished": len(done),
        "success": sum(r["status"] == "success" for r in done),
        "unsuccessful": sum(r["status"] == "unsuccessful" for r in done),
        "failed": sum(r["status"] == "failed" for r in done),
        "median_cost": _median([r["cost"] for r in done if r["cost"] > 0]),
        "max_cost": round(max((r["cost"] for r in done), default=0.0), 4),
        "median_steps": _median(steps),
        "median_duration_s": _median([r["duration_s"] for r in done if r["duration_s"] > 0]),
        "median_s_per_step": _median(
            [
                r["duration_s"] / r["own_steps"]
                for r in done
                if r["own_steps"] > 0 and r["duration_s"] > 0
            ]
        ),
        "models": dict(Counter(r["model"] for r in runs)),
    }


def _runs_by_signature(task_ids: list[str], signatures: dict[str, str]) -> dict[str, list[str]]:
    """Return ``{sea name: [task ids]}`` of *task_ids* whose system prompt carries a SEA signature.

    *signatures* maps a SEA name to a distinctive prefix of the prompt
    text its getter returns; a SEA's prompt is persisted verbatim in
    the run's ``system_prompt`` event (alone for ``system_prompt()``
    SEAs, appended to the default prompt for ``append_to_system_prompt()``
    ones), so the prefix identifies runs the server dispatched without a
    ``run_agent`` tool call (the side-channel task-update reports, runs
    started through ``sorcar.run(extension_agent_path=...)``).
    """
    found: dict[str, list[str]] = defaultdict(list)
    if not signatures:
        return found
    seen: set[str] = set()  # a task may persist several system_prompt events (set_model, resume)
    for tid, ev in _events_like(task_ids, '%"type": "system_prompt"%'):
        if ev.get("type") != "system_prompt" or tid in seen:
            continue
        text = str(ev.get("text") or "")
        for name, sig in signatures.items():
            if sig and sig in text:
                found[name].append(tid)
                seen.add(tid)
                break
    return found


def _mine_sea_runs(days: float = 7, signatures: dict[str, str] | None = None) -> dict[str, Any]:
    """Return the SEA runs of the last *days* days grouped by SEA name.

    A run is recognised through its parent's ``run_agent`` tool call
    (``agent`` argument = SEA path or channel name, ``task`` argument =
    the child's verbatim prompt) or, failing that, through *signatures*
    (``{sea name: prompt prefix}``, see :func:`_runs_by_signature`).

    Returns ``{"days", "window_start", "tasks_in_window", "seas": {name:
    {"agents": [distinct agent arguments], "stats": {...}, "runs": [...]}},
    "unmatched_dispatches"}``.  ``runs`` are newest first.  A dispatch
    whose child row cannot be found (the child was never persisted, or
    ran on another machine whose history was not synced) counts in
    ``unmatched_dispatches``.
    """
    rows = _rows_since(days)
    by_id = {r["id"]: r for r in rows}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        parent = str(r.get("parent_task_id") or "")
        if parent in by_id:
            children[parent].append(r)
    dispatches = [
        (tid, ev)
        for tid, ev in _events_like(list(by_id), '%"tool_call"%', '%"run_agent"%')
        if ev.get("type") == "tool_call" and ev.get("name") == "run_agent"
    ]
    claimed: set[str] = set()
    matched: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    unmatched = 0
    for parent_id, ev in dispatches:
        raw_extras = ev.get("extras")
        extras: dict[str, Any] = raw_extras if isinstance(raw_extras, dict) else {}
        name = sea_name_of(extras.get("agent"))
        if not name:
            continue  # a plain sub-agent, not a SEA
        wanted = str(extras.get("task") or "")
        candidates = [c for c in children[parent_id] if c["id"] not in claimed]
        # Channel and cron dispatches wrap the task in a preamble and
        # guidance (``agent_dispatch``), so fall back to containment.
        child = next((c for c in candidates if str(c.get("task") or "") == wanted), None)
        if child is None and wanted:
            child = next((c for c in candidates if wanted in str(c.get("task") or "")), None)
        if child is None:
            unmatched += 1
            continue
        claimed.add(child["id"])
        matched[name].append((str(extras.get("agent")), child))
    unclaimed = [tid for tid in by_id if tid not in claimed]
    for name, ids in _runs_by_signature(unclaimed, signatures or {}).items():
        for tid in ids:
            claimed.add(tid)
            matched[name].append(("(system prompt signature)", by_id[tid]))
    flags = _final_success(sorted(claimed))
    seas: dict[str, Any] = {}
    for name in sorted(matched):
        recs = [
            _run_record(child, children[child["id"]], flags.get(child["id"]))
            for _agent, child in matched[name]
        ]
        recs.sort(key=lambda r: r["started"], reverse=True)
        seas[name] = {
            "agents": sorted({agent for agent, _c in matched[name]}),
            "stats": _aggregate(recs),
            "runs": recs,
        }
    return {
        "days": days,
        "window_start": task_digest._fmt_ts(int((time.time() - days * DAY_S) * 1000)),
        "tasks_in_window": len(rows),
        "seas": seas,
        "unmatched_dispatches": unmatched,
    }


def _signal(index: int, kind: str, detail: str) -> dict[str, Any]:
    return {"entry": index, "kind": kind, "detail": task_digest.clip(detail, 240)}


def _run_findings(task_id: str) -> dict[str, Any] | str:
    """Return the deterministic mistake signals of one persisted run.

    The result has ``status``, ``cost``, ``steps``, ``duration_s``,
    ``tool_calls`` (calls per tool), ``signals`` (each with the digest
    ``entry`` index, a ``kind`` and a clipped ``detail``) and ``counts``
    (signals per kind).  Signal kinds: ``tool_error``, ``edit_rejected``,
    ``timeout``, ``stall_or_retry``, ``reviewer_misuse``,
    ``repeated_call``, ``huge_result``, ``shell_fetch`` (informational:
    curl/wget of a URL is fine for downloading a file, a mistake when it
    replaces ``go_to_url`` for research), ``no_summary``, ``error_event``,
    ``not_successful``.  Returns an error string when
    the task is unknown.
    """
    task = task_digest.load_task(task_id)
    if task is None:
        return f"Error: no task with id {task_id!r}"
    entries, _spend = task_digest.digest_events(task["events"])
    signals: list[dict[str, Any]] = []
    tool_calls: Counter[str] = Counter()
    seen_calls: dict[tuple[str, str], list[int]] = defaultdict(list)
    last_call = ""
    for i, entry in enumerate(entries):
        text = entry.text
        is_error = False
        if entry.kind == "TOOL CALL":
            tool_calls[entry.name] += 1
            last_call = entry.name
            seen_calls[(entry.name, text)].append(i)
            if entry.name == "Bash" and _SHELL_FETCH.search(text):
                signals.append(_signal(i, "shell_fetch", text))
        elif entry.kind.startswith("RESULT"):
            is_error = entry.kind.endswith("(error)") or bool(_ERROR_PREFIX.match(text))
            if is_error and last_call == "Edit" and _EDIT_REJECTED.search(text):
                signals.append(_signal(i, "edit_rejected", text))
            elif is_error and _REVIEWER_MISUSE.search(text):
                signals.append(_signal(i, "reviewer_misuse", f"{last_call}: {text}"))
            elif _TIMEOUT.search(text[:400]) and (is_error or last_call in _LONG_RUNNING_TOOLS):
                signals.append(_signal(i, "timeout", f"{last_call}: {text}"))
            elif is_error:
                signals.append(_signal(i, "tool_error", f"{last_call}: {text}"))
            if len(text) > _HUGE_RESULT_CHARS:
                signals.append(_signal(i, "huge_result", f"{last_call}: {len(text)} chars"))
        elif _is_error_kind(entry.kind):
            signals.append(_signal(i, "error_event", f"{entry.kind}: {text}"))
        if _STALL.search(text[:2000]) and (
            entry.kind == "TASK RESULT"
            or (entry.kind.startswith("RESULT") and is_error)
            or _is_error_kind(entry.kind)
        ):
            signals.append(_signal(i, "stall_or_retry", f"{entry.kind}: {text}"))
    for (name, text), idxs in seen_calls.items():
        if len(idxs) > 1 and name not in ("summary", "screenshot", "get_page_content"):
            detail = f"{name} x{len(idxs)} at {idxs}: {text}"
            signals.append(_signal(idxs[-1], "repeated_call", detail))
    steps = sum(1 for e in entries if e.kind in ("TOOL CALL", "SUMMARY", "FINISH"))
    if steps >= 10 and not any(e.kind == "SUMMARY" for e in entries):
        detail = f"{steps} steps without a summary call"
        signals.append(_signal(len(entries) - 1, "no_summary", detail))
    final = _final_success([task_id]).get(task_id)
    status = _status(task, final)
    if status != "success":
        detail = f"{status}: {task.get('result') or ''}"
        signals.append(_signal(len(entries) - 1, "not_successful", detail))
    signals.sort(key=lambda s: s["entry"])
    return {
        "task_id": task_id,
        "model": str(task.get("model") or ""),
        "status": status,
        "cost": round(float(task.get("cost") or 0), 4),
        "steps": int(task.get("steps") or 0),
        "duration_s": _duration_s(task),
        "entries": len(entries),
        "tool_calls": dict(tool_calls.most_common()),
        "signals": signals,
        "counts": dict(Counter(s["kind"] for s in signals)),
    }


def _error_reason(result: str) -> str:
    """Normalise a ``Task failed: ...`` result into a short reason (ids and numbers stripped)."""
    text = re.sub(r"<[^>]+>", " ", _TASK_ERROR.sub("", result))
    text = re.sub(r"Agent .*? Session-\S+ ", "Agent ", text)
    text = re.sub(r"[0-9a-f]{12,}|\$?\d[\d.,]*", "N", text)
    return " ".join(text.split())[:160]


def _role(row: dict[str, Any]) -> str:
    """``reviewer`` / ``subagent`` / ``top`` role of a task row."""
    if str(row.get("task") or "").startswith(_REVIEWER_TASK_PREFIX):
        return "reviewer"
    return "subagent" if str(row.get("parent_task_id") or "") else "top"


def _model_scorecard(days: float = 7) -> dict[str, Any]:
    """Return per-model speed, cost and reliability over every task of the last *days* days.

    Each model entry has ``tasks``, ``roles`` (top / subagent / reviewer
    counts), ``failed`` (killed, errored or stopped by the user),
    ``unsuccessful`` (finished with ``success: false``), ``median_own_cost``
    (row cost minus the children's), ``median_cost_per_step``,
    ``median_s_per_step``, ``median_steps``, ``median_tokens_per_step`` (all
    per-step figures use the row's own steps and tokens, children's excluded),
    ``tool_error_rate`` (errored tool results per step), ``task_errors``
    (runs that ended in a ``Task failed: ...`` error such as budget
    exhaustion, consecutive model errors or stream stalls) and
    ``error_reasons`` (the three most common of them).  Entries are
    sorted by task count.
    """
    rows = _rows_since(days)
    by_id = {r["id"]: r for r in rows}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        parent = str(r.get("parent_task_id") or "")
        if parent in by_id:
            children[parent].append(r)
    ids = list(by_id)
    flags = _final_success(ids)
    tool_errors: Counter[str] = Counter()
    for tid, ev in _events_like(ids, '%"tool_result"%', any_of=_TOOL_ERROR_PATTERNS):
        if ev.get("type") == "tool_result" and (
            ev.get("is_error") or _ERROR_PREFIX.match(str(ev.get("content") or ""))
        ):
            tool_errors[tid] += 1
    per_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per_model[str(r.get("model") or "")].append(r)
    cards: list[dict[str, Any]] = []
    for model, group in per_model.items():
        recs = [(_run_record(r, children[r["id"]], flags.get(r["id"])), r) for r in group]
        done = [(rec, r) for rec, r in recs if rec["status"] != "running"]
        with_steps = [(rec, r) for rec, r in done if rec["own_steps"] > 0]
        total_steps = sum(rec["own_steps"] for rec, _r in with_steps)
        errors = [
            _error_reason(str(r.get("result") or ""))
            for _rec, r in done
            if _TASK_ERROR.match(str(r.get("result") or ""))
        ]
        cards.append(
            {
                "model": model,
                "tasks": len(group),
                "roles": dict(Counter(_role(r) for r in group)),
                "failed": sum(rec["status"] == "failed" for rec, _r in done),
                "unsuccessful": sum(rec["status"] == "unsuccessful" for rec, _r in done),
                "median_own_cost": _median(
                    [rec["own_cost"] for rec, _r in done if rec["own_cost"] > 0]
                ),
                "median_cost_per_step": _median(
                    [
                        rec["own_cost"] / rec["own_steps"]
                        for rec, _r in with_steps
                        if rec["own_cost"] > 0
                    ]
                ),
                "median_s_per_step": _median(
                    [
                        rec["duration_s"] / rec["own_steps"]
                        for rec, _r in with_steps
                        if rec["duration_s"] > 0
                    ]
                ),
                "median_steps": _median([rec["own_steps"] for rec, _r in with_steps]),
                "median_tokens_per_step": _median(
                    [
                        rec["own_tokens"] / rec["own_steps"]
                        for rec, _r in with_steps
                        if rec["own_tokens"] > 0
                    ]
                ),
                "tool_error_rate": round(
                    sum(tool_errors[r["id"]] for _rec, r in done) / total_steps, 4
                )
                if total_steps
                else 0.0,
                "task_errors": len(errors),
                "error_reasons": dict(Counter(errors).most_common(3)),
            }
        )
    cards.sort(key=lambda c: -c["tasks"])
    return {
        "days": days,
        "window_start": task_digest._fmt_ts(int((time.time() - days * DAY_S) * 1000)),
        "tasks_in_window": len(rows),
        "models": cards,
    }


# Copies of four small helpers of ``skillopt_sea`` (kept identical): the
# daemon loads a SEA under the installed package, whose ``skillopt_sea``
# may predate them.
def _assigned_value(node: ast.stmt, name: str) -> ast.expr | None:
    """Return the value *node* assigns to *name* (``X = ...`` or ``X: str = ...``) or ``None``."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target: ast.expr = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target = node.target
    else:
        return None
    return node.value if isinstance(target, ast.Name) and target.id == name else None


def _string_literal(text: str) -> str:
    """Return a Python literal evaluating to *text* (triple-quoted when multi-line)."""
    body = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    literal = '"""\\\n' + body + '"""' if "\n" in text else repr(text)
    if ast.literal_eval(literal) != text:
        literal = repr(text)
    return literal


def _format_fields(text: str) -> set[str]:
    """Return the ``str.format`` field names of *text* (``ValueError`` on unbalanced braces)."""
    return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}


def _execute_sea(path: Path) -> dict[str, Any]:
    """Execute the SEA file at *path* and return its namespace."""
    return execute_python_file(str(path), ValueError, "SEA")


def system_prompt() -> str:
    """Replace the default Sorcar system prompt with the rsi7d procedure."""
    return SYSTEM_PROMPT


def max_budget() -> float:
    """Sweeping several SEAs needs room for replays of past runs that cost up to $500 each.

    A sub-agent's spend counts toward this task's total, so the cap must
    hold the mining work plus a few $500-class replays (the procedure
    limits replays to 60% of the remaining budget).
    """
    return 2000.0


def use_memory() -> bool:
    """Lessons about SEA failure modes are worth remembering across sweeps."""
    return True


def use_web_tools() -> bool:
    """Replays and evidence mining are local; no browsing."""
    return False


@dataclass(frozen=True)
class Scope:
    """What a ``/rsi7d`` task text restricts the sweep to (see :func:`parse_scope`)."""

    seas_dir: Path | None = None
    """The ``--seas-dir`` folder, resolved; ``None`` for the checkout's editable folders."""
    names: tuple[str, ...] = ()
    """The SEA names the task text starts with; empty for every SEA of the folders."""
    error: str = ""
    """Why the scope is unusable (``--seas-dir`` without a folder, or not a directory):
    nothing is then listed, mined or editable."""

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form ``indexed_seas`` reports: ``seas_dir`` (``""`` when unset),
        ``names`` and, when set, ``error``."""
        block: dict[str, Any] = {
            "seas_dir": str(self.seas_dir) if self.seas_dir is not None else "",
            "names": list(self.names),
        }
        if self.error:
            block["error"] = self.error
        return block


_SEAS_DIR_OPTION_RE = re.compile(re.escape(SEAS_DIR_OPTION) + r"(?:=|[ \t]+|(?=\s|$))(\S*)")
"""``--seas-dir <folder>`` or ``--seas-dir=<folder>`` anywhere in the task text (the
folder group is empty when the option ends the line or has no argument)."""
_SEA_NAME_RE = sea_commands._COMMAND_NAME_RE
"""A bare SEA name (the registry's command-name pattern): no path separators, no ``..``."""


def parse_scope(text: str) -> Scope:
    """Return the :class:`Scope` a ``/rsi7d`` task *text* names.

    ``--seas-dir <folder>`` (``~`` and a relative path resolve against
    the task's work dir) may appear anywhere; the SEA names are the
    leading whitespace-separated tokens (trailing ``.,;:`` ignored) that
    are SEAs of that folder or, without a folder, registered slash
    commands or SEAs of the checkout's editable folders.  Parsing stops
    at the first other token, so ``all``, ``all. Work inside …`` or
    ``Optimize review_paper`` name no SEA and keep every SEA in scope;
    ``review_paper write_paper. Replay …`` names two.  Runs are
    identified by SEA name (a bundled SEA runs from many worktree paths),
    so a same-named SEA of another folder shares the mined runs.
    """
    seas_dir: Path | None = None
    match = _SEAS_DIR_OPTION_RE.search(text)
    if match:
        if not match.group(1):
            return Scope(error=f"{SEAS_DIR_OPTION} needs a folder")
        try:
            seas_dir = (_work_root() / Path(match.group(1)).expanduser()).resolve()
        except (OSError, RuntimeError, ValueError) as e:  # ``~nobody``, a NUL byte, ...
            return Scope(error=f"{SEAS_DIR_OPTION} {match.group(1)!r}: {e}")
        if not seas_dir.is_dir():
            return Scope(seas_dir, error=f"{SEAS_DIR_OPTION} {seas_dir} is not a directory")
        text = text[: match.start()] + text[match.end() :]
    known: set[str] = set() if seas_dir is not None else set(sea_commands.list_commands())
    for folder in _dirs_of(seas_dir):
        known.update(sea_commands._scan_folder(folder))
    names: list[str] = []
    for token in text.split():
        token = token.rstrip(".,;:")
        if token not in known:
            break
        if token not in names:
            names.append(token)
    return Scope(seas_dir, tuple(names))


def _scope() -> Scope:
    """Return the scope of the running task's text (everything outside a task)."""
    agent = current_agent()
    return parse_scope(str(getattr(agent, "last_user_prompt", "") or ""))


def _checkout_seas_dir() -> Path:
    """Return the checkout's ``seas`` directory: the task work dir's, else this file's.

    Any ``src/kiss/agents/seas`` directory of the work dir counts, even
    one from an older layout without ``rsi7d/rsi7d_sea.py``: a replay of
    a past sweep in a clone (``replay_in_clone``) must edit the clone,
    never fall through to the checkout this file was loaded from.
    """
    agent = current_agent()
    bases = [Path(agent.work_dir)] if agent is not None and agent.work_dir else []
    bases.append(Path.cwd())
    for base in list(bases):  # a task in a sub-directory of a checkout edits that checkout
        top = _toplevel(base)
        if top is not None:
            bases.append(top)
    for base in bases:
        candidate = base / "src" / "kiss" / "agents" / "seas"
        if candidate.is_dir():
            return candidate.resolve()
    return Path(__file__).resolve().parents[1]


def _dirs_of(seas_dir: Path | None) -> list[Path]:
    """Return the editable folders for a scope's *seas_dir*: that folder alone when given,
    else the checkout's ``seas`` and, next to it, the bundled channel SEAs in
    ``third_party_agents`` (their prompts are assembled at run time by
    ``_channel_agent_utils``, so only the few with a prompt constant, e.g. ``ask``, end up
    patchable; the rest are analysed and reported on)."""
    if seas_dir is not None:
        return [seas_dir]
    seas = _checkout_seas_dir()
    third_party = seas.parent / "third_party_agents"
    return [seas, third_party] if third_party.is_dir() else [seas]


def _editable_dirs() -> list[Path]:
    """Return the folders whose SEAs the running task may edit (:func:`_dirs_of`; none
    when the scope is unusable)."""
    scope = _scope()
    return [] if scope.error else _dirs_of(scope.seas_dir)


def _editable_seas() -> dict[str, Path]:
    """Return ``{name: file}`` of every SEA in scope in the editable folders."""
    found: dict[str, Path] = {}
    for folder in reversed(_editable_dirs()):  # ``seas`` wins a name clash
        found.update(sea_commands._scan_folder(folder))
    names = _scope().names
    return {name: path for name, path in found.items() if name in names} if names else found


def _editable_path(name: str) -> Path | None:
    """Return the editable file of SEA *name*, or ``None`` when it is not an editable SEA
    in scope.

    *name* must be a bare module stem (``review_paper``); a path or ``..``
    can never escape the editable folders.
    """
    scope = _scope()
    if not _SEA_NAME_RE.match(name) or (scope.names and name not in scope.names):
        return None
    for folder in _editable_dirs():
        path = sea_commands.sea_script_in(folder / name)
        if path.is_file():
            return path
    return None


def _not_editable(name: str) -> str:
    """Return the error for a *name* :func:`_editable_path` rejected."""
    scope = _scope()
    if scope.error:
        return f"Error: {scope.error}"
    if scope.names and name not in scope.names:
        return f"Error: {name!r} is outside this run's scope {list(scope.names)}"
    folders = ", ".join(str(folder) for folder in _dirs_of(scope.seas_dir))
    return f"Error: {name!r} is not an editable SEA under {folders}"


def _prompt_constant(source: str) -> tuple[str, str]:
    """Return ``(getter, constant)`` of the prompt getter that returns a module constant.

    ``("system_prompt", "SYSTEM_PROMPT")`` for the common shape; the
    constant is ``""`` when the getter returns something other than a
    module-level string constant (an inline literal, an expression) and
    both are ``""`` when the file defines no prompt getter.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PROMPT_GETTERS:
            last = node.body[-1] if node.body else None
            if isinstance(last, ast.Return) and isinstance(last.value, ast.Name):
                try:
                    _prompt_node(tree, last.value.id)
                except ValueError:
                    return node.name, ""
                return node.name, last.value.id
            return node.name, ""
    return "", ""


def _prompt_node(tree: ast.Module, name: str) -> ast.Constant | ast.JoinedStr:
    """Return the string literal or f-string of the last module-level assignment to *name*."""
    values = [v for v in (_assigned_value(n, name) for n in tree.body) if v is not None]
    if not values:
        raise ValueError(f"no module-level assignment to {name}")
    value = values[-1]
    if (isinstance(value, ast.Constant) and isinstance(value.value, str)) or isinstance(
        value, ast.JoinedStr
    ):
        return value
    raise ValueError(f"{name} must be assigned a string literal or an f-string")


def _node_span(source: str, node: ast.expr) -> tuple[int, int]:
    """Return the ``(start, end)`` byte offsets of *node* in *source* (AST offsets are bytes)."""
    lines = source.splitlines(keepends=True)
    start = sum(len(line.encode()) for line in lines[: node.lineno - 1]) + node.col_offset
    end_lineno = node.end_lineno or node.lineno
    end = sum(len(line.encode()) for line in lines[: end_lineno - 1]) + (node.end_col_offset or 0)
    return start, end


def _prompt_text(source: str, constant: str) -> tuple[ast.Constant | ast.JoinedStr, str]:
    """Return the prompt node of *constant* and its editable text.

    The editable text is the value of a plain string constant and the
    raw source literal (``f\"\"\"...\"\"\"``, ``{NAME}`` placeholders included)
    of an f-string, because an f-string's value only exists at run time.
    """
    node = _prompt_node(ast.parse(source), constant)
    if isinstance(node, ast.Constant):
        return node, str(node.value)
    start, end = _node_span(source, node)
    return node, source.encode()[start:end].decode()


def _fingerprint(source: str, constant: str) -> str:
    """Return the AST dump of *source* with the prompt text of *constant* blanked.

    An f-string keeps its ``{...}`` placeholders (their expressions are
    code), so an edit that adds, drops or rewrites one changes the
    fingerprint and is rejected.
    """
    tree = ast.parse(source)
    node = _prompt_node(tree, constant)
    if isinstance(node, ast.Constant):
        node.value = ""
    else:
        node.values = [v for v in node.values if isinstance(v, ast.FormattedValue)]
    return ast.dump(tree)


def _sea_info(name: str, registered: Path | None) -> dict[str, Any]:
    """Describe one indexed SEA: registered path, editable path and prompt shape."""
    editable = _editable_path(name)
    source_path = editable or registered
    getter, constant = "", ""
    chars = 0
    if source_path is not None and source_path.is_file():
        source = source_path.read_text(encoding="utf-8")
        try:
            getter, constant = _prompt_constant(source)
            if constant:
                chars = len(_prompt_text(source, constant)[1])
        except (SyntaxError, ValueError):
            getter, constant = "", ""
    return {
        "name": name,
        "registered_path": str(registered) if registered else "",
        "editable_path": str(editable) if editable else "",
        "prompt_getter": getter,
        "prompt_constant": constant,
        "prompt_chars": chars,
    }


def indexed_seas() -> str:
    """List the indexed SEAs in scope with registered path, editable path and prompt shape.

    Returns JSON ``{"scope": {"seas_dir", "names"[, "error"]}, "seas": [...]}``:
    ``scope`` is what the task text restricts the sweep to (see
    :func:`parse_scope`; ``error`` when its ``--seas-dir`` is not a
    directory) and ``seas`` the rows ``{"name", "registered_path",
    "editable_path", "prompt_getter", "prompt_constant", "prompt_chars"}``.
    Without a scope every registered SEA and every SEA of the editable
    folders is listed; with ``--seas-dir`` only that folder's SEAs; with
    names only those; with an ``error`` none.  ``editable_path`` is the file ``patch_sea_prompt``
    edits (empty when the SEA is not in an editable folder);
    ``prompt_constant`` is the module constant the prompt getter returns
    (empty when the SEA has no editable prompt).
    """
    scope = _scope()
    names = set(scope.names)
    if not names and not scope.error:
        names = set(_editable_seas())
        if scope.seas_dir is None:
            names.update(sea_commands.list_commands())
    rows = [_sea_info(name, sea_commands.get_command(name)) for name in sorted(names)]
    return json.dumps({"scope": scope.as_dict(), "seas": rows}, indent=1)


def _signatures() -> dict[str, str]:
    """Return ``{sea name: prompt prefix}`` of every editable SEA whose prompt getter loads.

    The prefix (the first 200 characters of what the getter returns)
    identifies a run's ``system_prompt`` event when no ``run_agent``
    tool call links it to the SEA.
    """
    signatures: dict[str, str] = {}
    for name, path in sorted(_editable_seas().items()):
        try:
            namespace = _execute_sea(path)
        except Exception:  # noqa: BLE001 - a SEA that does not load has no runs to mine
            continue
        getter = next((namespace[g] for g in PROMPT_GETTERS if callable(namespace.get(g))), None)
        if getter is None:
            continue
        try:
            text = str(getter()).strip()
        except Exception:  # noqa: BLE001 - same
            continue
        if len(text) >= SIGNATURE_CHARS:
            signatures[name] = text[:SIGNATURE_CHARS]
    return signatures


def _sea_runs(days: float) -> dict[str, Any]:
    """Mine the SEA runs of the window, matching by dispatch and by prompt signature, and
    keep only the SEAs in scope (the named ones, or the ``--seas-dir`` folder's)."""
    scope = _scope()
    if scope.error:
        return {"error": scope.error, "seas": {}}
    data = _mine_sea_runs(days, _signatures())
    if scope.names or scope.seas_dir is not None:
        keep = set(scope.names) or set(_editable_seas())
        data["seas"] = {name: sea for name, sea in data["seas"].items() if name in keep}
    return data


def sea_runs(days: float = DEFAULT_DAYS, name: str = "") -> str:
    """Return the SEA runs of the last *days* days (every SEA in scope, or just *name*) as JSON.

    Per SEA: ``agents`` (the ``run_agent`` agent arguments seen, or
    ``(system prompt signature)`` for runs the server started directly),
    ``stats`` (runs, success / unsuccessful / failed counts, median and
    max cost, median steps, duration and seconds per step, models) and
    ``runs`` (newest first, at most 40: task_id, model, task, status,
    result, cost, own_cost, steps, tokens, duration_s, children, started).
    """
    data = _sea_runs(days)
    if name:
        data["seas"] = {k: v for k, v in data["seas"].items() if k == name}
    for entry in data["seas"].values():
        entry["runs"] = entry["runs"][:MAX_RUNS_LISTED]
    return json.dumps(data, indent=1)


def sea_findings(name: str, runs: int = 8, days: float = DEFAULT_DAYS) -> str:
    """Aggregate the mistake signals over the newest *runs* finished runs of SEA *name*.

    Returns JSON with ``runs_scanned``, ``by_status``, ``tool_calls``
    (calls per tool over the scanned runs), ``signal_counts`` (per kind)
    and ``examples`` (up to 5 per kind: task_id, entry, detail) to drill
    into with ``run_entry``.
    """
    data = _sea_runs(days)
    if data.get("error"):
        return f"Error: {data['error']}"
    entry = data["seas"].get(name)
    if entry is None:
        return f"Error: no runs of SEA {name!r} in the last {days} days"
    finished = [r for r in entry["runs"] if r["status"] != "running"][:runs]
    counts: dict[str, int] = {}
    tools: dict[str, int] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for run in finished:
        found = _run_findings(run["task_id"])
        if isinstance(found, str):
            continue
        for tool, n in found["tool_calls"].items():
            tools[tool] = tools.get(tool, 0) + n
        for sig in found["signals"]:
            counts[sig["kind"]] = counts.get(sig["kind"], 0) + 1
            bucket = examples.setdefault(sig["kind"], [])
            if len(bucket) < 5:
                bucket.append({"task_id": run["task_id"], **sig})
    return json.dumps(
        {
            "sea": name,
            "runs_scanned": [r["task_id"] for r in finished],
            "by_status": {
                s: sum(r["status"] == s for r in finished)
                for s in ("success", "unsuccessful", "failed")
            },
            "tool_calls": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
            "signal_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "examples": examples,
        },
        indent=1,
    )


def run_findings(task_id: str) -> str:
    """Return the deterministic mistake signals of one run (see :func:`_run_findings`) as JSON."""
    found = _run_findings(task_id)
    return found if isinstance(found, str) else json.dumps(found, indent=1)


def run_overview(task_id: str) -> str:
    """Return the compact overview of a run: header, sub-agents, summaries, result."""
    return task_digest.overview(task_id)


def run_transcript(task_id: str, start: int = 0, count: int = 150, contains: str = "") -> str:
    """Return one page of a run's digest (numbered entries) from entry *start*.

    *contains* keeps only entries containing one of its ``|``-separated
    terms (case-insensitive), e.g. ``"Error|USER|SUMMARY"``.
    """
    return task_digest.transcript_page(task_id, start, count, contains)


def run_entry(task_id: str, index: int) -> str:
    """Return the full text of digest entry *index* of a run (up to 20,000 characters)."""
    return task_digest.entry_detail(task_id, index)


def model_scorecard(days: float = DEFAULT_DAYS) -> str:
    """Return per-model speed, cost and reliability over every task of the last *days* days as JSON.

    Per model: tasks, roles (top / subagent / reviewer), failed,
    unsuccessful, median_own_cost, median_cost_per_step, median_s_per_step,
    median_steps, median_tokens_per_step, tool_error_rate, task_errors and
    error_reasons.  Compare with the catalog prices in
    ``~/.kiss/MODEL_INFO.json`` when reasoning about cost.
    """
    return json.dumps(_model_scorecard(days), indent=1)


def sea_prompt(name: str) -> str:
    """Return the prompt constant of editable SEA *name*: getter, constant name and text."""
    path = _editable_path(name)
    if path is None:
        return _not_editable(name)
    source = path.read_text(encoding="utf-8")
    getter, constant = _prompt_constant(source)
    if not constant:
        return f"Error: {path} has no prompt getter returning a module-level string constant"
    node, text = _prompt_text(source, constant)
    shape = (
        "an f-string: the text below is its source literal, {NAME} are placeholders evaluated "
        "at load time, and patch_sea_prompt matches old/new against this source text"
        if isinstance(node, ast.JoinedStr)
        else "a plain string: patch_sea_prompt matches old/new against this value"
    )
    return f"# {path}\n# {getter}() returns {constant}, {shape} ({len(text)} chars)\n\n{text}"


_BULLET = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _wrap_markdown(text: str) -> str:
    """Wrap the prose lines of Markdown *text* at :data:`WRAP_COLUMNS`.

    Table rows (``|``), fenced code and lines that already fit are kept;
    a bullet's continuation lines are indented under its text.  Returns
    the wrapped text; lines that cannot be wrapped (no spaces) stay long.
    """
    out: list[str] = []
    fence = ""  # the opening fence of the code block being copied, "" outside one
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match and not fence:
            fence = match.group(1)
        elif match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
            out.append(line)
            fence = ""
            continue
        if fence or len(line) <= WRAP_COLUMNS or line.lstrip().startswith("|"):
            out.append(line)
            continue
        match = _BULLET.match(line)
        first_indent = match.group(0) if match else line[: len(line) - len(line.lstrip())]
        rest = line[len(first_indent) :]
        out.extend(
            textwrap.wrap(
                rest,
                WRAP_COLUMNS,
                initial_indent=first_indent,
                subsequent_indent=" " * len(first_indent),
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [line]
        )
    return "\n".join(out)


def _too_long(text: str) -> str:
    """Return the first line of *text* longer than :data:`WRAP_COLUMNS`, or ``""``."""
    return next((line for line in text.splitlines() if len(line) > WRAP_COLUMNS), "")


def _segments(literal: str) -> list[tuple[int, int, bool]]:
    """Return ``(start, end, is_fstring)`` of each segment of an implicitly concatenated literal."""
    out: list[tuple[int, int, bool]] = []
    i, n = 0, len(literal)
    while i < n:
        if literal[i] in " \t\r\n\\":  # whitespace or a line continuation between segments
            i += 1
            continue
        if literal[i] == "#":  # a comment between segments
            newline = literal.find("\n", i)
            i = n if newline < 0 else newline
            continue
        j = i
        while j < n and literal[j] in "rRbBfFuU":
            j += 1
        prefix = literal[i:j].lower()
        quote = literal[j : j + 3] if literal[j : j + 3] in ('"""', "'''") else literal[j]
        k = j + len(quote)
        while k < n and not literal.startswith(quote, k):
            k += 2 if literal[k] == "\\" else 1  # an escaped quote never closes the segment
        k = min(k + len(quote), n)
        out.append((i, k, "f" in prefix))
        i = k
    return out


def _fstring_segment(literal: str, start: int, length: int) -> bool | None:
    """Return whether ``literal[start:start + length]`` lies in an f-string segment of *literal*.

    ``None`` when the span is not inside a single segment (it crosses a
    boundary or covers a quote).
    """
    for seg_start, seg_end, is_f in _segments(literal):
        if seg_start < start and start + length < seg_end:
            return is_f
    return None


def _gate(
    path: Path,
    constant: str,
    source: str,
    candidate: str,
    old_text: str,
    new_text: str,
    fstring: bool,
) -> str:
    """Return why *candidate* may not replace *source* at *path*, or ``""`` when it may."""
    try:
        compile(candidate, str(path), "exec")
        if _fingerprint(candidate, constant) != _fingerprint(source, constant):
            return (
                f"candidate changes code outside the {constant} constant "
                "or its {...} placeholders"
            )
    except (SyntaxError, ValueError) as exc:
        return f"candidate does not compile: {exc}"
    if fstring:
        return ""  # an f-string has no str.format fields to preserve
    try:
        old_fields = _format_fields(old_text)
    except ValueError:
        return ""  # the original is not a format template; nothing more to preserve
    try:
        new_fields = _format_fields(new_text)
    except ValueError as exc:
        return f"candidate breaks the prompt's str.format template: {exc}"
    if new_fields != old_fields:
        return (
            "candidate changes the template's replacement fields: "
            f"{sorted(new_fields)} != {sorted(old_fields)}"
        )
    return ""


def patch_sea_prompt(name: str, old: str, new: str) -> str:
    """Replace *old* with *new* inside the prompt constant of editable SEA *name*.

    An empty *old* appends *new* as a new paragraph at the end of the
    constant (as an adjacent plain string literal).  Otherwise *old* must
    occur exactly once in the text ``sea_prompt`` shows: the value of a
    plain string, or the source literal of an f-string, where *new* is
    inserted with its braces doubled so it can never add a placeholder
    and *old* must not contain a ``{...}`` placeholder.  The edit is
    rejected when the resulting module does not compile, when anything
    outside the constant's text changes (code, f-string placeholders),
    when the prompt's ``str.format`` fields change or when the SEA no
    longer loads or its prompt getter fails; the file is untouched then.
    Prose lines of *new* are wrapped at 92 columns (table rows are kept
    and must already fit).  Returns a one-line description of the change.
    """
    path = _editable_path(name)
    if path is None:
        return _not_editable(name)
    source = path.read_text(encoding="utf-8")
    getter, constant = _prompt_constant(source)
    if not constant:
        return f"Error: {path} has no prompt getter returning a module-level string constant"
    node, text = _prompt_text(source, constant)
    if old and text.count(old) != 1:
        return (
            f"Error: old text occurs {text.count(old)} times in {constant} (must be exactly once)"
        )
    new = _wrap_markdown(new)
    long_line = _too_long(new)
    if long_line:
        return (
            f"Error: a line of the new text is longer than {WRAP_COLUMNS} characters and cannot "
            f"be wrapped (shorten it or break it into several lines): {long_line[:60]}..."
        )
    fstring = isinstance(node, ast.JoinedStr)
    start, end = _node_span(source, node)
    raw = source.encode()
    if not old:
        # Append as an adjacent plain literal: Python joins it to the
        # constant, whatever its quoting, and it can never hold a placeholder.
        paragraph = "\n\n" + new.strip("\n") + "\n"
        literal = raw[start:end].decode() + " " + _string_literal(paragraph)
        updated = text + paragraph
    elif fstring:
        inside = _fstring_segment(text, text.index(old), len(old))
        if inside is None:
            return "Error: old text spans several string segments; replace a shorter piece"
        if inside:
            new = new.replace("{", "{{").replace("}", "}}")
        updated = literal = text.replace(old, new)
    else:
        updated = text.replace(old, new)
        literal = _string_literal(updated)
    candidate = (raw[:start] + literal.encode() + raw[end:]).decode()
    why = _gate(path, constant, source, candidate, text, updated, fstring)
    if why:
        return f"Error: {why}"
    path.write_text(candidate, encoding="utf-8")
    try:
        str(_execute_sea(path)[getter]())
    except Exception as exc:  # noqa: BLE001 - any load failure must roll the file back
        path.write_text(source, encoding="utf-8")
        return f"Error: the patched SEA no longer loads ({exc}); file restored"
    return (
        f"Patched {constant} of {path} via {getter}(): {len(text)} -> {len(updated)} chars, "
        f"{len(updated.splitlines()) - len(text.splitlines()):+d} lines"
    )


def write_autorouter_evidence(text: str) -> str:
    """Replace the observed-model-evidence block of the autorouter SEA's prompt with *text*.

    *text* is Markdown (a table plus a few bullets) describing what the
    task history shows about each model's cost, speed and reliability;
    the block is stamped with the current UTC date.  Goes through the
    same gate and wrapping as ``patch_sea_prompt``, so every table row
    must fit in 92 columns.
    """
    current = sea_prompt("autorouter")
    if current.startswith("Error:"):
        return current
    start = current.find(EVIDENCE_START)
    end = current.find(EVIDENCE_END)
    if start < 0 or end < start:
        return f"Error: the autorouter prompt has no {EVIDENCE_START} ... {EVIDENCE_END} block"
    old = current[start : end + len(EVIDENCE_END)]
    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    if text.lstrip().startswith(STAMP_PREFIX):  # the caller repeated the stamp line
        text = text.lstrip().split("\n", 1)[1] if "\n" in text.lstrip() else ""
    new = (
        f"{EVIDENCE_START}\n{STAMP_PREFIX}, refreshed {stamp} by /rsi7d._\n\n"
        f"{text.strip()}\n{EVIDENCE_END}"
    )
    return patch_sea_prompt("autorouter", old, new)


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git *args`` in *cwd* and return the completed process (no exception on failure)."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _task_row(task_id: str) -> dict[str, Any] | None:
    """Return the task_history row of *task_id* as a dict, or ``None``."""
    with persistence._rw_lock.read_lock():
        row = (
            persistence._get_db()
            .execute(persistence._HISTORY_SELECT + "WHERE id = ?", (task_id,))
            .fetchone()
        )
    return persistence._history_row_to_dict(row) if row is not None else None


def _toplevel(path: Path) -> Path | None:
    """Return the root of the git checkout containing *path*, or ``None``."""
    proc = _git("rev-parse", "--show-toplevel", cwd=path) if path.is_dir() else None
    return Path(proc.stdout.strip()) if proc is not None and proc.returncode == 0 else None


def _repo_of(tree: Path) -> Path:
    """Return the repository owning checkout *tree* (*tree* itself unless a linked worktree)."""
    proc = _git("rev-parse", "--git-common-dir", cwd=tree)
    common = (tree / proc.stdout.strip()).resolve() if proc.returncode == 0 else None
    if common is not None and common.name == ".git" and common.parent != tree.resolve():
        return common.parent
    return tree.resolve()


def _task_tree(work_dir: str) -> tuple[Path, Path, Path] | None:
    """Return ``(repo, tree, sub)`` for the directory a task ran in, or ``None``.

    ``repo`` is the repository root, ``tree`` the checkout the task ran
    in — ``repo`` itself or a worktree of it, even one removed since —
    and ``sub`` the task's directory relative to ``tree`` (``.`` for the
    root).  An existing directory is resolved through git (a repository
    nested inside another checkout is its own tree; a linked worktree's
    repository is the owner of its common git dir); a missing
    ``<repo>/.kiss-worktrees/kiss_wt-*/<sub>`` path is resolved by name.
    """
    path = Path(work_dir)
    tree = _toplevel(path) if work_dir else None
    if tree is not None:
        return _repo_of(tree), tree.resolve(), path.resolve().relative_to(tree.resolve())
    repo_text = strip_worktree_suffix(work_dir) if work_dir else work_dir
    repo = _toplevel(Path(repo_text)) if repo_text != work_dir else None
    if repo is None:
        return None
    depth = len(Path(repo_text).parts) + 2  # <repo>/.kiss-worktrees/kiss_wt-*
    return repo.resolve(), Path(*path.parts[:depth]), Path(*path.parts[depth:])


def _prompt_block(body: str) -> str | None:
    """Return the prompt an auto-commit message *body* quotes under ``User prompt:``, or ``None``.

    The canonical blocks end the message: the last ``User prompt:``
    heading opens the prompt, which runs to the ``Result:`` heading or
    the end of the message.
    """
    at = body.rfind(USER_PROMPT_HEADING)
    if at < 0:
        return None
    block = body[at + len(USER_PROMPT_HEADING) :]
    end = block.find(TASK_RESULT_HEADING)
    return block if end < 0 else block[:end]


def _base_commit(repo: Path, task: str, start_ms: int) -> tuple[str, str]:
    """Return ``(commit, source)``: the commit of *repo* the task started from.

    A task that changed files was auto-committed with its prompt under a
    ``User prompt:`` heading; the earliest commit made after the task
    started that quotes exactly *task* there (compared with whitespace
    runs collapsed, as git normalises messages) marks its end, and its
    first parent the state it started from.  Without such a commit the
    newest first-parent commit of ``HEAD`` from before *start_ms* is
    used.  ``("", "")`` when the repository has no commit from before
    the task.
    """
    start_s = start_ms // 1000
    wanted = _normalized(task)
    proc = _git(
        "log", "--all", f"--since=@{start_s}", "--format=%H%x1f%ct%x1f%P%x1f%B%x1e", cwd=repo
    )
    matches: list[tuple[int, str, str]] = []
    for record in proc.stdout.split("\x1e"):
        fields = record.strip("\n").split("\x1f", 3)
        if len(fields) < 4 or int(fields[1]) < start_s:
            continue
        block = _prompt_block(fields[3])
        if block is not None and _normalized(block) == wanted:
            parents = fields[2].split()
            matches.append((int(fields[1]), fields[0], parents[0] if parents else fields[0]))
    if matches:
        _committed, sha, parent = min(matches)
        return parent, f"first parent of the task's auto-commit {sha[:12]}"
    proc = _git("rev-list", "-1", "--first-parent", f"--before=@{start_s}", "HEAD", cwd=repo)
    sha = proc.stdout.strip()
    return sha, "HEAD of the repository when the task started" if sha else ""


def _normalized(text: str) -> str:
    """Return *text* with every whitespace run collapsed to one space and the ends stripped."""
    return " ".join(text.split())


def _work_root() -> Path:
    """Return this task's work directory (the cwd outside a task)."""
    agent = current_agent()
    return Path(agent.work_dir) if agent is not None and agent.work_dir else Path.cwd()


def prepare_replay_clone(task_id: str, name: str = "") -> dict[str, Any] | str:
    """Clone the repository of past run *task_id* at the commit it started from.

    Returns the replay parameters (``sea``, ``sea_file`` of this checkout,
    ``repo``, ``commit``, ``commit_source``, ``clone``, ``work_dir`` — the
    task's directory inside the clone — ``model`` and the ``task`` with
    repository paths rewritten to the clone) or an ``Error: ...`` string.
    *name* overrides the SEA recorded on the run.  An existing clone of
    the same run is replaced.
    """
    row = _task_row(task_id)
    if row is None:
        return f"Error: unknown task id {task_id!r}"
    sea = name or sea_name_of(row.get("sea") or "")
    if not sea:
        return f"Error: run {task_id} does not record its SEA; pass name=<sea>"
    sea_file = _editable_path(sea)
    if sea_file is None:
        return _not_editable(sea)
    work_dir = str(row.get("work_dir") or "")
    located = _task_tree(work_dir)
    if located is None:
        return (
            f"Error: the run's work dir {work_dir!r} is not inside a git repository, so there "
            "is nothing to clone; mark the run not replay-verified"
        )
    repo, tree, sub = located
    start_ms = int(row.get("start_ts") or 0) or int(float(row.get("timestamp") or 0) * 1000)
    commit, source = _base_commit(repo, str(row.get("task") or ""), start_ms)
    if not commit:
        return f"Error: {repo} has no commit from before the task started"
    clone = _work_root() / REPLAY_DIR / f"{sea}-{task_id[:8]}"
    shutil.rmtree(clone, ignore_errors=True)
    clone.parent.mkdir(parents=True, exist_ok=True)
    for label, args in (
        ("clone", ("clone", "--quiet", "--no-checkout", str(repo), str(clone))),
        ("checkout", ("-C", str(clone), "checkout", "--quiet", "--detach", commit)),
    ):
        proc = _git(*args)
        if proc.returncode:
            return f"Error: git {label} failed: {proc.stderr.strip()}"
    # One pass, longest alternative first: any worktree of the repository the task text
    # may name (the history strips worktree suffixes from work_dir, so the tree may be
    # the repository while the text names its worktree), the tree, the repository.
    worktrees = re.escape(str(repo)) + r"[/\\]\.kiss-worktrees[/\\]kiss_wt-[^/\\\s'\"]+"
    roots = sorted({str(tree), str(repo)}, key=len, reverse=True)
    task = re.sub(
        "|".join([worktrees, *map(re.escape, roots)]),
        str(clone).replace("\\", "\\\\"),
        str(row.get("task") or ""),
    )
    return {
        "sea": sea,
        "sea_file": str(sea_file),
        "repo": str(repo),
        "commit": commit,
        "commit_source": source,
        "clone": str(clone),
        "work_dir": str(clone / sub),
        "model": str(row.get("model") or ""),
        "task": task,
    }


def replay_in_clone(
    task_id: str, max_budget: float, timeout: float = 3600.0, name: str = "", model: str = ""
) -> str:
    """Replay past run *task_id* in a fresh clone of its repository at the commit it started from.

    Use it for tasks that modify files (papers, code, reports): the replay
    runs the SEA file of this checkout (with its patched prompt) inside
    ``tmp/rsi7d/replays/<sea>-<task id prefix>``, so it never touches this
    checkout.  The commit is the first parent of the task's own auto-commit
    (found through the ``User prompt:`` block of the commit message) or,
    without one, the repository HEAD at the task's start; repository paths
    in the task text are rewritten to the clone and the replay runs in the
    same directory of the clone the task ran in.  The replay runs without
    a worktree and without auto-commit, with the original run's model
    unless *model* is given, capped by *max_budget* (USD) and *timeout*
    (seconds); *name* overrides the SEA recorded on the run.  Returns JSON
    with ``replay_task_id`` (pass it to ``run_findings``), ``clone``,
    ``commit``, ``commit_source`` and the replay's ``result``.  Inspect
    what the replay changed with ``git -C <clone> status --short`` and
    delete the clone when done.
    """
    prepared = prepare_replay_clone(task_id, name)
    if isinstance(prepared, str):
        return prepared
    result = agent_dispatch.dispatch_result(
        prepared["sea"],
        prepared["task"],
        prepared["sea_file"],
        prepared["work_dir"],
        model or prepared["model"],
        max_budget,
        timeout,
        parent_agent=current_agent(),
        scope_work_dir=str(_work_root()),
        options=agent_dispatch.RunOptions(use_worktree=False, auto_commit=False),
    )
    if isinstance(result, str):
        prepared["replay_task_id"] = ""
        prepared["result"] = result
    else:
        prepared["replay_task_id"] = result.task_id
        prepared["result"] = {
            "success": result.success, "summary": result.text, "cost": result.cost,
            "steps": result.steps,
        }
    return json.dumps(prepared, indent=1)


def tools() -> list[Any]:
    """Trajectory mining, prompt inspection, the gated prompt editors and clone replays."""
    return [
        indexed_seas,
        sea_runs,
        sea_findings,
        run_findings,
        run_overview,
        run_transcript,
        run_entry,
        model_scorecard,
        sea_prompt,
        patch_sea_prompt,
        write_autorouter_evidence,
        replay_in_clone,
    ]
