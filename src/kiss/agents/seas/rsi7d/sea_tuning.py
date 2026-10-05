# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Deterministic ``settings()`` tuning and eval-set mining for ``rsi7d``.

Pure functions over the run records :mod:`rsi7d_sea` mines from the task
history, so the ``rsi7d`` tools stay thin and this module is testable
without a database:

* :func:`propose_settings` turns a SEA's runs of the window into
  proposals for ``timeout`` (``2 x p95 duration``) and ``max_budget``
  (``1.5 x p95 cost``), never below the current value when a run was
  stopped by that limit, and flags ``tool_profile`` / ``use_web_tools``
  when runs failed on a missing tool.
* :func:`patch_settings_literal` rewrites one key of the dict literal a
  script's ``settings()`` returns (an AST-confined edit: nothing outside
  that one value changes), and :func:`accepts_runs` is the acceptance
  test a proposed limit must pass: no successful run of the window
  would have been cut off by it.
* :func:`eval_candidates` turns successful runs into ``skillopt`` eval
  tasks (``prompt`` = the task text, ``expect`` = stable substrings of
  the result, ``split`` by recency) and :func:`frequent_task_templates`
  lists repeated task texts as candidates for new commands.
* The settings change log closes the loop: :func:`record_change` /
  :func:`load_changes` keep one JSON line per ``patch_sea_settings``
  (what changed, from what, when, and the replay that settled it),
  :func:`pending_change` names the change that still awaits its
  keep-or-revert, and :func:`compare_change` measures the runs before
  and after a settled change so :func:`propose_settings` can propose a
  revert when the SEA got worse.
"""

from __future__ import annotations

import ast
import html
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

MIN_RUNS = 3
"""Finished runs needed before a limit is proposed (fewer is no evidence)."""

TIMEOUT_FACTOR = 2.0
BUDGET_FACTOR = 1.5
MIN_TOOL_FAILURES = 2
"""Runs failing on a missing tool before the tool settings are flagged."""

_LIMIT_STOP = re.compile(r"(?i)did not finish within|timed out|budget exceeded|was stopped")
_TOOL_MISSING = re.compile(
    r"(?i)tool (?:'[^']*' |\"[^\"]*\" )?(?:is )?not (?:available|found)|unknown tool|no tool named"
)
_HTML_TAG = re.compile(r"<[^>]+>")
_SENTENCE = re.compile(r"[^.!?\n]{24,160}[.!?]")


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile *fraction* (``0.95``) of *values*; ``0.0`` when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def stopped_by_limit(run: dict[str, Any]) -> bool:
    """Return whether *run* (a ``_run_record`` dict) ended by a timeout, budget or stop."""
    return run.get("status") != "success" and bool(_LIMIT_STOP.search(str(run.get("result") or "")))


def propose_settings(
    runs: list[dict[str, Any]],
    current: dict[str, Any],
    changes: list[dict[str, Any]] | None = None,
    tools_used: Counter[str] | None = None,
) -> dict[str, Any]:
    """Return the settings proposals for a SEA from its *runs* and *current* settings.

    Args:
        runs: The SEA's run records of the window (``duration_s``,
            ``cost``, ``status``, ``result``, ``started_ms``).
        current: The SEA's effective ``settings()`` (``timeout``,
            ``max_budget``, ``tool_profile``, ``use_web_tools`` when set).
        changes: The SEA's settings change log (:func:`load_changes`),
            oldest first; ``None`` when there is none.
        tools_used: How often each tool was called across *runs*
            (``None``: unknown, no narrowing is proposed).

    Returns:
        ``{"runs": n, "finished": m, "stopped_by_limit": k, "proposals":
        {key: {"current", "proposed", "basis"}}, "flags": [...],
        "pending": <change or None>}``.  A key is proposed only when
        the window holds at least :data:`MIN_RUNS` finished runs and
        the proposal differs from the current value; a limit is never
        lowered below the current value when a run was stopped by a
        limit.  A settled change whose runs got worse afterwards
        (:func:`compare_change`) becomes a revert proposal for its key,
        replacing any limit proposal; a change still awaiting its
        keep-or-revert is returned as ``pending`` and blocks every
        proposal for the SEA.  A SEA with the full toolset whose
        :data:`NARROW_MIN_RUNS` or more finished runs never called a
        tool outside a narrower profile gets that profile proposed.
    """
    done = [r for r in runs if r.get("status") != "running"]
    stopped = sum(stopped_by_limit(r) for r in done)
    durations = [
        float(r.get("duration_s") or 0) for r in done if float(r.get("duration_s") or 0) > 0
    ]
    costs = [float(r.get("cost") or 0) for r in done if float(r.get("cost") or 0) > 0]
    proposals: dict[str, dict[str, Any]] = {}
    if len(done) >= MIN_RUNS:
        for key, values, factor, rounding in (
            ("timeout", durations, TIMEOUT_FACTOR, 60),
            ("max_budget", costs, BUDGET_FACTOR, 0.5),
        ):
            if not values:
                continue
            p95 = percentile(values, 0.95)
            proposed = math.ceil(p95 * factor / rounding) * rounding
            cur = current.get(key)
            if stopped and isinstance(cur, int | float) and proposed < cur:
                proposed = cur
            if proposed != cur:
                proposals[key] = {
                    "current": cur,
                    "proposed": proposed,
                    "basis": (
                        f"{factor:g} x p95 of {len(values)} runs ({p95:.1f}), "
                        f"rounded up to {rounding}"
                    ),
                }
    tool_failures = [
        r
        for r in done
        if r.get("status") != "success" and _TOOL_MISSING.search(str(r.get("result") or ""))
    ]
    flags: list[str] = []
    if len(tool_failures) >= MIN_TOOL_FAILURES:
        flags.append(
            f"{len(tool_failures)} runs failed on a missing tool "
            f"(tool_profile={current.get('tool_profile', 'full')!r}, "
            f"use_web_tools={current.get('use_web_tools', 'default')!r}); widen the profile or "
            f"fix the prompt that asks for the tool"
        )
    narrowed = narrower_profile(done, current, tools_used)
    if narrowed:
        proposals["tool_profile"] = narrowed
    for change in latest_settled(changes or []):
        if current.get(change["key"]) != change["new"]:
            continue  # superseded: the file no longer holds the value this change wrote
        compared = compare_change(runs, change)
        if compared["worse"]:
            proposals[str(change["key"])] = {
                "current": change["new"],
                "proposed": change["old"],
                "basis": "revert: " + compared["basis"],
            }
    pending = pending_change(changes or [])
    if pending:
        proposals = {}
        flags.append(
            f"settings()[{pending['key']!r}] changed {pending['old']!r} -> {pending['new']!r} "
            f"at {pending['at']} and is not settled yet: replay a past run and call "
            f"settle_sea_settings before changing anything else"
        )
    return {
        "runs": len(runs),
        "finished": len(done),
        "stopped_by_limit": stopped,
        "proposals": proposals,
        "flags": flags,
        "pending": pending,
    }


NARROW_MIN_RUNS = 10
"""Finished runs of a full-toolset SEA needed before a narrower ``tool_profile`` is proposed."""

NARROWING_PROFILES = ("shell", "assistant", "review")
"""Profiles tried, smallest first, when every run's tools fit inside one of them."""


def narrower_profile(
    done: list[dict[str, Any]], current: dict[str, Any], tools_used: Counter[str] | None
) -> dict[str, Any] | None:
    """Return the ``tool_profile`` proposal for a full-toolset SEA whose runs used fewer tools.

    Args:
        done: The finished run records.
        current: The SEA's effective settings.
        tools_used: Tool-call names counted across *done*.

    Returns:
        ``{"current", "proposed", "basis"}`` naming the narrowest profile
        of :data:`NARROWING_PROFILES` that holds every tool the runs
        called, or ``None`` when the SEA already sets a profile, the
        evidence is thin (fewer than :data:`NARROW_MIN_RUNS` runs or no
        tool calls), or the runs used a tool no narrower profile has.
        Narrowing only: a profile is never widened here (a missing tool
        is a flag, not a proposal).
    """
    from kiss.agents.sorcar.sorcar_agent import TOOL_PROFILES

    if current.get("tool_profile") or len(done) < NARROW_MIN_RUNS or not tools_used:
        return None
    # Tools no profile governs (``finish``, a SEA's own tools) are always present.
    governed = set().union(*(names for names in TOOL_PROFILES.values() if names))
    used = {name for name, count in tools_used.items() if count > 0} & governed
    if not used:
        return None
    for profile in NARROWING_PROFILES:
        allowed = TOOL_PROFILES.get(profile)
        if allowed is not None and used <= allowed:
            return {
                "current": "",
                "proposed": profile,
                "basis": (
                    f"{len(done)} runs called only {', '.join(sorted(used))}; "
                    f"every one is in the {profile!r} profile"
                ),
            }
    return None


def change_log_path(home: Path) -> Path:
    """Return the settings change log of the Sorcar home *home*."""
    return home / "rsi7d" / "settings_changes.jsonl"


def load_changes(log: Path, sea: str) -> list[dict[str, Any]]:
    """Return the change records of *sea* in *log*, oldest first (empty when there is no log)."""
    if not log.is_file():
        return []
    records = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("sea") == sea:
                records.append(record)
    return records


def record_change(log: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append *record* to *log* with its ``at`` / ``at_ms`` timestamps filled in; return it.

    A record is ``{"sea", "key", "old", "new", "task_id"}`` for a
    change (``replay_task_id`` ``""`` until settled, ``reverted``
    ``False``) or the same keys plus ``"reverted": True`` for a revert.
    """
    now = time.time()
    record = {
        **record,
        "at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
        "at_ms": int(now * 1000),
    }
    record.setdefault("replay_task_id", "")
    record.setdefault("reverted", False)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def settle_change(log: Path, sea: str, replay_task_id: str) -> dict[str, Any] | None:
    """Mark the pending change of *sea* settled by *replay_task_id*; return it or ``None``.

    The log is rewritten with the record's ``replay_task_id`` filled.
    """
    pending = pending_change(load_changes(log, sea))
    if pending is None:
        return None
    lines = log.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() and json.loads(line) == pending:
            pending = {**pending, "replay_task_id": replay_task_id}
            lines[i] = json.dumps(pending, sort_keys=True)
            break
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pending


def latest_settled(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the newest settled (replayed, not reverted) change of each key, oldest first."""
    latest: dict[str, dict[str, Any]] = {}
    for change in changes:
        if change.get("replay_task_id") and not change.get("reverted"):
            latest[str(change["key"])] = change
    return list(latest.values())


def pending_change(changes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the newest change of *changes* that no replay settled and no later record
    reverted, else ``None``."""
    reverted_later: set[str] = set()
    for change in reversed(changes):
        if change.get("reverted"):
            reverted_later.add(str(change["key"]))
        elif not change.get("replay_task_id") and change["key"] not in reverted_later:
            return change
    return None


def compare_change(runs: list[dict[str, Any]], change: dict[str, Any]) -> dict[str, Any]:
    """Compare the runs before and after *change* (by ``started_ms``).

    Returns ``{"before": n, "after": m, "stop_rate_before", "stop_rate_after",
    "p95_cost_before", "p95_cost_after", "worse": bool, "basis": str}``.
    ``worse`` holds when at least :data:`MIN_RUNS` runs finished after
    the change and their limit-stop rate rose, or their p95 cost rose by
    more than :data:`BUDGET_FACTOR` over the runs before (the change
    made the SEA slower to finish or costlier, not better).
    """
    at = int(change.get("at_ms") or 0)
    done = [r for r in runs if r.get("status") != "running"]
    before = [r for r in done if int(r.get("started_ms") or 0) < at]
    after = [r for r in done if int(r.get("started_ms") or 0) >= at]

    def stop_rate(group: list[dict[str, Any]]) -> float:
        return sum(stopped_by_limit(r) for r in group) / len(group) if group else 0.0

    def p95_cost(group: list[dict[str, Any]]) -> float:
        return percentile([float(r.get("cost") or 0) for r in group], 0.95)

    rate_b, rate_a = stop_rate(before), stop_rate(after)
    cost_b, cost_a = p95_cost(before), p95_cost(after)
    worse = len(after) >= MIN_RUNS and (
        rate_a > rate_b or (cost_b > 0 and cost_a > cost_b * BUDGET_FACTOR)
    )
    basis = (
        f"{len(before)} runs before / {len(after)} after the change of {change.get('at')}: "
        f"limit-stop rate {rate_b:.0%} -> {rate_a:.0%}, p95 cost {cost_b:.2f} -> {cost_a:.2f}"
    )
    return {
        "before": len(before),
        "after": len(after),
        "stop_rate_before": rate_b,
        "stop_rate_after": rate_a,
        "p95_cost_before": cost_b,
        "p95_cost_after": cost_a,
        "worse": worse,
        "basis": basis,
    }


def accepts_runs(key: str, value: float, runs: list[dict[str, Any]]) -> str:
    """Return why a proposed *key* = *value* limit fails the acceptance test, or ``""``.

    The test is deterministic and replay-free: every successful run of
    the window must fit under the new ``timeout`` (duration) or
    ``max_budget`` (cost); a limit that would have cut one off is refused.
    """
    field = {"timeout": "duration_s", "max_budget": "cost"}.get(key)
    if field is None:
        return ""
    cut = [r for r in runs if r.get("status") == "success" and float(r.get(field) or 0) > value]
    if not cut:
        return ""
    worst = max(float(r.get(field) or 0) for r in cut)
    return (
        f"{key}={value:g} would have cut off {len(cut)} successful run(s) of the window "
        f"(largest {field} {worst:g}); refused"
    )


def start_of(node: ast.expr, offsets: list[int]) -> int:
    """Return the byte offset of *node*'s first character (*offsets*: line starts)."""
    return offsets[node.lineno - 1] + node.col_offset


def settings_literal(tree: ast.Module) -> ast.Dict | None:
    """Return the dict literal the module-level ``settings()`` returns, or ``None``.

    Only a ``return {...}`` of a literal dict is editable: a computed
    dict (``return {**base, ...}`` counts as a literal; ``return
    build()`` does not) has no place to write a key.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "settings":
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            if len(returns) == 1 and isinstance(returns[0].value, ast.Dict):
                return returns[0].value
    return None


REMOVE = object()
"""Pass as the *value* of :func:`patch_settings_literal` to delete the entry instead."""


def literal_value(source: str, key: str) -> Any:
    """Return the value ``settings()``'s dict literal writes under *key*; ``None`` when absent.

    Raises:
        ValueError: The entry's value is computed (a name, a call), so
            it cannot be recorded for a later revert.
    """
    literal = settings_literal(ast.parse(source))
    if literal is None:
        return None
    for k, v in zip(literal.keys, literal.values, strict=True):
        if isinstance(k, ast.Constant) and k.value == key:
            try:
                return ast.literal_eval(v)
            except ValueError:
                raise ValueError(
                    f"settings()[{key!r}] is computed ({ast.unparse(v)}); change it with "
                    "patch_sea_code"
                ) from None
    return None


def patch_settings_literal(source: str, key: str, value: Any) -> str:
    """Return *source* with ``settings()``'s literal entry *key* set to *value*.

    The entry's value is replaced in place when the key is a string
    literal of the dict; otherwise ``key: value`` is inserted before the
    closing brace.  :data:`REMOVE` deletes the entry (with its trailing
    comma and, when it stood alone, its line); removing an absent key
    changes nothing.  Nothing outside the dict literal changes; the
    result is re-parsed before it is returned.

    Raises:
        ValueError: ``settings()`` does not return one dict literal, or
            the rewritten module does not parse.
    """
    tree = ast.parse(source)
    literal = settings_literal(tree)
    if literal is None or literal.end_lineno is None or literal.end_col_offset is None:
        raise ValueError("settings() does not return a single dict literal")
    data = source.encode("utf-8")
    offsets = [0]
    for line in data.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    rendered = repr(value)
    for k, v in zip(literal.keys, literal.values, strict=True):
        if (
            isinstance(k, ast.Constant)
            and k.value == key
            and v.end_lineno is not None
            and v.end_col_offset is not None
        ):
            start = offsets[v.lineno - 1] + v.col_offset
            end = offsets[v.end_lineno - 1] + v.end_col_offset
            if value is REMOVE:
                start = offsets[k.lineno - 1] + k.col_offset
                key_end = offsets[(k.end_lineno or k.lineno) - 1] + (k.end_col_offset or 0)
                for _ in range(data[key_end:start_of(v, offsets)].count(b"(")):
                    closing = data[end:].lstrip(b" ")  # a parenthesised value: ``(7200)``
                    if closing.startswith(b")"):
                        end = len(data) - len(closing) + 1
                rest = data[end:]
                after_spaces = rest.lstrip(b" \t\n")
                if after_spaces.startswith(b","):
                    end += len(rest) - len(after_spaces) + 1
                line_start = data.rfind(b"\n", 0, start) + 1
                newline = data.find(b"\n", end)
                line_end = len(data) if newline == -1 else newline + 1
                tail = data[end:line_end].strip()
                head = data[:start].rstrip()
                if not data[line_start:start].strip() and (not tail or tail.startswith(b"#")):
                    start, end = line_start, line_end  # the entry (and its comment) had a line
                elif tail.startswith(b"}") and head.endswith(b","):
                    start = len(head) - 1  # the last inline entry: its leading comma goes too
                elif data[end : end + 1] == b" ":
                    end += 1  # inline entry: drop the space that followed it
                data = data[:start] + data[end:]
            else:
                data = data[:start] + rendered.encode("utf-8") + data[end:]
            break
    else:
        if value is REMOVE:
            return source
        close = offsets[literal.end_lineno - 1] + literal.end_col_offset - 1  # the ``}``
        # Quote the new key like the dict's first string key (``"`` by default).
        first = next(
            (k for k in literal.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)),
            None,
        )
        quote = (
            chr(data[offsets[first.lineno - 1] + first.col_offset]) if first is not None else '"'
        )
        quote = quote if quote in "'\"" else '"'
        entry = f"{quote}{key}{quote}: {rendered}".encode()
        line_start = offsets[literal.end_lineno - 1]
        if data[line_start:close].strip() == b"":
            # The brace closes a multi-line dict: the entry gets its own line, so a
            # trailing ``# comment`` on the previous line can never swallow it.
            indent = data[line_start:close] + b"    "
            if first is not None and first.lineno != literal.lineno:
                key_line = offsets[first.lineno - 1]
                indent = data[key_line : key_line + first.col_offset]
            if literal.values:
                last = literal.values[-1]
                if last.end_lineno is not None and last.end_col_offset is not None:
                    tail_start = offsets[last.end_lineno - 1] + last.end_col_offset
                    gap_code = b"\n".join(
                        line.split(b"#", 1)[0] for line in data[tail_start:line_start].split(b"\n")
                    )
                    if b"," not in gap_code:  # the last entry has no trailing comma yet
                        data = data[:tail_start] + b"," + data[tail_start:]
                        line_start += 1
            data = data[:line_start] + indent + entry + b",\n" + data[line_start:]
        else:
            before = data[:close].rstrip()
            sep = b"" if before.endswith((b"{", b",")) else b","
            pad = data[len(before) : close]
            data = before + sep + b" " + entry + pad + data[close:]
    result = data.decode("utf-8")
    ast.parse(result)
    return result


def result_summary(result: str) -> str:
    """Return the ``summary`` of a persisted result's YAML envelope, or *result* itself.

    A finished task's result is ``finish()``'s YAML (``success:``,
    ``is_continue:``, ``summary:``), possibly behind the ``ran:`` line
    :func:`~kiss.agents.sorcar.run_config.with_run_config` prepends.
    """
    try:
        parsed = yaml.safe_load(result or "")
    except yaml.YAMLError:
        return result or ""
    if isinstance(parsed, dict) and "summary" in parsed:
        return str(parsed["summary"] or "")
    return result or ""


def plain_text(result: str) -> str:
    """Strip HTML tags and entities the way ``skillopt.verify`` does before matching ``expect``."""
    return html.unescape(_HTML_TAG.sub("", result))


def expectations(result: str, limit: int = 3) -> list[str]:
    """Return up to *limit* stable sentences of *result*'s summary usable as ``expect``.

    The text is normalised exactly as ``skillopt`` normalises a rollout
    result before matching (:func:`plain_text`), so a sentence picked
    here matches the same output verbatim.  Sentences with digits,
    dates, paths or ids are skipped: they change from run to run.
    """
    text = plain_text(result_summary(result))
    picked: list[str] = []
    for match in _SENTENCE.finditer(text):
        sentence = " ".join(match.group(0).split())
        if re.search(r"\d|/|\\|_|\$", sentence):
            continue
        if sentence not in picked:
            picked.append(sentence)
        if len(picked) == limit:
            break
    return picked


def eval_candidates(rows: list[dict[str, Any]], select_fraction: float = 0.3) -> dict[str, Any]:
    """Return a ``skillopt`` eval set built from successful task rows.

    Args:
        rows: Task-history rows of one SEA's successful runs (``id``,
            ``task``, ``result``), newest first.
        select_fraction: Share of the newest rows marked ``split:
            "select"`` (the held-out acceptance split); the rest are
            ``train``, and at least one row always is (``skillopt``
            refuses an eval set without a training task).

    Returns:
        ``{"rollout": {...}, "tasks": [{"id", "prompt", "expect",
        "split", "source_task_id"}]}``; a row whose result yields no
        stable sentence gets no ``expect`` (it passes when the rollout
        finishes successfully).  Rows with an empty task are skipped.
    """
    usable = [r for r in rows if str(r.get("task") or "").strip()]
    n_select = max(0, min(math.ceil(len(usable) * select_fraction), len(usable) - 1))
    tasks = []
    for index, row in enumerate(usable):
        task: dict[str, Any] = {
            "id": f"hist-{row.get('id')}",
            "prompt": str(row["task"]).strip(),
            "split": "select" if index < n_select else "train",
            "source_task_id": str(row.get("id")),
        }
        expect = expectations(str(row.get("result") or ""))
        if expect:
            task["expect"] = expect
        tasks.append(task)
    return {
        "rollout": {"web_tools": False, "is_parallel": False, "use_memory": False},
        "tasks": tasks,
    }


def frequent_task_templates(
    rows: list[dict[str, Any]], min_repeats: int = 3
) -> list[dict[str, Any]]:
    """Return task texts repeated at least *min_repeats* times in *rows*, most frequent first.

    Each entry is ``{"task", "count", "models"}``: a repeated task is a
    candidate ``/<name>`` whose ``prompt(task)`` is the template (the
    history rows record the model a run used, not its tool profile).
    """
    counts: Counter[str] = Counter()
    models: dict[str, Counter[str]] = {}
    for row in rows:
        text = " ".join(str(row.get("task") or "").split())
        if not text:
            continue
        counts[text] += 1
        models.setdefault(text, Counter())[str(row.get("model") or "unknown")] += 1
    return [
        {"task": text, "count": count, "models": dict(models[text])}
        for text, count in counts.most_common()
        if count >= min_repeats
    ]
