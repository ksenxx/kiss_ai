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
"""

from __future__ import annotations

import ast
import html
import math
import re
from collections import Counter
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


def propose_settings(runs: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    """Return the settings proposals for a SEA from its *runs* and *current* settings.

    Args:
        runs: The SEA's run records of the window (``duration_s``,
            ``cost``, ``status``, ``result``).
        current: The SEA's effective ``settings()`` (``timeout``,
            ``max_budget``, ``tool_profile``, ``use_web_tools`` when set).

    Returns:
        ``{"runs": n, "finished": m, "stopped_by_limit": k, "proposals":
        {key: {"current", "proposed", "basis"}}, "flags": [...]}``.  A
        key is proposed only when the window holds at least
        :data:`MIN_RUNS` finished runs and the proposal differs from
        the current value; a limit is never lowered below the current
        value when a run was stopped by a limit.
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
    return {
        "runs": len(runs),
        "finished": len(done),
        "stopped_by_limit": stopped,
        "proposals": proposals,
        "flags": flags,
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


def patch_settings_literal(source: str, key: str, value: Any) -> str:
    """Return *source* with ``settings()``'s literal entry *key* set to *value*.

    The entry's value is replaced in place when the key is a string
    literal of the dict; otherwise ``key: value`` is inserted before the
    closing brace.  Nothing outside the dict literal changes; the result
    is re-parsed before it is returned.

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
            data = data[:start] + rendered.encode("utf-8") + data[end:]
            break
    else:
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
