# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Argument checks and task classification for sub-agent fan-outs.

Shared by ``run_parallel`` / ``run_agent`` and the daemon:

* ``tasks`` must be a JSON array of non-empty task strings
  (:func:`parse_tasks_json`).  A literal ``"$(cat tasks.json)"`` passed
  as the argument used to be dispatched as one sub-agent task.
* A task counts as a *review* task when its text names reviewing,
  auditing, inspecting, or bug/regression hunting (:func:`is_review_task`).
  This is a lexical heuristic: paraphrases that avoid every listed stem
  escape it, and mentions such as "audit log" trip it.  A review task
  that does not also ask for changes (:func:`is_implementation_task`)
  puts the child on the read-only ``review`` tool profile.  The
  reviewer marker is inherited down the sub-tree and across daemon
  dispatches so a reviewer's helpers get the same reduced toolset.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

_REVIEW_WORDS = re.compile(
    r"\b(review\w*|audit\w*|critiqu\w*|inspect\w*|regression\w*"
    r"|read[- ]only|bug[- ]?hunt\w*|vulnerabilit\w*|adversarial\w*"
    r"|(find|hunt|check|look|search|scan)\w*\s+(for\s+)?"
    r"(bugs?|defects?|flaws?|issues?|mistakes?))\b",
    re.IGNORECASE,
)


def is_review_task(task: str) -> bool:
    """Return whether *task* reads like a review / audit / bug-hunt task.

    Args:
        task: A sub-agent task description.

    Returns:
        True when the text contains a reviewer stem (``review``,
        ``audit``, ``critique``, ``inspect``, ``regression``,
        ``read-only``, ``bug-hunt``, ``vulnerabilit…``, ``adversarial``)
        or a bug-hunting phrase such as "find bugs" / "check for
        defects" (case-insensitive).
    """
    return _REVIEW_WORDS.search(task) is not None


REVIEW_BUDGET_BASE_USD = 5.0
"""Budget (USD) of a read-only review of an empty diff.  The 2,535
reviewer sub-agents of the three weeks before 2026-10-25 cost $1.57 at
the median and $3.27 at the 75th percentile while holding 28x their
spend; the base alone covers three reviews in four."""
REVIEW_BUDGET_PER_KLINE_USD = 5.0
"""Budget (USD) added per 1,000 changed lines the review has to read."""


def changed_lines(work_dir: str) -> int:
    """Return the lines added and deleted in *work_dir* against ``HEAD``, untracked files included.

    ``0`` when *work_dir* is not a git repository or git is missing;
    binary files count as none.

    Args:
        work_dir: The directory whose diff a review is to read.
    """
    try:
        numstat = subprocess.run(
            ["git", "diff", "--numstat", "HEAD"], cwd=work_dir, capture_output=True,
            text=True, timeout=30, check=False,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=work_dir,
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    lines = 0
    for row in numstat.splitlines():
        # ``<added>\t<deleted>\t<path>``; binary files show ``-`` counts.
        columns = row.split("\t", 2)
        for count in columns[:2]:
            lines += int(count) if count.isdigit() else 0
    for name in untracked.splitlines():
        try:
            with open(os.path.join(work_dir, name), "rb") as handle:
                lines += sum(1 for _ in handle)
        except OSError:
            continue
    return lines


def review_budget(work_dir: str) -> float:
    """Return the default budget (USD) of a read-only review dispatched from *work_dir*.

    :data:`REVIEW_BUDGET_BASE_USD` plus :data:`REVIEW_BUDGET_PER_KLINE_USD`
    per thousand changed lines (:func:`changed_lines`): a review's
    cost follows the size of the diff it reads, not the caller's
    remaining budget.

    Args:
        work_dir: The directory the review runs in.
    """
    return REVIEW_BUDGET_BASE_USD + REVIEW_BUDGET_PER_KLINE_USD * changed_lines(work_dir) / 1000


_IMPLEMENTATION_WORDS = re.compile(
    r"\b(implement\w*|fix\w*|patch\w*|refactor\w*|create|add|write|modify|edit|update"
    r"|delete|remove|rename|migrate|install|train\w*|build|generate|rewrite)\b",
    re.IGNORECASE,
)


def is_implementation_task(task: str) -> bool:
    """Return whether *task* asks for changes, not just a verdict.

    Used to keep the reduced ``review`` tool profile away from tasks
    that merely mention a review topic ("implement a regression test",
    "patch the vulnerability", "review X and fix what you find"): when
    in doubt the child keeps the full toolset.

    Args:
        task: A sub-agent task description.

    Returns:
        True when the text contains an implementation verb.
    """
    return _IMPLEMENTATION_WORDS.search(task) is not None


# A ``\\`` pair, or a lone backslash before a character that is not a JSON
# escape (``\|``, ``\(``, ``\.`` in grep/sed patterns).  Replacing every
# match with a pair leaves valid pairs alone and repairs the lone ones.
_LONE_BACKSLASH = re.compile(r'\\\\|\\(?![/"bfnrtu])')


def _decode_json_leniently(text: str) -> object:
    """Decode *text* as JSON, repairing two mistakes models keep making.

    Shell commands inside JSON strings carry ``\\|`` / ``\\(`` escapes
    that are invalid JSON, and heredocs carry raw newlines.  Strict
    decoding failed 21 reviewer calls in one day, each a wasted step, so
    the raw control characters are accepted (``strict=False``) and, when
    that still fails, lone backslashes are doubled before a second try.

    Args:
        text: The raw JSON text.

    Returns:
        The decoded value.

    Raises:
        ValueError: If the text is not JSON even after the repair.
    """
    try:
        return json.loads(text, strict=False)
    except ValueError:
        return json.loads(_LONE_BACKSLASH.sub(r"\\\\", text), strict=False)


def parse_tasks_json(tasks: str, name: str = "tasks") -> list[str]:
    """Parse a JSON-array-of-strings tool argument strictly.

    Used for the ``tasks`` argument of ``run_parallel`` and the
    ``commands`` argument of ``run_commands_parallel``.

    Args:
        tasks: The raw string the model passed.
        name: The argument's name, used in the error messages.

    Returns:
        The decoded, non-empty list of strings.

    Raises:
        ValueError: If *tasks* is not valid JSON, is not an array, is
            empty, or holds anything but non-empty strings.  The message
            tells the model how to fix the call, including the case of a
            shell substitution such as ``$(cat file)`` that was never
            expanded.
    """
    stripped = tasks.strip()
    try:
        parsed = _decode_json_leniently(stripped)
    except ValueError:
        hint = (
            " Inside a JSON string every backslash must be doubled (write "
            '\\\\| for grep\'s \\|) and a newline written as \\n.'
        )
        if stripped.startswith("$(") or stripped.startswith("`"):
            hint = (
                " Shell substitutions are not expanded in tool arguments; "
                "Read the file first and paste its JSON array here."
            )
        raise ValueError(
            f"{name} must be a JSON array of strings, got {stripped[:80]!r}."
            + hint
        ) from None
    if not isinstance(parsed, list):
        raise ValueError(
            f"{name} must be a JSON array of strings, got a JSON "
            f"{type(parsed).__name__}."
        )
    if not parsed:
        raise ValueError(f"{name} is an empty array; nothing to run.")
    if not all(isinstance(t, str) and t.strip() for t in parsed):
        raise ValueError(
            f"{name} must be a JSON array of non-empty strings."
        )
    return parsed
