# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Mechanical guardrails for spawning sub-agents (``run_parallel`` / ``run_agent``).

Three waste patterns showed up in a 24-hour audit of ``sorcar.db``
(2026-09-18): a review/fix loop that ran twelve "final" review rounds,
reviewer sub-agents that recursively fanned out into 156 reviewers, and
a literal ``"$(cat tasks.json)"`` passed as the ``tasks`` argument and
dispatched as one sub-agent task.  Prompting alone did not stop any of
them, so the rules live here as code:

* ``tasks`` must be a JSON array of non-empty task strings
  (:func:`parse_tasks_json`).
* A task counts as a *review* task when its text names reviewing,
  auditing, inspecting, or bug/regression hunting (:func:`is_review_task`).
  This is a lexical heuristic: paraphrases that avoid every listed stem
  escape it, and mentions such as "audit log" trip it.  It is the
  mechanical backstop, not a semantic classifier.
* One task tree launches at most :data:`MAX_REVIEW_ROUNDS` review
  fan-outs (:class:`ReviewQuota`, shared by every in-process agent of
  the tree), and an agent inside a reviewer's sub-tree may launch none.
  The quota cannot cross a daemon dispatch (``run_agent`` runs the child
  in another process), but the reviewer marker does, so a reviewer's
  descendants stay reviewer-bound there too.
"""

from __future__ import annotations

import json
import re
import threading

MAX_REVIEW_ROUNDS = 3

REVIEWER_SPAWN_REFUSAL = (
    "You are a reviewer sub-agent and may not spawn further reviewers. "
    "Do the review yourself with the read-only tools and report your "
    "findings."
)

REVIEW_CAP_REFUSAL = (
    f"Review-round cap reached: this task has already launched "
    f"{MAX_REVIEW_ROUNDS} review rounds via run_parallel and may not "
    "launch another. Verify the remaining fixes yourself and finish "
    "with the findings you have."
)

_REVIEW_WORDS = re.compile(
    r"\b(review\w*|audit\w*|critiqu\w*|inspect\w*|regression\w*"
    r"|read[- ]only|bug[- ]?hunt\w*|vulnerabilit\w*|adversarial\w*"
    r"|(find|hunt|check|look|search|scan)\w*\s+(for\s+)?"
    r"(bugs?|defects?|flaws?|issues?|mistakes?))\b",
    re.IGNORECASE,
)


class ReviewQuota:
    """Task-tree-wide budget of review fan-outs, shared by nested agents.

    One instance is created when a top-level task starts and handed to
    every ``run_parallel`` child (see ``run_tasks_parallel``), so
    intermediate helper sub-agents draw from the same budget instead of
    each getting a fresh one.  Reservation is atomic: concurrent
    fan-outs cannot jointly exceed the cap.
    """

    def __init__(self, limit: int = MAX_REVIEW_ROUNDS) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._used = 0

    @property
    def used(self) -> int:
        """Number of review rounds reserved so far."""
        with self._lock:
            return self._used

    def try_reserve(self) -> bool:
        """Atomically reserve one review round.

        Returns:
            True when a round was reserved; False when the cap is
            already exhausted.
        """
        with self._lock:
            if self._used >= self._limit:
                return False
            self._used += 1
            return True


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
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        hint = ""
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
