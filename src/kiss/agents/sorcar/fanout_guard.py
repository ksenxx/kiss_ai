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
* The same quota caps the USD handed to reviewers over the whole tree
  when the user's prompt names an allowance ("at most 40% of the budget
  for reviewing", "spend under $30 on the review", "a third of the
  budget"; :func:`review_budget_from_prompt`): a review
  fan-out is clipped to what is left and refused once nothing useful
  is left (:meth:`ReviewQuota.reserve_budget`).
  The quota cannot cross a daemon dispatch (``run_agent`` runs the child
  in another process), but the reviewer marker does, so a reviewer's
  descendants stay reviewer-bound there too.
"""

from __future__ import annotations

import json
import re
import threading

from kiss.core.config import DEFAULT_CONFIG

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

REVIEWER_TASK_TEMPLATE = (
    "Verify the listed changes: {changes}. Report only demonstrated issues "
    "with file:line evidence and a reproduction or reasoning for each. Do "
    "not seek novel regressions elsewhere and do not invent problems; say "
    "so when a change is correct."
)
"""Canonical wording for a reviewer sub-agent task (mirrors SYSTEM.md).

Rounds 4+ of the audited review loops mostly re-flagged the same diff or
hunted for new problems; asking for verification of a named list keeps a
round bounded.  Callers format ``{changes}`` with the files/functions.
"""

REVIEW_BUDGET_REFUSAL = (
    "Review-budget cap reached: reviewer sub-agents have already been "
    "handed their share of this task's budget and may not receive more. "
    "Verify the remaining fixes yourself and finish with the findings you "
    "have."
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

    def __init__(
        self, limit: int = MAX_REVIEW_ROUNDS, budget: float | None = None,
    ) -> None:
        """Create a quota.

        Args:
            limit: Maximum number of review fan-outs in the task tree.
            budget: Total USD the tree's reviewer sub-agents may be
                handed (:meth:`reserve_budget`); ``None`` = unlimited.
        """
        self._limit = limit
        self._budget = budget
        self._lock = threading.Lock()
        self._used = 0
        self._reserved = 0.0

    @property
    def used(self) -> int:
        """Number of review rounds reserved so far."""
        with self._lock:
            return self._used

    @property
    def budget_left(self) -> float | None:
        """USD still available for reviewers, or ``None`` when unlimited."""
        with self._lock:
            return None if self._budget is None else max(0.0, self._budget - self._reserved)

    def reserve_budget(self, requested: float) -> float:
        """Atomically reserve up to *requested* USD of the review budget.

        Args:
            requested: USD the fan-out wants to hand to its reviewers.

        Returns:
            The USD granted: *requested* itself when the quota is
            unlimited or has room, otherwise whatever is left (``0.0``
            once exhausted).  Call :meth:`release` with the unspent part
            when the fan-out returns.
        """
        with self._lock:
            if self._budget is None:
                return requested
            granted = max(0.0, min(requested, self._budget - self._reserved))
            self._reserved += granted
            return granted

    def release(self, unspent: float) -> None:
        """Hand back the part of a reservation the reviewers did not spend."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - max(0.0, unspent))

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


_CURRENT_TASK_MARKER = "# Task (work on it now)"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.\n])\s+")
_REVIEW_SENTENCE_RE = re.compile(r"review|debug|audit", re.IGNORECASE)
_MONEY_SENTENCE_RE = re.compile(r"budget|spend|cost|\$|usd|dollar", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_USD_RE = re.compile(
    r"(?:\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|dollars?)\b)", re.IGNORECASE
)
_WORD_FRACTIONS = {
    "half": 0.5, "a third": 1 / 3, "one third": 1 / 3, "two thirds": 2 / 3,
    "a quarter": 0.25, "one quarter": 0.25, "three quarters": 0.75,
    "a fifth": 0.2, "one fifth": 0.2, "a tenth": 0.1, "one tenth": 0.1,
}
_WORD_FRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_FRACTIONS) + r")\b", re.IGNORECASE
)


def review_budget_from_prompt(prompt: str, max_budget: float) -> float | None:
    """Return the USD the user's task text allows reviewers to spend.

    The user states the cap in the prompt in whatever form comes
    naturally — "at most 50% of the task budget in gpt-5.6-sol for
    reviewing", "spend no more than $40 on the review", "keep the
    reviewers under a third of the budget" — so it is read from there
    rather than fixed in code.  Only the current task is searched (the
    chat prefix repeats earlier tasks' instructions) and only sentences
    that mention both reviewing/debugging/auditing and money or budget.
    The first such sentence wins.

    Args:
        prompt: The task prompt (possibly with the chat-history prefix).
        max_budget: The top-level task's budget cap in USD (fractions
            and percentages are taken of it; absolute amounts are
            clipped to it).

    Returns:
        The reviewers' USD allowance, or ``None`` when the text names none.
    """
    current = prompt.rsplit(_CURRENT_TASK_MARKER, 1)[-1]
    for sentence in _SENTENCE_SPLIT_RE.split(current):
        if not (_REVIEW_SENTENCE_RE.search(sentence) and _MONEY_SENTENCE_RE.search(sentence)):
            continue
        percent = _PERCENT_RE.search(sentence)
        if percent is not None:
            fraction = float(percent.group(1)) / 100.0
            return max_budget * fraction if 0.0 < fraction < 1.0 else None
        money = _USD_RE.search(sentence)
        if money is not None:
            amount = float(money.group(1) or money.group(2))
            return min(amount, max_budget) if amount > 0 else None
        words = _WORD_FRACTION_RE.search(sentence)
        if words is not None:
            return max_budget * _WORD_FRACTIONS[words.group(1).lower()]
    return None


def review_budget_for(max_budget: float, prompt: str = "") -> float | None:
    """Return the USD reviewers may be handed in a task with *max_budget*.

    The allowance comes from the task text
    (:func:`review_budget_from_prompt`); when the prompt names none,
    ``DEFAULT_CONFIG.review_budget_fraction`` applies, which is off (no
    cap) unless ``KISS_REVIEW_BUDGET_FRACTION`` is set.

    Args:
        max_budget: The top-level task's budget cap in USD.
        prompt: The task prompt to read the allowance from.

    Returns:
        The reviewers' USD allowance, or ``None`` for no cap.
    """
    from_prompt = review_budget_from_prompt(prompt, max_budget)
    if from_prompt is not None:
        return from_prompt
    fraction = DEFAULT_CONFIG.review_budget_fraction
    if not 0.0 < fraction < 1.0:
        return None
    return max_budget * fraction


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
    except (ValueError, TypeError):
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
