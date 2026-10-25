# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""LLM commit-message generation for auto-committed agent work.

These helpers live in sorcar (not the server layer) because sorcar's
auto-commit path (:mod:`kiss.agents.sorcar.sorcar_agent` /
``git_worktree``) is their primary consumer and sorcar code must only
depend on itself and ``kiss.core``.
"""

from __future__ import annotations

import logging
from typing import Any

from kiss.agents.sorcar.git_worktree import _ensure_task_metadata
from kiss.core.models.model_info import get_fast_model

logger = logging.getLogger(__name__)

FALLBACK_SUBJECT = "kiss: auto-commit agent work"
"""Commit subject used when no model-written message is available."""


def clean_llm_output(text: str) -> str:
    """Strip whitespace and *paired* surrounding quotes from LLM output.

    LLM responses routinely carry surrounding whitespace (a trailing
    newline is near-universal) *and* wrap the answer in quotes, e.g.
    ``"feat: add widget"\\n``.  The quote characters and the whitespace
    must both be removed regardless of their order: stripping quotes
    alone would leave the stray newline (and, when the newline sits
    *outside* the closing quote, would fail to reach the quote pair at
    all, leaving a dangling quote in the result).  Whitespace is
    therefore stripped first, then surrounding quotes, then any
    whitespace the quotes were hiding.

    Only **paired** quotes (the same quote character at both ends) are
    stripped.  ``str.strip('"')`` removes leading and trailing quote
    characters *independently*, so a message that legitimately ends
    (or starts) with a quoted word — e.g. ``feat: rename "foo"`` — was
    corrupted to ``feat: rename "foo`` with a dangling opening quote.
    An unpaired boundary quote is real content, not LLM decoration,
    and is preserved.
    """
    s = text.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def generate_commit_message_from_diff(
    diff_text: str,
    user_prompt: str | None = None,
    task_result: str | None = None,
    bill_to: Any = None,
) -> str:
    """Generate a git commit message from a diff via LLM.

    Uses a fast/cheap model to produce a conventional-commit-style
    message.  When *user_prompt* is provided, the user's original
    task prompt is included in the LLM context so the generated
    subject/body reflect the *intent* of the change (not just the
    mechanical diff) and the full prompt is appended at the end of
    the commit message body for traceability.  When *task_result*
    is provided, the task's result summary is appended after the
    user prompt under a ``Result:`` heading.  Returns a fallback
    string on any failure.

    Args:
        diff_text: Output of ``git diff --cached`` or similar.
        user_prompt: The user's task prompt that produced the diff,
            or ``None`` when not available (e.g. user-invoked manual
            commit-message generation from the UI).
        task_result: The task's result summary to append to the
            commit message, or ``None`` when not available.
        bill_to: The :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent`
            whose task the commit is made for; the model call's spend
            and tokens are attributed to it.  ``None`` (a manual commit
            outside any task) leaves the spend unattributed.

    Returns:
        The cleaned commit-message string, or
        :func:`fallback_commit_message` on failure.
    """
    subject = FALLBACK_SUBJECT
    if diff_text:
        subject = _commit_message_from_llm(diff_text, user_prompt, subject, bill_to)
    return _ensure_task_metadata(subject, user_prompt, task_result)


def fallback_commit_message(
    user_prompt: str | None = None, task_result: str | None = None,
) -> str:
    """The commit message used when no model-written one can be produced.

    The fixed :data:`FALLBACK_SUBJECT` plus the same ``User prompt:`` /
    ``Result:`` trailers :func:`generate_commit_message_from_diff`
    appends, so a commit made after the diff could not even be read
    still records what the agent was asked and what it reported.

    Args:
        user_prompt: The user's task prompt, or ``None``.
        task_result: The task's result summary, or ``None``.

    Returns:
        The fallback commit message.
    """
    return _ensure_task_metadata(FALLBACK_SUBJECT, user_prompt, task_result)


def _commit_message_from_llm(
    diff_text: str, user_prompt: str | None, fallback: str, bill_to: Any,
) -> str:
    """Ask the fast model for a commit message describing *diff_text*.

    Args:
        diff_text: The non-empty diff to describe.
        user_prompt: The task prompt behind the diff, when known; the
            model is told to phrase the subject after its intent.
        fallback: Returned when no model is available or the call fails.
        bill_to: The agent the call's spend is attributed to, or ``None``.
    """
    if user_prompt:
        context = f"User task prompt:\n{user_prompt}\n\nDiff:\n{diff_text}"
        intent = (
            "The user's task prompt is provided for context — use it to "
            "phrase the subject line in terms of the user's INTENT, not "
            "just the mechanical diff. "
        )
        no_quote = "Do NOT quote or repeat the user prompt — it will be appended separately. "
    else:
        context = f"Diff:\n{diff_text}"
        intent = no_quote = ""
    template = (
        f"Generate a concise git commit message for these changes. {intent}"
        "Use conventional commit format with a clear subject line "
        "(type: description) and optionally a body with bullet points for "
        f"multiple changes. {no_quote}Return ONLY the commit message text, "
        "no quotes or markdown fences.\n\n{context}"
    )
    from kiss.core.kiss_agent import KISSAgent

    generator = KISSAgent("Commit Message Generator")
    try:
        raw = generator.run(
            model_name=get_fast_model(),
            prompt_template=template,
            arguments={"context": context},
            is_agentic=False,
            verbose=False,
        )
        return clean_llm_output(raw) or fallback
    except Exception:
        logger.debug("Commit message generation failed", exc_info=True)
        return fallback
    finally:
        if bill_to is not None and (generator.budget_used > 0 or generator.total_tokens_used > 0):
            from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

            # Steps stay the task's own: the call is not one of its steps.
            _attribute_sub_usage(
                bill_to, generator.budget_used, generator.total_tokens_used, 0,
            )

