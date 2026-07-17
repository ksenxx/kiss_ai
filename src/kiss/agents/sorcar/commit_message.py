# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""LLM commit-message generation for auto-committed agent work.

These helpers live in sorcar (not the server layer) because sorcar's
auto-commit path (:mod:`kiss.agents.sorcar.sorcar_agent` /
``git_worktree``) is their primary consumer and sorcar code must only
depend on itself and ``kiss.core``.  The server layer re-exports them
from :mod:`kiss.server.helpers` for its own callers.
"""

from __future__ import annotations

import logging

from kiss.agents.sorcar.git_worktree import (
    TASK_RESULT_HEADING,
    USER_PROMPT_HEADING,
)
from kiss.core.models.model_info import get_fast_model

logger = logging.getLogger(__name__)


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


def _run_oneshot_llm(
    agent_name: str,
    prompt_template: str,
    arguments: dict[str, str],
    model: str,
    fallback: str,
    failure_log: str,
) -> str:
    """Run a single non-agentic LLM call and return the cleaned text.

    Wraps the boilerplate shared by *every* short LLM helper here and
    in :mod:`kiss.server.helpers`: construct a
    :class:`~kiss.core.kiss_agent.KISSAgent`, call ``run`` with
    ``is_agentic=False`` and ``verbose=False``, clean the response,
    and fall back to *fallback* on any exception or empty response.

    Args:
        agent_name: Display name for the underlying ``KISSAgent``
            instance.
        prompt_template: The prompt template passed to ``KISSAgent.run``.
        arguments: Template-arguments dict passed to ``KISSAgent.run``.
        model: The model name to use for generation.
        fallback: Returned verbatim on any exception or empty
            response.
        failure_log: Message used for the ``logger.debug`` call on
            exception (with ``exc_info=True``).

    Returns:
        Cleaned LLM output text, or *fallback* on failure.
    """
    from kiss.core.kiss_agent import KISSAgent

    try:
        agent = KISSAgent(agent_name)
        raw = agent.run(
            model_name=model,
            prompt_template=prompt_template,
            arguments=arguments,
            is_agentic=False,
            verbose=False,
        )
        return clean_llm_output(raw) or fallback
    except Exception:
        logger.debug(failure_log, exc_info=True)
        return fallback


def generate_commit_message_from_diff(
    diff_text: str,
    user_prompt: str | None = None,
    task_result: str | None = None,
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

    Returns:
        The cleaned commit-message string, or ``"kiss: auto-commit agent work"``
        on failure.
    """
    fallback = "kiss: auto-commit agent work"
    if not diff_text:
        msg = (
            _append_user_prompt(fallback, user_prompt)
            if user_prompt
            else fallback
        )
        return _append_task_result(msg, task_result) if task_result else msg
    if user_prompt:
        context = f"User task prompt:\n{user_prompt}\n\nDiff:\n{diff_text}"
        template = (
            "Generate a concise git commit message for these "
            "changes. The user's task prompt is provided for "
            "context — use it to phrase the subject line in "
            "terms of the user's INTENT, not just the mechanical "
            "diff. Use conventional commit format with a clear "
            "subject line (type: description) and optionally a "
            "body with bullet points for multiple changes. Do "
            "NOT quote or repeat the user prompt — it will be "
            "appended separately. Return ONLY the commit message "
            "text, no quotes or markdown fences.\n\n{context}"
        )
    else:
        context = f"Diff:\n{diff_text}"
        template = (
            "Generate a concise git commit message for these "
            "changes. Use conventional commit format with a "
            "clear subject line (type: description) and "
            "optionally a body with bullet points for multiple "
            "changes. Return ONLY the commit message text, no "
            "quotes or markdown fences.\n\n{context}"
        )
    msg = _run_oneshot_llm(
        agent_name="Commit Message Generator",
        prompt_template=template,
        arguments={"context": context},
        model=get_fast_model(),
        fallback=fallback,
        failure_log="Commit message generation failed",
    )
    msg = _append_user_prompt(msg, user_prompt) if user_prompt else msg
    return _append_task_result(msg, task_result) if task_result else msg


def _append_user_prompt(message: str, user_prompt: str) -> str:
    """Append the user's task prompt to a commit message body.

    Trims whitespace from *user_prompt* and appends it under a
    ``User prompt:`` heading separated by a blank line.  If the
    prompt is empty after trimming, *message* is returned unchanged.

    Args:
        message: The base commit message (subject + optional body).
        user_prompt: The user's original task prompt string.

    Returns:
        The combined commit message with the user prompt appended.
    """
    trimmed = user_prompt.strip()
    if not trimmed:
        return message
    # USER_PROMPT_HEADING is single-sourced in git_worktree.py: its
    # ``_ensure_task_metadata`` dedup detection depends on byte-exact
    # agreement with the block appended here.
    return f"{message.rstrip()}{USER_PROMPT_HEADING}{trimmed}"


def _append_task_result(message: str, task_result: str) -> str:
    """Append the task's result summary to a commit message body.

    Trims whitespace from *task_result* and appends it under a
    ``Result:`` heading separated by a blank line.  If the result
    is empty after trimming, *message* is returned unchanged.

    Args:
        message: The base commit message (subject + optional body).
        task_result: The task's result summary string.

    Returns:
        The combined commit message with the task result appended.
    """
    trimmed = task_result.strip()
    if not trimmed:
        return message
    # TASK_RESULT_HEADING is single-sourced in git_worktree.py (see
    # ``_append_user_prompt``).
    return f"{message.rstrip()}{TASK_RESULT_HEADING}{trimmed}"
