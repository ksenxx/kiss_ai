# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Helper utilities for Sorcar agent backends (autocomplete, model info, file ranking)."""

from __future__ import annotations

import logging
from typing import Any

from kiss.agents.sorcar.git_worktree import (
    TASK_RESULT_HEADING,
    USER_PROMPT_HEADING,
)
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.core.models.model_info import _OPENAI_PREFIXES, get_fast_model

logger = logging.getLogger(__name__)


def tab_owns_answer_queue(tab: Any, task_key: str) -> bool:
    """Return whether *tab*'s live answer queue belongs to *task_key*.

    Task-ownership filter (BUG-TR2-2) shared by
    ``commands._resolve_user_answer_queue`` and
    ``task_runner._resolve_task_answer_queue``:
    ``JsonPrinter.cleanup_task`` intentionally preserves subscriber
    sets of FINISHED tasks, so a tab that co-subscribed to an old,
    finished task may now be running a brand-new UNRELATED task — its
    live ``user_answer_queue`` belongs to THAT task.  Delivering a
    stale answer there would answer the wrong question and dismiss the
    wrong task's askUser modal.  A tab is disqualified only when its
    live agent is actively running a task other than *task_key*.

    Args:
        tab: The ``_RunningAgentState`` whose queue is a candidate.
        task_key: The coerced task id the answer/question belongs to.

    Returns:
        True when the tab's queue may serve *task_key*.
    """
    agent = getattr(tab, "agent", None)
    agent_task = (
        JsonPrinter._coerce_task_id(getattr(agent, "_last_task_id", None))
        if agent is not None
        else ""
    )
    return not (tab.is_task_active and agent_task and agent_task != task_key)


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

    Wraps the boilerplate shared by *every* short LLM helper in this
    module: construct a :class:`KISSAgent`, call ``run`` with
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


def clip_autocomplete_suggestion(query: str, suggestion: str) -> str:
    """Normalise an autocomplete continuation suffix for ghost display.

    *suggestion* is always a continuation **suffix** (a prefix-matched
    history task minus the query, or an identifier candidate minus the
    typed partial) — both call sites (``_AutocompleteMixin._complete``
    and ``CliCompleter._active_file_suffix``) strip the query before
    calling.  It must therefore NOT be prefix-stripped again here: a
    legitimate suffix can itself begin with the query text (active file
    holds ``quxqux_token``, user typed ``qux`` → suffix ``qux_token``),
    and re-stripping would corrupt the accepted completion (``qux`` +
    ``_token`` types the non-existent ``qux_token``).

    For the same reason it must NOT strip quote characters at the
    suffix boundary: they are real characters of the matched history
    task, not LLM decoration (history ``run "make test"`` typed as
    ``run "make`` continues with `` test"`` — stripping the closing
    quote would make the accepted completion diverge from the task).
    Quote-stripping belongs only to :func:`clean_llm_output`, which
    handles raw LLM responses.

    Stops at newlines — including CR/CRLF and unicode line
    boundaries, so a CRLF-sourced suggestion never leaks a trailing
    ``"\\r"`` (or an embedded lone CR) into the ghost text.

    Normalises the cursor-to-ghost gap so the overlay (which uses
    ``white-space: pre-wrap``) never renders visible extra spaces
    between the user's cursor and the start of the ghost text:

    - When the user's query is empty or already ends in whitespace, the
      user's cursor (or empty input) already provides the gap, so any
      leading whitespace on the suggestion would render as visible
      *extra* spaces.  All leading whitespace is stripped.

    - When the query ends in a non-whitespace character, exactly one
      space is allowed as the legitimate cursor-to-ghost separator
      (e.g. query ``"fix"`` + suggestion ``" the bug"`` reads as
      ``"fix the bug"``).  Any *additional* leading whitespace is the
      same visible-padding bug — it happens when the prefix-matched
      history task contains consecutive spaces (e.g. user types
      ``"parse"`` and history holds ``"parse  arguments"`` with two
      spaces) — and is collapsed away, leaving exactly one separator
      space.  A suggestion that starts with non-whitespace gets no
      separator prepended (e.g. identifier completion ``"os.pa"`` →
      ``"th"``).
    """
    s = suggestion
    if not s:
        return ""
    s = s.splitlines()[0]
    if not query or (query[-1:].isspace()):
        s = s.lstrip()
    else:
        stripped = s.lstrip()
        if len(stripped) != len(s):
            s = " " + stripped
    return s


def model_vendor(name: str) -> tuple[str, int]:
    """Return (vendor_display_name, sort_order) for a model name.

    Args:
        name: The model name string.

    Returns:
        Tuple of (display name, numeric sort order).
    """
    if name.startswith("claude-") or name.startswith("cc/"):
        return "Anthropic", 0
    if name.startswith("openai/") or name.startswith(_OPENAI_PREFIXES):
        return "OpenAI", 1
    if name.startswith("gemini-"):
        return "Gemini", 2
    if name.startswith("glm-"):
        return "Z.AI", 3
    if name.startswith("kimi-") or name.startswith("moonshot-"):
        return "Moonshot", 4
    if name.startswith("openrouter/"):
        return "OpenRouter", 5
    return "Together AI", 6


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


def generate_followup_text(task: str, result: str, model: str) -> str:
    """Generate a follow-up task suggestion via LLM.

    Args:
        task: The completed task description.
        result: The task result summary.
        model: The model to use for generation.

    Returns:
        Suggestion text, or empty string on failure.
    """
    return _run_oneshot_llm(
        agent_name="Followup Proposer",
        prompt_template=(
            "A developer just completed this task:\n"
            "Task: {task}\n"
            "Result summary: {result}\n\n"
            "Suggest ONE short, concrete follow-up task they "
            "might want to do next. Return ONLY the task "
            "description as a single plain-text sentence."
        ),
        arguments={"task": task, "result": result},
        model=model,
        fallback="",
        failure_log="Followup generation failed",
    )


# Maximum number of dropdown suggestion items emitted to the webview
# per request.  Single-sourced here and shared by the @-mention file
# picker (:func:`rank_file_suggestions`) and the fast-complete
# dropdown (``autocomplete``) so the two pickers stay scrollable
# without UI tuning differences between them.
SUGGESTION_LIMIT = 20


def rank_file_suggestions(
    file_cache: list[str],
    query: str,
    usage: dict[str, int],
    limit: int = SUGGESTION_LIMIT,
) -> list[dict[str, str]]:
    """Rank and filter file paths by query match, recency, and usage.

    Args:
        file_cache: List of file paths to search.
        query: Case-sensitive substring to match against paths.
        usage: File usage counts keyed by path (insertion order
            encodes recency, last key = most recently used).
        limit: Maximum number of results to return.

    Returns:
        Sorted list of dicts with ``type`` (``"frequent"`` or ``"file"``)
        and ``text`` keys.
    """
    frequent: list[dict[str, str]] = []
    rest: list[dict[str, str]] = []
    for path in file_cache:
        if not query or query in path:
            item: dict[str, str] = {"type": "file", "text": path}
            if usage.get(path, 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)

    def _end_dist(text: str) -> int:
        if not query:
            return 0
        pos = text.rfind(query)
        if pos < 0:  # pragma: no cover — files are pre-filtered by query match
            return len(text)
        return len(text) - (pos + len(query))

    _usage_keys = list(usage.keys())
    _recency = {k: i for i, k in enumerate(reversed(_usage_keys))}
    _n = len(_usage_keys)
    frequent.sort(
        key=lambda m: (
            _end_dist(m["text"]),
            _recency.get(m["text"], _n),
            -usage.get(m["text"], 0),
        )
    )
    rest.sort(key=lambda m: _end_dist(m["text"]))
    for f in frequent:
        f["type"] = "frequent"
    return (frequent + rest)[:limit]
