"""Helper utilities for Sorcar agent backends (autocomplete, model info, file ranking)."""

from __future__ import annotations

import logging

from kiss.core.models.model_info import _OPENAI_PREFIXES, get_fast_model

logger = logging.getLogger(__name__)


def clean_llm_output(text: str) -> str:
    """Strip whitespace and surrounding quotes from LLM output."""
    return text.strip('"').strip("'")


def clip_autocomplete_suggestion(query: str, suggestion: str) -> str:
    """Return the autocomplete continuation, stripped of the query prefix.

    Removes the query prefix if the LLM echoed it, strips surrounding
    whitespace, and stops at newlines.

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
    s = clean_llm_output(suggestion)
    if not s:
        return ""
    if s.lower().startswith(query.lower()):
        s = s[len(query) :]
    s = s.split("\n")[0]
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
    if name.startswith("minimax-"):
        return "MiniMax", 3
    if name.startswith("openrouter/"):
        return "OpenRouter", 4
    return "Together AI", 5


def generate_commit_message_from_diff(
    diff_text: str, user_prompt: str | None = None,
) -> str:
    """Generate a git commit message from a diff via LLM.

    Uses a fast/cheap model to produce a conventional-commit-style
    message.  When *user_prompt* is provided, the user's original
    task prompt is included in the LLM context so the generated
    subject/body reflect the *intent* of the change (not just the
    mechanical diff) and the full prompt is appended at the end of
    the commit message body for traceability.  Returns a fallback
    string on any failure.

    Args:
        diff_text: Output of ``git diff --cached`` or similar.
        user_prompt: The user's task prompt that produced the diff,
            or ``None`` when not available (e.g. user-invoked manual
            commit-message generation from the UI).

    Returns:
        The cleaned commit-message string, or ``"kiss: auto-commit agent work"``
        on failure.
    """
    from kiss.core.kiss_agent import KISSAgent

    fallback = "kiss: auto-commit agent work"
    if not diff_text:
        if user_prompt:
            return _append_user_prompt(fallback, user_prompt)
        return fallback
    try:
        agent = KISSAgent("Commit Message Generator")
        if user_prompt:
            context = (
                f"User task prompt:\n{user_prompt}\n\nDiff:\n{diff_text}"
            )
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
        raw = agent.run(
            model_name=get_fast_model(),
            prompt_template=template,
            arguments={"context": context},
            is_agentic=False,
            verbose=False,
        )
        msg = clean_llm_output(raw) or fallback
        if user_prompt:
            msg = _append_user_prompt(msg, user_prompt)
        return msg
    except Exception:
        logger.debug("Commit message generation failed", exc_info=True)
        if user_prompt:
            return _append_user_prompt(fallback, user_prompt)
        return fallback


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
    return f"{message.rstrip()}\n\nUser prompt:\n{trimmed}"


def generate_followup_text(task: str, result: str, model: str) -> str:
    """Generate a follow-up task suggestion via LLM.

    Args:
        task: The completed task description.
        result: The task result summary.
        model: The model to use for generation.

    Returns:
        Suggestion text, or empty string on failure.
    """
    from kiss.core.kiss_agent import KISSAgent

    try:
        agent = KISSAgent("Followup Proposer")
        raw = agent.run(
            model_name=model,
            prompt_template=(
                "A developer just completed this task:\n"
                "Task: {task}\n"
                "Result summary: {result}\n\n"
                "Suggest ONE short, concrete follow-up task they "
                "might want to do next. Return ONLY the task "
                "description as a single plain-text sentence."
            ),
            arguments={"task": task, "result": result},
            is_agentic=False,
            verbose=False,
        )
        return clean_llm_output(raw)
    except Exception:
        logger.debug("Followup generation failed", exc_info=True)
        return ""


def rank_file_suggestions(
    file_cache: list[str],
    query: str,
    usage: dict[str, int],
    limit: int = 20,
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
