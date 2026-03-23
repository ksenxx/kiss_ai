"""Shared utilities for Sorcar agent backends (chatbot UI and VS Code)."""

from __future__ import annotations

import logging

from kiss.core.models.model_info import _OPENAI_PREFIXES

logger = logging.getLogger(__name__)


def clean_llm_output(text: str) -> str:
    """Strip whitespace and surrounding quotes from LLM output."""
    return text.strip().strip('"').strip("'")


def clip_autocomplete_suggestion(query: str, suggestion: str) -> str:
    """Return the autocomplete continuation, stripped of the query prefix.

    Removes the query prefix if the LLM echoed it, strips surrounding
    whitespace, and stops at newlines.
    """
    s = clean_llm_output(suggestion)
    if not s:
        return ""
    if s.lower().startswith(query.lower()):
        s = s[len(query) :]
    s = s.split("\n")[0].strip()
    return s


def model_vendor(name: str) -> tuple[str, int]:
    """Return (vendor_display_name, sort_order) for a model name.

    Args:
        name: The model name string.

    Returns:
        Tuple of (display name, numeric sort order).
    """
    if name.startswith("claude-"):
        return "Anthropic", 0
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/"):
        return "OpenAI", 1
    if name.startswith("gemini-"):
        return "Gemini", 2
    if name.startswith("minimax-"):
        return "MiniMax", 3
    if name.startswith("openrouter/"):
        return "OpenRouter", 4
    return "Together AI", 5


def generate_followup_text(task: str, result: str, model: str) -> str:
    """Generate a follow-up task suggestion via LLM.

    Args:
        task: The completed task description.
        result: The task result summary (truncated to 500 chars internally).
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
        query: Case-insensitive substring to match against paths.
        usage: File usage counts keyed by path (insertion order
            encodes recency, last key = most recently used).
        limit: Maximum number of results to return.

    Returns:
        Sorted list of dicts with ``type`` (``"frequent"`` or ``"file"``)
        and ``text`` keys.
    """
    q = query.lower()
    frequent: list[dict[str, str]] = []
    rest: list[dict[str, str]] = []
    for path in file_cache:
        if not q or q in path.lower():
            item: dict[str, str] = {"type": "file", "text": path}
            if usage.get(path, 0) > 0:
                frequent.append(item)
            else:
                rest.append(item)

    def _end_dist(text: str) -> int:
        if not q:
            return 0
        pos = text.lower().rfind(q)
        if pos < 0:
            return len(text)
        return len(text) - (pos + len(q))

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
