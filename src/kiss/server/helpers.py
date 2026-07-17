# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Helper utilities for Sorcar agent backends (autocomplete, model info, file ranking)."""

from __future__ import annotations

import logging
from typing import Any

# The commit-message helpers moved to
# ``kiss.agents.sorcar.commit_message`` (sorcar's auto-commit path is
# their primary consumer and sorcar must not depend on the server
# layer); they are re-exported here for this module's historical
# importers.
from kiss.agents.sorcar.commit_message import (  # noqa: F401 — re-exported
    _append_task_result,
    _append_user_prompt,
    _run_oneshot_llm,
    clean_llm_output,
    generate_commit_message_from_diff,
)
from kiss.core.models.model_info import _OPENAI_PREFIXES
from kiss.server.json_printer import JsonPrinter

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
