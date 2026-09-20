# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Helper utilities for Sorcar agent backends (autocomplete, model info, file ranking)."""

from __future__ import annotations

import posixpath
from collections import Counter

# ``generate_commit_message_from_diff`` is re-exported: production
# callers (``server.py``, ``merge_flow.py``) import it from here.
from kiss.agents.sorcar.commit_message import (
    generate_commit_message_from_diff,  # noqa: F401 — re-exported
)
from kiss.core.models.model_info import _OPENAI_PREFIXES
from kiss.server.diff_merge import _is_under


def clip_autocomplete_suggestion(query: str, suggestion: str) -> str:
    """Normalise an autocomplete continuation suffix for ghost display.

    *suggestion* is always a continuation **suffix** (a prefix-matched
    history task minus the query, or an identifier candidate minus the
    typed partial) — the call site (``_AutocompleteMixin._complete``)
    strips the query before calling.  It must therefore NOT be prefix-stripped again here: a
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


SUGGESTION_LIMIT = 20

# A directory with at least this many immediate subdirectories is a container
# of generated runs (``artifacts/<run_id>/``, ``jobs/<date>/``): source trees
# rarely have more than a couple of dozen sibling packages, while such
# containers hold hundreds of near-identical copies of the same files.
WIDE_DIR_MIN_CHILDREN = 50


def _wide_dirs(file_cache: list[str]) -> set[str]:
    """Find directories holding at least ``WIDE_DIR_MIN_CHILDREN`` subdirectories.

    Args:
        file_cache: Scanned relative paths; directory entries end with ``/``.

    Returns:
        Relative directory paths (no trailing ``/``) whose immediate
        subdirectory count reaches the threshold.  The repository root is
        never returned.
    """
    children = Counter(
        posixpath.dirname(p.rstrip("/")) for p in file_cache if p.endswith("/")
    )
    return {d for d, n in children.items() if d and n >= WIDE_DIR_MIN_CHILDREN}


def _path_depth(path: str) -> int:
    """Return how many directories deep *path* sits.

    Args:
        path: Slash-separated relative path; directories end with ``/``.

    Returns:
        Number of parent directories: ``README.md`` and ``src/`` are 0,
        ``src/kiss/`` and ``src/main.py`` are 1.
    """
    return path.rstrip("/").count("/")


def rank_file_suggestions(
    file_cache: list[str],
    query: str,
    usage: dict[str, int],
    limit: int = SUGGESTION_LIMIT,
) -> list[dict[str, str]]:
    """Rank and filter file paths by query match, recency, and usage.

    Files inside a wide container of generated runs (see ``_wide_dirs``)
    rank after every other match, so ``@test`` lists the project's own
    tests before the hundreds of ``tests/test_*.py`` copies under
    ``artifacts/<run_id>/``; they still appear when nothing else matches.
    Otherwise matches are ordered by how close to the end of the path
    the query occurs, and equally good matches shallowest first, so
    ``README.md`` precedes ``docs/guides/README.md``.

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
    wide = _wide_dirs(file_cache)
    rest.sort(
        key=lambda m: (
            _is_under(m["text"], wide),
            _end_dist(m["text"]),
            _path_depth(m["text"]),
        )
    )
    for f in frequent:
        f["type"] = "frequent"
    return (frequent + rest)[:limit]
