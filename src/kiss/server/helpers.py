# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Helper utilities for Sorcar agent backends (autocomplete, model info)."""

from __future__ import annotations

from kiss.core.models.model_info import _OPENAI_PREFIXES


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
        Tuple of (display name, numeric sort order).  Each vendor has its
        own order, so a sort on it keeps every vendor's models contiguous
        (the model picker draws a group header wherever the vendor changes).
    """
    if name.startswith(("claude-", "cc/")):
        return "Anthropic", 0
    if name.startswith("bedrock/"):
        return "Bedrock", 1
    if name.startswith("openai/") or name.startswith(_OPENAI_PREFIXES):
        return "OpenAI", 2
    if name.startswith("gemini-"):
        return "Gemini", 3
    if name.startswith("glm-"):
        return "Z.AI", 4
    if name.startswith("kimi-") or name.startswith("moonshot-"):
        return "Moonshot", 5
    if name.startswith("openrouter/"):
        return "OpenRouter", 6
    return "Together AI", 7


SUGGESTION_LIMIT = 20
