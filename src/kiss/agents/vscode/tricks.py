# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Load and prefix-match user "Inject instruction" tricks.

The user's trick strings live at ``~/.kiss/INJECTIONS.md`` (seeded
from the package copy bundled at ``src/kiss/INJECTIONS.md`` on first
install).  They power two UI features:

* The "Inject" sidebar panel — exposed via ``window.__TRICKS__`` by
  the HTML builder.
* Ghost-text fast-complete suggestions — surfaced at the start of
  each sentence by :func:`prefix_match_trick`.

Both consumers parse the same ``## Trick`` section format so the two
features stay in sync as the file is edited.
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.agents.vscode.user_assets import ensure_user_asset

# ``[.!?]`` followed by whitespace marks a sentence boundary.  The
# trailing ``\s+`` is greedy: any run of whitespace (including
# newlines) belongs to the boundary, so the partial begins at the
# first non-whitespace character of the next sentence.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+")


def read_tricks() -> list[str]:
    """Parse ``~/.kiss/INJECTIONS.md`` and return the trick texts.

    The user-local copy at ``~/.kiss/INJECTIONS.md`` is the runtime
    source of truth — ``install.sh`` seeds it from the package copy
    bundled at ``src/kiss/INJECTIONS.md`` on first install, and
    :func:`ensure_user_asset` seeds it again the first time it is
    read after a user wipes it; once present, user edits survive
    every read.

    The file contains a series of ``## Trick`` sections, each followed
    by a blank line and the trick text.  Returns an empty list if the
    file is missing or unparseable, so a deployment without
    INJECTIONS.md still degrades gracefully (no tricks rendered, no
    ghost suggestions).

    Returns:
        Ordered list of trick text strings (one per ``## Trick``
        section), preserving file order.
    """
    try:
        package_path = Path(__file__).parent.parent.parent / "INJECTIONS.md"
        tfile = ensure_user_asset("INJECTIONS.md", package_path)
        text = tfile.read_text()
    except (OSError, UnicodeDecodeError):
        # ``read_text`` raises ``UnicodeDecodeError`` (a ``ValueError``,
        # NOT an ``OSError``) when INJECTIONS.md has been corrupted into
        # a binary blob.  Letting it propagate would kill the singleton
        # autocomplete worker thread (which has no restart path), so
        # ghost text for the daemon's whole lifetime.
        return []
    tricks: list[str] = []
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines or lines[0].strip() != "Trick":
            continue
        body = "\n".join(lines[1:]).strip()
        if body:
            tricks.append(body)
    return tricks


def current_sentence_partial(query: str) -> str:
    """Return the partial of *query* that lies at the current sentence start.

    The partial is everything after the *last* sentence-ending
    punctuation (``.``, ``?``, ``!``) followed by whitespace.  Leading
    whitespace at the very start of *query* is also trimmed so a user
    typing ``"  Reproduce"`` is still treated as being at the start of
    the first sentence.

    Args:
        query: The full input string from the chat textarea.

    Returns:
        Substring of *query* starting at the current sentence boundary.
        Empty string when *query* itself is empty.

    Examples:
        >>> current_sentence_partial("Reproduce the issue")
        'Reproduce the issue'
        >>> current_sentence_partial("Hello. Reproduce")
        'Reproduce'
        >>> current_sentence_partial("What is it? Use")
        'Use'
        >>> current_sentence_partial("Done.\\nReproduce")
        'Reproduce'
        >>> current_sentence_partial("  Reproduce")
        'Reproduce'
    """
    if not query:
        return ""
    # Locate the *last* sentence boundary in the query: we want the
    # partial relative to the most recent sentence, not the first.
    last_boundary_end = 0
    for m in _SENTENCE_BOUNDARY.finditer(query):
        last_boundary_end = m.end()
    partial = query[last_boundary_end:]
    if last_boundary_end == 0:
        # No interior sentence boundary — strip any leading whitespace
        # so an indented first sentence still matches.
        partial = partial.lstrip()
    return partial


def prefix_match_trick(query: str, min_partial_len: int = 2) -> str:
    """Return the first INJECTIONS.md trick whose prefix matches *query*'s sentence start.

    Identifies the partial currently being typed *at the start of the
    most recent sentence* (see :func:`current_sentence_partial`) and
    returns the first trick whose case-sensitive prefix is exactly
    that partial.  Mirrors :func:`_prefix_match_task`'s
    case-sensitivity so capitalised tricks aren't auto-completed by
    lowercase keystrokes.

    Args:
        query: The full input string from the chat textarea.
        min_partial_len: Minimum non-whitespace length the sentence
            partial must have before a match is attempted.  Defaults
            to 2 — the same threshold ``_AutocompleteMixin._complete``
            uses for ghost text generally, so a single keystroke at
            the start of a sentence does not pop a suggestion.

    Returns:
        The full trick string that the partial prefixes, or ``""`` if
        no trick matches (or the partial is too short, or
        INJECTIONS.md is unavailable).
    """
    partial = current_sentence_partial(query)
    if len(partial) < min_partial_len:
        return ""
    for trick in read_tricks():
        if trick.startswith(partial) and len(trick) > len(partial):
            return trick
    return ""
