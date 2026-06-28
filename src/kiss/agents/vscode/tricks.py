# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Load and prefix-match user "Inject instruction" tricks.

The "Inject instruction" panel and the ghost-text fast-complete
suggestions both consume a single ordered list of trick strings
returned by :func:`read_tricks`, built from two sources:

1. ``~/.kiss/MY_INJECTION.md`` — user-curated tricks.  Auto-seeded on
   first read with the default starter

       ## Trick

       Write end-to-end 100% coverage tests for the feature first.  Then implement the feature.

   so a fresh install always shows at least one user-editable trick.
   Never overwritten once it exists — user edits survive every read.

2. The bundled ``src/kiss/INJECTIONS.md`` shipped with the package.
   Read **directly from the package**; no copy is ever written into
   ``~/.kiss/``.  This way every extension upgrade automatically
   delivers the latest bundled tricks without clobbering the user's
   curated list.

Order matters — user-curated tricks come first so a user who adds
their own trick at the top of MY_INJECTION.md sees it before the
bundled defaults in both the panel and ghost-text suggestions.

The bundled-file path can be overridden via the ``KISS_INJECTIONS_PATH``
environment variable, which the test suite uses to pin a known set of
bundled tricks for assertions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from kiss.agents.vscode.user_assets import ensure_user_asset_from_default

# ``[.!?]`` followed by whitespace marks a sentence boundary.  The
# trailing ``\s+`` is greedy: any run of whitespace (including
# newlines) belongs to the boundary, so the partial begins at the
# first non-whitespace character of the next sentence.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+")

#: User-visible body of the default ``## Trick`` section auto-seeded
#: into ``~/.kiss/MY_INJECTION.md`` on first read.  Two spaces between
#: sentences (matches the task spec verbatim).
MY_INJECTION_DEFAULT_BODY = (
    "Write end-to-end 100% coverage tests for the feature first."
    "  Then implement the feature."
)

#: Full default file content for ``~/.kiss/MY_INJECTION.md``.  A single
#: ``## Trick`` section whose body is :data:`MY_INJECTION_DEFAULT_BODY`.
#: A trailing newline matches the convention used by
#: ``MY_TASK_TEMPLATES.md`` (``## Task\n\nHi!\n``).
DEFAULT_MY_INJECTION = "## Trick\n\n" + MY_INJECTION_DEFAULT_BODY + "\n"


def _parse_trick_sections(text: str) -> list[str]:
    """Return the body of every ``## Trick`` section in *text*.

    Bodies are trimmed; empty bodies are skipped.  Mirrors the
    TypeScript ``readMarkdownSections`` parser used by ``SorcarTab.ts``.
    """
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


def _read_my_injection_tricks() -> list[str]:
    """Return the user-curated tricks from ``~/.kiss/MY_INJECTION.md``.

    Auto-seeds the file with :data:`DEFAULT_MY_INJECTION` on first read.
    Returns an empty list when ``~/.kiss/`` is not writable (so the
    seed cannot be written) or when the file is unreadable / corrupt.
    """
    user_path = ensure_user_asset_from_default(
        "MY_INJECTION.md", DEFAULT_MY_INJECTION,
    )
    if user_path is None:
        return []
    try:
        text = user_path.read_text()
    except (OSError, UnicodeDecodeError):
        # A corrupted MY_INJECTION.md (binary blob, bad encoding) must
        # not kill the singleton autocomplete worker thread — let the
        # daemon keep serving the bundled tricks.
        return []
    return _parse_trick_sections(text)


def _bundled_injections_path() -> Path:
    """Return the path to the bundled ``src/kiss/INJECTIONS.md``.

    Honours the ``KISS_INJECTIONS_PATH`` env override (used by the test
    suite to pin a deterministic set of bundled tricks), falling back
    to the file shipped inside the package.
    """
    override = os.environ.get("KISS_INJECTIONS_PATH")
    if override:
        return Path(override)
    # ``__file__`` is ``…/kiss/agents/vscode/tricks.py``; the bundled
    # INJECTIONS.md lives at ``…/kiss/INJECTIONS.md`` (two ``parent``s
    # up from ``vscode/``).
    return Path(__file__).parent.parent.parent / "INJECTIONS.md"


def _read_bundled_tricks() -> list[str]:
    """Return the tricks shipped in the bundled ``src/kiss/INJECTIONS.md``.

    Read directly from the package — no copy into ``~/.kiss/`` ever
    happens.  Returns an empty list if the file is missing or
    unreadable (graceful degradation).
    """
    path = _bundled_injections_path()
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    return _parse_trick_sections(text)


def read_tricks() -> list[str]:
    """Return the ordered "Inject instruction" trick list.

    The list is built by concatenating, in order:

    1. ``~/.kiss/MY_INJECTION.md`` (user-curated, auto-seeded with the
       default test-first trick on first read).
    2. The bundled ``src/kiss/INJECTIONS.md`` (read directly from the
       package; never copied into ``~/.kiss/``).

    Empty list if both files are unavailable so a deployment without
    either still degrades gracefully (no tricks rendered, no ghost
    suggestions).

    Returns:
        Ordered list of trick text strings, MY_INJECTION first then
        bundled.
    """
    return _read_my_injection_tricks() + _read_bundled_tricks()


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


def prefix_match_tricks(query: str, min_partial_len: int = 2) -> list[str]:
    """Return every trick whose prefix matches *query*'s current-sentence start.

    Identifies the partial currently being typed *at the start of the
    most recent sentence* (see :func:`current_sentence_partial`) and
    returns every trick whose case-sensitive prefix equals that
    partial.  Mirrors :func:`_prefix_match_tasks`'s case-sensitivity
    AND its "return up to N alternatives" contract so a dropdown menu
    can offer the user a choice when several tricks share a prefix
    (e.g. the bundled INJECTIONS.md ships two ``Reproduce the issue
    by writing …`` tricks — one for integration tests, one for
    end-to-end tests).

    Tricks are returned in file order (MY_INJECTION.md first, then
    bundled); deduplication handles the case where an editor
    inadvertently duplicates a trick.

    Args:
        query: The full input string from the chat textarea.
        min_partial_len: Minimum length the sentence partial must have
            before a match is attempted.  Defaults to 2 — the same
            threshold ``_AutocompleteMixin._complete`` uses for ghost
            text generally, so a single keystroke at the start of a
            sentence does not pop suggestions.

    Returns:
        Ordered list of full trick strings that the partial prefixes,
        empty when no trick matches (or the partial is too short, or
        the tricks files are unavailable).
    """
    partial = current_sentence_partial(query)
    if len(partial) < min_partial_len:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for trick in read_tricks():
        if trick in seen:
            continue
        if trick.startswith(partial) and len(trick) > len(partial):
            seen.add(trick)
            out.append(trick)
    return out


def prefix_match_trick(query: str, min_partial_len: int = 2) -> str:
    """Return the first trick whose prefix matches *query*'s current-sentence start.

    Singular convenience wrapper over :func:`prefix_match_tricks`,
    returning only the first match in file order.  The VS Code
    ghost-text pipeline (which can only display one ghost suffix at a
    time) uses this; the CLI dropdown — which can show every
    alternative — uses :func:`prefix_match_tricks`.

    Args:
        query: The full input string from the chat textarea.
        min_partial_len: Minimum length the sentence partial must
            have before a match is attempted (default 2).

    Returns:
        The first matching trick string, or ``""`` when no trick
        matches.
    """
    matches = prefix_match_tricks(query, min_partial_len)
    return matches[0] if matches else ""
