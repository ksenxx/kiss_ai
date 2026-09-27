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
import threading
from pathlib import Path
from typing import TypedDict

from kiss.server.user_assets import ensure_user_asset_from_default

_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+")

# CommonMark §2.4 backslash escapes: only ASCII punctuation may be
# escaped.  Byte-for-byte the character class of ``unescapeMarkdown``
# in ``SorcarTab.ts``, so the two parsers can never disagree on which
# ``\X`` sequences are decoration and which are literal text.
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>|~\"'$%&,/:;=?@^])")
# The inverse for writing: a backslash that *would* be read as an
# escape is doubled so the body round-trips through the parser above.
_MARKDOWN_ESCAPABLE_BACKSLASH = re.compile(
    r"\\(?=[\\`*_{}\[\]()#+\-.!<>|~\"'$%&,/:;=?@^])"
)
# Serialises the read-check-append of the Add button: the daemon runs
# client commands on a thread pool, so two windows adding at once must
# not both pass the duplicate check.
_APPEND_LOCK = threading.Lock()

MY_INJECTION_DEFAULT_BODY = (
    "Write end-to-end 100% coverage tests for the feature first."
    "  Then implement the feature."
)

DEFAULT_MY_INJECTION = "## Trick\n\n" + MY_INJECTION_DEFAULT_BODY + "\n"


class TricksData(TypedDict):
    """The Inject promptlet list and how many leading entries the user owns."""

    tricks: list[str]
    userCount: int


def _parse_trick_sections(text: str) -> list[str]:
    """Return the body of every ``## Trick`` section in *text*.

    Bodies are trimmed and backslash-unescaped (``mdformat`` writes
    ``<<x>>`` as ``\\<<x>>`` and ``snake_case`` as ``snake\\_case``;
    the trick the author wrote is what every surface must show);
    empty bodies are skipped.  Mirrors the TypeScript
    ``readMarkdownSections`` parser used by ``SorcarTab.ts`` — the VS
    Code webview's Trick panel — so the remote webapp's panel and the
    daemon's ghost-text ``trick`` completions offer the same text.
    """
    tricks: list[str] = []
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines or lines[0].strip() != "Trick":
            continue
        body = _MARKDOWN_ESCAPE.sub(r"\1", "\n".join(lines[1:]).strip())
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
        text = user_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
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
    return Path(__file__).parent.parent / "INJECTIONS.md"


def _read_bundled_tricks() -> list[str]:
    """Return the tricks shipped in the bundled ``src/kiss/INJECTIONS.md``.

    Read directly from the package — no copy into ``~/.kiss/`` ever
    happens.  Returns an empty list if the file is missing or
    unreadable (graceful degradation).
    """
    path = _bundled_injections_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return _parse_trick_sections(text)


def read_tricks_data() -> TricksData:
    """Return the Inject promptlet list plus how many entries the user owns.

    The list is built by concatenating, in order:

    1. ``~/.kiss/MY_INJECTION.md`` (user-curated, auto-seeded with the
       default test-first trick on first read).
    2. The bundled ``src/kiss/INJECTIONS.md`` (read directly from the
       package; never copied into ``~/.kiss/``).

    ``userCount`` is the length of the first part: the panel shows a
    delete button only on those leading rows, since only the user
    file can be edited.  Empty list if both files are unavailable so a
    deployment without either still degrades gracefully (no tricks
    rendered, no ghost suggestions).

    Returns:
        ``{"tricks": [...], "userCount": n}`` — the ordered trick
        strings (MY_INJECTION first then bundled) and the number of
        leading MY_INJECTION entries.  The shape of the ``tricksData``
        event and of the page's ``window.__TRICKS__`` /
        ``window.__MY_TRICKS_COUNT__`` globals.
    """
    user = _read_my_injection_tricks()
    return {"tricks": user + _read_bundled_tricks(), "userCount": len(user)}


def read_tricks() -> list[str]:
    """Return the ordered "Inject instruction" trick list.

    See :func:`read_tricks_data` for the sources and their order.

    Returns:
        Ordered list of trick text strings, MY_INJECTION first then
        bundled.
    """
    return read_tricks_data()["tricks"]


def delete_my_injection_trick(text: str) -> str | None:
    """Remove the ``## Trick`` section(s) whose body is *text* from ``~/.kiss/MY_INJECTION.md``.

    Services the delete button of the Inject promptlet panel.  Every
    section whose parsed body (trimmed, backslash-unescaped — what the
    panel shows) equals *text* is dropped; everything else in the file
    (text before the first ``##`` heading, other ``##`` sections in
    their original order and spelling) is kept, ending in a single
    newline (an empty file when nothing is left, so the starter
    promptlet is not re-seeded behind the user's back); the rewrite
    goes through text mode, so CRLF line endings come out as LF.  The
    file is rewritten in place under the same process-wide lock as
    :func:`append_my_injection_trick`, so a concurrent add from another
    window is never lost.

    Args:
        text: The promptlet body as shown in the panel.  Surrounding
            whitespace is trimmed and CRLF line breaks are read as LF
            (the VS Code page lists a CRLF file's bodies verbatim, this
            module's parser normalises them).

    Returns:
        ``None`` on success, else a user-facing error message: an
        unreadable or missing file, a file that is not UTF-8 text, or
        a body that is not in the file (already deleted from another
        window, or a bundled promptlet).  ``OSError`` from the write
        propagates to the caller.
    """
    body = text.replace("\r\n", "\n").strip()
    with _APPEND_LOCK:
        user_path = ensure_user_asset_from_default(
            "MY_INJECTION.md", DEFAULT_MY_INJECTION,
        )
        if user_path is None:
            return "Could not read ~/.kiss/MY_INJECTION.md"
        try:
            existing = user_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "~/.kiss/MY_INJECTION.md is not UTF-8 text"
        starts = [m.start() for m in re.finditer(r"^##\s+", existing, re.MULTILINE)]
        kept = [existing[: starts[0]]] if starts else [existing]
        removed = 0
        for i, start in enumerate(starts):
            section = existing[start : starts[i + 1] if i + 1 < len(starts) else None]
            if _parse_trick_sections(section) == [body]:
                removed += 1
            else:
                kept.append(section)
        if removed == 0:
            return "That promptlet is not in ~/.kiss/MY_INJECTION.md"
        # Trailing blank lines belonged to the dropped section's spacing;
        # one newline keeps add/delete cycles from growing the file.
        rest = "".join(kept)
        rest = rest.rstrip("\n") + "\n" if rest.strip() else ""
        user_path.write_text(rest, encoding="utf-8")
    return None


def append_my_injection_trick(text: str) -> str | None:
    """Append *text* as a new ``## Trick`` section of ``~/.kiss/MY_INJECTION.md``.

    Services the "Add" button of the Inject promptlet panel.  The file
    is seeded first when missing (so the user's default trick is never
    lost), then the new section is appended after the existing ones —
    the panel and the ghost-text completions list it in file order.

    The body is stored Markdown-escaped (backslashes that the reader
    would strip are doubled), so the panel shows exactly what the user
    typed after the reload.  The whole read-check-append runs under a
    process-wide lock: commands from different windows execute
    concurrently on the daemon's thread pool.

    Args:
        text: The promptlet body.  Surrounding whitespace is trimmed.

    Returns:
        ``None`` on success, else a user-facing error message: an
        empty body, a body that would itself start a new ``##``
        section (which the parser would split), a duplicate of a
        trick already in the file, an unwritable ``~/.kiss/``, or a
        file that is not UTF-8 text.  ``OSError`` from the read or
        write propagates to the caller.
    """
    body = text.strip()
    if not body:
        return "Promptlet must not be empty"
    if re.search(r"^##\s", body, flags=re.MULTILINE):
        return "Promptlet must not contain a line starting with '## '"
    with _APPEND_LOCK:
        user_path = ensure_user_asset_from_default(
            "MY_INJECTION.md", DEFAULT_MY_INJECTION,
        )
        if user_path is None:
            return "Could not write ~/.kiss/MY_INJECTION.md"
        try:
            existing = user_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "~/.kiss/MY_INJECTION.md is not UTF-8 text"
        if body in _parse_trick_sections(existing):
            return "That promptlet is already in ~/.kiss/MY_INJECTION.md"
        escaped = _MARKDOWN_ESCAPABLE_BACKSLASH.sub(r"\\\\", body)
        separator = "" if existing == "" or existing.endswith("\n") else "\n"
        with user_path.open("a", encoding="utf-8") as fh:
            fh.write(separator + "\n## Trick\n\n" + escaped + "\n")
    return None


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
    last_boundary_end = 0
    for m in _SENTENCE_BOUNDARY.finditer(query):
        last_boundary_end = m.end()
    partial = query[last_boundary_end:]
    if last_boundary_end == 0:
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
