# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Redaction of the key store's secrets in tool output.

A tool can print a credential of ``$KISS_HOME/api_keys.env``: a ``cat`` of
the store, an ``env`` or ``ps`` listing, a test failure that quotes it.
Tool output is stored in ``history.db`` and sent to the model's provider,
so before either happens every value of the store in it is replaced by the
name of its variable (``ANTHROPIC_API_KEY``): :func:`redact_secrets` for a
whole tool result, :class:`StreamRedactor` for the output shown while a
command runs.  What a variable is called does not decide whether its value
is a secret.  Only a value under 8 characters (:data:`_MIN_SECRET_CHARS`)
and the region (:data:`_NOT_SECRET`) are left alone.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path

from kiss.core.locked import locked_call
from kiss.core.user_config import _parse_env_assignment, _read_user_text, api_keys_env_path

_NOT_SECRET = frozenset({"AWS_REGION"})
"""The one variable of the store whose value is left alone.

The store keeps the Bedrock region beside the Bedrock credentials
(:data:`~kiss.core.user_config.API_KEY_ENV_VARS`).  Its value, ``us-east-1``,
is a public word of source files, documentation and command output: shown as
``AWS_REGION`` there, a file would read differently from what it holds, and
an edit made from that reading would fail or write the name into the file.
"""

_MIN_SECRET_CHARS = 8
"""A shorter value is not replaced: ordinary output would hold it by chance."""


class _Secrets:
    """The secrets of one reading of the key store, and those read from it earlier.

    Args:
        store_text: The text of the key store.
        earlier: The ``names`` of the reading before this one.  A value the
            store held then stays a secret: a command started before a key was
            rotated or removed still prints the old one.

    Attributes:
        names: The variable name that replaces each secret value.  A value two
            variables hold goes by the first name in sorted order, and a value
            the store no longer holds keeps the name it had.
        pattern: Matches a secret value, the longest first, so that a value
            which contains another is replaced whole.
        tail_chars: The characters a cut-off secret can end with: output that
            ends with any other character holds back nothing.
    """

    def __init__(self, store_text: str, earlier: dict[str, str]) -> None:
        self.names: dict[str, str] = {}
        assignments = filter(None, map(_parse_env_assignment, store_text.splitlines()))
        for name, value in sorted(assignments):
            if len(value) >= _MIN_SECRET_CHARS and name not in _NOT_SECRET:
                self.names.setdefault(value, name)
        self.names = earlier | self.names
        values = sorted(self.names, key=len, reverse=True)
        self.pattern = re.compile("|".join(map(re.escape, values)))
        self.tail_chars = frozenset("".join(value[:-1] for value in values))

    def name(self, match: re.Match[str]) -> str:
        """Return the variable name that replaces the secret value *match* found."""
        return self.names[match.group()]

    def redact(self, text: str) -> str:
        """Return *text* with every secret value replaced by its variable's name."""
        return self.pattern.sub(self.name, text) if self.names else text

    def split(self, text: str) -> tuple[str, str]:
        """Return *text* redacted up to the end that may open a secret, and that end.

        *text* is the beginning of an output still arriving.  Its end may open a
        secret when it is a proper prefix of a secret value, which the rest of
        the output can complete; it is returned as it is, to be split again with
        what arrives next.
        """
        if not self.names:
            return text, ""
        held = self._held_from(text, 0)
        shown = ""
        at = 0
        for match in self.pattern.finditer(text):
            if match.start() >= held:
                break
            if match.end() > held:
                # A whole secret ends past the held-back point: only what follows it may wait.
                held = self._held_from(text, match.end())
            shown += text[at : match.start()] + self.name(match)
            at = match.end()
        return shown + text[at:held], text[held:]

    def _held_from(self, text: str, start: int) -> int:
        """Return the first index from *start* on where the rest of *text* may open a secret.

        ``len(text)`` when no rest is a proper prefix of a secret value.
        """
        held = len(text)
        if not text or text[-1] not in self.tail_chars:
            return held
        for value in self.names:
            at = text.find(value[0], max(start, len(text) - len(value) + 1), held)
            while at != -1 and not value.startswith(text[at:]):
                at = text.find(value[0], at + 1, held)
            if at != -1:
                held = at
        return held


_NO_SECRETS = _Secrets("", {})
_cache: tuple[tuple[str, int, int, int] | None, _Secrets] = (None, _NO_SECRETS)
"""The last reading of the key store, under the state of the file it was read from."""
_reading = threading.RLock()
"""Held while a changed store is read and merged into :data:`_cache`.

Reentrant for ``locked_call``: tool results are redacted on the task thread,
where a Stop injection can land right after a plain ``with`` took the lock
(CPython 3.14 has a check point there) and leave it held for good, with every
later tool result of the daemon waiting on it.
"""


def _secrets() -> _Secrets:
    """Return the secrets of the key store, read again only when the file has changed.

    A call costs one ``stat``.  The store is read again when its modification
    time, size or inode differs from those of the last reading (a save replaces the
    file, so the inode tells two saves within one tick of the clock apart), which is
    how a key saved while the daemon runs is known from the next tool call on.

    What was read from the store stays known while the process lives: a key
    rotated or removed while a command runs is still replaced in what that
    command prints, and a store that goes missing or unreadable keeps the
    secrets last read from it.  A store at another path starts with none.
    Two tool calls that find the store changed read it one after the other,
    each on top of what the one before has published, so neither puts back
    an older reading.
    """
    path = api_keys_env_path()
    try:
        stat = path.stat()
        found = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except OSError:
        found = None
    state, known = _cache
    if found is not None and found == state:
        return known
    return locked_call(_reading, _read_changed_store, path, found)


def _read_changed_store(path: Path, found: tuple[str, int, int, int] | None) -> _Secrets:
    """Return the secrets of the store at *path*, read again unless *found* is the cached state.

    Runs under :data:`_reading`.  *found* is the state of the file now, ``None``
    when it is missing.
    """
    global _cache
    state, known = _cache
    if state is None or state[0] != str(path):
        state, known = None, _NO_SECRETS
    if found is not None and found != state:
        try:
            known = _Secrets(_read_user_text(path), known.names)
        except OSError:
            return known
        _cache = (found, known)
    return known


def redact_secrets(text: str) -> str:
    """Return *text* with every secret of the key store replaced by its variable's name.

    A secret is the value, of at least 8 characters, of any assignment of
    ``api_keys.env`` but that of the region (``AWS_REGION``), whatever the
    variable is called, including a value the store held earlier in the life
    of this process.  Called on every tool result before it is printed (the
    event ``history.db`` stores) or handed to the model.

    Args:
        text: The output of a tool.

    Returns:
        *text* itself when it holds no secret; otherwise *text* with
        ``ANTHROPIC_API_KEY`` where the value of that variable stood.
    """
    return _secrets().redact(text)


class StreamRedactor:
    """Shows the output of one command, redacted, while it arrives in pieces.

    Two pieces can cut a secret between them, so the end of what has arrived is
    held back while it may open a secret (it is a proper prefix of one), until
    the next piece or :meth:`flush`.  Nothing else waits, and what is shown adds
    up to :func:`redact_secrets` of the whole output.  The store is consulted
    for every piece, so a key saved while the command runs is replaced in the
    output that follows, and one rotated out meanwhile stays replaced.

    Args:
        show: Receives the redacted output in pieces, none of them empty (the
            stream callback of the tool); ``None`` when nobody watches.
    """

    def __init__(self, show: Callable[[str], None] | None) -> None:
        self._show = show
        self._held = ""

    def feed(self, piece: str) -> None:
        """Show *piece*, redacted, except an end that may open a secret.

        Args:
            piece: The next piece of the command's output.
        """
        if self._show is None:
            return
        shown, self._held = _secrets().split(self._held + piece)
        if shown:
            self._show(shown)

    def flush(self) -> None:
        """Show the end still held back, redacted: the command has ended."""
        held, self._held = self._held, ""
        if held and self._show is not None:
            self._show(_secrets().redact(held))
