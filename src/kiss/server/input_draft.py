# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The text the chat page's Save button keeps: ``$KISS_HOME/saved_input.txt``.

The contract.  The composer's Save button sends its text with ``saveInputDraft`` and a
page that starts asks for it with ``getInputDraft``, so the text survives a crash of the
page or of the daemon, and a lost connection.  The file is all the daemon keeps: plain
text, exactly the saved text; no file when nothing is saved.  It is one file for every
browser and window, and the last save to arrive wins.  A removal (an empty text: the page
sends one when the saved text is sent or cleared) names the texts its browser knows the
file may hold (``was``) and deletes the file only while it holds one of them, so a text
another browser saved since stays.  Every command may be repeated.

Outside the contract: any order between the saves of different browsers but that of their
arrival (an older save that arrives later wins), and what two windows of one browser do
in the same instant.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home
from kiss.core.utils import atomic_write_text

MAX_INPUT_DRAFT_BYTES = 1_000_000
"""The longest text kept, in UTF-8 bytes; ``media/main.js`` refuses a longer one unsent."""

_LOCK = threading.Lock()  # held while a command looks at the file and changes it


def input_draft_path() -> Path:
    """Return the file, ``$KISS_HOME/saved_input.txt``, resolved on every call."""
    return kiss_home() / "saved_input.txt"


def read_reply() -> dict[str, Any]:
    """Return the ``inputDraft`` event that answers ``getInputDraft``.

    Returns:
        ``{"type": "inputDraft", "text": <the saved text>, "path": <the
        file>}``; the text is ``""`` when no file holds one.
    """
    path = input_draft_path()
    try:
        # ``surrogatepass``: a JavaScript string may hold a lone surrogate.
        text = path.read_text(encoding="utf-8", errors="surrogatepass")
    except (OSError, UnicodeDecodeError):
        text = ""
    return {"type": "inputDraft", "text": text, "path": str(path)}


def save_reply(text: object, was: object, seq: object) -> dict[str, Any]:
    """Apply a ``saveInputDraft`` command and return the event that answers it.

    Args:
        text: The text to keep.  An empty one deletes the file while it
            holds one of the texts in *was*.
        was: The texts the sender knows the file may hold, a list.
        seq: The sender's number of the command, echoed in the answer.

    Returns:
        :func:`read_reply` after the command, with ``"seq"``, and with
        ``"error"`` (one sentence) when the file is as it was: *text* is no
        string or over :data:`MAX_INPUT_DRAFT_BYTES`, or a change failed.
    """
    path = input_draft_path()
    error = ""
    with _LOCK:
        try:
            if not isinstance(text, str):
                error = "Not saved: the text is no string."
            elif len(text.encode("utf-8", "surrogatepass")) > MAX_INPUT_DRAFT_BYTES:
                error = "Not saved: the text is longer than 1 MB."
            elif text:
                atomic_write_text(path, text, errors="surrogatepass")
            else:
                held = read_reply()["text"]
                if held and isinstance(was, list) and held in was:
                    path.unlink()
        except OSError as exc:
            error = f"The saved text in {path} was not changed: {exc}"
        reply = {**read_reply(), "seq": seq}
    return {**reply, "error": error} if error else reply
