# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt test: multi-byte UTF-8 split across reads must not be lost.

:meth:`kiss.agents.sorcar.cli_steering._InputBox.feed` receives raw
``os.read`` chunks.  A terminal paste (or a slow link) can split a
multi-byte UTF-8 character across two reads.  ``feed`` used to call
``data.decode("utf-8", "ignore")`` per chunk, so both halves of the
character decoded to nothing and the typed character silently vanished
from the steering buffer.
"""

from __future__ import annotations

import io
import threading

import pytest

from kiss.agents.sorcar.cli_steering import _InputBox


def _box() -> _InputBox:
    """Return a fresh, inactive input box writing to a throwaway buffer."""
    return _InputBox(threading.RLock(), io.StringIO())


def test_emoji_split_across_reads_is_preserved() -> None:
    """A 4-byte emoji split into two chunks must still type the emoji."""
    box = _box()
    box.feed("😀".encode()[:2], lambda _line: None, lambda: None)
    box.feed("😀".encode()[2:], lambda _line: None, lambda: None)
    assert box.buf == "😀", (
        f"split emoji was destroyed by per-chunk decoding: {box.buf!r}"
    )


def test_two_byte_char_split_across_reads_is_preserved() -> None:
    """A 2-byte char (é) split across chunks must still type correctly."""
    box = _box()
    box.feed(b"caf\xc3", lambda _line: None, lambda: None)
    box.feed(b"\xa9", lambda _line: None, lambda: None)
    assert box.buf == "café", (
        f"split 2-byte UTF-8 char was destroyed: {box.buf!r}"
    )


def test_unsplit_utf8_still_types() -> None:
    """Sanity: whole multi-byte characters in one chunk keep working."""
    box = _box()
    box.feed("汉字 ok".encode(), lambda _line: None, lambda: None)
    assert box.buf == "汉字 ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
