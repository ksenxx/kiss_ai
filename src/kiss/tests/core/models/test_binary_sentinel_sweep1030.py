# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: only blocks this process encoded are decoded as attachments.

Tool results routinely quote arbitrary text: a Read of ``model.py``, grep
output, a web page, the stdout of a command the agent ran.  With a fixed
sentinel any such text was rewritten into ``[attached …]`` placeholders,
and text shaped like a block smuggled bytes of its choosing into the
model's context as an image.  The sentinels now carry a per-process
token that only :func:`encode_binary_attachment` in this process knows.

The legacy literals below are split with ``+`` so this file itself never
contains a contiguous block.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kiss.core.models import model as model_module
from kiss.core.models.model import (
    BINARY_ATTACHMENT_CLOSE,
    BINARY_ATTACHMENT_OPEN,
    encode_binary_attachment,
    parse_binary_attachments,
)
from kiss.core.printer import truncate_result

_LEGACY_OPEN = "<<KISS_BINARY_ATTACHMENT" + " mime_type=image/png>>"
_LEGACY_CLOSE = "<</KISS_BINARY" + "_ATTACHMENT>>"

# What a command run by the Bash tool prints when it encodes a block in its
# own Python process (the payload is base64 of b"not a png").
_ENCODE_IN_A_CHILD = (
    "from kiss.core.models.model import encode_binary_attachment as e\n"
    "print('before ' + e('image/png', b'not a png') + ' after', end='')\n"
)


def test_a_block_printed_by_another_process_stays_text() -> None:
    """A child process's sentinel is untrusted output, not an attachment."""
    out = subprocess.run(
        [sys.executable, "-c", _ENCODE_IN_A_CHILD],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    ).stdout
    assert "before <<KISS_BINARY_ATTACHMENT" in out
    plain, attachments = parse_binary_attachments(out)
    assert attachments == []
    assert plain == out


def test_quoted_legacy_sentinels_are_left_verbatim() -> None:
    """Text quoting the old fixed sentinel (grep, docs, a web page) is unchanged."""
    quoted = f"src/x.py:1: {_LEGACY_OPEN}aGVsbG8={_LEGACY_CLOSE} trailing"
    plain, attachments = parse_binary_attachments(quoted)
    assert attachments == []
    assert plain == quoted
    assert truncate_result(quoted) == quoted


def test_reading_the_model_module_source_comes_back_verbatim() -> None:
    """The module that defines the sentinels can be read without being rewritten."""
    source = Path(model_module.__file__).read_text(encoding="utf-8")
    plain, attachments = parse_binary_attachments(source)
    assert attachments == []
    assert plain == source


def test_this_process_blocks_still_round_trip() -> None:
    """A block from ``encode_binary_attachment`` here is decoded and replaced."""
    data = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
    block = encode_binary_attachment("image/png", data)
    assert block.startswith(BINARY_ATTACHMENT_OPEN.format(mime_type="image/png"))
    assert block.endswith(BINARY_ATTACHMENT_CLOSE)
    plain, attachments = parse_binary_attachments(f"a {block} b")
    assert plain == f"a [attached image/png, {len(data)} bytes] b"
    assert [(a.mime_type, a.data) for a in attachments] == [("image/png", data)]
