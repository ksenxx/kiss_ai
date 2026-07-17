# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for the ``Read`` tool's binary-attachment size cap.

Bug (bughunt round 9): the text path of ``UsefulTools.Read`` is guarded
by ``max_lines``, but the binary-attachment path had NO size guard — a
``Read`` of a large supported binary (e.g. a video or a huge PNG) would
``read_bytes()`` the whole file and base64-encode it (+33%) into the
tool-result string, blowing up process memory and the model
conversation.  The fix refuses binaries larger than
``_MAX_BINARY_READ_BYTES`` up front (via ``stat``, without reading the
content) with an actionable error, while small supported binaries keep
round-tripping as inline attachments.
"""

import struct
import zlib

from kiss.core.models.model import parse_binary_attachments
from kiss.core.useful_tools import _MAX_BINARY_READ_BYTES, UsefulTools


def _png_bytes(payload_size: int) -> bytes:
    """Build a syntactically plausible PNG file of roughly *payload_size* bytes.

    Starts with the real 8-byte PNG signature (so the content is
    definitively non-UTF-8 and hits the binary branch) followed by a
    minimal IHDR chunk and junk filler bytes.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    )
    filler = b"\xff\xfe\x00\x01" * (max(payload_size - len(sig) - len(ihdr), 4) // 4)
    return sig + ihdr + filler


def test_read_oversized_binary_is_refused_without_loading(tmp_path):
    """A supported binary above the cap returns a small, actionable error."""
    big = tmp_path / "big.png"
    big.write_bytes(_png_bytes(_MAX_BINARY_READ_BYTES + 1024 * 1024))

    tools = UsefulTools(work_dir=str(tmp_path))
    result = tools.Read(str(big))

    assert result.startswith("Error:"), result[:200]
    assert "too large" in result.lower()
    # The pre-fix behaviour returned the whole base64 payload (~28MB for
    # a 21MB file); the fixed error message must stay tiny.
    assert len(result) < 2000
    # And it must not smuggle an attachment sentinel through.
    _, attachments = parse_binary_attachments(result)
    assert attachments == []


def test_read_small_binary_still_round_trips(tmp_path):
    """A small supported binary is still embedded as an inline attachment."""
    small = tmp_path / "small.png"
    data = _png_bytes(4096)
    small.write_bytes(data)

    tools = UsefulTools(work_dir=str(tmp_path))
    result = tools.Read(str(small))

    assert "content attached below" in result
    plain, attachments = parse_binary_attachments(result)
    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/png"
    assert attachments[0].data == data


def test_read_binary_exactly_at_cap_is_allowed(tmp_path):
    """A binary exactly at the cap boundary is still readable."""
    at_cap = tmp_path / "cap.png"
    payload = _png_bytes(_MAX_BINARY_READ_BYTES)
    payload = payload.ljust(_MAX_BINARY_READ_BYTES, b"\xfe")[:_MAX_BINARY_READ_BYTES]
    at_cap.write_bytes(payload)
    assert at_cap.stat().st_size == _MAX_BINARY_READ_BYTES

    tools = UsefulTools(work_dir=str(tmp_path))
    result = tools.Read(str(at_cap))

    assert "content attached below" in result
    _, attachments = parse_binary_attachments(result)
    assert len(attachments) == 1
    assert len(attachments[0].data) == _MAX_BINARY_READ_BYTES
