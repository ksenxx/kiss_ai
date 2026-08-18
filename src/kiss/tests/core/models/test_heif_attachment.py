# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: HEIC/HEIF photos become JPEG attachments.

The remote webapp converts an iPhone camera photo in the browser, but a
client whose engine cannot decode HEIC (every engine except Safari 17+) still
uploads the raw bytes, and ``sorcar`` can be handed a ``.HEIC`` path
directly.  ``Attachment`` therefore transcodes HEIF at the boundary: OpenAI
and Anthropic reject ``image/heic`` outright, so anything else would drop the
photo silently.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from kiss.core.models.heif import heif_to_jpeg, is_heif
from kiss.core.models.model import (
    READ_TOOL_BINARY_MIME_TYPES,
    Attachment,
    encode_binary_attachment,
    parse_binary_attachments,
)

_JPEG_SOI = b"\xff\xd8\xff"


def _write_gradient_png(path: Path, width: int = 64, height: int = 48) -> None:
    """Write a small RGB gradient PNG without an image library."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            rows += bytes((x * 4 % 256, y * 5 % 256, 96))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def heic_photo(tmp_path: Path) -> Path:
    """Return a real HEIC file, skipping where none can be produced."""
    sips = shutil.which("sips")
    if sips is None:
        pytest.skip("no HEIF encoder available to build the fixture")
    png = tmp_path / "IMG_4242.png"
    heic = tmp_path / "IMG_4242.HEIC"
    _write_gradient_png(png)
    subprocess.run(
        [sips, "-s", "format", "heic", str(png), "--out", str(heic)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return heic


def test_is_heif_reads_the_container_header(heic_photo: Path, tmp_path: Path) -> None:
    """The sniff keys off the ISO-BMFF brand, not the file name."""
    png = tmp_path / "plain.png"
    _write_gradient_png(png)
    assert is_heif(heic_photo.read_bytes())
    assert not is_heif(png.read_bytes())
    assert not is_heif(b"")
    assert not is_heif(b"\x00\x00\x00\x18ftypmp42short")


def test_heif_to_jpeg_transcodes(heic_photo: Path) -> None:
    """A host converter turns HEIF bytes into a real JPEG."""
    jpeg = heif_to_jpeg(heic_photo.read_bytes())
    assert jpeg is not None, "no HEIF decoder on this host"
    assert jpeg.startswith(_JPEG_SOI)


def test_from_bytes_converts_an_uploaded_camera_photo(heic_photo: Path) -> None:
    """An upload labelled image/heic reaches the model as JPEG."""
    att = Attachment.from_bytes(heic_photo.read_bytes(), "image/heic")
    assert att.mime_type == "image/jpeg"
    assert att.data.startswith(_JPEG_SOI)
    assert att.to_data_url().startswith("data:image/jpeg;base64,")


def test_from_bytes_trusts_the_header_over_the_label(heic_photo: Path) -> None:
    """iOS sometimes reports no MIME type at all; the bytes decide."""
    att = Attachment.from_bytes(heic_photo.read_bytes(), "")
    assert att.mime_type == "image/jpeg"


def test_from_bytes_leaves_other_formats_untouched(tmp_path: Path) -> None:
    """Non-HEIF uploads are passed through byte for byte."""
    png = tmp_path / "shot.png"
    _write_gradient_png(png)
    data = png.read_bytes()
    att = Attachment.from_bytes(data, "image/png")
    assert att.mime_type == "image/png"
    assert att.data == data


def test_from_file_accepts_a_heic_path(heic_photo: Path) -> None:
    """Attaching IMG_4242.HEIC no longer raises ValueError."""
    att = Attachment.from_file(str(heic_photo))
    assert att.mime_type == "image/jpeg"
    assert att.data.startswith(_JPEG_SOI)


def test_from_file_still_rejects_unsupported_types(tmp_path: Path) -> None:
    """The MIME allow-list is intact for genuinely unusable files."""
    bad = tmp_path / "notes.xyz"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="Unsupported MIME type"):
        Attachment.from_file(str(bad))


def test_from_bytes_ignores_a_wrong_heif_label(tmp_path: Path) -> None:
    """A mislabelled file keeps its own bytes and label."""
    png = tmp_path / "mislabelled.png"
    _write_gradient_png(png)
    data = png.read_bytes()
    att = Attachment.from_bytes(data, "image/heic")
    assert att.data == data
    assert att.mime_type == "image/heic"


def test_read_tool_can_embed_a_heic_photo(heic_photo: Path) -> None:
    """Read('photo.HEIC') reaches the model as JPEG, not as HEIC."""
    assert "image/heic" in READ_TOOL_BINARY_MIME_TYPES
    encoded = encode_binary_attachment("image/heic", heic_photo.read_bytes())
    text, attachments = parse_binary_attachments("look:\n" + encoded)
    assert "[attached image/heic," in text
    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/jpeg"
    assert attachments[0].data.startswith(_JPEG_SOI)
