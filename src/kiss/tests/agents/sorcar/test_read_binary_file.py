"""Integration test: Read tool must not crash on binary files (e.g. PNG screenshots).

Reproduces the bug where reading a PNG file produced:
    Error: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte
because ``Path.read_text()`` blindly decodes as UTF-8.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools

# 1x1 transparent PNG — starts with the classic 0x89 PNG signature byte that
# triggers the UTF-8 decode failure reported by the user.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp()).resolve()
    cwd = Path.cwd()
    os.chdir(d)
    yield d
    os.chdir(cwd)
    shutil.rmtree(d, ignore_errors=True)


def test_read_png_does_not_emit_utf8_decode_error(temp_dir):
    """Reading a PNG must not surface a cryptic UTF-8 decode traceback."""
    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)

    result = UsefulTools().Read(str(png_path))

    assert "utf-8" not in result.lower()
    assert "invalid start byte" not in result
    # And it should clearly tell the caller the file is binary so the model
    # can react sensibly.
    assert "binary" in result.lower()
    assert "screenshot.png" in result


def test_read_text_file_still_works(temp_dir):
    """Plain text reads must continue to return the raw contents unchanged."""
    p = temp_dir / "hello.txt"
    p.write_text("hello\nworld\n")
    assert UsefulTools().Read(str(p)) == "hello\nworld\n"


def test_read_utf8_text_with_non_ascii(temp_dir):
    """UTF-8 text with non-ASCII bytes must still decode as text, not be flagged binary."""
    p = temp_dir / "u.txt"
    p.write_text("héllo — wörld\n", encoding="utf-8")
    out = UsefulTools().Read(str(p))
    assert out == "héllo — wörld\n"
