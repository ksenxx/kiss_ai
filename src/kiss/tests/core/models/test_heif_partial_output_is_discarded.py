# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a failed HEIF converter never feeds the next candidate.

``heif_to_jpeg`` tries up to four host converters in turn and hands them all
the *same* output path.  ``ffmpeg`` and ImageMagick write JPEG data
incrementally, so a converter that dies (non-zero exit, or a kill by the
conversion timeout) can leave a truncated file on that shared path.  Unless
the path is cleared before the next candidate runs, a later converter that
exits ``0`` without writing anything makes the leftover bytes look like a
successful conversion, and the truncated image is shipped to a vision API as
the user's photo.

The converters here are real executables on a real ``PATH``: each test writes
small shell scripts named after the entries of ``_CONVERTERS`` and prepends
their directory to ``PATH``, so ``shutil.which`` resolves them exactly the way
it resolves the host's own tools.  Stand-ins are provided for *every* name in
``_CONVERTERS`` so the outcome does not depend on which decoders happen to be
installed on the machine running the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.core.models.heif import _CONVERTERS, heif_to_jpeg

_JPEG_SOI = b"\xff\xd8\xff"

# A one-pixel grey JPEG, the smallest real output a converter could produce.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffdb004300ff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffc2000b080001000101011100ffc4001400"
    "01000000000000000000000000000000ffda0008010100000131ffd9"
)


def _script(path: Path, body: str) -> None:
    """Write ``body`` as an executable ``/bin/sh`` script at ``path``."""
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _dst_of(argv_template: tuple[str, ...]) -> str:
    """Return the shell expression for the output path of one converter.

    Each ``_CONVERTERS`` entry formats ``{dst}`` into exactly one argument, so
    the destination is the positional shell parameter at that index.
    """
    index = next(i for i, arg in enumerate(argv_template) if "{dst}" in arg)
    return f'"${index + 1}"'


def _fake_converter_dir(
    tmp_path: Path, bodies: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Install a stand-in for every known converter and put it first on PATH.

    Args:
        tmp_path: Directory to create the ``bin`` folder in.
        bodies: Shell body per converter name; names left out exit 0 silently
            without producing any output file.
        monkeypatch: Fixture used to prepend the folder to ``PATH``.

    Returns:
        The directory holding the stand-in executables.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool, _ in _CONVERTERS:
        _script(bin_dir / tool, bodies.get(tool, "exit 0\n"))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def test_partial_output_of_a_failed_converter_is_not_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated file from a crashed converter is discarded, not shipped."""
    sips_dst = _dst_of(dict(_CONVERTERS)["sips"])
    bodies = {
        # Writes a partial JPEG (correct SOI, no EOI) and then dies, exactly
        # like a converter killed part-way through encoding.
        "sips": f"printf '\\377\\330\\377 truncated' > {sips_dst}\nexit 1\n",
    }
    _fake_converter_dir(tmp_path, bodies, monkeypatch)

    assert heif_to_jpeg(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64) is None


def test_a_later_converter_still_supplies_the_jpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing the shared path does not break the fallback chain."""
    sips_dst = _dst_of(dict(_CONVERTERS)["sips"])
    convert_dst = _dst_of(dict(_CONVERTERS)["heif-convert"])
    bodies = {
        "sips": f"printf '\\377\\330\\377 truncated' > {sips_dst}\nexit 1\n",
        "heif-convert": (
            "printf '"
            + "".join(f"\\{byte:03o}" for byte in _TINY_JPEG)
            + f"' > {convert_dst}\nexit 0\n"
        ),
    }
    _fake_converter_dir(tmp_path, bodies, monkeypatch)

    jpeg = heif_to_jpeg(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
    assert jpeg == _TINY_JPEG
    assert jpeg is not None and jpeg.startswith(_JPEG_SOI)


def test_all_converters_exiting_without_output_yields_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every candidate exiting 0 with no file at all still means failure."""
    _fake_converter_dir(tmp_path, {}, monkeypatch)

    assert heif_to_jpeg(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64) is None


def test_an_empty_output_file_is_treated_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-byte result is not a conversion, and is cleared for the next try."""
    sips_dst = _dst_of(dict(_CONVERTERS)["sips"])
    convert_dst = _dst_of(dict(_CONVERTERS)["heif-convert"])
    bodies = {
        "sips": f": > {sips_dst}\nexit 0\n",
        "heif-convert": (
            "printf '"
            + "".join(f"\\{byte:03o}" for byte in _TINY_JPEG)
            + f"' > {convert_dst}\nexit 0\n"
        ),
    }
    _fake_converter_dir(tmp_path, bodies, monkeypatch)

    assert heif_to_jpeg(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64) == _TINY_JPEG


def test_a_converter_that_cannot_be_executed_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError from exec is a converter failure, not a crash of the caller."""
    bin_dir = _fake_converter_dir(tmp_path, {}, monkeypatch)
    # Present on PATH and marked executable, but not a runnable program: the
    # kernel refuses it with OSError rather than a non-zero exit status.
    (bin_dir / "sips").write_bytes(b"\x7fELF garbage")
    (bin_dir / "sips").chmod(0o755)
    convert_dst = _dst_of(dict(_CONVERTERS)["heif-convert"])
    _script(
        bin_dir / "heif-convert",
        "printf '"
        + "".join(f"\\{byte:03o}" for byte in _TINY_JPEG)
        + f"' > {convert_dst}\nexit 0\n",
    )

    assert heif_to_jpeg(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64) == _TINY_JPEG
