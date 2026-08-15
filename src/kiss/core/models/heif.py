# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""HEIF/HEIC recognition and transcoding for prompt attachments.

An iPhone whose *Settings > Camera > Formats* is "High Efficiency" -- the
factory default since the iPhone 7 -- stores every photo as HEIC.  Of the
vision APIs this framework talks to, only Gemini accepts that format: OpenAI
and Anthropic reject ``image/heic`` outright, so a camera photo has to be
transcoded to JPEG before it reaches a model.

The conversion is delegated to whatever HEIF decoder the host already has
(macOS ships ``sips``; Linux distributions ship ``heif-convert`` with
libheif's tools, and ``ffmpeg`` or ImageMagick are common), which keeps the
framework free of a native image-codec dependency.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

HEIF_MIME_TYPES = frozenset(
    {
        "image/heic",
        "image/heif",
        "image/heic-sequence",
        "image/heif-sequence",
    }
)

HEIF_SUFFIXES = frozenset({".heic", ".heif", ".hif"})

# ISO base media file format major brands (bytes 8..12 of the file) used by
# HEIF containers, including the burst and Live-Photo-still variants.
_HEIF_BRANDS = frozenset(
    {
        b"heic",
        b"heix",
        b"heim",
        b"heis",
        b"hevc",
        b"hevx",
        b"hevm",
        b"hevs",
        b"mif1",
        b"msf1",
    }
)

# Each entry builds the argument list of one candidate converter, tried in
# order.  ``sips`` is first because it is always present on macOS, where the
# daemon most often runs, and needs no extra package.
_CONVERTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sips", ("-s", "format", "jpeg", "{src}", "--out", "{dst}")),
    ("heif-convert", ("-q", "88", "{src}", "{dst}")),
    ("magick", ("{src}", "{dst}")),
    ("ffmpeg", ("-loglevel", "error", "-y", "-i", "{src}", "{dst}")),
)

_CONVERT_TIMEOUT_S = 60


def is_heif(data: bytes) -> bool:
    """Return whether ``data`` starts with a HEIF/HEIC container header.

    The byte header is authoritative: iOS reports an empty MIME type for
    some pictures and browsers disagree on ``image/heic`` vs ``image/heif``.

    Args:
        data: Raw file bytes (only the first 12 are inspected).

    Returns:
        True if the ISO base media file format major brand identifies a HEIF
        container.
    """
    return len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS


def heif_to_jpeg(data: bytes) -> bytes | None:
    """Transcode HEIF/HEIC bytes to JPEG using a host converter.

    Args:
        data: Raw HEIF/HEIC file bytes.

    Returns:
        JPEG bytes, or ``None`` if no converter is installed or every
        available one failed.
    """
    with tempfile.TemporaryDirectory(prefix="kiss-heif-") as tmp:
        src = Path(tmp) / "in.heic"
        dst = Path(tmp) / "out.jpg"
        src.write_bytes(data)
        for tool, template in _CONVERTERS:
            exe = shutil.which(tool)
            if exe is None:
                continue
            argv = [exe] + [
                a.format(src=str(src), dst=str(dst)) for a in template
            ]
            # Every candidate writes to the same path, and a converter that is
            # killed by the timeout or exits non-zero can leave a truncated
            # JPEG behind (ffmpeg and ImageMagick both encode incrementally).
            # Clearing the path first keeps the success check below about the
            # output of *this* converter only.
            dst.unlink(missing_ok=True)
            try:
                subprocess.run(
                    argv,
                    check=True,
                    capture_output=True,
                    timeout=_CONVERT_TIMEOUT_S,
                )
            except (subprocess.SubprocessError, OSError):
                logger.warning("HEIF conversion via %s failed", tool, exc_info=True)
                continue
            if dst.exists() and dst.stat().st_size > 0:
                return dst.read_bytes()
            logger.warning("HEIF conversion via %s produced no output", tool)
    logger.warning(
        "No working HEIF decoder found; tried %s",
        ", ".join(tool for tool, _ in _CONVERTERS),
    )
    return None
