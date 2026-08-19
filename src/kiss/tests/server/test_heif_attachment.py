# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `heic_photo` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""End-to-end tests: HEIC/HEIF photos become JPEG attachments.

The remote webapp converts an iPhone camera photo in the browser, but a
client whose engine cannot decode HEIC (every engine except Safari 17+) still
uploads the raw bytes, and ``sorcar`` can be handed a ``.HEIC`` path
directly.  ``Attachment`` therefore transcodes HEIF at the boundary: OpenAI
and Anthropic reject ``image/heic`` outright, so anything else would drop the
photo silently.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from kiss.server.task_runner import decode_attachments
from kiss.tests.core.models.test_heif_attachment import (  # noqa: F401
    _JPEG_SOI,
    _write_gradient_png,
    heic_photo,
)


def test_uploaded_photo_is_converted_on_the_way_in(heic_photo: Path) -> None:
    """The daemon transcodes what the webapp uploaded as image/heic."""
    payload = [
        {
            "name": "IMG_4242.HEIC",
            "mimeType": "image/heic",
            "data": base64.b64encode(heic_photo.read_bytes()).decode("ascii"),
        }
    ]
    attachments = decode_attachments(payload)
    assert attachments is not None
    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/jpeg"
    assert attachments[0].data.startswith(_JPEG_SOI)


@pytest.mark.parametrize("raw", [[], None, "nonsense", {"a": 1}])
def test_decode_attachments_without_a_payload(raw: object) -> None:
    """An absent or malformed field yields no attachments at all."""
    assert decode_attachments(raw) is None
