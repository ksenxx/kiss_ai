"""Integration tests for multimodal (image/PDF/audio/video) support across all model providers."""

import io
import os
import struct
import unittest
import zlib

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import SUPPORTED_MIME_TYPES, Attachment, transcribe_audio
from kiss.tests.conftest import (
    requires_gemini_api_key,
)

TEST_TIMEOUT = 120


def _create_png_bytes(width: int = 2, height: int = 2, color: tuple = (255, 0, 0)) -> bytes:
    """Create a minimal valid PNG image in memory."""
    r, g, b = color
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"
        for _ in range(width):
            raw_data += bytes([r, g, b])
    compressed = zlib.compress(raw_data)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


def _create_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG image using PIL if available, else raw bytes."""
    try:
        from PIL import Image  # type: ignore[import-not-found]

        buf = io.BytesIO()
        img = Image.new("RGB", (4, 4), color=(0, 0, 255))
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _create_minimal_pdf() -> bytes:
    """Create a minimal valid PDF with text 'Hello World'."""
    return (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n431\n%%EOF\n"
    )


def _create_silent_wav(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file with silence."""
    num_samples = sample_rate * duration_ms // 1000
    data_size = num_samples * 2  # 16-bit mono
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


class TestAttachment(unittest.TestCase):

    def test_supported_mime_types(self) -> None:
        # Images and PDF
        assert "image/jpeg" in SUPPORTED_MIME_TYPES
        assert "image/png" in SUPPORTED_MIME_TYPES
        assert "image/gif" in SUPPORTED_MIME_TYPES
        assert "image/webp" in SUPPORTED_MIME_TYPES
        assert "application/pdf" in SUPPORTED_MIME_TYPES
        # Audio
        assert "audio/mpeg" in SUPPORTED_MIME_TYPES
        assert "audio/wav" in SUPPORTED_MIME_TYPES
        assert "audio/x-wav" in SUPPORTED_MIME_TYPES
        assert "audio/ogg" in SUPPORTED_MIME_TYPES
        assert "audio/webm" in SUPPORTED_MIME_TYPES
        assert "audio/flac" in SUPPORTED_MIME_TYPES
        assert "audio/aac" in SUPPORTED_MIME_TYPES
        assert "audio/mp4" in SUPPORTED_MIME_TYPES
        # Video
        assert "video/mp4" in SUPPORTED_MIME_TYPES
        assert "video/webm" in SUPPORTED_MIME_TYPES
        assert "video/ogg" in SUPPORTED_MIME_TYPES
        assert "video/mpeg" in SUPPORTED_MIME_TYPES
        assert "video/quicktime" in SUPPORTED_MIME_TYPES


class TestAnthropicModelAudioVideoAttachments(unittest.TestCase):
    """Unit tests: Anthropic model transcribes audio or skips with warning."""

    def test_audio_attachment_transcribed_when_api_key_set(self) -> None:
        """Audio is transcribed to text when OPENAI_API_KEY is available."""
        from kiss.core.models.anthropic_model import AnthropicModel

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Use a minimal but valid WAV so Whisper can process it
        wav_data = _create_silent_wav()
        m = AnthropicModel("claude-sonnet-4-20250514", api_key="test-key")
        audio_att = Attachment(data=wav_data, mime_type="audio/wav")
        m.initialize("Transcribe this audio", attachments=[audio_att])
        content = m.conversation[0]["content"]
        assert isinstance(content, list)
        # Should have transcription text block + prompt text block
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"].startswith("[Audio transcription]")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Transcribe this audio"

    def test_video_attachment_skipped_with_warning(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-sonnet-4-20250514", api_key="test-key")
        video_att = Attachment(data=b"\x00\x00\x00\x1c", mime_type="video/mp4")
        with self.assertLogs("kiss.core.models.anthropic_model", level="WARNING") as log:
            m.initialize("Describe this video", attachments=[video_att])
        assert any("video/mp4" in msg for msg in log.output)
        content = m.conversation[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_mixed_attachments_image_audio_no_key(self) -> None:
        """With no OpenAI key, audio is skipped but image is kept."""
        from kiss.core.models.anthropic_model import AnthropicModel

        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            m = AnthropicModel("claude-sonnet-4-20250514", api_key="test-key")
            png_data = _create_png_bytes()
            img_att = Attachment(data=png_data, mime_type="image/png")
            audio_att = Attachment(data=b"\xff\xfb\x90\x00", mime_type="audio/wav")
            with self.assertLogs("kiss.core.models.anthropic_model", level="WARNING") as log:
                m.initialize("Analyze these", attachments=[img_att, audio_att])
            assert any("audio/wav" in msg for msg in log.output)
            content = m.conversation[0]["content"]
            assert isinstance(content, list)
            types = [b["type"] for b in content]
            assert "image" in types
            assert "text" in types
            assert len(content) == 2
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


class TestOpenAICompatibleModelAudioVideoAttachments(unittest.TestCase):
    """Unit tests: OpenAI model handles audio via input_audio, skips video."""

    def test_mixed_attachments_image_audio_video(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        m = OpenAICompatibleModel("gpt-audio", base_url="http://localhost", api_key="k")
        png_data = _create_png_bytes()
        img_att = Attachment(data=png_data, mime_type="image/png")
        audio_att = Attachment(data=b"\xff\xfb\x90\x00", mime_type="audio/mpeg")
        video_att = Attachment(data=b"\x00" * 10, mime_type="video/webm")
        with self.assertLogs(
            "kiss.core.models.openai_compatible_model", level="WARNING"
        ) as log:
            m.initialize(
                "Analyze all", attachments=[img_att, audio_att, video_att]
            )
        assert any("video/webm" in msg for msg in log.output)
        content = m.conversation[-1]["content"]
        types = [p["type"] for p in content]
        assert "image_url" in types
        assert "input_audio" in types
        assert "text" in types
        # Video should be skipped
        assert types.count("text") == 1
        assert len(content) == 3  # image + audio + text (video skipped)


class TestAudioMimeToFormat(unittest.TestCase):
    """Unit tests for the _audio_mime_to_format helper."""

    def test_unknown_format_uses_subtype(self) -> None:
        from kiss.core.models.openai_compatible_model import _audio_mime_to_format

        assert _audio_mime_to_format("audio/amr") == "amr"
        assert _audio_mime_to_format("audio/opus") == "opus"


class TestTranscribeAudio(unittest.TestCase):
    """Unit tests for the transcribe_audio helper."""

    def test_raises_with_invalid_api_key(self) -> None:
        with pytest.raises(RuntimeError, match="transcription failed"):
            transcribe_audio(b"\xff\xfb\x90\x00", "audio/mpeg", api_key="sk-invalid-key")

    def test_mime_to_ext_mapping(self) -> None:
        from kiss.core.models.model import _AUDIO_MIME_TO_EXT

        assert _AUDIO_MIME_TO_EXT["audio/mpeg"] == ".mp3"
        assert _AUDIO_MIME_TO_EXT["audio/wav"] == ".wav"
        assert _AUDIO_MIME_TO_EXT["audio/x-wav"] == ".wav"
        assert _AUDIO_MIME_TO_EXT["audio/ogg"] == ".ogg"
        assert _AUDIO_MIME_TO_EXT["audio/webm"] == ".webm"
        assert _AUDIO_MIME_TO_EXT["audio/flac"] == ".flac"
        assert _AUDIO_MIME_TO_EXT["audio/aac"] == ".aac"
        assert _AUDIO_MIME_TO_EXT["audio/mp4"] == ".m4a"


@requires_gemini_api_key
class TestGeminiMultimodal(unittest.TestCase):
    """Integration tests for Gemini model with image attachments."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_multiple_attachments(self) -> None:
        red_png = _create_png_bytes(width=4, height=4, color=(255, 0, 0))
        blue_png = _create_png_bytes(width=4, height=4, color=(0, 0, 255))
        agent = KISSAgent("Gemini Multi-Attach Test")
        result = agent.run(
            model_name="gemini-2.0-flash",
            prompt_template=(
                "I'm sending you two images. What are their primary colors? Answer briefly."
            ),
            is_agentic=False,
            max_budget=0.50,
            attachments=[
                Attachment(data=red_png, mime_type="image/png"),
                Attachment(data=blue_png, mime_type="image/png"),
            ],
        )
        assert result is not None
        result_lower = result.lower()
        assert "red" in result_lower or "blue" in result_lower


if __name__ == "__main__":
    unittest.main()
