"""Integration tests for multimodal (image/PDF) support across all model providers."""

import io
import struct
import tempfile
import unittest
import zlib

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import SUPPORTED_MIME_TYPES, Attachment
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)

TEST_TIMEOUT = 120


def _create_png_bytes(width: int = 2, height: int = 2, color: tuple = (255, 0, 0)) -> bytes:
    """Create a minimal valid PNG image in memory."""
    r, g, b = color
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter byte
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


class TestAttachment(unittest.TestCase):
    def test_from_file_png(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = _create_png_bytes()
            f.write(png_data)
            f.flush()
            att = Attachment.from_file(f.name)
            assert att.mime_type == "image/png"
            assert att.data == png_data

    def test_from_file_jpeg(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpeg_data = _create_jpeg_bytes()
            f.write(jpeg_data)
            f.flush()
            att = Attachment.from_file(f.name)
            assert att.mime_type == "image/jpeg"
            assert att.data == jpeg_data

    def test_from_file_pdf(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_data = _create_minimal_pdf()
            f.write(pdf_data)
            f.flush()
            att = Attachment.from_file(f.name)
            assert att.mime_type == "application/pdf"
            assert att.data == pdf_data

    def test_from_file_unsupported(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported MIME type"):
                Attachment.from_file(f.name)

    def test_from_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Attachment.from_file("/nonexistent/file.png")

    def test_to_base64(self) -> None:
        import base64

        data = b"test data"
        att = Attachment(data=data, mime_type="image/png")
        assert att.to_base64() == base64.b64encode(data).decode("ascii")

    def test_to_data_url(self) -> None:
        import base64

        data = b"test data"
        att = Attachment(data=data, mime_type="image/jpeg")
        expected = f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}"
        assert att.to_data_url() == expected

    def test_supported_mime_types(self) -> None:
        assert "image/jpeg" in SUPPORTED_MIME_TYPES
        assert "image/png" in SUPPORTED_MIME_TYPES
        assert "image/gif" in SUPPORTED_MIME_TYPES
        assert "image/webp" in SUPPORTED_MIME_TYPES
        assert "application/pdf" in SUPPORTED_MIME_TYPES

    def test_from_file_gif(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            f.write(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;\x00")
            f.flush()
            att = Attachment.from_file(f.name)
            assert att.mime_type == "image/gif"

    def test_from_file_webp(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WEBP")
            f.flush()
            att = Attachment.from_file(f.name)
            assert att.mime_type == "image/webp"


class TestAttachmentFromFileExtensions(unittest.TestCase):
    """Test that Attachment.from_file works with .jpeg extension too."""

    def test_jpeg_extension(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f:
            f.write(_create_jpeg_bytes())
            f.flush()
            att = Attachment.from_file(f.name)
            assert att.mime_type == "image/jpeg"


@requires_gemini_api_key
class TestGeminiMultimodal(unittest.TestCase):
    """Integration tests for Gemini model with image attachments."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_describe_png_image(self) -> None:
        png_data = _create_png_bytes(width=4, height=4, color=(255, 0, 0))
        att = Attachment(data=png_data, mime_type="image/png")
        agent = KISSAgent("Gemini Image Test")
        result = agent.run(
            model_name="gemini-2.0-flash",
            prompt_template=(
                "Describe this image. What color is it? Answer with just the color name."
            ),
            is_agentic=False,
            max_budget=0.50,
            attachments=[att],
        )
        assert result is not None
        assert len(result) > 0
        assert "red" in result.lower()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_describe_pdf(self) -> None:
        pdf_data = _create_minimal_pdf()
        att = Attachment(data=pdf_data, mime_type="application/pdf")
        agent = KISSAgent("Gemini PDF Test")
        result = agent.run(
            model_name="gemini-2.0-flash",
            prompt_template=("What text does this PDF contain? Answer with just the text content."),
            is_agentic=False,
            max_budget=0.50,
            attachments=[att],
        )
        assert result is not None
        assert "hello" in result.lower() or "world" in result.lower()

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


@requires_anthropic_api_key
class TestAnthropicMultimodal(unittest.TestCase):
    """Integration tests for Anthropic (Claude) model with image attachments."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_describe_png_image(self) -> None:
        png_data = _create_png_bytes(width=4, height=4, color=(0, 255, 0))
        att = Attachment(data=png_data, mime_type="image/png")
        agent = KISSAgent("Claude Image Test")
        result = agent.run(
            model_name="claude-haiku-4-5",
            prompt_template=(
                "Describe this image. What color is it? Answer with just the color name."
            ),
            is_agentic=False,
            max_budget=0.50,
            attachments=[att],
        )
        assert result is not None
        assert len(result) > 0
        assert "green" in result.lower()

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_describe_pdf(self) -> None:
        pdf_data = _create_minimal_pdf()
        att = Attachment(data=pdf_data, mime_type="application/pdf")
        agent = KISSAgent("Claude PDF Test")
        result = agent.run(
            model_name="claude-haiku-4-5",
            prompt_template=("What text does this PDF contain? Answer with just the text content."),
            is_agentic=False,
            max_budget=0.50,
            attachments=[att],
        )
        assert result is not None
        assert "hello" in result.lower() or "world" in result.lower()


@requires_openai_api_key
class TestOpenAIMultimodal(unittest.TestCase):
    """Integration tests for OpenAI model with image attachments."""

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_describe_png_image(self) -> None:
        png_data = _create_png_bytes(width=32, height=32, color=(0, 0, 255))
        att = Attachment(data=png_data, mime_type="image/png")
        agent = KISSAgent("OpenAI Image Test")
        result = agent.run(
            model_name="gpt-4o-mini",
            prompt_template=(
                "This image is a solid color square. What color is it? "
                "Answer with ONLY the color name, nothing else."
            ),
            is_agentic=False,
            max_budget=0.50,
            attachments=[att],
        )
        assert result is not None
        assert len(result) > 0
        assert "blue" in result.lower()


class TestNoAttachments(unittest.TestCase):
    """Verify that passing no attachments still works (regression)."""

    @requires_gemini_api_key
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_no_attachments(self) -> None:
        agent = KISSAgent("No Attachment Test")
        result = agent.run(
            model_name="gemini-2.0-flash",
            prompt_template="What is 3 + 5? Answer with just the number.",
            is_agentic=False,
            max_budget=0.50,
        )
        assert "8" in result


if __name__ == "__main__":
    unittest.main()
