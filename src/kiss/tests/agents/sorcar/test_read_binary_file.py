"""Integration tests for binary file handling in the ``Read`` tool.

Originally reproduced:
    Error: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte

The tool now embeds supported image/PDF binaries as sentinel-wrapped base64
that the Anthropic model layer lifts into a real ``image`` / ``document``
content block, so the LLM can actually *see* a screenshot it was asked to
read.
"""

import base64
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.models.model import parse_binary_attachments
from kiss.core.printer import truncate_result

# 1x1 transparent PNG — starts with the classic 0x89 PNG signature byte that
# triggers the UTF-8 decode failure reported by the user.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)

# Minimal PDF skeleton (raw bytes start with ``%PDF-1.4`` which is ASCII —
# decode succeeds, so to exercise the binary branch we mix in a couple of
# 0x80+ bytes that would fail strict UTF-8 decoding).
_PDF_BYTES = b"%PDF-1.4\n\x89\xc3\x28binarystream\n%%EOF"


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
    assert "screenshot.png" in result


def test_read_png_returns_sentinel_with_base64_payload(temp_dir):
    """A supported binary (PNG) round-trips through the sentinel parser."""
    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)

    result = UsefulTools().Read(str(png_path))

    assert "<<KISS_BINARY_ATTACHMENT mime_type=image/png>>" in result
    assert "<</KISS_BINARY_ATTACHMENT>>" in result
    expected_b64 = base64.b64encode(_PNG_1x1).decode("ascii")
    assert expected_b64 in result

    plain, attachments = parse_binary_attachments(result)
    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/png"
    assert attachments[0].data == _PNG_1x1
    # Sentinel + base64 payload must be removed from the plain text view.
    assert "<<KISS_BINARY_ATTACHMENT" not in plain
    assert expected_b64 not in plain
    assert "[attached image/png," in plain


def test_read_pdf_returned_as_document_attachment(temp_dir):
    """PDFs are a supported binary type and decoded back to bytes."""
    pdf_path = temp_dir / "doc.pdf"
    pdf_path.write_bytes(_PDF_BYTES)

    result = UsefulTools().Read(str(pdf_path))

    _, attachments = parse_binary_attachments(result)
    assert len(attachments) == 1
    assert attachments[0].mime_type == "application/pdf"
    assert attachments[0].data == _PDF_BYTES


def test_read_unsupported_binary_returns_friendly_error(temp_dir):
    """An unknown binary (e.g. ``.bin``) keeps the legacy guard message."""
    blob = temp_dir / "weights.bin"
    blob.write_bytes(b"\x00\x01\x02\x89\xff\xfeRAW")

    result = UsefulTools().Read(str(blob))

    assert "utf-8" not in result.lower()
    assert "Cannot read binary file" in result
    assert "weights.bin" in result
    # No sentinel, because we refuse to inline arbitrary binaries.
    assert "<<KISS_BINARY_ATTACHMENT" not in result


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


def test_anthropic_tool_result_includes_image_block(temp_dir):
    """Read → AnthropicModel produces a ``tool_result`` with an ``image`` block."""
    from kiss.core.models.anthropic_model import AnthropicModel

    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)
    read_output = UsefulTools().Read(str(png_path))

    model = AnthropicModel("claude-sonnet-4-5", api_key="sk-test")
    # Prime the conversation with an assistant tool_use so the id matching
    # path is exercised end to end.
    model.conversation = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "Read",
                    "input": {"file_path": str(png_path)},
                }
            ],
        },
    ]

    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    last = model.conversation[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    block = last["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_abc"
    content_blocks = block["content"]
    assert isinstance(content_blocks, list)
    types = [b["type"] for b in content_blocks]
    assert "image" in types
    image_block = next(b for b in content_blocks if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["type"] == "base64"
    assert base64.b64decode(image_block["source"]["data"]) == _PNG_1x1
    # Sentinel must not leak into the text block(s).
    for b in content_blocks:
        if b["type"] == "text":
            assert "<<KISS_BINARY_ATTACHMENT" not in b["text"]


def test_anthropic_tool_result_pdf_becomes_document_block(temp_dir):
    """PDFs flow through as ``document`` blocks, not ``image`` blocks."""
    from kiss.core.models.anthropic_model import AnthropicModel

    pdf_path = temp_dir / "doc.pdf"
    pdf_path.write_bytes(_PDF_BYTES)
    read_output = UsefulTools().Read(str(pdf_path))

    model = AnthropicModel("claude-sonnet-4-5", api_key="sk-test")
    model.conversation = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_pdf",
                    "name": "Read",
                    "input": {"file_path": str(pdf_path)},
                }
            ],
        },
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    block = model.conversation[-1]["content"][0]
    types = [b["type"] for b in block["content"]]
    assert "document" in types
    doc_block = next(b for b in block["content"] if b["type"] == "document")
    assert doc_block["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(doc_block["source"]["data"]) == _PDF_BYTES


def test_printer_truncate_result_masks_base64_payload(temp_dir):
    """The terminal/webview must never render the raw base64 blob."""
    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)
    read_output = UsefulTools().Read(str(png_path))
    expected_b64 = base64.b64encode(_PNG_1x1).decode("ascii")
    assert expected_b64 in read_output  # sanity

    rendered = truncate_result(read_output)
    assert expected_b64 not in rendered
    assert "<<KISS_BINARY_ATTACHMENT" not in rendered
    assert "[attached image/png," in rendered


def test_default_tool_result_strips_base64_to_avoid_context_bloat(temp_dir):
    """For non-image-capable model backends the base64 payload is dropped."""
    from kiss.core.models.model import Model

    # Build a minimal concrete Model so we can drive the default tool-result
    # path without instantiating a real provider client.
    class _StubModel(Model):
        def initialize(self, prompt, attachments=None):  # pragma: no cover
            self.conversation = [{"role": "user", "content": prompt}]

        def generate(self):  # pragma: no cover
            return "", None

        def generate_and_process_with_tools(self, function_map, tools_schema=None):  # pragma: no cover  # noqa: E501
            return [], "", None

        def extract_input_output_token_counts_from_response(self, response):  # pragma: no cover  # noqa: E501
            return 0, 0, 0, 0

        def get_embedding(self, text, embedding_model=None):  # pragma: no cover
            return []

    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)
    read_output = UsefulTools().Read(str(png_path))
    expected_b64 = base64.b64encode(_PNG_1x1).decode("ascii")

    m = _StubModel("stub")
    m.conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    ]
    m.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    appended = m.conversation[-1]
    assert appended["role"] == "tool"
    assert appended["tool_call_id"] == "call_1"
    assert expected_b64 not in appended["content"]
    assert "<<KISS_BINARY_ATTACHMENT" not in appended["content"]
    assert "[attached image/png," in appended["content"]
