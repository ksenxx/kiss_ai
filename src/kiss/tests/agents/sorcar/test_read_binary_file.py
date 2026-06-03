# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

# Tiny synthetic WAV (44-byte RIFF header + a couple of PCM samples).  The
# header begins with the ASCII tag ``RIFF`` but the embedded length fields
# include non-ASCII bytes that fail UTF-8 decoding, so the Read tool hits
# the binary branch.
_WAV_BYTES = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00"
    b"data\x00\x00\x00\x00"
)

# Tiny synthetic MP4 (just an ``ftyp`` box).  Triggers the binary branch
# because byte ``\x18`` etc. are valid UTF-8 but the box payload contains
# non-UTF-8 bytes.
_MP4_BYTES = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomavc1\xff\xfe"


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


def test_read_audio_file_encodes_as_attachment(temp_dir):
    """Audio files (WAV) are now embedded as sentinel-wrapped base64."""
    wav = temp_dir / "clip.wav"
    wav.write_bytes(_WAV_BYTES)
    result = UsefulTools().Read(str(wav))

    assert "<<KISS_BINARY_ATTACHMENT mime_type=audio/" in result
    _, attachments = parse_binary_attachments(result)
    assert len(attachments) == 1
    assert attachments[0].mime_type.startswith("audio/")
    assert attachments[0].data == _WAV_BYTES


def test_read_video_file_encodes_as_attachment(temp_dir):
    """Video files (MP4) are now embedded as sentinel-wrapped base64."""
    mp4 = temp_dir / "movie.mp4"
    mp4.write_bytes(_MP4_BYTES)
    result = UsefulTools().Read(str(mp4))

    _, attachments = parse_binary_attachments(result)
    assert len(attachments) == 1
    assert attachments[0].mime_type == "video/mp4"
    assert attachments[0].data == _MP4_BYTES


def test_openai_tool_result_appends_image_user_message(temp_dir):
    """OpenAICompatibleModel lifts a PNG attachment into a follow-up user msg."""
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)
    read_output = UsefulTools().Read(str(png_path))

    model = OpenAICompatibleModel(
        "openrouter/openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
    )
    model.conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_42",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    # Tool message: plain text only, no base64.
    tool_msg = model.conversation[-2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_42"
    assert "<<KISS_BINARY_ATTACHMENT" not in tool_msg["content"]
    assert "[attached image/png," in tool_msg["content"]

    # Follow-up user message with image_url part carrying the data URL.
    user_msg = model.conversation[-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    parts = user_msg["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    data_url = image_parts[0]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == _PNG_1x1


def test_openai_tool_result_appends_audio_user_message(temp_dir):
    """OpenAICompatibleModel lifts an audio attachment as input_audio part."""
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

    wav = temp_dir / "clip.wav"
    wav.write_bytes(_WAV_BYTES)
    read_output = UsefulTools().Read(str(wav))

    model = OpenAICompatibleModel(
        "openai/gpt-4o-audio-preview",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
    )
    model.conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_99",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    user_msg = model.conversation[-1]
    audio_parts = [p for p in user_msg["content"] if p.get("type") == "input_audio"]
    assert len(audio_parts) == 1
    assert audio_parts[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio_parts[0]["input_audio"]["data"]) == _WAV_BYTES


def test_openai_tool_result_drops_video_with_no_follow_up(temp_dir):
    """OpenAICompatibleModel drops video attachments (no MIME support)."""
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

    mp4 = temp_dir / "movie.mp4"
    mp4.write_bytes(_MP4_BYTES)
    read_output = UsefulTools().Read(str(mp4))

    model = OpenAICompatibleModel(
        "openai/gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
    )
    model.conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_v",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    # Only the tool message should be appended — no follow-up user message
    # because video has no supported content-part type.
    assert model.conversation[-1]["role"] == "tool"
    assert "[attached video/mp4," in model.conversation[-1]["content"]


def test_gemini_tool_result_routes_attachment_via_user_message(temp_dir):
    """GeminiModel lifts a PNG attachment into a follow-up user msg.

    The attachment must be present on the new user message so that
    :meth:`_convert_conversation_to_gemini_contents` can render it via
    ``Part.from_bytes`` for the API call.
    """
    from kiss.core.models.gemini_model import GeminiModel

    png_path = temp_dir / "screenshot.png"
    png_path.write_bytes(_PNG_1x1)
    read_output = UsefulTools().Read(str(png_path))

    model = GeminiModel("gemini-2.5-pro", api_key="key")
    model.conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_g",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    # Tool message comes first, follow-up user message with attachments next.
    tool_msg = model.conversation[-2]
    assert tool_msg["role"] == "tool"
    assert "<<KISS_BINARY_ATTACHMENT" not in tool_msg["content"]

    user_msg = model.conversation[-1]
    assert user_msg["role"] == "user"
    attachments = user_msg["attachments"]
    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/png"
    assert attachments[0].data == _PNG_1x1

    # Confirm the conversion path produces a Part with inline_data of the
    # right MIME type.
    contents = model._convert_conversation_to_gemini_contents()
    last = contents[-1]
    assert last.role == "user"
    has_inline = False
    for part in last.parts or []:
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.mime_type == "image/png":
            has_inline = True
            break
    assert has_inline


def test_gemini_tool_result_routes_video_attachment(temp_dir):
    """GeminiModel passes video bytes through (Gemini accepts video/mp4)."""
    from kiss.core.models.gemini_model import GeminiModel

    mp4 = temp_dir / "movie.mp4"
    mp4.write_bytes(_MP4_BYTES)
    read_output = UsefulTools().Read(str(mp4))

    model = GeminiModel("gemini-2.5-pro", api_key="key")
    model.conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_gv",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    user_msg = model.conversation[-1]
    assert user_msg["role"] == "user"
    assert user_msg["attachments"][0].mime_type == "video/mp4"


def test_anthropic_tool_result_drops_video(temp_dir):
    """Anthropic skips video attachments in tool results (no native support)."""
    from kiss.core.models.anthropic_model import AnthropicModel

    mp4 = temp_dir / "movie.mp4"
    mp4.write_bytes(_MP4_BYTES)
    read_output = UsefulTools().Read(str(mp4))

    model = AnthropicModel("claude-sonnet-4-5", api_key="sk-test")
    model.conversation = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_vid",
                    "name": "Read",
                    "input": {"file_path": str(mp4)},
                }
            ],
        },
    ]
    model.add_function_results_to_conversation_and_return(
        [("Read", {"result": read_output})]
    )

    block = model.conversation[-1]["content"][0]
    types = [b["type"] for b in block["content"]]
    # No image/document/audio block — video is dropped.  Only text remains.
    assert types == ["text"]
    assert "[attached video/mp4," in block["content"][0]["text"]
