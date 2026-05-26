# Author: Koushik Sen (ksen@berkeley.edu)

"""Tests that AnthropicModel._normalize_content_blocks drops whitespace-only text blocks.

The Anthropic API rejects messages that contain text content blocks with
only whitespace text (``invalid_request_error: messages: text content
blocks must contain non-whitespace text``).  These tests verify that
_normalize_content_blocks silently filters such blocks.
"""

from types import SimpleNamespace

from kiss.core.models.anthropic_model import AnthropicModel


def _make_model() -> AnthropicModel:
    return AnthropicModel(
        model_name="claude-sonnet-4-20250514",
        api_key="test-key",
    )


def test_empty_text_block_is_dropped():
    """A text block with empty string text is dropped."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [SimpleNamespace(type="text", text="")]
    )
    assert blocks == []


def test_whitespace_only_text_block_is_dropped():
    """A text block with only whitespace is dropped."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [SimpleNamespace(type="text", text="   \n\t  ")]
    )
    assert blocks == []


def test_non_empty_text_block_is_kept():
    """A text block with real text is kept."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [SimpleNamespace(type="text", text="Hello world")]
    )
    assert len(blocks) == 1
    assert blocks[0] == {"type": "text", "text": "Hello world"}


def test_tool_use_blocks_are_never_dropped():
    """tool_use blocks must not be filtered regardless of text content."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [
            SimpleNamespace(type="text", text=""),
            SimpleNamespace(type="tool_use", id="t1", name="Bash", input={"command": "ls"}),
        ]
    )
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["name"] == "Bash"


def test_thinking_blocks_are_never_dropped():
    """thinking blocks must not be filtered even when thinking text is empty."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [SimpleNamespace(type="thinking", thinking="", signature="sig123")]
    )
    assert len(blocks) == 1
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["signature"] == "sig123"


def test_dict_text_block_with_whitespace_is_dropped():
    """A pre-existing dict text block with whitespace-only text is dropped."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [{"type": "text", "text": "  "}]
    )
    assert blocks == []


def test_dict_non_text_block_is_kept():
    """A pre-existing dict non-text block is always kept."""
    m = _make_model()
    tool_block = {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
    blocks = m._normalize_content_blocks([tool_block])
    assert len(blocks) == 1
    assert blocks[0] == tool_block


def test_mixed_blocks_only_empty_text_is_dropped():
    """In a mix of block types, only empty/whitespace text blocks are dropped."""
    m = _make_model()
    blocks = m._normalize_content_blocks(
        [
            SimpleNamespace(type="thinking", thinking="I will run Bash", signature="s1"),
            SimpleNamespace(type="text", text=""),
            SimpleNamespace(type="text", text="Here is my plan"),
            SimpleNamespace(type="text", text="   "),
            SimpleNamespace(type="tool_use", id="t1", name="Bash", input={"command": "ls"}),
        ]
    )
    assert len(blocks) == 3
    types = [b["type"] for b in blocks]
    assert types == ["thinking", "text", "tool_use"]
    assert blocks[1]["text"] == "Here is my plan"


def test_model_dump_whitespace_text_is_dropped():
    """A block with model_dump that produces a whitespace-only text dict is dropped."""
    m = _make_model()

    class FakeBlock:
        type = "text"

        def model_dump(self, **_kwargs):
            return {"type": "text", "text": "  \n  "}

    blocks = m._normalize_content_blocks([FakeBlock()])
    assert blocks == []


def test_model_dump_non_text_is_kept():
    """A block with model_dump that produces a non-text type is kept."""
    m = _make_model()

    class FakeBlock:
        type = "server_tool_use"

        def model_dump(self, **_kwargs):
            return {"type": "server_tool_use", "id": "x"}

    blocks = m._normalize_content_blocks([FakeBlock()])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "server_tool_use"


def test_fallback_str_whitespace_is_dropped():
    """A block that falls through to str() with whitespace result is dropped."""
    m = _make_model()

    class WeirdBlock:
        def __str__(self):
            return "  "

    blocks = m._normalize_content_blocks([WeirdBlock()])
    assert blocks == []


def test_fallback_str_with_content_is_kept():
    """A block that falls through to str() with real content is kept."""
    m = _make_model()

    class WeirdBlock:
        def __str__(self):
            return "some content"

    blocks = m._normalize_content_blocks([WeirdBlock()])
    assert len(blocks) == 1
    assert blocks[0] == {"type": "text", "text": "some content"}
