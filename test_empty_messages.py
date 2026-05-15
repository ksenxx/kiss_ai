"""Test that empty messages raise a proper error."""

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel


def test_empty_messages_after_normalization_raises_error():
    """When all messages have whitespace-only content, a KISSError is raised."""
    m = AnthropicModel(
        model_name="claude-sonnet-4-20250514",
        api_key="test-key",
    )
    
    # Initialize with whitespace-only prompt
    m.initialize("   \n\t  ")
    
    # Now add another message with only whitespace content
    m.conversation.append(
        {"role": "assistant", "content": [{"type": "text", "text": "   "}]}
    )
    
    # This should raise a KISSError with a clear message
    with pytest.raises(KISSError) as exc_info:
        m._build_create_kwargs()
    
    error_msg = str(exc_info.value)
    assert "whitespace-only" in error_msg
    assert "at least one message" in error_msg


if __name__ == "__main__":
    test_empty_messages_after_normalization_raises_error()
    print("✓ Test passed! Error is raised correctly.")
