"""Tests for ClaudeCodeModel — Claude Code CLI backend."""

import shutil

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.model_info import MODEL_INFO, model

_has_claude = shutil.which("claude") is not None
requires_claude_cli = pytest.mark.skipif(not _has_claude, reason="claude CLI not installed")


class TestFindClaudeCli:

    def test_find_claude_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(KISSError, match="not found"):
            _find_claude_cli()


class TestBuildPrompt:

    def test_tool_result_messages(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.conversation = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "Calling tool"},
            {"role": "tool", "tool_call_id": "call_1", "content": "tool output"},
        ]
        prompt = m._build_prompt()
        assert "[Tool Result]: tool output" in prompt


class TestGenerateAndProcessWithTools:

    def test_system_prompt_restored_when_originally_empty(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        try:
            m.generate_and_process_with_tools({"finish": lambda result: result})
        except Exception:
            pass
        assert "system_instruction" not in m.model_config


class TestUnsupportedMethods:
    def test_get_embedding_raises(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("test")


class TestTokenExtraction:

    def test_extract_from_non_dict(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        assert m.extract_input_output_token_counts_from_response("bad") == (
            0, 0, 0, 0,
        )


class TestModelRouting:
    def test_cc_prefix_creates_claude_code_model(self) -> None:
        m = model("cc/opus")
        assert isinstance(m, ClaudeCodeModel)
        assert m._cli_model == "opus"


class TestModelInfoEntries:
    def test_cc_models_in_model_info(self) -> None:
        assert "cc/opus" in MODEL_INFO
        assert "cc/sonnet" in MODEL_INFO
        assert "cc/haiku" in MODEL_INFO

    def test_cc_models_support_function_calling(self) -> None:
        for name in ("cc/opus", "cc/sonnet", "cc/haiku"):
            assert MODEL_INFO[name].is_function_calling_supported


@requires_claude_cli
class TestGenerateIntegration:
    """Integration tests that actually call the claude CLI."""

    @pytest.mark.timeout(60)
    def test_generate_token_counts(self) -> None:
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("Say 'hi'")
        _, response = m.generate()
        inp, out, _, _ = m.extract_input_output_token_counts_from_response(response)
        assert inp > 0
        assert out > 0

    @pytest.mark.timeout(60)
    def test_generate_streaming(self) -> None:
        tokens: list[str] = []
        m = ClaudeCodeModel("cc/haiku", token_callback=tokens.append)
        m.initialize("Reply with exactly the word 'pong'. Nothing else.")
        content, response = m.generate()
        assert "pong" in content.lower()
        assert len(tokens) > 0
