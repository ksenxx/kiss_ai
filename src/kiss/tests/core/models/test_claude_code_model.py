"""Tests for ClaudeCodeModel — Claude Code CLI backend."""

import shutil

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.model_info import MODEL_INFO, model

# Skip entire module if 'claude' CLI is not on PATH
_has_claude = shutil.which("claude") is not None
requires_claude_cli = pytest.mark.skipif(not _has_claude, reason="claude CLI not installed")


class TestFindClaudeCli:
    def test_find_claude_cli_returns_path(self) -> None:
        if not _has_claude:
            pytest.skip("claude CLI not installed")
        path = _find_claude_cli()
        assert "claude" in path

    def test_find_claude_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(KISSError, match="not found"):
            _find_claude_cli()


class TestClaudeCodeModelInit:
    def test_cc_prefix_stripped(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        assert m._cli_model == "opus"
        assert m.model_name == "cc/opus"

    def test_no_prefix(self) -> None:
        m = ClaudeCodeModel("opus")
        assert m._cli_model == "opus"

    def test_model_config_defaults(self) -> None:
        m = ClaudeCodeModel("cc/sonnet")
        assert m.model_config == {}


class TestClaudeCodeModelInitialize:
    def test_basic_init(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("Hello")
        assert len(m.conversation) == 1
        assert m.conversation[0] == {"role": "user", "content": "Hello"}

    def test_attachments_ignored(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("Hello", attachments=[])
        assert len(m.conversation) == 1


class TestBuildPrompt:
    def test_single_message(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("Hello world")
        assert m._build_prompt() == "Hello world"

    def test_multi_turn(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = m._build_prompt()
        assert "[User]: Hi" in prompt
        assert "[Assistant]: Hello!" in prompt
        assert "[User]: How are you?" in prompt


class TestBuildCliArgs:
    def test_basic_args(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        args = m._build_cli_args()
        assert "--print" in args
        assert "--tools" in args
        assert "--bare" in args
        assert "--model" in args
        idx = args.index("--model")
        assert args[idx + 1] == "opus"
        assert "--output-format" in args
        idx = args.index("--output-format")
        assert args[idx + 1] == "json"

    def test_streaming_args(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        args = m._build_cli_args(use_streaming=True)
        idx = args.index("--output-format")
        assert args[idx + 1] == "stream-json"
        assert "--verbose" in args
        assert "--include-partial-messages" in args

    def test_system_prompt(self) -> None:
        m = ClaudeCodeModel(
            "cc/opus", model_config={"system_instruction": "Be concise."}
        )
        m.initialize("test")
        args = m._build_cli_args()
        assert "--system-prompt" in args
        idx = args.index("--system-prompt")
        assert args[idx + 1] == "Be concise."


class TestUnsupportedMethods:
    def test_get_embedding_raises(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("test")


class TestTokenExtraction:
    def test_extract_from_valid_response(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            }
        }
        inp, out, cr, cw = m.extract_input_output_token_counts_from_response(
            response
        )
        assert inp == 100
        assert out == 50
        assert cr == 10
        assert cw == 5

    def test_extract_from_empty_response(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        assert m.extract_input_output_token_counts_from_response({}) == (0, 0, 0, 0)

    def test_extract_from_non_dict(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        assert m.extract_input_output_token_counts_from_response("bad") == (
            0, 0, 0, 0,
        )

    def test_extract_partial_usage(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        response = {"usage": {"input_tokens": 42}}
        inp, out, cr, cw = m.extract_input_output_token_counts_from_response(
            response
        )
        assert inp == 42
        assert out == 0
        assert cr == 0
        assert cw == 0


class TestModelRouting:
    def test_cc_prefix_creates_claude_code_model(self) -> None:
        m = model("cc/opus")
        assert isinstance(m, ClaudeCodeModel)
        assert m._cli_model == "opus"

    def test_cc_prefix_with_full_name(self) -> None:
        m = model("cc/claude-opus-4-6")
        assert isinstance(m, ClaudeCodeModel)
        assert m._cli_model == "claude-opus-4-6"


class TestModelInfoEntries:
    def test_cc_models_in_model_info(self) -> None:
        assert "cc/opus" in MODEL_INFO
        assert "cc/sonnet" in MODEL_INFO
        assert "cc/haiku" in MODEL_INFO

    def test_cc_models_no_function_calling(self) -> None:
        for name in ("cc/opus", "cc/sonnet", "cc/haiku"):
            assert not MODEL_INFO[name].is_function_calling_supported


@requires_claude_cli
class TestGenerateIntegration:
    """Integration tests that actually call the claude CLI."""

    @pytest.mark.timeout(60)
    def test_generate_simple(self) -> None:
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("Reply with exactly the word 'pong'. Nothing else.")
        content, response = m.generate()
        assert "pong" in content.lower()
        assert isinstance(response, dict)
        assert "result" in response

    @pytest.mark.timeout(60)
    def test_generate_with_system_prompt(self) -> None:
        m = ClaudeCodeModel(
            "cc/haiku",
            model_config={"system_instruction": "Always reply in uppercase."},
        )
        m.initialize("Say hello")
        content, response = m.generate()
        assert content  # non-empty
        assert isinstance(response, dict)

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
        # Streaming should have produced at least one token callback
        assert len(tokens) > 0

    @pytest.mark.timeout(60)
    def test_multi_turn(self) -> None:
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("My name is Alice.")
        content1, _ = m.generate()
        assert content1
        # Add follow-up
        m.add_message_to_conversation("user", "What is my name?")
        content2, _ = m.generate()
        assert "alice" in content2.lower()

    @pytest.mark.timeout(60)
    def test_conversation_appended_after_generate(self) -> None:
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("Say 'ok'")
        m.generate()
        assert len(m.conversation) == 2
        assert m.conversation[1]["role"] == "assistant"

    @pytest.mark.timeout(60)
    def test_reset_conversation(self) -> None:
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("Say 'ok'")
        m.generate()
        m.reset_conversation()
        assert m.conversation == []

    @pytest.mark.timeout(120)
    def test_generate_and_process_with_tools_agentic(self, tmp_path: object) -> None:
        """Test that generate_and_process_with_tools runs the CLI as a full agent."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as work_dir:
            m = ClaudeCodeModel(
                "cc/haiku",
                model_config={
                    "system_instruction": f"Work dir: {work_dir}. "
                    "Create files in the work directory.",
                },
            )
            m.initialize(
                f"Create a file called test_output.txt in {work_dir} "
                "containing 'agent works'. Do nothing else."
            )
            calls, text, response = m.generate_and_process_with_tools({})
            # Should return a synthetic finish call
            assert len(calls) == 1
            assert calls[0]["name"] == "finish"
            assert calls[0]["arguments"]["success"] == "true"
            assert calls[0]["arguments"]["is_continue"] == "false"
            assert isinstance(calls[0]["arguments"]["summary"], str)
            # The CLI agent should have created the file
            output_file = Path(work_dir) / "test_output.txt"
            assert output_file.exists()
            assert "agent works" in output_file.read_text()
