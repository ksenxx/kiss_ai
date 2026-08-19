# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ClaudeCodeModel — Claude Code CLI backend."""

import shutil

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model_info import MODEL_INFO, model
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401

_has_claude = shutil.which("claude") is not None


requires_claude_cli = pytest.mark.skipif(not _has_claude, reason="claude CLI not installed")


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

    @pytest.mark.slow
    def test_system_prompt_restored_when_originally_empty(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        try:
            m.generate_and_process_with_tools({"finish": lambda result: result})
        except Exception:
            pass
        assert "system_instruction" not in m.model_config


class TestTokenExtraction:

    def test_extract_from_non_dict(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        assert m.extract_input_output_token_counts_from_response("bad") == (
            0, 0, 0, 0, 0,
        )

    def test_extract_split_cache_creation(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 40,
                    "ephemeral_1h_input_tokens": 50,
                },
            }
        }
        assert m.extract_input_output_token_counts_from_response(response) == (
            100, 20, 30, 40, 50,
        )

    def test_extract_aggregate_cache_creation_is_conservative_one_hour(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 50,
            }
        }
        assert m.extract_input_output_token_counts_from_response(response) == (
            100, 20, 30, 0, 50,
        )


class TestModelRouting:
    def test_cc_prefix_creates_claude_code_model(self) -> None:
        m = model("cc/opus")
        assert isinstance(m, ClaudeCodeModel)
        assert m._cli_model == "opus"


class TestBuildCliArgs:
    """Verify CLI argument construction — agentic mode, like CodexModel."""

    def test_agentic_mode_enables_builtin_tools(self) -> None:
        """Native tools stay enabled; permission prompts are bypassed.

        Like ``codex exec --dangerously-bypass-approvals-and-sandbox``,
        the ``claude`` CLI runs agentically (Bash/Edit/Read) with
        ``--dangerously-skip-permissions``, so ``--tools`` must not
        appear — an empty ``--tools ""`` would disable agentic mode.
        """
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        args = m._build_cli_args()
        assert "--tools" not in args
        assert "--dangerously-skip-permissions" in args

    def test_no_session_persistence_flag_present(self) -> None:
        """``--no-session-persistence`` keeps each invocation self-contained."""
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        args = m._build_cli_args()
        assert "--no-session-persistence" in args


class TestModelInfoEntries:
    def test_cc_models_in_model_info(self) -> None:
        assert "cc/opus" in MODEL_INFO
        assert "cc/sonnet" in MODEL_INFO
        assert "cc/haiku" in MODEL_INFO

    def test_cc_models_support_function_calling(self) -> None:
        for name in ("cc/opus", "cc/sonnet", "cc/haiku"):
            assert MODEL_INFO[name].is_function_calling_supported


@requires_claude_cli
@pytest.mark.slow
@pytest.mark.live_cli
class TestGenerateIntegration:
    """Integration tests that actually call the claude CLI."""

    @pytest.mark.timeout(60)
    def test_generate_token_counts(self) -> None:
        m = ClaudeCodeModel("cc/haiku")
        m.initialize("Say 'hi'")
        _, response = m.generate()
        inp, out, _, _, _ = m.extract_input_output_token_counts_from_response(response)
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
