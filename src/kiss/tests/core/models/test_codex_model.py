"""Tests for CodexModel — Codex CLI backend."""

import shutil

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.codex_model import CodexModel, _find_codex_cli
from kiss.core.models.model_info import MODEL_INFO, model

_has_codex = shutil.which("codex") is not None
requires_codex_cli = pytest.mark.skipif(not _has_codex, reason="codex CLI not installed")


class TestFindCodexCli:

    def test_find_codex_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(KISSError, match="not found"):
            _find_codex_cli()


class TestCliModelName:

    def test_default_strips_to_default(self) -> None:
        m = CodexModel("codex/default")
        assert m._cli_model == "default"

    def test_explicit_model_after_prefix(self) -> None:
        m = CodexModel("codex/gpt-5-codex")
        assert m._cli_model == "gpt-5-codex"

    def test_no_prefix_kept_as_is(self) -> None:
        m = CodexModel("gpt-5")
        assert m._cli_model == "gpt-5"


class TestBuildCliArgs:

    def test_default_omits_model_flag(self) -> None:
        m = CodexModel("codex/default")
        args = m._build_cli_args()
        assert "-m" not in args

    def test_explicit_model_adds_flag(self) -> None:
        m = CodexModel("codex/gpt-5-codex")
        args = m._build_cli_args()
        assert "-m" in args
        assert args[args.index("-m") + 1] == "gpt-5-codex"

    def test_default_args_present(self) -> None:
        m = CodexModel("codex/default")
        args = m._build_cli_args()
        assert "exec" in args
        assert "--json" in args
        assert "--skip-git-repo-check" in args
        assert "--sandbox" in args
        assert args[args.index("--sandbox") + 1] == "read-only"


class TestBuildPrompt:

    def test_single_user_message(self) -> None:
        m = CodexModel("codex/default")
        m.initialize("hello there")
        assert m._build_prompt() == "hello there"

    def test_tool_result_messages(self) -> None:
        m = CodexModel("codex/default")
        m.conversation = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "Calling tool"},
            {"role": "tool", "tool_call_id": "call_1", "content": "tool output"},
        ]
        prompt = m._build_prompt()
        assert "[Tool Result]: tool output" in prompt
        assert "[User]: Do something" in prompt
        assert "[Assistant]: Calling tool" in prompt

    def test_system_instruction_is_prepended(self) -> None:
        m = CodexModel("codex/default", model_config={"system_instruction": "Be brief."})
        m.initialize("hi")
        prompt = m._build_prompt()
        assert prompt.startswith("[System]: Be brief.")
        assert "[User]: hi" in prompt


class TestParseStreamEvents:

    def test_agent_message_emits_text_and_calls_callback(self) -> None:
        tokens: list[str] = []
        m = CodexModel("codex/default", token_callback=tokens.append)
        lines = [
            '{"type":"thread.started","thread_id":"abc"}',
            '{"type":"turn.started"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}',
            '{"type":"turn.completed","usage":{"input_tokens":10,'
            '"cached_input_tokens":4,"output_tokens":3,"reasoning_output_tokens":1}}',
        ]
        content, result, err = m._parse_stream_events(iter(lines))
        assert content == "hello"
        assert tokens == ["hello"]
        assert err is None
        assert result["thread_id"] == "abc"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 3

    def test_agent_reasoning_wraps_with_thinking_callback(self) -> None:
        thinking_states: list[bool] = []
        tokens: list[str] = []
        m = CodexModel(
            "codex/default",
            token_callback=tokens.append,
            thinking_callback=thinking_states.append,
        )
        lines = [
            '{"type":"item.completed","item":{"type":"agent_reasoning",'
            '"text":"thinking..."}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
        ]
        content, _, err = m._parse_stream_events(iter(lines))
        assert content == "answer"
        assert thinking_states == [True, False]
        assert tokens == ["thinking...", "answer"]
        assert err is None

    def test_turn_failed_returns_error_message(self) -> None:
        m = CodexModel("codex/default")
        lines = [
            '{"type":"thread.started","thread_id":"x"}',
            '{"type":"error","message":"boom"}',
            '{"type":"turn.failed","error":{"message":"boom"}}',
        ]
        _content, _result, err = m._parse_stream_events(iter(lines))
        assert err == "boom"

    def test_blank_and_invalid_json_lines_are_skipped(self) -> None:
        m = CodexModel("codex/default")
        lines = [
            "",
            "not json",
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        ]
        content, _result, err = m._parse_stream_events(iter(lines))
        assert content == "ok"
        assert err is None


class TestGenerateAndProcessWithTools:

    def test_system_prompt_restored_when_originally_empty(self) -> None:
        m = CodexModel("codex/default")
        m.initialize("test")
        try:
            m.generate_and_process_with_tools({"finish": lambda result: result})
        except Exception:
            pass
        assert "system_instruction" not in m.model_config


class TestUnsupportedMethods:
    def test_get_embedding_raises(self) -> None:
        m = CodexModel("codex/default")
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("test")


class TestTokenExtraction:

    def test_extract_from_non_dict(self) -> None:
        m = CodexModel("codex/default")
        assert m.extract_input_output_token_counts_from_response("bad") == (
            0, 0, 0, 0,
        )

    def test_extract_subtracts_cached_from_input(self) -> None:
        m = CodexModel("codex/default")
        response = {
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 70,
                "output_tokens": 30,
                "reasoning_output_tokens": 10,
            }
        }
        inp, out, cr, cw = m.extract_input_output_token_counts_from_response(response)
        assert inp == 30
        assert out == 30
        assert cr == 70
        assert cw == 0

    def test_extract_handles_missing_usage(self) -> None:
        m = CodexModel("codex/default")
        assert m.extract_input_output_token_counts_from_response({}) == (0, 0, 0, 0)


class TestModelRouting:
    def test_codex_prefix_creates_codex_model(self) -> None:
        m = model("codex/default")
        assert isinstance(m, CodexModel)
        assert m._cli_model == "default"

    def test_codex_explicit_model_routed_correctly(self) -> None:
        m = model("codex/gpt-5-codex")
        assert isinstance(m, CodexModel)
        assert m._cli_model == "gpt-5-codex"


class TestModelInfoEntries:
    def test_codex_models_in_model_info(self) -> None:
        assert "codex/default" in MODEL_INFO
        assert "codex/gpt-5" in MODEL_INFO
        assert "codex/gpt-5-codex" in MODEL_INFO
        assert "codex/gpt-5.1-codex" in MODEL_INFO

    def test_codex_models_support_function_calling(self) -> None:
        for name in (
            "codex/default",
            "codex/gpt-5",
            "codex/gpt-5-codex",
            "codex/gpt-5.1-codex",
        ):
            assert MODEL_INFO[name].is_function_calling_supported


@requires_codex_cli
class TestGenerateIntegration:
    """Integration tests that actually call the codex CLI."""

    @pytest.mark.timeout(120)
    def test_generate_token_counts(self) -> None:
        m = CodexModel("codex/default")
        m.initialize("Reply with only the word 'hi'.")
        _, response = m.generate()
        inp, out, _, _ = m.extract_input_output_token_counts_from_response(response)
        assert inp >= 0
        assert out > 0

    @pytest.mark.timeout(120)
    def test_generate_streaming(self) -> None:
        tokens: list[str] = []
        m = CodexModel("codex/default", token_callback=tokens.append)
        m.initialize("Reply with exactly the word 'pong'. Nothing else.")
        content, _response = m.generate()
        assert "pong" in content.lower()
        assert len(tokens) > 0

    @pytest.mark.timeout(120)
    def test_generate_failure_raises(self) -> None:
        m = CodexModel("codex/this-model-does-not-exist-xyz")
        m.initialize("hi")
        with pytest.raises(KISSError, match="Codex CLI failed"):
            m.generate()

    @pytest.mark.timeout(180)
    def test_generate_and_process_with_tools_runs(self) -> None:
        tokens: list[str] = []
        m = CodexModel("codex/default", token_callback=tokens.append)
        m.initialize("Call the finish tool with result='done'.")

        def finish(result: str) -> str:
            """Finish the task.

            Args:
                result: The final result.
            """
            return result

        function_calls, content, _response = m.generate_and_process_with_tools(
            {"finish": finish}
        )
        assert content
        if function_calls:
            assert function_calls[0]["name"] == "finish"
            assert m.conversation[-1].get("tool_calls")
