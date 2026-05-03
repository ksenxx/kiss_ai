"""Tests for CodexModel — Codex CLI backend."""

import shutil
import stat
from pathlib import Path

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models import codex_model as codex_module
from kiss.core.models.codex_model import (
    CodexModel,
    _find_codex_cli,
    _find_in_candidate_paths,
    find_codex_executable,
)
from kiss.core.models.model_info import MODEL_INFO, model

_has_codex = shutil.which("codex") is not None
requires_codex_cli = pytest.mark.skipif(not _has_codex, reason="codex CLI not installed")


class TestFindCodexCli:

    def test_find_codex_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(codex_module, "_UI_CANDIDATE_PATHS", ())
        with pytest.raises(KISSError, match="not found"):
            _find_codex_cli()


class TestFindInCandidatePaths:
    """Tests for ``_find_in_candidate_paths`` using real files."""

    def test_returns_none_when_no_candidate_exists(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "missing-codex"
        assert _find_in_candidate_paths([str(nonexistent)]) is None

    def test_skips_non_executable_files(self, tmp_path: Path) -> None:
        non_exec = tmp_path / "codex-noexec"
        non_exec.write_text("#!/bin/sh\necho hi\n")
        non_exec.chmod(stat.S_IRUSR | stat.S_IWUSR)
        assert _find_in_candidate_paths([str(non_exec)]) is None

    def test_returns_first_executable_match(self, tmp_path: Path) -> None:
        first = tmp_path / "first-codex"
        second = tmp_path / "second-codex"
        for f in (first, second):
            f.write_text("#!/bin/sh\necho hi\n")
            f.chmod(0o755)
        result = _find_in_candidate_paths([str(first), str(second)])
        assert result == str(first)

    def test_skips_missing_then_finds_existing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-codex"
        existing = tmp_path / "real-codex"
        existing.write_text("#!/bin/sh\necho hi\n")
        existing.chmod(0o755)
        result = _find_in_candidate_paths([str(missing), str(existing)])
        assert result == str(existing)

    def test_expands_user_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        bin_path = tmp_path / "codex-home"
        bin_path.write_text("#!/bin/sh\necho hi\n")
        bin_path.chmod(0o755)
        result = _find_in_candidate_paths(["~/codex-home"])
        assert result == str(bin_path)


class TestFindCodexExecutable:
    """Tests for the public ``find_codex_executable`` helper."""

    def test_prefers_path_when_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path_codex = tmp_path / "codex"
        path_codex.write_text("#!/bin/sh\necho hi\n")
        path_codex.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        ui_codex = tmp_path / "ui-codex"
        ui_codex.write_text("#!/bin/sh\necho hi\n")
        ui_codex.chmod(0o755)
        monkeypatch.setattr(codex_module, "_UI_CANDIDATE_PATHS", (str(ui_codex),))
        assert find_codex_executable() == str(path_codex)

    def test_falls_back_to_ui_when_not_on_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setenv("PATH", str(empty_dir))
        ui_codex = tmp_path / "ui-codex"
        ui_codex.write_text("#!/bin/sh\necho hi\n")
        ui_codex.chmod(0o755)
        monkeypatch.setattr(codex_module, "_UI_CANDIDATE_PATHS", (str(ui_codex),))
        assert find_codex_executable() == str(ui_codex)

    def test_returns_none_when_neither_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setenv("PATH", str(empty_dir))
        monkeypatch.setattr(codex_module, "_UI_CANDIDATE_PATHS", ())
        assert find_codex_executable() is None


class TestUiCandidatePaths:
    """Sanity checks on the hard-coded UI candidate path list."""

    def test_includes_macos_system_app_bundle(self) -> None:
        assert (
            "/Applications/Codex.app/Contents/Resources/codex"
            in codex_module._UI_CANDIDATE_PATHS
        )

    def test_includes_macos_user_app_bundle(self) -> None:
        assert (
            "~/Applications/Codex.app/Contents/Resources/codex"
            in codex_module._UI_CANDIDATE_PATHS
        )

    def test_includes_windows_install_path(self) -> None:
        assert any(
            "AppData/Local/Programs" in p and p.endswith("codex.exe")
            for p in codex_module._UI_CANDIDATE_PATHS
        )

    def test_includes_linux_install_path(self) -> None:
        assert "/opt/Codex/resources/codex" in codex_module._UI_CANDIDATE_PATHS


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
        assert "--dangerously-bypass-approvals-and-sandbox" in args

    def test_no_read_only_sandbox(self) -> None:
        # Regression: --sandbox read-only previously blocked codex from
        # making any file modifications when KISS asked it to fix code.
        m = CodexModel("codex/default")
        args = m._build_cli_args()
        assert "read-only" not in args
        # The sandbox flag must not be present at all; bypass replaces it.
        assert "--sandbox" not in args


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

    def test_command_execution_started_streams_command(self) -> None:
        # Regression: previously item.started events were ignored, so the
        # user saw nothing while codex was running shell commands and
        # appeared to wait silently for a long time.
        tokens: list[str] = []
        thinking_states: list[bool] = []
        m = CodexModel(
            "codex/default",
            token_callback=tokens.append,
            thinking_callback=thinking_states.append,
        )
        lines = [
            '{"type":"item.started","item":{"id":"item_0",'
            '"type":"command_execution","command":"/bin/zsh -lc ls",'
            '"status":"in_progress"}}',
            '{"type":"item.completed","item":{"id":"item_0",'
            '"type":"command_execution","command":"/bin/zsh -lc ls",'
            '"aggregated_output":"file1\\nfile2\\n","exit_code":0,'
            '"status":"completed"}}',
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"done"}}',
        ]
        content, _result, err = m._parse_stream_events(iter(lines))
        assert content == "done"
        assert err is None
        # Command-line text was streamed before the final answer.
        assert any("/bin/zsh -lc ls" in t for t in tokens)
        # Command output was streamed too.
        assert any("file1" in t for t in tokens)
        # Streaming was wrapped in thinking-callback pairs.
        assert thinking_states.count(True) == thinking_states.count(False)
        assert thinking_states.count(True) >= 2


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
        assert "codex/gpt-5.1-codex-max" in MODEL_INFO
        assert "codex/gpt-5.1-codex-mini" in MODEL_INFO
        assert "codex/gpt-5.2-codex" in MODEL_INFO
        assert "codex/gpt-5.3-codex" in MODEL_INFO
        assert "codex/gpt-5.5" in MODEL_INFO
        assert "codex/gpt-5.5-codex" in MODEL_INFO
        assert "codex/gpt-5.5-pro" in MODEL_INFO

    def test_codex_models_support_function_calling(self) -> None:
        for name in (
            "codex/default",
            "codex/gpt-5",
            "codex/gpt-5-codex",
            "codex/gpt-5.1-codex",
            "codex/gpt-5.1-codex-max",
            "codex/gpt-5.1-codex-mini",
            "codex/gpt-5.2-codex",
            "codex/gpt-5.3-codex",
            "codex/gpt-5.5",
            "codex/gpt-5.5-codex",
            "codex/gpt-5.5-pro",
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

    @pytest.mark.timeout(240)
    def test_generate_can_modify_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: codex must be able to make filesystem modifications.

        With ``--sandbox read-only`` codex would refuse to write the file
        and emit an ``agent_message`` saying it could not create it, which
        is what the user reported when running ``uv run check --full and
        fix`` against the codex/gpt-5.5 model.
        """
        monkeypatch.chdir(tmp_path)
        m = CodexModel("codex/default", model_config={"timeout": 240})
        m.initialize(
            f"Create a file at the absolute path "
            f"{tmp_path / 'result.txt'} with exactly the text "
            f"'ok' and nothing else. Do not ask for permission."
        )
        m.generate()
        result = tmp_path / "result.txt"
        assert result.exists(), (
            "Codex CLI should have been able to create the file, but it "
            "wasn't created — the sandbox is likely still read-only."
        )
        assert result.read_text().strip() == "ok"

    @pytest.mark.timeout(240)
    def test_generate_streams_command_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: progress events must stream while codex is working.

        Previously only ``item.completed`` agent_message events fired the
        token callback, so users saw nothing while codex was busy running
        shell commands — the symptom reported as "long waits".
        """
        monkeypatch.chdir(tmp_path)
        tokens: list[str] = []
        thinking_states: list[bool] = []
        m = CodexModel(
            "codex/default",
            token_callback=tokens.append,
            thinking_callback=thinking_states.append,
            model_config={"timeout": 240},
        )
        m.initialize(
            "Run the shell command 'ls -la' in the current directory and "
            "report the output."
        )
        m.generate()
        # The shell command should have been streamed via the token
        # callback while codex was executing it.
        assert any("ls" in t for t in tokens), (
            "Expected shell command progress to be streamed, but token "
            f"callback only saw: {tokens!r}"
        )

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
