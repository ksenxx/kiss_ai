# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for CodexModel — Codex CLI backend."""

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from kiss.core.models import codex_model as codex_module
from kiss.core.models.codex_model import (
    CODEX_LOCKDOWN_ARGS,
    CodexModel,
    _find_in_candidate_paths,
    find_codex_executable,
)
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, model
from kiss.scripts.update_models import SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.conftest import IS_WINDOWS, posix_only
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_has_codex = shutil.which("codex") is not None


requires_codex_cli = pytest.mark.skipif(not _has_codex, reason="codex CLI not installed")


# Catalog entries the Codex backend must expose.  ``codex/gpt-5.4`` was
# retired upstream and dropped by the 2026-09-26 ``update_models.py`` refresh.
_CODEX_MODEL_NAMES = (
    "codex/codex-auto-review",
    "codex/default",
    "codex/gpt-5.5",
    "codex/gpt-5.6-luna",
    "codex/gpt-5.6-sol",
    "codex/gpt-5.6-terra",
    "codex/gpt-6-astra",
)

# Slugs rejected by the Codex CLI when authenticated with a ChatGPT
# subscription (HTTP 400: "not supported when using Codex with a ChatGPT
# account"), verified live on 2026-09-02. They must stay out of the
# catalog. Derived from the updater's exclusion set so the two cannot
# drift (a refresh once resurrected them when this list was standalone).
_SUBSCRIPTION_INCOMPATIBLE_NAMES = tuple(
    sorted(
        f"codex/{slug}" for slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS
    )
)


class TestFindInCandidatePaths:
    """Tests for ``_find_in_candidate_paths`` using real files."""

    def test_returns_none_when_no_candidate_exists(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "missing-codex"
        assert _find_in_candidate_paths([str(nonexistent)]) is None

    @posix_only("Windows has no executable bit: every existing file passes X_OK")
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
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what expanduser reads on Windows
        bin_path = tmp_path / "codex-home"
        bin_path.write_text("#!/bin/sh\necho hi\n")
        bin_path.chmod(0o755)
        result = _find_in_candidate_paths(["~/codex-home"])
        assert result is not None
        assert Path(result) == bin_path  # expanduser keeps the "/" on Windows


class TestFindCodexExecutable:
    """Tests for the public ``find_codex_executable`` helper."""

    def test_prefers_path_when_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # shutil.which only returns PATHEXT names on Windows, as for npm's codex.cmd.
        path_codex = tmp_path / ("codex.cmd" if IS_WINDOWS else "codex")
        path_codex.write_text("#!/bin/sh\necho hi\n")
        path_codex.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        ui_codex = tmp_path / "ui-codex"
        ui_codex.write_text("#!/bin/sh\necho hi\n")
        ui_codex.chmod(0o755)
        monkeypatch.setattr(codex_module, "_UI_CANDIDATE_PATHS", (str(ui_codex),))
        found = find_codex_executable()
        assert found is not None
        # shutil.which spells the extension the way PATHEXT does (``.CMD``).
        assert os.path.normcase(found) == os.path.normcase(str(path_codex))

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
        args = m._build_cli_args("/tmp/sys.md")
        assert "-m" not in args

    def test_explicit_model_adds_flag(self) -> None:
        m = CodexModel("codex/gpt-5-codex")
        args = m._build_cli_args("/tmp/sys.md")
        assert "-m" in args
        assert args[args.index("-m") + 1] == "gpt-5-codex"

    def test_lockdown_args_present(self) -> None:
        """Every native capability is off: the model has only KISS's tools."""
        m = CodexModel("codex/default")
        args = m._build_cli_args("/tmp/sys.md")
        assert args[1:4] == ["exec", "--json", "--skip-git-repo-check"]
        assert tuple(args[4 : 4 + len(CODEX_LOCKDOWN_ARGS)]) == CODEX_LOCKDOWN_ARGS
        assert "--dangerously-bypass-approvals-and-sandbox" not in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert "--ephemeral" in args
        assert "--ignore-user-config" in args
        for feature in ("shell_tool", "unified_exec", "hooks"):
            assert args[args.index(feature) - 1] == "--disable"
        # The code-mode host routes the ``kiss`` MCP server's calls (codex 0.163).
        assert "code_mode_host" not in args
        # Outside a ``native_tools_exposed`` block no MCP server is registered.
        assert not any(a.startswith("mcp_servers.") for a in args)
        assert "agents.enabled=false" in args
        assert 'web_search="disabled"' in args
        assert "skills.max_context_tokens=1" in args
        assert "project_doc_max_bytes=0" in args
        # Codex emits ``reasoning`` items (shown as thinking) only when a
        # reasoning summary is configured, and the user config is ignored.
        assert args[args.index('model_reasoning_summary="detailed"') - 1] == "-c"

    def test_instructions_file_is_passed_as_a_toml_string(self) -> None:
        m = CodexModel("codex/default")
        args = m._build_cli_args("/tmp/kiss codex/sys.md")
        value = next(a for a in args if a.startswith("model_instructions_file="))
        assert value == 'model_instructions_file="/tmp/kiss codex/sys.md"'
        assert args[args.index(value) - 1] == "-c"


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

    def test_system_instruction_never_rides_in_the_prompt(self) -> None:
        m = CodexModel("codex/default", model_config={"system_instruction": "Be brief."})
        m.initialize("hi")
        assert m._build_prompt() == "hi"


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

    @pytest.mark.parametrize("item_type", ["reasoning", "agent_reasoning"])
    def test_agent_reasoning_wraps_with_thinking_callback(self, item_type: str) -> None:
        """A completed reasoning item is shown as thinking.

        ``codex exec --json`` (codex-cli 0.163.0) names the item
        ``reasoning``; ``agent_reasoning`` is the older name.  Neither
        counts as a tool item (an extra request).
        """
        thinking_states: list[bool] = []
        tokens: list[str] = []
        m = CodexModel(
            "codex/default",
            token_callback=tokens.append,
            thinking_callback=thinking_states.append,
        )
        lines = [
            f'{{"type":"item.completed","item":{{"type":"{item_type}","text":"thinking..."}}}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
        ]
        content, result, err = m._parse_stream_events(iter(lines))
        assert content == "answer"
        assert thinking_states == [True, False]
        assert tokens == ["thinking...", "answer"]
        assert err is None
        assert result["requests"] == 1

    def test_reasoning_title_lines_lose_their_bold_markers(self) -> None:
        """Codex summaries are bold titles; the Thinking panel shows plain text.

        Only a line that is wholly one bold title is unwrapped; bold inside
        a sentence, or two bold spans on one line, stays as written.
        """
        tokens: list[str] = []
        m = CodexModel("codex/default", token_callback=tokens.append)
        text = "**Checking descent**\\n**Verifying bounds.**\\nA **key** step\\n**a** and **b**"
        m._parse_stream_events(
            iter([f'{{"type":"item.completed","item":{{"type":"reasoning","text":"{text}"}}}}'])
        )
        assert tokens == ["Checking descent\nVerifying bounds.\nA **key** step\n**a** and **b**"]

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

    def test_startup_error_items_are_ignored(self) -> None:
        """The locked-down CLI announces its disabled code-mode host as an
        ``item.type == "error"`` item; that is not a turn failure."""
        tokens: list[str] = []
        m = CodexModel("codex/default", token_callback=tokens.append)
        lines = [
            '{"type":"item.completed","item":{"id":"item_0","type":"error",'
            '"message":"Code Mode is unavailable because code-mode host is disabled."}}',
            '{"type":"turn.started"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
        ]
        content, _result, err = m._parse_stream_events(iter(lines))
        assert content == "done"
        assert err is None
        assert tokens == ["done"]


class TestGenerateAndProcessWithTools:

    @pytest.mark.slow
    def test_system_prompt_restored_when_originally_empty(self) -> None:
        m = CodexModel("codex/default")
        m.initialize("test")
        try:
            m.generate_and_process_with_tools({"finish": lambda result: result})
        except Exception:
            pass
        assert "system_instruction" not in m.model_config


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

    def test_extract_splits_cache_writes_out_of_input(self) -> None:
        """``turn.completed`` usage: input 100 = cached 40 + written 60 (codex-rs
        ``responses.rs`` test); the written subset is billed at the cache-write rate,
        not as uncached input."""
        m = CodexModel("codex/default")
        response = {
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "cache_write_input_tokens": 60,
                "output_tokens": 30,
                "reasoning_output_tokens": 10,
            }
        }
        assert m.extract_input_output_token_counts_from_response(response) == (0, 30, 40, 60)

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


_CODEX_RECORDS_ITS_ARGUMENTS = """
    import json
    import os
    import pathlib
    import sys

    sys.stdin.read()
    pathlib.Path(os.environ["KISS_TEST_ARGV_FILE"]).write_text(json.dumps(sys.argv[1:]))
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "seen"}}), flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 5, "cached_input_tokens": 0,
                                "output_tokens": 1}}), flush=True)
"""


@pytest.mark.parametrize(
    ("name", "config", "cli_model", "effort"),
    [
        ("codex/gpt-6-astra-xhigh", None, "gpt-6-astra", "xhigh"),
        ("codex/gpt-6-astra-xhigh", {"reasoning_effort": "low"}, "gpt-6-astra", "low"),
        ("codex/gpt-6-astra", None, "gpt-6-astra", None),
        ("codex/default", {"reasoning_effort": "high"}, None, "high"),
        # Not an alias: a level suffix the catalog does not mark is the model's own name.
        ("codex/o3-mini-high", None, "o3-mini-high", None),
    ],
)
def test_an_effort_alias_runs_its_base_model_at_that_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    config: dict[str, Any] | None,
    cli_model: str | None,
    effort: str | None,
) -> None:
    """``codex/gpt-6-astra-xhigh`` is GPT-6 Astra at ``xhigh`` effort, not a model of the CLI's.

    ``-xhigh`` is the alias suffix every lookup of the catalog strips, and
    the CLI was handed the alias as ``-m gpt-6-astra-xhigh``: a model it
    does not know, with no effort set.  The alias now runs its base model
    with ``model_reasoning_effort`` set, and ``reasoning_effort`` in the
    ``model_config`` (the framework's key) outranks the alias's level.
    """
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_RECORDS_ITS_ARGUMENTS)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("KISS_TEST_ARGV_FILE", str(argv_file))
    m = model(name, model_config=config)
    m.initialize("hi")

    content, _ = m.generate()

    argv = json.loads(argv_file.read_text())
    assert content == "seen"
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == "-m"] == (
        [cli_model] if cli_model else []
    )
    assert [arg for arg in argv if arg.startswith("model_reasoning_effort=")] == (
        [f'model_reasoning_effort="{effort}"'] if effort else []
    )


_CODEX_RECORDS_ITS_INPUT = """
    import base64
    import json
    import os
    import sys

    argv = sys.argv[1:]
    record = {"stdin": sys.stdin.read(), "images": {}}
    for path in [argv[i + 1] for i, arg in enumerate(argv) if arg == "-i"]:
        with open(path, "rb") as image:
            record["images"][path] = base64.b64encode(image.read()).decode()
    with open(os.environ["KISS_TEST_INPUT_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\\n")
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "seen"}}), flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 5, "cached_input_tokens": 0,
                                "output_tokens": 1}}), flush=True)
"""


def test_user_images_reach_the_cli_as_image_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Images the user attached are handed to ``codex exec`` with ``-i``.

    ``initialize`` dropped them with a log line and the transcript ignored
    the ``attachments`` of later user messages, while Sorcar's prompt told
    the model the files were "included in this message as inline content
    that you can see directly".  Codex 0.163 sends the N-th ``-i`` file
    ahead of the prompt as ``[Image #N]``, the label the transcript uses;
    a file that is no image (``-i`` cannot carry a PDF) is named in the
    prompt as not shown.  The files live only as long as the turn.
    """
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_RECORDS_ITS_INPUT)
    log = tmp_path / "input.jsonl"
    monkeypatch.setenv("KISS_TEST_INPUT_LOG", str(log))
    png = Attachment(b"\x89PNG\r\n\x1a\npng-bytes", "image/png")
    jpeg = Attachment(b"\xff\xd8\xffjpeg-bytes", "image/jpeg")
    m = CodexModel("codex/default")
    m.initialize("describe these", [png, Attachment(b"%PDF-1.4 pdf-bytes", "application/pdf")])

    assert m.generate()[0] == "seen"
    m.add_message_to_conversation("user", "and this one", attachments=[jpeg])
    m.generate()

    first, second = (json.loads(line) for line in log.read_text().splitlines())
    task = (
        "describe these\n[Image #1]\n[1 attachment(s) of this message could not be shown "
        "to this model: application/pdf (18 bytes)]"
    )
    assert list(first["images"].values()) == [png.to_base64()]
    assert first["stdin"] == task
    assert list(second["images"].values()) == [png.to_base64(), jpeg.to_base64()]
    assert second["stdin"] == (
        f"[User]: {task}\n\n[Assistant]: seen\n\n[User]: and this one\n[Image #2]"
    )
    assert not any(os.path.exists(path) for path in [*first["images"], *second["images"]])


class TestModelInfoEntries:
    def test_codex_models_in_model_info(self) -> None:
        for name in _CODEX_MODEL_NAMES:
            assert name in MODEL_INFO

    def test_codex_models_support_function_calling(self) -> None:
        for name in _CODEX_MODEL_NAMES:
            assert MODEL_INFO[name].is_function_calling_supported

    def test_subscription_incompatible_models_not_in_model_info(self) -> None:
        for name in _SUBSCRIPTION_INCOMPATIBLE_NAMES:
            assert name not in MODEL_INFO


@requires_codex_cli
@pytest.mark.slow
@pytest.mark.live_cli
@pytest.mark.live_api
class TestGenerateIntegration:
    """Integration tests that actually call the codex CLI."""

    @pytest.fixture(autouse=True)
    def _bill_the_api_key_when_not_logged_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a ChatGPT login, run on ``OPENAI_API_KEY`` or skip."""
        if codex_module.codex_auth_status()["loggedIn"]:
            return
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("codex is not logged in and OPENAI_API_KEY is not set")
        from kiss.core.config import DEFAULT_CONFIG

        monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)

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

    @pytest.mark.timeout(600)
    def test_generate_streams_reasoning_as_thinking(self) -> None:
        """A turn that needs reasoning shows its summary titles as thinking.

        The backend sends Codex summaries as bold titles only; they must
        reach the thinking callback, as plain text, before the answer.
        The model skips the summary when it reasons briefly (about one
        turn in eight on this prompt), so up to four turns are tried.
        """
        events: list[object] = []
        for _ in range(4):
            m = CodexModel(
                "codex/default",
                token_callback=events.append,
                thinking_callback=events.append,
                model_config={"timeout": 240},
            )
            m.initialize(
                "Find all integer solutions (x, y) of x^2 - 7y^2 = 2 with |x| < 200. "
                "Answer briefly."
            )
            content, _ = m.generate()
            assert content
            if True in events:
                break
        start = events.index(True)
        thinking = events[start + 1]
        assert events[start + 2] is False
        assert isinstance(thinking, str) and thinking.strip()
        assert "**" not in thinking

    @pytest.mark.timeout(240)
    def test_generate_has_no_native_file_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Locked down, codex is a bare model: asked to create a file with no
        KISS tool available, it cannot touch the file system (KISS-driven
        writes are covered by ``test_cli_models_live.py``)."""
        monkeypatch.chdir(tmp_path)
        m = CodexModel("codex/default", model_config={"timeout": 240})
        m.initialize(
            f"Create a file at the absolute path "
            f"{tmp_path / 'result.txt'} with exactly the text "
            f"'ok' and nothing else. Do not ask for permission."
        )
        content, _ = m.generate()
        assert content
        assert not (tmp_path / "result.txt").exists(), (
            "the locked-down codex CLI must have no native file tools"
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
