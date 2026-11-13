# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ClaudeCodeModel — Claude Code CLI backend."""

import json
import os
import shutil
from pathlib import Path

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel, _system_prompt_file
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import MODEL_INFO, model
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

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
    """Verify CLI argument construction — a bare model, like CodexModel."""

    def test_native_tools_and_customizations_are_off(self) -> None:
        """``--tools ""`` removes the CLI's own tools; ``--safe-mode`` and
        ``--strict-mcp-config`` drop the user's CLAUDE.md, skills, hooks
        and MCP servers, so KISS's tools are the only ones the model has.
        """
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        args = m._build_cli_args()
        assert args[args.index("--tools") + 1] == ""
        assert "--safe-mode" in args
        assert "--strict-mcp-config" in args
        assert "--dangerously-skip-permissions" not in args
        assert "--bare" not in args  # --bare would also disable OAuth login

    def test_system_instruction_replaces_the_cli_system_prompt(self) -> None:
        """KISS's system prompt goes to ``--system-prompt-file`` (replace), not
        ``--append-system-prompt``, and never into the user prompt."""
        m = ClaudeCodeModel("cc/opus", model_config={"system_instruction": "be terse"})
        m.initialize("do it")
        with _system_prompt_file(m.system_instruction_text()) as path:
            args = m._build_cli_args(path)
            with open(args[args.index("--system-prompt-file") + 1], encoding="utf-8") as f:
                assert f.read() == "be terse"
        assert "--append-system-prompt" not in args
        assert m._build_prompt() == "do it"

    def test_no_session_persistence_flag_present(self) -> None:
        """``--no-session-persistence`` keeps each invocation self-contained."""
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        args = m._build_cli_args()
        assert "--no-session-persistence" in args


def _flag(args: list[str], flag: str) -> str | None:
    """Return the value after ``flag`` in a CLI argument list, or None when absent."""
    return args[args.index(flag) + 1] if flag in args else None


class TestEffortFlag:
    """``--effort`` follows the catalog ``thinking`` level; ``reasoning_effort`` outranks it."""

    def test_entry_without_a_level_sends_no_flag_and_keeps_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``cc/opus`` names whichever Opus the CLI picks and carries no level:
        the CLI's own default applies and a ``CLAUDE_CODE_EFFORT_LEVEL`` the
        user set stays in force, whichever way the subscription switch is."""
        from kiss.core.config import DEFAULT_CONFIG

        assert MODEL_INFO["cc/opus"].thinking is None
        monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
        m = ClaudeCodeModel("cc/opus")
        m.initialize("test")
        assert "--effort" not in m._build_cli_args()
        monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", True)
        env = m._cli_env()
        assert env is not None and env["CLAUDE_CODE_EFFORT_LEVEL"] == "low"
        monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
        assert m._cli_env()["CLAUDE_CODE_EFFORT_LEVEL"] == "low"  # a copy of the daemon's env

    def test_catalog_level_of_the_base_and_of_an_alias(self) -> None:
        """A base entry runs at the level the catalog stores; a generated
        ``-{level}`` alias runs at its own level on the base's ``--model``."""
        assert MODEL_INFO["cc/claude-opus-5-5"].thinking == "high"
        m = ClaudeCodeModel("cc/claude-opus-5-5")
        m.initialize("test")
        args = m._build_cli_args()
        assert _flag(args, "--model") == "claude-opus-5-5"
        assert _flag(args, "--effort") == "high"

        for level in ("low", "medium", "high", "xhigh", "max"):
            alias = model(f"cc/claude-opus-5-5-{level}")
            assert isinstance(alias, ClaudeCodeModel)
            assert MODEL_INFO[alias.model_name].alias_of == "cc/claude-opus-5-5"
            alias.initialize("test")
            args = alias._build_cli_args()
            assert _flag(args, "--model") == "claude-opus-5-5"
            assert _flag(args, "--effort") == level

    def test_reasoning_effort_outranks_the_catalog_and_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``model_config["reasoning_effort"]`` (what ``KISSAgent._turn_effort``
        sets for one turn) wins over the alias's level, is read at each
        launch, and the flag is not outranked by ``CLAUDE_CODE_EFFORT_LEVEL``:
        the variable is left out of the child environment, whichever way
        the subscription switch is, without touching the daemon's own."""
        from kiss.core.config import DEFAULT_CONFIG

        monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
        m = ClaudeCodeModel("cc/claude-opus-5-5-xhigh", model_config={"reasoning_effort": "low"})
        m.initialize("test")
        assert _flag(m._build_cli_args(), "--effort") == "low"
        del m.model_config["reasoning_effort"]
        assert _flag(m._build_cli_args(), "--effort") == "xhigh"

        monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", True)
        env = m._cli_env()
        assert env is not None and "CLAUDE_CODE_EFFORT_LEVEL" not in env
        assert m._use_subscription is True
        monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
        env = m._cli_env()
        assert env is not None and "CLAUDE_CODE_EFFORT_LEVEL" not in env
        assert m._use_subscription is False
        assert os.environ["CLAUDE_CODE_EFFORT_LEVEL"] == "max"


_CLAUDE_RECORDS_ITS_INPUT = """
    import json
    import os
    import sys

    with open(os.environ["KISS_TEST_INPUT_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read()}) + "\\n")
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1", "content": [{"type": "text", "text": "seen"}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "seen",
                      "usage": {"input_tokens": 10, "output_tokens": 5}}), flush=True)
"""


def _block(kind: str, attachment: Attachment) -> dict[str, object]:
    """Return the Anthropic content block that carries *attachment*."""
    source = {
        "type": "base64",
        "media_type": attachment.mime_type,
        "data": attachment.to_base64(),
    }
    return {"type": kind, "source": source}


def test_user_attachments_reach_the_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Images and PDFs the user attached are handed to ``claude`` as content blocks.

    ``initialize`` dropped them with a log line and the transcript ignored
    the ``attachments`` of later user messages, while Sorcar's prompt told
    the model the files were "included in this message as inline content
    that you can see directly".  The CLI takes them as one
    ``--input-format stream-json`` user message (the shape claude 2.1.308
    forwards to the API unchanged); a type it cannot take is named in the
    prompt as not shown, and a turn without attachments is plain text as
    before.  ``application/pdf; version=1.7`` is such a type: the block
    builder reads only the bare ``application/pdf`` as a PDF, so a label
    for it would name a file the model never gets.
    """
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_RECORDS_ITS_INPUT)
    log = tmp_path / "input.jsonl"
    monkeypatch.setenv("KISS_TEST_INPUT_LOG", str(log))
    png = Attachment(b"\x89PNG\r\n\x1a\npng-bytes", "image/png")
    pdf = Attachment(b"%PDF-1.4 pdf-bytes", "application/pdf")
    m = ClaudeCodeModel("cc/opus")
    unread = [
        Attachment(b"mp3-bytes", "audio/mpeg"),
        Attachment(pdf.data, "application/pdf; version=1.7"),
    ]
    m.initialize("describe these", [png, unread[0], pdf, unread[1]])

    assert m.generate()[0] == "seen"
    m.add_message_to_conversation("user", "and this one", attachments=[png])
    m.generate()
    plain = ClaudeCodeModel("cc/opus")
    plain.initialize("no files")
    plain.generate()

    first, second, third = (json.loads(line) for line in log.read_text().splitlines())
    shown = [
        {"type": "text", "text": "[Image #1]"},
        _block("image", png),
        {"type": "text", "text": "[Document #2]"},
        _block("document", pdf),
    ]
    task = (
        "describe these\n[Image #1]\n[Document #2]\n"
        "[2 attachment(s) of this message could not be shown to this model: "
        "audio/mpeg (9 bytes), application/pdf; version=1.7 (18 bytes)]"
    )
    assert _flag(first["argv"], "--input-format") == "stream-json"
    # One line is one turn: a second line would be a second API request.
    assert first["stdin"].endswith("\n") and first["stdin"].count("\n") == 1
    assert json.loads(first["stdin"]) == {
        "type": "user",
        "message": {"role": "user", "content": [*shown, {"type": "text", "text": task}]},
        "parent_tool_use_id": None,
    }
    assert json.loads(second["stdin"])["message"]["content"] == [
        *shown,
        {"type": "text", "text": "[Image #3]"},
        _block("image", png),
        {
            "type": "text",
            "text": f"[User]: {task}\n\n[Assistant]: seen\n\n[User]: and this one\n[Image #3]",
        },
    ]
    assert "--input-format" not in third["argv"]
    assert third["stdin"] == "no files"


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

    @pytest.fixture(autouse=True)
    def _bill_the_api_key_when_not_logged_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a Claude login, run on ``ANTHROPIC_API_KEY`` or skip."""
        from kiss.core.config import DEFAULT_CONFIG
        from kiss.core.models.claude_code_model import claude_auth_status

        if claude_auth_status()["loggedIn"]:
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("claude is not logged in and ANTHROPIC_API_KEY is not set")
        monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)

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
