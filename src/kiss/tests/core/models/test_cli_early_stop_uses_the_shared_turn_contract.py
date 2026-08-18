# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: one mechanism marks a tool-bearing CLI turn.

Text-based tool calling makes a CLI turn special in two ways at once: the
raw ``{"tool_calls": ...}`` JSON must be kept out of the chat panel, and a
reasoning model that keeps writing after that block — inventing the tool
output it has not received yet — must be cut off at the block.  Both CLI
adapters already share the filter that solves the first half, so the second
half belongs to the same mechanism instead of a widened ``generate()``
signature that only one adapter has and only its own tool path can reach.

With the capability behind the shared contract, a caller that holds a plain
:class:`~kiss.core.models.model.Model` gets it: entering the tool-turn
filter and calling ``generate()`` — the abstract signature, no extra
arguments — stops the turn at the tool call.

Every test runs a REAL stand-in ``claude`` / ``codex`` executable installed
on ``PATH``: real subprocesses, real streams, no mocks, patches or doubles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model import Model, _ToolCallFilteredStream
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_TOOL_CALL_JSON = (
    '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
)
_HALLUCINATION = "The command printed three files, so we are done."

_CLAUDE_KEEPS_TALKING_AFTER_THE_TOOL_CALL = f"""
    import json
    import sys

    sys.stdin.read()
    text = 'Listing now.\\n{_TOOL_CALL_JSON}\\n{_HALLUCINATION}'
    for chunk in text.split(" "):
        print(json.dumps({{"type": "content_block_delta",
                           "delta": {{"type": "text_delta",
                                      "text": chunk + " "}}}}),
              flush=True)
    print(json.dumps({{"type": "result", "result": text,
                       "usage": {{"input_tokens": 10, "output_tokens": 5,
                                  "cache_read_input_tokens": 0}}}}), flush=True)
"""

_CODEX_KEEPS_TALKING_AFTER_THE_TOOL_CALL = f"""
    import json
    import sys

    sys.stdin.read()
    text = 'Listing now.\\n{_TOOL_CALL_JSON}\\n{_HALLUCINATION}'
    print(json.dumps({{"type": "item.completed",
                       "item": {{"type": "agent_message", "text": text}}}}),
          flush=True)
    print(json.dumps({{"type": "turn.completed",
                       "usage": {{"input_tokens": 10,
                                  "cached_input_tokens": 0,
                                  "output_tokens": 5}}}}), flush=True)
"""


def _list_files(command: str) -> str:
    """Pretend to run a shell command.

    Args:
        command: The command the model asked for.

    Returns:
        A fixed listing.
    """
    return f"ran {command}"


class TestTheSharedTurnContractCarriesEarlyStop:
    """``generate()`` needs no extra argument to stop at a tool call."""

    def test_claude_code_stops_inside_the_tool_turn_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The abstract ``generate()`` call stops at the tool-call block."""
        install_cli(
            tmp_path, monkeypatch, "claude",
            _CLAUDE_KEEPS_TALKING_AFTER_THE_TOOL_CALL,
        )
        model = ClaudeCodeModel("cc/opus")
        model.initialize("list the files")
        # Typed as the base class on purpose: the capability must be
        # reachable without knowing which adapter this is.
        through_the_contract: Model = model
        with _ToolCallFilteredStream(model):
            content, _response = through_the_contract.generate()
        assert _TOOL_CALL_JSON in content
        assert _HALLUCINATION not in content

    def test_claude_code_keeps_the_whole_turn_outside_the_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain turn is not truncated: there is no tool call to stop at."""
        install_cli(
            tmp_path, monkeypatch, "claude",
            _CLAUDE_KEEPS_TALKING_AFTER_THE_TOOL_CALL,
        )
        model = ClaudeCodeModel("cc/opus")
        model.initialize("list the files")
        content, _response = model.generate()
        assert _HALLUCINATION in content

    def test_codex_accepts_the_same_call_inside_the_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other CLI adapter answers the identical abstract call."""
        install_cli(
            tmp_path, monkeypatch, "codex",
            _CODEX_KEEPS_TALKING_AFTER_THE_TOOL_CALL,
        )
        model = CodexModel("codex/default")
        model.initialize("list the files")
        through_the_contract: Model = model
        with _ToolCallFilteredStream(model):
            content, _response = through_the_contract.generate()
        assert _TOOL_CALL_JSON in content


class TestTheToolPathStillBehavesForBothAdapters:
    """Unifying the mechanism must not change what the tool path returns."""

    def test_claude_code_parses_the_call_and_hides_the_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call is parsed, and the user never sees its JSON."""
        install_cli(
            tmp_path, monkeypatch, "claude",
            _CLAUDE_KEEPS_TALKING_AFTER_THE_TOOL_CALL,
        )
        tokens: list[str] = []
        model = ClaudeCodeModel("cc/opus", token_callback=tokens.append)
        model.initialize("list the files")
        calls, content, _response = model.generate_and_process_with_tools(
            {"Bash": _list_files}
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert calls[0]["arguments"] == {"command": "ls"}
        assert "tool_calls" not in "".join(tokens)
        assert _HALLUCINATION not in content

    def test_codex_parses_the_call_and_hides_the_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex's tool path is unchanged by the shared mechanism."""
        install_cli(
            tmp_path, monkeypatch, "codex",
            _CODEX_KEEPS_TALKING_AFTER_THE_TOOL_CALL,
        )
        tokens: list[str] = []
        model = CodexModel("codex/default", token_callback=tokens.append)
        model.initialize("list the files")
        calls, _content, _response = model.generate_and_process_with_tools(
            {"Bash": _list_files}
        )
        assert [call["name"] for call in calls] == ["Bash"]
        assert "tool_calls" not in "".join(tokens)

    def test_the_filter_leaves_no_turn_state_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain turn after a tool turn is not truncated by leftover state."""
        install_cli(
            tmp_path, monkeypatch, "claude",
            _CLAUDE_KEEPS_TALKING_AFTER_THE_TOOL_CALL,
        )
        model = ClaudeCodeModel("cc/opus")
        model.initialize("list the files")
        model.generate_and_process_with_tools({"Bash": _list_files})
        model.reset_conversation()
        model.initialize("list the files")
        content, _response = model.generate()
        assert _HALLUCINATION in content
