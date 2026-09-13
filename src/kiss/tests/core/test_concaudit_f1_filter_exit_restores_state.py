# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression: a Stop raised during the filter's exit flush.

``_ToolCallFilteredStream.__exit__`` releases whatever the turn ended in
the middle of by calling the *original* token callback.  In production
that callback is ``JsonPrinter.token_callback``, which raises
``KeyboardInterrupt`` when the user has pressed Stop.  The exit path used
to flush first and restore the model's callbacks / clear
``_tool_bearing_turn`` afterwards, so a Stop that landed exactly at the
end of a tool-bearing turn left the reused adapter with:

* ``token_callback`` / ``thinking_callback`` still bound to the dead
  filter object (the next turn streams through stale filter state), and
* ``_tool_bearing_turn`` still ``True`` (Claude Code then ends an
  ordinary later turn early at any tool-call-shaped fragment).

The scenario is forced for real: a stand-in ``claude`` on ``PATH`` streams
text that ends in two backticks — a fragment the filter must hold back as
a possible split fence opener, so it is only released by the exit flush —
and the token callback is a plain function that raises
``KeyboardInterrupt`` when it receives that held fragment, exactly like
the production printer would after Stop.  No mocks, patches or doubles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_HELD_FRAGMENT = "``"
_VISIBLE_TEXT = "Working on it "

_CLAUDE_ENDS_ON_A_HELD_FRAGMENT = f"""
    import json
    import sys

    sys.stdin.read()
    for chunk in [{_VISIBLE_TEXT!r}, {_HELD_FRAGMENT!r}]:
        print(json.dumps({{"type": "content_block_delta",
                           "delta": {{"type": "text_delta",
                                      "text": chunk}}}}),
              flush=True)
    print(json.dumps({{"type": "result",
                       "result": {_VISIBLE_TEXT + _HELD_FRAGMENT!r},
                       "usage": {{"input_tokens": 10, "output_tokens": 5,
                                  "cache_read_input_tokens": 0}}}}), flush=True)
"""

_CODEX_ENDS_ON_A_HELD_FRAGMENT = f"""
    import json
    import sys

    sys.stdin.read()
    print(json.dumps({{"type": "item.completed",
                       "item": {{"type": "agent_message",
                                 "text": {_VISIBLE_TEXT + _HELD_FRAGMENT!r}}}}}),
          flush=True)
    print(json.dumps({{"type": "turn.completed",
                       "usage": {{"input_tokens": 10,
                                  "cached_input_tokens": 0,
                                  "output_tokens": 5}}}}), flush=True)
"""


class _StopsOnTheFlushedFragment:
    """A token sink that behaves like the production printer after Stop.

    ``JsonPrinter.token_callback`` checks the task's stop flag and raises
    ``KeyboardInterrupt`` once it is set.  Here the "flag" is set the
    moment the exit flush hands over the fragment the filter was holding,
    so the interrupt is raised from inside ``__exit__`` and nowhere else.
    """

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def __call__(self, token: str) -> None:
        """Record *token*; raise on the held fragment released by the flush.

        Args:
            token: The text the filter released.
        """
        self.tokens.append(token)
        if token == _HELD_FRAGMENT:
            raise KeyboardInterrupt


def _noop_thinking(is_start: bool) -> None:
    """Accept the thinking bracket without doing anything."""


def _list_files(command: str) -> str:
    """Pretend to run a shell command.

    Args:
        command: The command the model asked for.

    Returns:
        A fixed listing.
    """
    return f"ran {command}"


def _assert_turn_state_released(
    model: ClaudeCodeModel | CodexModel, sink: _StopsOnTheFlushedFragment,
) -> None:
    """Check the adapter is left exactly as it was before the tool turn.

    Args:
        model: The adapter whose tool turn was interrupted at exit.
        sink: The callback installed on *model* before the turn.
    """
    assert sink.tokens == [_VISIBLE_TEXT, _HELD_FRAGMENT]
    assert model.token_callback is sink, "token_callback still bound to the filter"
    assert model.thinking_callback is _noop_thinking, (
        "thinking_callback still bound to the filter"
    )
    assert model._tool_bearing_turn is False, "turn still marked tool-bearing"


def test_claude_code_exit_flush_interrupt_restores_callbacks_and_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop raised by the exit flush still restores the adapter's state."""
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_ENDS_ON_A_HELD_FRAGMENT)
    sink = _StopsOnTheFlushedFragment()
    model = ClaudeCodeModel(
        "cc/opus", token_callback=sink, thinking_callback=_noop_thinking,
    )
    model.initialize("list the files")
    with pytest.raises(KeyboardInterrupt):
        model.generate_and_process_with_tools({"Bash": _list_files})
    _assert_turn_state_released(model, sink)


def test_codex_exit_flush_interrupt_restores_callbacks_and_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other CLI adapter releases its turn state the same way."""
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_ENDS_ON_A_HELD_FRAGMENT)
    sink = _StopsOnTheFlushedFragment()
    model = CodexModel(
        "codex/default", token_callback=sink, thinking_callback=_noop_thinking,
    )
    model.initialize("list the files")
    with pytest.raises(KeyboardInterrupt):
        model.generate_and_process_with_tools({"Bash": _list_files})
    _assert_turn_state_released(model, sink)


def test_a_later_plain_turn_is_not_truncated_after_the_interrupted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reused adapter's next ordinary turn streams the whole answer.

    With ``_tool_bearing_turn`` left ``True`` the Claude Code adapter would
    stop this turn at the tool-call-shaped block and drop the text after
    it; the stale filter callback would also swallow that block.
    """
    tool_call = json.dumps(
        {"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}
    )
    tail = "That is what I would run."
    plain_turn = f"""
    import json
    import sys

    sys.stdin.read()
    text = {json.dumps("Quoting a call:\n" + tool_call + "\n" + tail)}
    for chunk in text.split(" "):
        print(json.dumps({{"type": "content_block_delta",
                           "delta": {{"type": "text_delta",
                                      "text": chunk + " "}}}}),
              flush=True)
    print(json.dumps({{"type": "result", "result": text,
                       "usage": {{"input_tokens": 10, "output_tokens": 5,
                                  "cache_read_input_tokens": 0}}}}), flush=True)
    """
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_ENDS_ON_A_HELD_FRAGMENT)
    sink = _StopsOnTheFlushedFragment()
    model = ClaudeCodeModel(
        "cc/opus", token_callback=sink, thinking_callback=_noop_thinking,
    )
    model.initialize("list the files")
    with pytest.raises(KeyboardInterrupt):
        model.generate_and_process_with_tools({"Bash": _list_files})

    install_cli(tmp_path, monkeypatch, "claude", plain_turn)
    streamed: list[str] = []
    model.reset_conversation()
    model.rebind_callbacks(streamed.append, _noop_thinking)
    model.initialize("what would you run?")
    content, _response = model.generate()
    assert tail in content
    assert tail in "".join(streamed)
    assert tool_call in "".join(streamed)
