# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: ``cc/*`` and ``codex/*`` run the KISS tool loop.

The CLI-backed models are driven as plain language models, not as the
coding agents their CLIs are: every KISSAgent step runs the CLI once with
its native tools switched off and KISS's system prompt — including the
text-based ``tool_calls`` protocol describing KISS's tools — installed as
the CLI's system prompt:

- ``claude``: ``--tools ""``, ``--safe-mode``, ``--strict-mcp-config``
  and ``--system-prompt-file <file holding the KISS prompt>``;
- ``codex``: the lock-down flags of ``CODEX_LOCKDOWN_ARGS``,
  ``-c model_instructions_file=<file holding the KISS prompt>`` and,
  within an agent step, the ``-c mcp_servers.kiss.*`` overrides that
  serve the KISS tools (``finish`` excepted) over MCP — the text prompt
  then describes ``finish`` alone (the MCP calls themselves are covered
  by ``test_codex_mcp_bridge.py``);
- the prompt on stdin is the task (first step) or the
  ``[User]/[Assistant]/[Tool Result]`` transcript (later steps), never the
  system prompt;
- a ``tool_calls`` block in the CLI's answer is executed by KISS, whose
  ``finish`` ends the run.

The tests run real fake ``claude`` / ``codex`` executables placed on PATH
that record their argv, stdin and the instructions file, and emit valid
event streams — no mocks.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.codex_model import (
    CODEX_INSTRUCTIONS_PREAMBLE,
    CODEX_LOCKDOWN_ARGS,
    CODEX_MCP_TOOLS_NOTE,
)
from kiss.core.models.model_info import model as model_factory
from kiss.core.utils import finish as structured_finish
from kiss.tests.conftest import install_cli_script

TASK = "Echo MARKER-42 through the tool, then stop."
SYSTEM_PROMPT = "Always answer in exactly three words."
FINAL_TEXT = "ECHOED done."
TOOL_CALL = json.dumps({"tool_calls": [{"name": "_dummy_tool", "arguments": {"x": "MARKER-42"}}]})
FINISH_CALL = json.dumps(
    {
        "tool_calls": [
            {
                "name": "finish",
                "arguments": {
                    "success": True,
                    "is_continue": False,
                    "summary_in_html": FINAL_TEXT,
                },
            }
        ]
    }
)


def _claude_events(text: str) -> list[dict]:
    """One ``claude --print`` stream that answers *text*.

    The per-message usage events come first, as in the real stream: a
    turn KISS cuts short at a ``tool_calls`` block is billed from them.
    """
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 0}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 7},
        },
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "result", "result": text, "usage": {"input_tokens": 12, "output_tokens": 7}},
    ]


def _codex_events(text: str) -> list[dict]:
    """One ``codex exec --json`` stream that answers *text*."""
    return [
        {"type": "thread.started", "thread_id": "t-1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 7}},
    ]


# The fake answers call N with turns[N] (the last turn repeats).  It
# records argv, the stdin prompt, the cwd, and the content of the system
# prompt file named in argv (claude's ``--system-prompt-file``, codex's
# ``model_instructions_file``), which the model deletes once the turn is
# over.
_FAKE_CLI_TEMPLATE = """#!/usr/bin/env python3
import json, os, re, sys, pathlib, time
record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("call-*.json")))
prompt = sys.stdin.read()
instructions = None
for i, arg in enumerate(sys.argv):
    m = re.match(r'model_instructions_file=(.*)$', arg)
    if m:
        instructions = pathlib.Path(json.loads(m.group(1))).read_text()
    elif arg == "--system-prompt-file":
        instructions = pathlib.Path(sys.argv[i + 1]).read_text()
(record_dir / f"call-{{n}}.json").write_text(json.dumps(
    {{"argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd(),
      "instructions": instructions}}))
turns = {turns!r}
for event in turns[min(n, len(turns) - 1)]:
    print(json.dumps(event), flush=True)
    time.sleep({delay!r})
time.sleep({final_sleep!r})
"""


def _install_fake_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    turns: list[list[dict]],
    delay: float = 0.0,
    final_sleep: float = 0.0,
) -> Path:
    """Install a fake CLI executable *name* on PATH and return its record dir.

    Args:
        tmp_path: The test's temporary directory.
        monkeypatch: Fixture used to prepend the fake's dir to PATH.
        name: Executable name to fake (``claude`` or ``codex``).
        turns: Per-call event lists; call N prints ``turns[N]`` (the last
            list repeats for further calls).
        delay: Seconds the fake sleeps after printing each event.
        final_sleep: Seconds the fake stays silent after the last event
            (to exercise stall-timeout behavior).

    Returns:
        The directory the fake writes its call records into.
    """
    bin_dir = tmp_path / "bin"
    record_dir = tmp_path / "records"
    bin_dir.mkdir(exist_ok=True)
    record_dir.mkdir(exist_ok=True)
    install_cli_script(
        bin_dir / name,
        _FAKE_CLI_TEMPLATE.format(
            record_dir=str(record_dir), turns=turns, delay=delay, final_sleep=final_sleep
        ),
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record_dir


def _read_calls(record_dir: Path) -> list[dict]:
    """Return the recorded CLI invocations, in call order."""
    return [json.loads(p.read_text()) for p in sorted(record_dir.glob("call-*.json"))]


def _dummy_tool(x: str) -> str:
    """Echo *x* back, upper-cased: a KISS tool the CLI must call through KISS.

    Args:
        x: Any string.

    Returns:
        ``x`` upper-cased with ``ECHOED:`` in front.
    """
    return f"ECHOED:{x.upper()}"


def _system_text(call: dict, cli_name: str) -> str:
    """The system prompt a recorded CLI call received."""
    argv = call["argv"]
    instructions = call["instructions"]
    assert instructions is not None, f"{cli_name} must get a system prompt file"
    if cli_name == "claude":
        return str(instructions)
    assert instructions.startswith(CODEX_INSTRUCTIONS_PREAMBLE)
    instructions = instructions[len(CODEX_INSTRUCTIONS_PREAMBLE) :]
    if any(arg.startswith("mcp_servers.kiss.") for arg in argv):
        # Within an agent step the KISS tools are served over MCP and the
        # preamble says so.
        assert instructions.startswith(CODEX_MCP_TOOLS_NOTE)
        instructions = instructions[len(CODEX_MCP_TOOLS_NOTE) :]
    return str(instructions)


@pytest.mark.parametrize(
    "model_name,cli_name,make_events",
    [
        ("cc/sonnet", "claude", _claude_events),
        ("codex/gpt-5.5", "codex", _codex_events),
    ],
)
def test_agentic_run_drives_the_cli_through_kiss_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    cli_name: str,
    make_events,
) -> None:
    """Step 1: the CLI gets the task and the KISS tools prompt and answers
    with a ``tool_calls`` block; KISS runs the tool.  Step 2: the CLI gets
    the transcript with the tool result and calls ``finish``.

    Codex is also handed ``_dummy_tool`` as the ``kiss`` MCP server (so
    the text prompt describes ``finish`` alone); a ``tool_calls`` block
    naming it is still executed by KISS, as here."""
    record_dir = _install_fake_cli(
        tmp_path, monkeypatch, cli_name, [make_events(TOOL_CALL), make_events(FINISH_CALL)]
    )
    agent = KISSAgent("kiss-driven cli e2e")
    result = agent.run(
        model_name=model_name,
        prompt_template=TASK,
        system_prompt=SYSTEM_PROMPT,
        tools=[structured_finish, _dummy_tool],
        is_agentic=True,
        verbose=False,
    )

    calls = _read_calls(record_dir)
    assert len(calls) == 2, "one CLI invocation per KISS step"
    first, second = calls
    # The system prompt (with the KISS tools) is the CLI's system prompt,
    # never part of the stdin prompt.
    for call in calls:
        system_text = _system_text(call, cli_name)
        assert system_text.startswith(SYSTEM_PROMPT)
        assert "tool_calls" in system_text and "finish" in system_text
        over_mcp = [a for a in call["argv"] if a.startswith("mcp_servers.kiss.")]
        if cli_name == "codex":
            assert "_dummy_tool" not in system_text
            assert any(a.startswith("mcp_servers.kiss.command=") for a in over_mcp)
            assert 'mcp_servers.kiss.default_tools_approval_mode="approve"' in over_mcp
        else:
            assert "_dummy_tool" in system_text and not over_mcp
        assert "--append-system-prompt" not in call["argv"]
    assert first["prompt"] == TASK  # no tool description, no system text
    assert "[Assistant]:" in second["prompt"]  # the transcript, tool call included
    assert "[Tool Result]" in second["prompt"]
    assert "ECHOED:MARKER-42" in second["prompt"]

    payload = yaml.safe_load(result)
    assert payload["success"] is True
    assert payload["is_continue"] is False
    assert FINAL_TEXT in payload["summary"]
    assert agent.total_tokens_used == 2 * 19


def test_claude_runs_as_a_bare_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``claude`` gets no native tools, no user customizations and no
    permission bypass: KISS's tools are the only ones."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "claude", [_claude_events("ok")])
    KISSAgent("bare claude").run(
        model_name="cc/sonnet", prompt_template=TASK, is_agentic=False, verbose=False
    )
    argv = _read_calls(record_dir)[0]["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert "--safe-mode" in argv
    assert "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv
    assert "--dangerously-skip-permissions" not in argv
    assert "--bare" not in argv


def test_codex_runs_as_a_bare_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``codex exec`` gets the lock-down flags, no sandbox bypass, and
    ``-m`` only for a named model."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "codex", [_codex_events("ok")])
    KISSAgent("bare codex").run(
        model_name="codex/default",
        prompt_template=TASK,
        system_prompt=SYSTEM_PROMPT,
        is_agentic=False,
        verbose=False,
    )
    call = _read_calls(record_dir)[0]
    argv = call["argv"]
    assert argv[:3] == ["exec", "--json", "--skip-git-repo-check"]
    assert list(CODEX_LOCKDOWN_ARGS) == argv[3 : 3 + len(CODEX_LOCKDOWN_ARGS)]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "-m" not in argv
    assert call["instructions"] == CODEX_INSTRUCTIONS_PREAMBLE + SYSTEM_PROMPT
    assert call["prompt"] == TASK
    # The instructions file is gone once the turn is over.
    file_arg = next(a for a in argv if a.startswith("model_instructions_file="))
    assert not Path(json.loads(file_arg.partition("=")[2])).exists()


def test_codex_without_system_prompt_still_gets_the_preamble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even without a KISS system prompt the built-in Codex instructions
    are replaced, so the model never acts as the coding agent."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "codex", [_codex_events("ok")])
    KISSAgent("codex no system prompt").run(
        model_name="codex/gpt-5.5", prompt_template=TASK, is_agentic=False, verbose=False
    )
    call = _read_calls(record_dir)[0]
    assert call["instructions"] == CODEX_INSTRUCTIONS_PREAMBLE
    assert call["argv"][-2:] == ["-m", "gpt-5.5"]


def test_work_dir_config_sets_cli_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_config["work_dir"] becomes the CLI subprocess's cwd."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "claude", [_claude_events("ok")])
    work_dir = tmp_path / "task-worktree"
    work_dir.mkdir()
    KISSAgent("cli cwd").run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=False,
        model_config={"work_dir": str(work_dir)},
        verbose=False,
    )
    calls = _read_calls(record_dir)
    assert Path(calls[0]["cwd"]).resolve() == work_dir.resolve()


def test_stall_timeout_bounds_silence_not_total_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI steadily producing output outlives the configured timeout."""
    events: list[dict] = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"part-{i} "}]}}
        for i in range(5)
    ] + [{"type": "result", "result": "steady", "usage": {"input_tokens": 3, "output_tokens": 2}}]
    # Total run time (6 events x 0.4 s) exceeds the 1 s timeout, but each
    # inter-event gap is well under it: an absolute deadline would kill
    # the run; the inactivity deadline must not.
    _install_fake_cli(tmp_path, monkeypatch, "claude", [events], delay=0.4)
    result = KISSAgent("steady output").run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=False,
        model_config={"timeout": 1},
        verbose=False,
    )
    assert "part-4" in result


def test_silent_cli_times_out_and_partial_usage_is_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled turn raises TimeoutError but keeps its observed usage."""
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "working..."}]}},
        {"type": "message_delta", "usage": {"input_tokens": 100, "output_tokens": 50}},
    ]
    _install_fake_cli(tmp_path, monkeypatch, "claude", [events], final_sleep=30.0)
    agent = KISSAgent("stall billing")
    # The inactivity clock starts before the fake CLI's Python interpreter is
    # up; 5 s leaves startup room and is still far below the fake's 30 s
    # silence.
    with pytest.raises(TimeoutError):
        agent.run(
            model_name="cc/sonnet",
            prompt_template=TASK,
            is_agentic=False,
            model_config={"timeout": 5},
            verbose=False,
        )
    assert agent.total_tokens_used == 150, "usage observed before the stall must still be billed"


def _serve_one_switch_turn() -> "HTTPServer":
    """Start a local chat-completions server whose first (and only) reply
    calls the ``switch_model`` tool.  Returns the running server."""
    body = {
        "id": "chatcmpl-switch",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "switch_model", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_mid_run_model_switch_keeps_the_tool_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching to a cc/* model mid-run (Sorcar's ``set_model``) continues
    the turn-by-turn loop: the CLI gets the transcript so far plus the
    KISS tools prompt, and its ``finish`` call ends the run."""
    record_dir = _install_fake_cli(
        tmp_path, monkeypatch, "claude", [_claude_events(FINISH_CALL)]
    )
    server = _serve_one_switch_turn()
    agent = KISSAgent("mid-run switch e2e")

    def switch_model() -> str:
        """Swap the live model to a Claude Code CLI model.

        Returns:
            A confirmation string.
        """
        old_model = agent.model
        new_model = model_factory("cc/sonnet", model_config={})
        new_model.initialize("")
        new_model.conversation = old_model.conversation
        agent.model = new_model
        agent.model_name = "cc/sonnet"
        return "switched to cc/sonnet"

    try:
        result = agent.run(
            model_name="gpt-4o-mini",
            prompt_template=TASK,
            system_prompt=SYSTEM_PROMPT,
            tools=[structured_finish, switch_model],
            max_steps=6,
            max_budget=5.0,
            verbose=False,
            model_config={
                "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                "api_key": "sk-test",
            },
        )
    finally:
        server.shutdown()

    calls = _read_calls(record_dir)
    assert len(calls) == 1
    prompt = calls[0]["prompt"]
    assert "[User]:" in prompt
    assert "switched to cc/sonnet" in prompt
    assert "switch_model" in _system_text(calls[0], "claude")
    payload = yaml.safe_load(result)
    assert payload["success"] is True
    assert FINAL_TEXT in payload["summary"]
