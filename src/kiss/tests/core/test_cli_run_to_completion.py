# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for run-to-completion CLI models in KISSAgent.

``cc/*`` and ``codex/*`` models are full coding agents, so an agentic
:class:`~kiss.core.kiss_agent.KISSAgent` run hands them the WHOLE task in
one CLI invocation instead of driving a turn-by-turn KISS tool loop:

- exactly one CLI subprocess is spawned for the whole run;
- the system prompt is appended to the task after
  ``CLI_SYSTEM_PROMPT_HEADER`` (``"\\n\\n# You new system prompt
  follows:\\n"``) — no ``--system-prompt`` flag, no ``[System]:`` prefix;
- KISS tool descriptions are never injected into the prompt;
- the CLI's final message is returned wrapped in the registered ``finish``
  tool's output contract (YAML for the structured ``kiss.core.utils.finish``,
  plain text for the built-in ``finish``).

The tests run real fake ``claude`` / ``codex`` executables placed on PATH
that record their argv and stdin and emit valid event streams — no mocks.
"""

import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import CLI_SYSTEM_PROMPT_HEADER
from kiss.core.models.model_info import model as model_factory
from kiss.core.utils import finish as structured_finish

TASK = "Say COMPLETED-TASK and stop."
SYSTEM_PROMPT = "Always answer in exactly three words."
FINAL_TEXT = "COMPLETED-TASK done."

# One stream-json transcript of a full agentic Claude Code run: an
# assistant message followed by the terminal ``result`` event.
_CLAUDE_EVENTS: list[dict] = [
    {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": FINAL_TEXT}]},
    },
    {
        "type": "result",
        "result": FINAL_TEXT,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    },
]

# One JSONL transcript of a full agentic Codex run.
_CODEX_EVENTS: list[dict] = [
    {"type": "thread.started", "thread_id": "t-1"},
    {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": FINAL_TEXT},
    },
    {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 7}},
]

_FAKE_CLI_TEMPLATE = """#!/usr/bin/env python3
import json, os, sys, pathlib, time
record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("call-*.json")))
prompt = sys.stdin.read()
(record_dir / f"call-{{n}}.json").write_text(
    json.dumps({{"argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd()}}))
for event in {events!r}:
    print(json.dumps(event), flush=True)
    time.sleep({delay!r})
time.sleep({final_sleep!r})
"""


def _install_fake_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    events: list[dict],
    delay: float = 0.0,
    final_sleep: float = 0.0,
) -> Path:
    """Install a fake CLI executable *name* on PATH and return its record dir.

    The fake records each invocation's argv, stdin prompt, and cwd to
    ``record_dir/call-N.json`` and prints *events* as JSON lines.

    Args:
        tmp_path: The test's temporary directory.
        monkeypatch: Fixture used to prepend the fake's dir to PATH.
        name: Executable name to fake (``claude`` or ``codex``).
        events: Event dicts the fake prints, one JSON object per line.
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
    cli = bin_dir / name
    cli.write_text(
        _FAKE_CLI_TEMPLATE.format(
            record_dir=str(record_dir),
            events=events,
            delay=delay,
            final_sleep=final_sleep,
        )
    )
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record_dir


def _read_calls(record_dir: Path) -> list[dict]:
    """Return the recorded CLI invocations, in call order."""
    return [
        json.loads(p.read_text())
        for p in sorted(record_dir.glob("call-*.json"))
    ]


def _dummy_tool(x: str) -> str:
    """Echo *x* back; a KISS tool that must never reach the CLI prompt.

    Args:
        x: Any string.

    Returns:
        The same string.
    """
    return x


@pytest.mark.parametrize(
    "model_name,cli_name,events",
    [
        ("cc/sonnet", "claude", _CLAUDE_EVENTS),
        ("codex/gpt-5.5", "codex", _CODEX_EVENTS),
    ],
)
def test_agentic_run_is_single_shot_with_appended_system_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    cli_name: str,
    events: list[dict],
) -> None:
    """One CLI call runs the whole task; system prompt rides in the prompt."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, cli_name, events)
    agent = KISSAgent("run-to-completion e2e")
    result = agent.run(
        model_name=model_name,
        prompt_template=TASK,
        system_prompt=SYSTEM_PROMPT,
        tools=[structured_finish, _dummy_tool],
        is_agentic=True,
        verbose=False,
    )

    calls = _read_calls(record_dir)
    assert len(calls) == 1, "the whole task must run in exactly one CLI invocation"
    prompt = calls[0]["prompt"]
    assert prompt == TASK + CLI_SYSTEM_PROMPT_HEADER + SYSTEM_PROMPT
    assert CLI_SYSTEM_PROMPT_HEADER == "\n\n# You new system prompt follows:\n"
    # KISS tools are never described to the CLI.
    assert "_dummy_tool" not in prompt
    assert "tool_calls" not in prompt
    # The system prompt is not passed out-of-band.
    assert "--system-prompt" not in calls[0]["argv"]
    assert not any("[System]" in a for a in calls[0]["argv"])

    payload = yaml.safe_load(result)
    assert payload["success"] is True
    assert payload["is_continue"] is False
    assert FINAL_TEXT in payload["summary"]
    assert agent.total_tokens_used == 19


def test_builtin_finish_contract_returns_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a structured finish, the CLI's final text is the result."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EVENTS)
    agent = KISSAgent("run-to-completion plain finish")
    result = agent.run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=True,
        verbose=False,
    )
    assert result == FINAL_TEXT
    assert len(_read_calls(record_dir)) == 1


def test_no_system_prompt_sends_bare_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a system prompt, the prompt is exactly the task text."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "codex", _CODEX_EVENTS)
    agent = KISSAgent("run-to-completion bare task")
    agent.run(
        model_name="codex/default",
        prompt_template=TASK,
        is_agentic=True,
        verbose=False,
    )
    calls = _read_calls(record_dir)
    assert calls[0]["prompt"] == TASK
    assert "-m" not in calls[0]["argv"]


def test_default_stall_timeout_is_raised_for_full_task_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-task run defaults to the 3600 s stall timeout, overridably,
    without mutating the caller's model_config dict."""
    _install_fake_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EVENTS)
    agent = KISSAgent("run-to-completion timeout default")
    caller_config: dict = {}
    agent.run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=True,
        model_config=caller_config,
        verbose=False,
    )
    assert agent.model.model_config["timeout"] == 3600
    assert caller_config == {}, "the caller's config dict must not be mutated"

    agent2 = KISSAgent("run-to-completion timeout override")
    agent2.run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=True,
        model_config={"timeout": 42},
        verbose=False,
    )
    assert agent2.model.model_config["timeout"] == 42


def test_work_dir_config_sets_cli_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_config["work_dir"] becomes the CLI subprocess's cwd."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EVENTS)
    work_dir = tmp_path / "task-worktree"
    work_dir.mkdir()
    agent = KISSAgent("run-to-completion cwd")
    agent.run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=True,
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
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": f"part-{i} "}]},
        }
        for i in range(5)
    ] + [
        {
            "type": "result",
            "result": "steady",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    ]
    # Total run time (6 events x 0.4 s) exceeds the 1 s timeout, but each
    # inter-event gap is well under it: an absolute deadline would kill
    # the run; the inactivity deadline must not.
    _install_fake_cli(tmp_path, monkeypatch, "claude", events, delay=0.4)
    agent = KISSAgent("run-to-completion steady output")
    result = agent.run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        is_agentic=True,
        model_config={"timeout": 1},
        verbose=False,
    )
    assert "part-4" in result


def test_silent_cli_times_out_and_partial_usage_is_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled run raises TimeoutError but keeps its observed usage."""
    events = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "working..."}]},
        },
        {
            "type": "message_delta",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    ]
    _install_fake_cli(
        tmp_path, monkeypatch, "claude", events, final_sleep=30.0
    )
    agent = KISSAgent("run-to-completion stall billing")
    with pytest.raises(TimeoutError):
        agent.run(
            model_name="cc/sonnet",
            prompt_template=TASK,
            is_agentic=True,
            model_config={"timeout": 1},
            verbose=False,
        )
    assert agent.total_tokens_used == 150, (
        "usage observed before the stall must still be billed"
    )
    # cc/* models are subscription-priced at $0/token in MODEL_INFO, so
    # only the token accounting (not budget_used) can be asserted here.


def test_non_agentic_run_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-agentic runs still return the raw generated text."""
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EVENTS)
    agent = KISSAgent("non-agentic cc")
    result = agent.run(
        model_name="cc/sonnet",
        prompt_template=TASK,
        system_prompt=SYSTEM_PROMPT,
        is_agentic=False,
        verbose=False,
    )
    assert result == FINAL_TEXT
    calls = _read_calls(record_dir)
    assert calls[0]["prompt"] == TASK + CLI_SYSTEM_PROMPT_HEADER + SYSTEM_PROMPT


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
                            "function": {
                                "name": "switch_model",
                                "arguments": "{}",
                            },
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


def test_mid_run_model_switch_hands_task_to_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching to a cc/* model mid-run ends the turn-by-turn loop.

    Mirrors Sorcar's ``set_model``: a tool swaps the live model to a CLI
    agent; the very next loop iteration must hand the accumulated
    conversation to the CLI in one shot instead of resuming KISS tool
    prompting.
    """
    record_dir = _install_fake_cli(tmp_path, monkeypatch, "claude", _CLAUDE_EVENTS)
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
    assert len(calls) == 1, "after the switch the CLI must be invoked exactly once"
    prompt = calls[0]["prompt"]
    # The accumulated conversation rides in the prompt as a transcript.
    assert "[User]:" in prompt
    assert "switched to cc/sonnet" in prompt
    payload = yaml.safe_load(result)
    assert payload["success"] is True
    assert FINAL_TEXT in payload["summary"]
