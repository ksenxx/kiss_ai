# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests for the KISS tools → Codex MCP bridge.

:class:`~kiss.core.models.codex_mcp_bridge.ToolBridge` serves KISS tools
from the agent process and ``python -m kiss.core.models.codex_mcp_bridge``
is the stdio proxy a Codex CLI spawns as the ``kiss`` MCP server.  A real
MCP client (``mcp.client.stdio``) spawns the proxy exactly as Codex would
and lists / calls the tools through it — real processes, no mocks.

The end-to-end half runs a ``KISSAgent`` on a ``codex/*`` model whose
``codex`` executable is a fake that behaves as Codex does with an MCP
server configured: it reads the ``-c mcp_servers.kiss.*`` overrides from
its argv, spawns the proxy, calls a KISS tool through it during the turn
and then prints the ``codex exec --json`` event stream.
"""

import asyncio
import json
import os
import socket
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kiss.core import config as config_module
from kiss.core import stop_signal
from kiss.core.base import collecting_trajectories
from kiss.core.kiss_agent import NATIVE_TOOLS_CONTINUE_MSG, KISSAgent
from kiss.core.kiss_error import BudgetExceededError, ContextWindowExceededError, KISSError
from kiss.core.llm_reply import LLMReply
from kiss.core.models.codex_mcp_bridge import (
    ServedCall,
    ToolBridge,
    _read_request_line,
    toml_value,
)
from kiss.core.models.codex_model import (
    CODEX_INSTRUCTIONS_PREAMBLE,
    CODEX_MCP_TOOL_TIMEOUT_SECONDS,
    CODEX_MCP_TOOLS_NOTE,
)
from kiss.core.models.model import encode_binary_attachment
from kiss.core.models.model_info import _NATIVE_PROVIDERS, OPENAI_COMPATIBLE_PROVIDERS
from kiss.core.printer import Printer
from kiss.core.tool_verdict import Verdict, end_run, refuse
from kiss.core.utils import finish as structured_finish
from kiss.tests.conftest import install_cli_script

ADD_SCHEMA: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}]


def _proxy_session(bridge: ToolBridge) -> Any:
    """Return the stdio client context for the proxy of *bridge*, spawned as Codex would."""
    return stdio_client(StdioServerParameters(
        command=sys.executable,
        args=["-m", "kiss.core.models.codex_mcp_bridge", str(bridge.port)],
        env=bridge.codex_env(),
    ))


def test_proxy_lists_and_calls_tools_through_the_bridge() -> None:
    """The proxy forwards list_tools / call_tool to the in-process executor."""
    calls: list[dict[str, Any]] = []

    def execute(call: dict[str, Any]) -> tuple[str, str]:
        calls.append(call)
        if call["arguments"]["a"] == 13:
            raise RuntimeError("unlucky")
        return call["name"], str(call["arguments"]["a"] + call["arguments"]["b"])

    async def drive() -> list[Any]:
        with ToolBridge(ADD_SCHEMA, execute) as bridge:
            async with _proxy_session(bridge) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    ok = await session.call_tool("add", {"a": 2, "b": 3})
                    failed = await session.call_tool("add", {"a": 13, "b": 0})
                    invalid = await session.call_tool("add", {"a": 1})
            served = bridge.take_served()
        return [tools, ok, failed, invalid, served]

    tools, ok, failed, invalid, served = asyncio.run(drive())
    assert [(t.name, t.description, t.inputSchema) for t in tools] == [
        ("add", "Add two integers.", ADD_SCHEMA[0]["function"]["parameters"])
    ]
    assert not ok.isError and ok.content[0].text == "5"
    assert failed.isError and "RuntimeError: unlucky" in failed.content[0].text
    # Argument validation happens in the proxy: the executor never sees the bad call.
    assert invalid.isError
    assert calls == [
        {"name": "add", "arguments": {"a": 2, "b": 3}},
        {"name": "add", "arguments": {"a": 13, "b": 0}},
    ]
    # Every answered call is on record, the failed one with its error text.
    assert served == [
        ServedCall("add", {"a": 2, "b": 3}, "5"),
        ServedCall("add", {"a": 13, "b": 0}, "RuntimeError: unlucky"),
    ]


def test_bridge_port_is_closed_after_the_block() -> None:
    """Leaving the ``with`` block stops the listener; nothing lingers on the port."""
    with ToolBridge(ADD_SCHEMA, lambda call: (call["name"], "")) as bridge:
        port = bridge.port
        with socket.create_connection(("127.0.0.1", port)):
            pass
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def _send_then_end(sock: socket.socket, data: bytes) -> None:
    """Send *data*, then shut the writing side so the reader sees end of stream."""
    sock.sendall(data)
    sock.shutdown(socket.SHUT_WR)


@pytest.mark.parametrize(
    ("cap", "size", "accepted"),
    [
        (1024, 1000, True),  # below the cap
        (1024, 1024, True),  # a newline-terminated line exactly at the cap
        (1024, 1131, False),  # over the cap within one receive
        (100_000, 110_000, False),  # over the cap in the last of several receives
    ],
)
def test_request_line_cap_holds_whatever_the_last_receive_brings(
    monkeypatch: pytest.MonkeyPatch, cap: int, size: int, accepted: bool
) -> None:
    """A request longer than the cap is dropped even when one receive crosses the cap (audit C2/F3).

    The cap used to be checked only before each receive, so a line that
    crossed it inside the final chunk came back whole, up to 64 KiB over.
    """
    monkeypatch.setattr("kiss.core.models.codex_mcp_bridge._MAX_REQUEST_BYTES", cap)
    body = json.dumps({"op": "tools", "padding": "x" * size}).encode()
    payload = body[: size - 1] + b"\n"
    assert len(payload) == size
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(2)
        # From a thread: a payload larger than the socket's buffer blocks the sender.
        sender = threading.Thread(target=_send_then_end, args=(writer, payload))
        sender.start()
        line = _read_request_line(reader, time.monotonic() + 2)
        assert reader.gettimeout() == 2  # the reply's write keeps the socket's own bound
        # A dropped line leaves the sender blocked in sendall: closing the
        # writer under it raised EBADF in the thread, so drain the rest.
        while reader.recv(1 << 16):
            pass
        sender.join(timeout=5)
        assert not sender.is_alive()
    assert line == (payload if accepted else b"")


def test_leaving_the_block_waits_for_a_call_still_executing() -> None:
    """A tool Codex called keeps the run open until it returns; it never outlives the block."""
    started = threading.Event()
    release = threading.Event()
    finished_at: list[float] = []

    def execute(call: dict[str, Any]) -> tuple[str, str]:
        started.set()
        release.wait()
        finished_at.append(time.monotonic())
        return call["name"], "done"

    bridge = ToolBridge(ADD_SCHEMA, execute).__enter__()
    assert bridge.tool_names == {"add"}
    with socket.create_connection(("127.0.0.1", bridge.port)) as client:
        request = {"op": "call", "name": "add", "arguments": {}, "token": bridge._server.token}
        client.sendall(json.dumps(request).encode() + b"\n")
        assert started.wait(5)
        assert bridge.calls_in_flight == 1  # what pauses the CLI stall timeout
        threading.Timer(0.3, release.set).start()
        bridge.__exit__(None, None, None)
        exited_at = time.monotonic()
        assert finished_at and finished_at[0] <= exited_at
        assert json.loads(client.makefile("rb").readline()) == {"text": "done"}
    assert bridge.calls_in_flight == 0
    assert bridge.take_served() == [ServedCall("add", {}, "done")]
    assert bridge.take_served() == []


@pytest.mark.parametrize("text", [
    'quotes "and" back\\slashes',
    "new\nlines\tand tabs",
    "emoji \U0001f642 and accents é",
    "control \x01 and DEL \x7f",
    "C:\\Users\\k\\AppData\\Local\\Programs\\python.exe",
])
def test_toml_value_round_trips_through_a_toml_parser(text: str) -> None:
    """Every ``-c key=value`` string survives Codex's TOML parsing unchanged."""
    assert tomllib.loads(f"v={toml_value(text)}")["v"] == text
    assert tomllib.loads(f"v={toml_value(['-m', text])}")["v"] == ["-m", text]


# ---------------------------------------------------------------------------
# End to end: a KISSAgent on codex/* with a fake ``codex`` that is an MCP
# client of the ``kiss`` server named in its ``-c`` overrides.
# ---------------------------------------------------------------------------

TASK = "Shout hi."
SYSTEM_PROMPT = "Be loud."
FINAL_TEXT = "SHOUTED done."
FINISH_BLOCK = "```json\n" + json.dumps({
    "tool_calls": [{
        "name": "finish",
        "arguments": {"success": True, "is_continue": False, "summary_in_html": FINAL_TEXT},
    }]
}) + "\n```"

# Call N of the fake plays ``turns[N]`` (the last turn repeats): it makes
# the turn's MCP calls through the proxy (dying with exit 1 if one takes
# longer than ``crash_after`` seconds, or right after the calls when the
# turn says ``die: "after-calls"``), stays silent for the turn's
# ``hang`` seconds, then answers with the turn's text, in which
# ``{names}`` stands for the sorted MCP tool list, ``{result}`` for the
# first tool result and ``{results}`` for all of them, and completes the
# turn with the turn's ``usage`` (3 input / 4 output tokens unless given),
# dying with exit 1 after that when it says ``die: "after-usage"``.  ``thread.started``
# goes out before anything else so the model's stall timeout starts from
# a line, as with Codex.  Like Codex, it hands the proxy only the
# ``env_vars`` the server config names (the bridge's secret).  It records
# argv, the stdin prompt and the instructions file per call.
_FAKE_CODEX = """#!{python}
import json, os, pathlib, sys
print(json.dumps({{"type": "thread.started", "thread_id": "t-1"}}), flush=True)
import asyncio, tomllib
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("call-*.json")))
argv = sys.argv[1:]
prompt = sys.stdin.read()
overrides = tomllib.loads("\\n".join(
    argv[i + 1] for i in range(len(argv) - 1) if argv[i] == "-c"))
instructions = pathlib.Path(overrides["model_instructions_file"]).read_text()
(record_dir / f"call-{{n}}.json").write_text(json.dumps(
    {{"argv": argv, "prompt": prompt, "instructions": instructions}}))
turns = {turns!r}
turn = turns[min(n, len(turns) - 1)]
kiss = overrides["mcp_servers"]["kiss"]

async def main():
    results = []
    env = {{name: os.environ[name] for name in kiss.get("env_vars", [])}}
    params = StdioServerParameters(command=kiss["command"], args=kiss["args"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = sorted(t.name for t in (await session.list_tools()).tools)
            for tool, arguments in turn["mcp_calls"]:
                print(json.dumps({{"type": "item.started", "item": {{
                    "type": "mcp_tool_call", "server": "kiss", "tool": tool,
                    "arguments": arguments}}}}), flush=True)
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(tool, arguments), turn.get("crash_after"))
                except asyncio.TimeoutError:
                    os._exit(1)  # Codex died while the tool was still running
                results.append(result.content[0].text)
            if turn.get("die") == "after-calls":
                os._exit(1)  # Codex died after its calls, before the turn completed
    await asyncio.sleep(turn.get("hang", 0))  # silence after the calls
    text = turn["text"].replace("{{names}}", str(names))
    text = text.replace("{{result}}", "".join(results[:1]))
    text = text.replace("{{results}}", " | ".join(results))
    print(json.dumps({{"type": "item.completed", "item": {{
        "type": "agent_message", "text": text}}}}), flush=True)
    usage = turn.get("usage") or {{"input_tokens": 3, "output_tokens": 4}}
    print(json.dumps({{"type": "turn.completed", "usage": usage}}), flush=True)
    if turn.get("die") == "after-usage":
        os._exit(1)  # Codex died after reporting the turn's usage

asyncio.run(main())
"""


class _RecordingPrinter(Printer):
    """Real Printer that records every event with the thread binding it was printed under.

    Like ``JsonPrinter`` it keeps a per-thread task binding; the test's
    thread binds itself to ``"task-42"`` and every event records what
    the printing thread had adopted.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, dict[str, Any]]] = []
        self.tokens: list[str] = []
        self.bound = threading.local()

    def thread_binding(self) -> str:
        """Return the calling thread's task id (``""`` when unbound)."""
        return getattr(self.bound, "task_id", "")

    def adopt_thread_binding(self, binding: Any) -> None:
        """Bind the calling thread to task *binding*."""
        self.bound.task_id = binding

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Record the event together with the printing thread's task binding.

        Args:
            content: The content to display.
            type: Content type (``tool_call``, ``tool_result``, ...).
            **kwargs: Type-specific options (``tool_input``, ...).
        """
        self.events.append((type, content, {**kwargs, "bound_to": self.thread_binding()}))

    def token_callback(self, token: str) -> None:
        """Record a streamed token.

        Args:
            token: The text token.
        """
        self.tokens.append(token)

    def thinking_callback(self, is_start: bool) -> None:
        """Ignore thinking boundaries.

        Args:
            is_start: Unused.
        """

    def reset(self) -> None:
        """Nothing to reset."""


SEEN_STOP_EVENTS: list[threading.Event | None] = []
SHOUT_DELAY_SECONDS = 0.0

# The stall timeout of the tests that make the fake codex stall.  It also
# bounds the fake's silent start-up (its ``mcp`` imports, the bridge
# server it spawns, the handshake), which takes over 1.5 s on Windows.
STALL_TIMEOUT_SECONDS = 4.0


def shout(text: str) -> str:
    """Return TEXT upper-cased with a bang, noting the thread's stop event.

    Args:
        text: The text to shout.

    Returns:
        The shouted text.
    """
    SEEN_STOP_EVENTS.append(stop_signal.get_thread_stop_event())
    time.sleep(SHOUT_DELAY_SECONDS)
    return text.upper() + "!"


AGENT_UNDER_TEST: list[KISSAgent] = []


def explode(text: str) -> str:
    """Raise the fatal error a sub-agent's exhausted budget raises.

    Args:
        text: Ignored.

    Returns:
        Never.
    """
    raise BudgetExceededError(f"sub-agent budget exceeded before {text}")


def exhaust(text: str) -> str:
    """Spend the running agent's whole budget, as a costly sub-agent would.

    Args:
        text: Ignored.

    Returns:
        ``"spent"``.
    """
    AGENT_UNDER_TEST[-1].budget_used = AGENT_UNDER_TEST[-1].max_budget
    return "spent"


def snapshot() -> str:
    """Return a tiny PNG the way screenshot tools do: text plus a binary attachment.

    Returns:
        The text ``captured`` with the image embedded after it.
    """
    return "captured " + encode_binary_attachment("image/png", b"\x89PNG\r\n\x1a\nfake")


def switch_things(name: str) -> str:
    """Stand in for a tool that must run between turns (Sorcar's ``set_model``).

    Args:
        name: Ignored.

    Returns:
        ``"switched"``.
    """
    return "switched"


switch_things.runs_between_turns = True  # type: ignore[attr-defined]


HELPER_TRAJECTORIES: list[Path] = []


def delegate(text: str) -> str:
    """Run a helper agent on *text*, as a delegating tool (``run_agent``) does.

    The helper is a second codex/* agent against the same fake, which
    plays it the next turn of the script.

    Args:
        text: The helper's task.

    Returns:
        ``delegated: `` and the helper's summary.
    """
    helper = KISSAgent("bridge helper")
    result = helper.run(
        model_name="codex/gpt-5.5",
        prompt_template=text,
        tools=[structured_finish],
        is_agentic=True,
        verbose=False,
    )
    HELPER_TRAJECTORIES.append(helper.get_trajectory_path())
    return "delegated: " + str(yaml.safe_load(result)["summary"])


SEEN_REPLIES: list[LLMReply] = []


def _record_reply(reply: LLMReply) -> Verdict:
    """Keep every reply the response hook sees; allow it (an episode's token accounting)."""
    SEEN_REPLIES.append(reply)
    return Verdict(allowed=True)


def _install_fake_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turns: list[dict[str, Any]]
) -> Path:
    """Put the fake ``codex`` playing *turns* on PATH; return its record directory."""
    bin_dir = tmp_path / "bin"
    record_dir = tmp_path / "records"
    bin_dir.mkdir()
    record_dir.mkdir()
    install_cli_script(
        bin_dir / "codex",
        _FAKE_CODEX.format(python=sys.executable, record_dir=str(record_dir), turns=turns),
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record_dir


def _recorded_calls(record_dir: Path) -> list[dict[str, Any]]:
    """The fake's per-call records (argv, stdin prompt, instructions file), in order."""
    return [json.loads(p.read_text()) for p in sorted(record_dir.glob("call-*.json"))]


def _run_fake_codex_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    printer: _RecordingPrinter,
    turns: list[dict[str, Any]],
    tool_call_hook: Any = None,
    model_config: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
    llm_call_hook: Any = None,
    llm_response_hook: Any = None,
) -> tuple[KISSAgent, dict[str, Any], list[dict[str, Any]]]:
    """Run a codex/* KISSAgent against the fake MCP-client ``codex`` playing *turns*.

    Returns:
        The agent, the parsed finish payload and the fake's per-call
        records.
    """
    record_dir = _install_fake_codex(tmp_path, monkeypatch, turns)
    agent = KISSAgent("codex mcp bridge e2e")
    AGENT_UNDER_TEST.append(agent)
    result = agent.run(
        model_name="codex/gpt-5.5",
        printer=printer,
        prompt_template=TASK,
        system_prompt=SYSTEM_PROMPT,
        tools=[structured_finish, shout, *(tools or [])],
        is_agentic=True,
        verbose=False,
        tool_call_hook=tool_call_hook,
        llm_call_hook=llm_call_hook,
        llm_response_hook=llm_response_hook,
        model_config=model_config,
    )
    payload = yaml.safe_load(result)
    assert isinstance(payload, dict)
    return agent, payload, _recorded_calls(record_dir)


def test_agentic_codex_run_calls_kiss_tools_over_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 1: Codex calls ``shout`` over MCP — executed on the bridge thread
    under the task's binding and stop event — and ends with text; KISS replays
    the call into the transcript and asks for ``finish``.  Turn 2: Codex
    calls ``finish`` through the text protocol."""
    printer = _RecordingPrinter()
    printer.adopt_thread_binding("task-42")
    stop_event = threading.Event()
    stop_signal.set_thread_stop_event(stop_event)
    SEEN_STOP_EVENTS.clear()
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [["shout", {"text": "hi"}]], "text": "tools={names} shout={result}"},
        {"mcp_calls": [], "text": "All done.\n" + FINISH_BLOCK},
    ]
    hook_saw: list[list[dict[str, Any]]] = []

    def llm_call_hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hook_saw.append([dict(m) for m in messages])
        return messages

    try:
        agent, payload, calls = _run_fake_codex_agent(
            tmp_path, monkeypatch, printer, turns, llm_call_hook=llm_call_hook
        )
    finally:
        stop_signal.set_thread_stop_event(None)

    assert payload["success"] is True and FINAL_TEXT in payload["summary"]
    assert len(calls) == 2 and agent.step_count == 2
    assert agent.total_tokens_used == 2 * 7
    assert agent.tool_calls_made == 1

    first, second = calls
    # The bridge is registered per turn; ``finish`` is not served.
    for call in calls:
        argv = call["argv"]
        assert argv[argv.index("-m") + 1] == "gpt-5.5"
        assert "code_mode_host" not in argv  # the MCP router needs it
        overrides = tomllib.loads("\n".join(
            argv[i + 1] for i in range(len(argv) - 1) if argv[i] == "-c"
        ))
        kiss = overrides["mcp_servers"]["kiss"]
        assert kiss["command"] == sys.executable
        assert kiss["args"][:2] == ["-m", "kiss.core.models.codex_mcp_bridge"]
        assert kiss["tool_timeout_sec"] == CODEX_MCP_TOOL_TIMEOUT_SECONDS
        assert kiss["default_tools_approval_mode"] == "approve"
        assert call["instructions"].startswith(
            CODEX_INSTRUCTIONS_PREAMBLE + CODEX_MCP_TOOLS_NOTE + SYSTEM_PROMPT
        )
        # The text protocol describes finish alone; shout is an MCP tool.
        assert "`finish`" in call["instructions"] and "shout" not in call["instructions"]
    assert first["prompt"] == TASK

    # The call ran in the KISS process through the usual tool executor ...
    tool_events = [(t, c, k) for t, c, k in printer.events if t in ("tool_call", "tool_result")]
    assert [(t, c) for t, c, _ in tool_events[:2]] == [
        ("tool_call", "shout"), ("tool_result", "HI!"),
    ]
    assert [c for t, c, _ in tool_events[2:] if t == "tool_call"] == ["finish"]
    assert tool_events[0][2]["tool_input"] == {"text": "hi"}
    assert tool_events[1][2]["is_error"] is False
    # ... on a bridge thread, yet under the task's binding and stop event ...
    assert [k["bound_to"] for _, _, k in tool_events[:2]] == ["task-42", "task-42"]
    assert SEEN_STOP_EVENTS == [stop_event]
    # ... and the fake saw the tool list and the result.
    assert "tools=['shout'] shout=HI!" in second["prompt"]

    # Turn 2's transcript carries the MCP call and its result ahead of the
    # answer, then the nudge to finish.
    transcript = second["prompt"]
    assert transcript.index(f"[User]: {TASK}") < transcript.index(
        "[Assistant]: ```python\nshout(text='hi')\n```"
    ) < transcript.index("[Tool Result]: HI!") < transcript.index(
        "[Assistant]: tools=['shout'] shout=HI!"
    ) < transcript.index(f"[User]: {NATIVE_TOOLS_CONTINUE_MSG}")
    assert transcript.count("[Assistant]:") == 2
    # The trajectory records the call as it happened, before the step's answer.
    roles_and_text = [(m["role"], m["content"]) for m in agent.messages]
    shout_at = roles_and_text.index(("model", "```python\nshout(text='hi')\n```"))
    assert roles_and_text[shout_at + 1] == ("user", "[shout]: HI!")
    assert "tools=['shout'] shout=HI!" in roles_and_text[shout_at + 2][1]
    assert not any("mcp" in token for token in printer.tokens)

    # The replayed call has the ordinary tool-call shape, valid for an
    # API model the conversation may later be handed to ...
    conversation = agent.model.conversation
    call_msg = next(m for m in conversation if m.get("tool_calls"))
    (tool_call,) = call_msg["tool_calls"]
    assert tool_call["function"] == {"name": "shout", "arguments": '{"text": "hi"}'}
    result_msg = conversation[conversation.index(call_msg) + 1]
    assert result_msg["role"] == "tool" and result_msg["tool_call_id"] == tool_call["id"]
    assert result_msg["content"].startswith("HI!")
    # ... and the second LLM call's hook saw it as new, with its result.
    assert len(hook_saw) == 2
    new_messages = hook_saw[1]
    assert new_messages[0]["tool_calls"] == call_msg["tool_calls"]
    assert [m["role"] for m in new_messages] == ["assistant", "tool", "assistant", "user"]
    assert new_messages[1]["content"].startswith("HI!")
    assert new_messages[3]["content"].startswith(NATIVE_TOOLS_CONTINUE_MSG)


def _refuse_shouting(name: str, args: dict[str, Any]) -> Verdict:
    """Refuse every ``shout`` call; a hook of the kind Sorcar installs."""
    return refuse("hook says no") if name == "shout" else Verdict(allowed=True)


def test_tool_call_hook_applies_to_mcp_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call Codex makes over MCP goes through tool_call_hook like any other;
    the refusal is the result Codex sees and the transcript records."""
    printer = _RecordingPrinter()
    SEEN_STOP_EVENTS.clear()
    turns = [{"mcp_calls": [["shout", {"text": "hi"}]], "text": "shout={result}"}]
    turns.append({"mcp_calls": [], "text": FINISH_BLOCK})
    agent, payload, calls = _run_fake_codex_agent(
        tmp_path, monkeypatch, printer, turns, tool_call_hook=_refuse_shouting
    )
    assert payload["success"] is True
    assert SEEN_STOP_EVENTS == [], "a refused call must not run the tool"
    assert agent.tool_calls_made == 1
    tool_results = [(c, k["is_error"]) for t, c, k in printer.events if t == "tool_result"]
    refused = "REFUSED: the shout call did NOT run and had no effect. hook says no"
    assert tool_results[0] == (refused, True)  # then finish's own result
    assert f"shout={refused}" in calls[1]["prompt"]
    assert f"[Tool Result]: {refused}" in calls[1]["prompt"]


def _end_on_stop(name: str, args: dict[str, Any]) -> Verdict:
    """End the run on ``shout("stop now")``; a SEA's domain rule."""
    if name == "shout" and args.get("text") == "stop now":
        return end_run("<p>Episode over.</p>")
    return Verdict(allowed=True)


def test_end_run_verdict_on_the_bridge_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``end_run`` verdict on a call Codex makes over MCP ends the run once the
    turn returns: the call does not run, the turn's later call is refused, and
    the verdict's text is the run's result in the finish contract."""
    printer = _RecordingPrinter()
    SEEN_STOP_EVENTS.clear()
    turns: list[dict[str, Any]] = [{
        "mcp_calls": [["shout", {"text": "hi"}], ["shout", {"text": "stop now"}],
                      ["shout", {"text": "again"}]],
        "text": "results={results}",
    }]
    turns.append({"mcp_calls": [], "text": FINISH_BLOCK})
    SEEN_REPLIES.clear()
    agent, payload, calls = _run_fake_codex_agent(
        tmp_path, monkeypatch, printer, turns, tool_call_hook=_end_on_stop,
        llm_response_hook=_record_reply,
    )
    assert payload == {"success": True, "is_continue": False, "summary": "<p>Episode over.</p>"}
    assert len(calls) == 1, "the run ended with the turn: no second Codex turn"
    assert len(SEEN_STOP_EVENTS) == 1, "only the first shout ran"
    assert agent.tool_calls_made == 2, "the third call was refused before the hook"
    assert [c for t, c, _ in printer.events if t == "tool_call"] == ["shout", "shout"]
    transcript = " ".join(str(m["content"]) for m in agent.messages)
    refused = "REFUSED: the shout call did NOT run and had no effect. <p>Episode over.</p>"
    assert refused in transcript
    # The response hook sees the reply that closed the turn, usage included,
    # even though the tool verdict (which keeps precedence) had ended the run.
    assert len(SEEN_REPLIES) == 1
    assert SEEN_REPLIES[0].text.startswith("results=HI!")
    assert SEEN_REPLIES[0].usage["output_tokens"] == 4 and SEEN_REPLIES[0].step == 1


def test_end_run_verdict_survives_a_turn_that_fails_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex dies after a call whose hook ended the run: the verdict is the run's
    result all the same, and the trajectory says what came after it."""
    printer = _RecordingPrinter()
    turns: list[dict[str, Any]] = [{
        "mcp_calls": [["shout", {"text": "stop now"}]], "text": "never sent",
        "die": "after-calls",
    }]
    agent, payload, calls = _run_fake_codex_agent(
        tmp_path, monkeypatch, printer, turns, tool_call_hook=_end_on_stop
    )
    assert payload == {"success": True, "is_continue": False, "summary": "<p>Episode over.</p>"}
    assert len(calls) == 1 and agent.step_count == 1
    contents = [str(m["content"]) for m in agent.messages]
    assert any(c.startswith("[the turn failed after the run ended: ") for c in contents), contents


@pytest.mark.parametrize("die", ["after-usage", None])
def test_a_turn_ended_by_a_verdict_still_meets_the_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, die: str | None
) -> None:
    """The usage a turn reports is billed whether the turn then returns or Codex
    dies, and a limit it crosses ends the run with the limit's error, not with the
    verdict a tool hook gave during the turn."""
    printer = _RecordingPrinter()
    # Past 70% of codex/gpt-5.5's 500k context in one turn.
    turns: list[dict[str, Any]] = [{
        "mcp_calls": [["shout", {"text": "stop now"}]], "text": "results={results}",
        "usage": {"input_tokens": 400_000, "output_tokens": 4}, "die": die,
    }]
    with pytest.raises(ContextWindowExceededError, match="400,004 of 500,000 context tokens"):
        _run_fake_codex_agent(tmp_path, monkeypatch, printer, turns, tool_call_hook=_end_on_stop)
    assert len(AGENT_UNDER_TEST) and AGENT_UNDER_TEST[-1].total_tokens_used == 400_004


@pytest.mark.parametrize("save", [True, False])
def test_a_helper_agent_a_bridged_tool_creates_belongs_to_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save: bool
) -> None:
    """A tool Codex calls over MCP runs on a bridge thread; a helper agent it
    creates there still belongs to the run's ``collecting_trajectories`` scope:
    its file is collected, or not written when the scope says so."""
    printer = _RecordingPrinter()
    HELPER_TRAJECTORIES.clear()
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [["delegate", {"text": "sub-task"}]], "text": "helper said {result}"},
        {"mcp_calls": [], "text": FINISH_BLOCK},  # the helper's own turn
        {"mcp_calls": [], "text": FINISH_BLOCK},  # the run's second turn
    ]
    with collecting_trajectories(save=save) as scope:
        agent, payload, calls = _run_fake_codex_agent(
            tmp_path, monkeypatch, printer, turns, tools=[delegate]
        )
    assert payload["success"] is True and len(calls) == 3
    assert "[Tool Result]: delegated: <p>SHOUTED done.</p>" in calls[2]["prompt"]
    assert len(HELPER_TRAJECTORIES) == 1
    own, helper = agent.get_trajectory_path(), HELPER_TRAJECTORIES[0]
    assert (own.exists(), helper.exists()) == (save, save)
    assert set(scope.paths) == ({own, helper} if save else set())


def test_stall_timeout_waits_while_a_bridged_tool_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex is silent while a KISS tool it called runs; that silence, longer
    than the model's stall timeout, does not end the turn."""
    global SHOUT_DELAY_SECONDS
    printer = _RecordingPrinter()
    turns = [{
        "mcp_calls": [["shout", {"text": "slow"}]],
        "text": "shout={result}\n" + FINISH_BLOCK,
    }]
    SHOUT_DELAY_SECONDS = 2 * STALL_TIMEOUT_SECONDS
    try:
        agent, payload, calls = _run_fake_codex_agent(
            tmp_path, monkeypatch, printer, turns,
            model_config={"timeout": STALL_TIMEOUT_SECONDS},
        )
    finally:
        SHOUT_DELAY_SECONDS = 0.0
    assert payload["success"] is True
    assert len(calls) == 1 and agent.step_count == 1
    assert [c for t, c, _ in printer.events if t == "tool_result"][0] == "SLOW!"


def test_calls_served_before_a_stalled_turn_survive_its_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that stalls after Codex called a tool is retried; the retry's
    transcript still carries that call and its result, so the work is not
    redone."""
    printer = _RecordingPrinter()
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [["shout", {"text": "once"}]], "text": "never sent", "hang": 30},
        {"mcp_calls": [], "text": FINISH_BLOCK},
    ]
    agent, payload, calls = _run_fake_codex_agent(
        tmp_path, monkeypatch, printer, turns, model_config={"timeout": STALL_TIMEOUT_SECONDS}
    )
    assert payload["success"] is True
    assert len(calls) == 2 and agent.step_count == 2
    assert [c for t, c, _ in printer.events if t == "tool_call"] == ["shout", "finish"]
    retry_prompt = calls[1]["prompt"]
    assert "[Assistant]: ```python\nshout(text='once')\n```" in retry_prompt
    assert "[Tool Result]: ONCE!" in retry_prompt
    assert "never sent" not in retry_prompt


def test_fatal_tool_error_on_the_bridge_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fatal error a natively run tool raises is not just an MCP error Codex
    talks past: later calls of the turn are refused and the run raises it."""
    printer = _RecordingPrinter()
    SEEN_STOP_EVENTS.clear()
    turns: list[dict[str, Any]] = [{
        "mcp_calls": [["explode", {"text": "boom"}], ["shout", {"text": "hi"}]],
        "text": "results={results}",
    }]
    with pytest.raises(BudgetExceededError, match="sub-agent budget exceeded before boom"):
        _run_fake_codex_agent(tmp_path, monkeypatch, printer, turns, tools=[explode])
    assert [c for t, c, _ in printer.events if t == "tool_call"] == ["explode"]
    assert SEEN_STOP_EVENTS == [], "a call after the fatal error must be refused"


def test_limits_are_checked_between_native_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is checked after every native call, as after every ordinary
    one: a turn whose first tool spends it all gets no second tool."""
    printer = _RecordingPrinter()
    SEEN_STOP_EVENTS.clear()
    turns: list[dict[str, Any]] = [{
        "mcp_calls": [["exhaust", {"text": "all"}], ["shout", {"text": "hi"}]],
        "text": "results={results}",
    }]
    with pytest.raises(BudgetExceededError, match="budget exceeded"):
        _run_fake_codex_agent(tmp_path, monkeypatch, printer, turns, tools=[exhaust])
    agent = AGENT_UNDER_TEST[-1]
    assert [c for t, c, _ in printer.events if t == "tool_call"] == ["exhaust"]
    assert SEEN_STOP_EVENTS == []
    # The call that ran is in the trajectory and in the model conversation.
    assert ("user", "[exhaust]: spent") in [(m["role"], m["content"]) for m in agent.messages]
    assert any(m.get("tool_calls") for m in agent.model.conversation)


def test_binary_tool_results_are_lifted_out_for_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A screenshot a tool returns is not fed to Codex as base64, neither in the
    MCP reply nor in the next turn's transcript: the result takes the path every
    tool result takes (the bytes lifted out, the picture handed to the vision
    bridge), and with no vision model configured both say it could not be described."""
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, provider.api_key_name, "")
    for _prefix, _label, api_key_name in _NATIVE_PROVIDERS:
        if api_key_name is not None:
            monkeypatch.setattr(config_module.DEFAULT_CONFIG, api_key_name, "")
    printer = _RecordingPrinter()
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [["snapshot", {}]], "text": "got={result}"},
        {"mcp_calls": [], "text": FINISH_BLOCK},
    ]
    _agent, payload, calls = _run_fake_codex_agent(
        tmp_path, monkeypatch, printer, turns, tools=[snapshot]
    )
    assert payload["success"] is True
    transcript = calls[1]["prompt"]
    assert "iVBOR" not in transcript and "base64" not in transcript  # no PNG bytes
    not_described = (
        "[image/png attachment 1 could not be described: no vision model is configured; "
        "the model you run on (codex/gpt-5.5) cannot see images, use a vision model or "
        "an OCR tool]"
    )
    lifted = "captured [attached image/png, 12 bytes]\n\n" + not_described
    # What Codex got back from the call, echoed in its answer ...
    assert f"[Assistant]: got={lifted}" in transcript
    # ... and the replayed tool result, recorded as it was served.
    assert f"[Tool Result]: {lifted}" in transcript
    assert transcript.count(not_described) == 2


def test_tools_that_run_between_turns_stay_on_the_text_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool marked ``runs_between_turns`` is not served over MCP; it is
    described in the text tools prompt with ``finish`` instead."""
    printer = _RecordingPrinter()
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [], "text": "tools={names}"},
        {"mcp_calls": [], "text": FINISH_BLOCK},
    ]
    _agent, payload, calls = _run_fake_codex_agent(
        tmp_path, monkeypatch, printer, turns, tools=[switch_things]
    )
    assert payload["success"] is True
    assert "tools=['shout']" in calls[1]["prompt"]
    assert "switch_things" in calls[0]["instructions"]
    assert "shout" not in calls[0]["instructions"]


def test_a_call_still_running_when_codex_dies_is_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex exits while a KISS tool it called is still running: the tool
    finishes, and the call lands in the conversation and the trajectory
    even though the turn failed."""
    global SHOUT_DELAY_SECONDS
    printer = _RecordingPrinter()
    turns: list[dict[str, Any]] = [{
        "mcp_calls": [["shout", {"text": "late"}]], "text": "never", "crash_after": 0.5,
    }]
    record_dir = _install_fake_codex(tmp_path, monkeypatch, turns)
    agent = KISSAgent("codex dies mid-call")
    SHOUT_DELAY_SECONDS = 2.0
    try:
        with pytest.raises(KISSError, match="Codex CLI failed"):
            agent.run(
                model_name="codex/gpt-5.5",
                printer=printer,
                prompt_template=TASK,
                tools=[structured_finish, shout],
                is_agentic=True,
                verbose=False,
            )
    finally:
        SHOUT_DELAY_SECONDS = 0.0
    assert len(_recorded_calls(record_dir)) == 1
    assert [c for t, c, _ in printer.events if t == "tool_result"] == ["LATE!"]
    tool_msgs = [m for m in agent.model.conversation if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["content"].startswith("LATE!")
    assert ("user", "[shout]: LATE!") in [(m["role"], m["content"]) for m in agent.messages]
