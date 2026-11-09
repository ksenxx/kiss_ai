# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a hook ends the run, the response hook sees every reply, ``plain_io``.

* ``end_run(result)`` from ``tool_call_hook`` ends the run at that call:
  the call does not run, the later calls of the turn do not run, and the
  result is rendered through the registered ``finish`` contract.
* The implicit-finish probe (text-only turns) honours ``end_run`` too.
* ``llm_response_hook`` receives every reply with its usage before any of
  its tool calls run, and ``end_run`` from it ends the run.
* ``model_config["plain_io"]`` leaves the usage line out of what the model
  reads (tool results and user messages) while the trajectory keeps it.

Every test runs the real ``KISSAgent.run`` against a local HTTP server
speaking the OpenAI chat-completions protocol — no mocks or patches.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.llm_reply import LLMReply
from kiss.core.tool_verdict import ALLOW, REFUSED_PREFIX, Verdict, end_run, refuse
from kiss.core.utils import finish as structured_finish

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_PROMPT = "Verify the build is green and report the status."


def _text_response(text: str) -> dict[str, Any]:
    """OpenAI-compatible response with text only, no tool calls."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": _USAGE,
    }


def _tool_calls_response(text: str, calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """OpenAI-compatible response with text plus the given (name, args) tool calls."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": f"call_{i + 1}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        for i, (name, arguments) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": _USAGE,
    }


def _serve(
    respond: Callable[[int, dict[str, Any]], dict[str, Any]],
    requests_log: list[dict[str, Any]],
) -> HTTPServer:
    """Start a local chat-completions server; ``respond(turn, request)`` builds each reply."""
    turn_counter = [0]

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(content_length)) if content_length else {}
            requests_log.append(request)
            reply = respond(turn_counter[0], request)
            turn_counter[0] += 1
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _Counting:
    """Tools whose executions are counted, to prove what did and did not run."""

    def __init__(self) -> None:
        self.check_build_runs = 0
        self.deploy_runs = 0

    def check_build(self) -> str:
        """Deterministic read-only verification tool: always reports success."""
        self.check_build_runs += 1
        return "build: all 42 tests passed"

    def deploy(self, target: str) -> str:
        """Deploy the build to *target*."""
        self.deploy_runs += 1
        return f"deployed to {target}"


def _run(
    respond: Callable[[int, dict[str, Any]], dict[str, Any]],
    tools: list[Callable[..., Any]],
    *,
    tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None,
    llm_response_hook: Callable[[LLMReply], Verdict] | None = None,
    model_config: dict[str, Any] | None = None,
    max_steps: int = 12,
) -> tuple[str, KISSAgent, list[dict[str, Any]]]:
    """Run a real KISSAgent against a local server; return (result, agent, requests)."""
    requests: list[dict[str, Any]] = []
    server = _serve(respond, requests)
    agent = KISSAgent("test-hook-ends-run")
    try:
        result = agent.run(
            model_name="gpt-4o-mini",
            prompt_template=_PROMPT,
            tools=tools,
            max_steps=max_steps,
            max_budget=5.0,
            verbose=False,
            model_config={
                "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                "api_key": "sk-test",
                **(model_config or {}),
            },
            tool_call_hook=tool_call_hook,
            llm_response_hook=llm_response_hook,
        )
    finally:
        server.shutdown()
    return result, agent, requests


class TestEndRunFromToolCallHook:
    """``end_run`` from ``tool_call_hook`` ends the run at the call."""

    def test_call_and_later_calls_do_not_run_and_result_is_the_verdict_text(self) -> None:
        counter = _Counting()

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            # One turn asks for three calls: the first runs, the second
            # ends the run, the third must never run.
            return _tool_calls_response(
                "Working.",
                [("check_build", {}), ("deploy", {"target": "prod"}), ("deploy", {"target": "eu"})],
            )

        def hook(name: str, args: dict[str, Any]) -> Verdict:
            if name == "deploy":
                return end_run("Episode over: the report was accepted.")
            return ALLOW

        result, agent, requests = _run(respond, [counter.check_build, counter.deploy],
                                       tool_call_hook=hook)
        assert result == "Episode over: the report was accepted."
        assert counter.check_build_runs == 1
        assert counter.deploy_runs == 0
        assert len(requests) == 1, "The run ends at the verdict: no further model call"
        # The trajectory records the turn: the allowed call's result, the
        # refused call (which did not run), and nothing for the third.
        results_text = agent.messages[-1]["content"]
        assert "42 tests passed" in results_text
        assert REFUSED_PREFIX.format(name="deploy") in results_text
        assert results_text.count("[deploy]") == 1

    def test_structured_finish_contract_and_success_flag(self) -> None:
        counter = _Counting()

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_calls_response("Checking.", [("check_build", {})])

        def hook(name: str, args: dict[str, Any]) -> Verdict:
            if name == "check_build":
                return end_run("<p>Token budget spent.</p>", success=False)
            return ALLOW

        # The structured ``finish`` first, as RelentlessAgent registers it.
        result, _agent, _requests = _run(
            respond, [structured_finish, counter.check_build], tool_call_hook=hook
        )
        payload = yaml.safe_load(result)
        assert payload["success"] is False
        assert payload["is_continue"] is False
        assert payload["summary"] == "<p>Token budget spent.</p>"
        assert counter.check_build_runs == 0

    def test_refuse_still_lets_the_run_go_on(self) -> None:
        """The default verdicts behave as before: a refusal is a tool result, not an end."""
        counter = _Counting()

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_calls_response("Deploying.", [("deploy", {"target": "prod"})])
            return _tool_calls_response("Done.", [("finish", {"result": "GREEN"})])

        def hook(name: str, args: dict[str, Any]) -> Verdict:
            return refuse("Not during the freeze.") if name == "deploy" else ALLOW

        result, _agent, requests = _run(respond, [counter.deploy], tool_call_hook=hook)
        assert result == "GREEN"
        assert counter.deploy_runs == 0
        assert len(requests) == 2
        second_turn_text = json.dumps(requests[1]["messages"])
        assert "Not during the freeze." in second_turn_text


class TestEndRunFromImplicitFinishProbe:
    """Text-only turns probe the hook with ``("finish", {})``: ``end_run`` ends the run."""

    def test_probe_end_run_is_the_result(self) -> None:
        probes: list[tuple[str, dict[str, Any]]] = []

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _text_response(f"Status {turn}.")

        def hook(name: str, args: dict[str, Any]) -> Verdict:
            probes.append((name, args))
            return end_run("The model only talks; the episode is over.")

        result, _agent, requests = _run(respond, [], tool_call_hook=hook)
        assert result == "The model only talks; the episode is over."
        assert probes == [("finish", {})]
        # Two text-only turns reach the probe (MAX_CONSECUTIVE_NO_TOOL_CALLS).
        assert len(requests) == 2


class TestFailedFinish:
    """A ``finish`` call its tool rejected is a tool error the model corrects, not the result."""

    def test_a_finish_missing_an_argument_is_reported_to_the_model(self) -> None:
        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                # The structured finish requires ``success``: this call raises.
                return _tool_calls_response(
                    "Done.", [("finish", {"summary_in_html": "<p>first</p>"})]
                )
            return _tool_calls_response(
                "Done.", [("finish", {"success": True, "summary_in_html": "<p>second</p>"})]
            )

        result, _agent, requests = _run(respond, [structured_finish])
        assert len(requests) == 2, "The failed finish is not the task's result"
        second_turn_text = json.dumps(requests[1]["messages"])
        assert "missing 1 required positional argument: 'success'" in second_turn_text
        payload = yaml.safe_load(result)
        assert payload["success"] is True
        assert payload["summary"] == "<p>second</p>"


class TestLLMResponseHook:
    """``llm_response_hook`` sees every reply with its usage; ``end_run`` ends the run."""

    def test_hook_sees_each_reply_before_its_tools_run(self) -> None:
        counter = _Counting()
        replies: list[LLMReply] = []
        runs_when_seen: list[int] = []

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_calls_response("Checking.", [("check_build", {})])
            return _tool_calls_response("Done.", [("finish", {"result": "GREEN"})])

        def hook(reply: LLMReply) -> Verdict:
            replies.append(reply)
            runs_when_seen.append(counter.check_build_runs)
            return ALLOW

        result, agent, _requests = _run(respond, [counter.check_build], llm_response_hook=hook)
        assert result == "GREEN"
        assert [r.text for r in replies] == ["Checking.", "Done."]
        assert replies[0].tool_calls == [{"name": "check_build", "arguments": {}}]
        assert replies[1].tool_calls == [{"name": "finish", "arguments": {"result": "GREEN"}}]
        assert runs_when_seen == [0, 1], "The hook runs before the reply's tools do"
        assert [r.step for r in replies] == [1, 2]
        assert replies[0].usage["input_tokens"] == 10
        assert replies[0].usage["output_tokens"] == 5
        assert replies[0].usage["stop_reason"] == "tool_calls"
        assert replies[0].total_tokens_used == 15
        assert replies[1].total_tokens_used == 30 == agent.total_tokens_used
        assert replies[1].budget_used == agent.budget_used

    def test_end_run_from_the_response_hook_skips_the_reply_tools(self) -> None:
        counter = _Counting()
        output_tokens = 0

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_calls_response("Deploying.", [("deploy", {"target": "prod"})])

        def hook(reply: LLMReply) -> Verdict:
            nonlocal output_tokens
            output_tokens += reply.usage["output_tokens"]
            if output_tokens >= 10:
                return end_run(f"Episode token cap reached after {output_tokens} output tokens.")
            return ALLOW

        result, agent, requests = _run(respond, [counter.deploy], llm_response_hook=hook)
        assert result == "Episode token cap reached after 10 output tokens."
        assert len(requests) == 2
        assert counter.deploy_runs == 1, "Only the first reply's call ran"
        assert agent.messages[-1]["role"] == "model"
        assert agent.messages[-1]["content"].startswith("Deploying.")

    def test_end_run_on_a_text_only_reply(self) -> None:
        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _text_response("REPORT: all green.")

        def hook(reply: LLMReply) -> Verdict:
            if reply.text.startswith("REPORT:"):
                return end_run(reply.text, success=True)
            return ALLOW

        result, _agent, requests = _run(respond, [], llm_response_hook=hook)
        assert result == "REPORT: all green."
        assert len(requests) == 1


class TestPlainIO:
    """``model_config["plain_io"]`` keeps the usage line out of what the model reads."""

    @staticmethod
    def _messages_text(requests: list[dict[str, Any]]) -> str:
        return json.dumps([m for r in requests for m in r["messages"]])

    def test_default_sends_the_usage_line(self) -> None:
        counter = _Counting()

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_calls_response("Checking.", [("check_build", {})])
            return _tool_calls_response("Done.", [("finish", {"result": "GREEN"})])

        _result, _agent, requests = _run(respond, [counter.check_build])
        assert "Steps: 1/12" in self._messages_text(requests)

    def test_plain_io_sends_no_usage_line_but_keeps_it_in_the_trajectory(self) -> None:
        counter = _Counting()

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_calls_response("Checking.", [("check_build", {})])
            if turn == 1:
                return _text_response("Thinking out loud.")
            return _tool_calls_response("Done.", [("finish", {"result": "GREEN"})])

        result, agent, requests = _run(
            respond, [counter.check_build], model_config={"plain_io": True}
        )
        assert result == "GREEN"
        wire = self._messages_text(requests)
        assert "Steps:" not in wire and "Budget:" not in wire
        # The tool result and the retry message reached the model as is.
        assert "build: all 42 tests passed" in wire
        assert "MUST have at least one function call" in wire
        # The trajectory still carries the usage footer of every turn.
        model_turns = [m["content"] for m in agent.messages if m["role"] == "model"]
        assert model_turns and all("Steps:" in turn for turn in model_turns)
