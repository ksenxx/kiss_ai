# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``@tool_schema``: a tool's declared JSON schema reaches the model.

The declared schema replaces the one derived from the signature and the
docstring, both in the native tool definitions a real ``KISSAgent.run``
sends to an OpenAI-protocol server (asserted on the wire) and in the text
catalog of tools built for models without native tool calling; the
function itself is unchanged and the model's arguments reach it as sent.
A tool marked ``unlisted`` is in neither, and still runs when called.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import _build_text_based_tools_prompt
from kiss.core.tool_schema import (
    TOOL_SCHEMA_ATTR,
    declared_schema,
    is_unlisted,
    tool_schema,
    unlisted,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["low", "high"], "description": "How bad."},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                },
                "required": ["file"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["severity", "findings"],
}

filed: list[dict[str, Any]] = []


@tool_schema(parameters=SCHEMA, description="File the findings.")
def file_report(severity: str, findings: list[dict[str, Any]], note: str = "") -> str:
    """Derived description, replaced by the declared one."""
    filed.append({"severity": severity, "findings": findings, "note": note})
    return f"filed {len(findings)}"


@tool_schema(parameters={"type": "object", "properties": {}})
def ping() -> str:
    """Answer pong."""
    return "pong"


@unlisted
def knock(times: int) -> str:
    """Knock on the door no request lists."""
    return f"knocked {times}"


def test_decorator_keeps_the_function_and_records_the_declaration() -> None:
    assert file_report.__name__ == "file_report"
    assert file_report("low", [], note="x") == "filed 0"
    assert declared_schema(file_report) == {
        "parameters": SCHEMA, "description": "File the findings."
    }
    assert declared_schema(ping) == {"parameters": {"type": "object", "properties": {}},
                                     "description": ""}
    assert declared_schema(test_decorator_keeps_the_function_and_records_the_declaration) is None
    assert getattr(file_report, TOOL_SCHEMA_ATTR)["parameters"] is SCHEMA
    assert knock.__name__ == "knock" and knock(3) == "knocked 3"
    assert is_unlisted(knock) and not is_unlisted(ping)


@pytest.mark.parametrize("bad", [{"type": "array"}, {"properties": {}}, ["type", "object"], None])
def test_a_schema_that_is_not_an_object_is_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="needs an object schema"):
        tool_schema(parameters=bad)  # type: ignore[arg-type]


def test_text_catalog_carries_the_declared_schema_whole() -> None:
    """The text catalog states the declared schema verbatim: nested constraints included."""
    catalog = _build_text_based_tools_prompt(
        {"file_report": file_report, "ping": ping, "knock": knock, "plain": _plain}
    )
    assert "knock" not in catalog
    assert (
        f"- **file_report**: File the findings.\n  Parameters (JSON schema): {json.dumps(SCHEMA)}"
        in catalog
    )
    assert '"line": {"type": "integer", "minimum": 1}' in catalog
    assert '"required": ["file"], "additionalProperties": false' in catalog
    assert (
        '- **ping**: Answer pong.\n  Parameters (JSON schema): {"type": "object", "properties": {}}'
        in catalog
    )
    # An undeclared tool keeps the catalog's signature-derived lines.
    assert "- **plain**: Echo a word.\n  Parameters:\n    - word (str)" in catalog


def _plain(word: str) -> str:
    """Echo a word."""
    return word


def _serve(requests: list[dict[str, Any]]) -> HTTPServer:
    """A chat-completions server: a ``file_report`` call, a ``knock`` call, then ``finish``."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            requests.append(json.loads(self.rfile.read(length)))
            arguments: dict[str, Any]
            if len(requests) == 1:
                name, arguments = "file_report", {
                    "severity": "high", "findings": [{"file": "a.py"}, {"file": "b.py"}],
                }
            elif len(requests) == 2:
                name, arguments = "knock", {"times": 2}
            else:
                name, arguments = "finish", {"result": "DONE"}
            body = json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{len(requests)}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }).encode()
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


def test_declared_schema_is_what_the_model_receives_and_arguments_reach_the_tool() -> None:
    requests: list[dict[str, Any]] = []
    server = _serve(requests)
    filed.clear()
    try:
        result = KISSAgent("test-tool-schema").run(
            model_name="gpt-4o-mini",
            prompt_template="Review and file.",
            tools=[file_report, ping, knock],
            max_steps=4,
            max_budget=1.0,
            verbose=False,
            model_config={
                "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                "api_key": "sk-test",
            },
        )
    finally:
        server.shutdown()
    assert result == "DONE"
    tools = {t["function"]["name"]: t["function"] for t in requests[0]["tools"]}
    # An unlisted tool is in no request, and runs when the model calls it all the same.
    assert sorted(tools) == ["file_report", "finish", "ping"]
    assert requests[0]["tools"] == requests[1]["tools"] == requests[2]["tools"]
    assert requests[2]["messages"][-1]["role"] == "tool"
    assert "knocked 2" in requests[2]["messages"][-1]["content"]
    assert tools["file_report"]["parameters"] == SCHEMA
    assert tools["file_report"]["description"] == "File the findings."
    assert tools["ping"] == {
        "name": "ping", "description": "Answer pong.",
        "parameters": {"type": "object", "properties": {}},
    }
    # A tool without a declaration keeps the derived schema.
    assert tools["finish"]["parameters"]["properties"]["result"]["type"] == "string"
    assert filed == [{"severity": "high", "findings": [{"file": "a.py"}, {"file": "b.py"}],
                      "note": ""}]
