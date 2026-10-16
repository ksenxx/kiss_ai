# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end: ``run_agent``, ``run_parallel`` and ``run_commands_parallel``
take a ``description`` argument, like ``Bash``.

The chat page folds a tool panel to ``<tool> <description>`` when the
``tool_call`` event carries a ``description``; the daemon stamps it from
the call's arguments.  These tests cover the Python side: the three tools
accept the argument (and ignore it, as ``Bash`` does), their generated
schema offers it to the model with its docstring text, and the daemon's
``JsonPrinter`` copies it into the ``tool_call`` event.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool, make_run_parallel_tool
from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL
from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.models.model_info import model
from kiss.server.json_printer import JsonPrinter


def _schema_params(tool: Callable[..., Any]) -> dict[str, Any]:
    schema = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(tool)["function"]
    params: dict[str, Any] = schema["parameters"]
    return params


def test_three_tools_offer_description_like_bash(tmp_path: Path) -> None:
    useful = UsefulTools(work_dir=str(tmp_path))
    tools: dict[str, Callable[..., Any]] = {
        "run_agent": make_run_agent_tool(str(tmp_path)),
        "run_parallel": make_run_parallel_tool(str(tmp_path)),
        "run_commands_parallel": useful.run_commands_parallel,
        "docker run_commands_parallel": DockerManager.run_commands_parallel,
    }
    bash_desc = _schema_params(useful.Bash)["properties"]["description"]
    assert bash_desc["type"] == "string"
    for name, tool in tools.items():
        param = inspect.signature(tool).parameters["description"]
        assert param.annotation in ("str", str), name
        assert param.default == "", name
        params = _schema_params(tool)
        prop = params["properties"]["description"]
        assert prop["type"] == "string", name
        assert prop["description"].endswith("shown as the title of this call in the chat."), name
        # Optional, as every existing caller passes no description.
        assert "description" not in params["required"], name
    # The ``Args:`` section of run_parallel reaches the model whole: its
    # ``tasks`` example used to be an indented block with a blank line,
    # which ended the section for the docstring parser and dropped every
    # later argument's description from the schema.
    props = _schema_params(tools["run_parallel"])["properties"]
    assert props["tasks"]["description"].startswith("A JSON-encoded list of task description")
    assert "summarize its purpose" in props["tasks"]["description"]
    assert props["agent"]["description"].startswith("The SEA every child runs as")
    assert props["options"]["description"].startswith("JSON object of run settings")


def test_run_commands_parallel_runs_with_a_description(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    report = tools.run_commands_parallel(
        '["echo described", "echo twice"]', description="Echo two words",
    )
    assert report.startswith("2 commands: 2 succeeded")
    assert "described" in report and "twice" in report
    # The description changes nothing about the run itself.
    assert "Echo two words" not in report


def test_run_agent_and_run_parallel_accept_a_description(tmp_path: Path) -> None:
    run_parallel = make_run_parallel_tool(str(tmp_path))
    assert run_parallel("not json", description="Fan out").startswith("Error: tasks")
    run_agent = make_run_agent_tool(str(tmp_path))
    assert run_agent("say hi", "no-such-agent-xyz", description="Greet").startswith(
        "Error: unknown agent 'no-such-agent-xyz'",
    )


def test_daemon_stamps_description_on_the_tool_call_event() -> None:
    printer = JsonPrinter()
    printer._thread_local.task_id = "t-desc"
    printer.start_recording()
    printer._format_tool_call(
        "run_parallel",
        {"tasks": '["a", "b"]', "description": "Review the diff from two angles"},
    )
    printer._format_tool_call(
        "run_commands_parallel",
        {"commands": '["make lint"]', "description": "Lint the tree"},
    )
    printer._format_tool_call("run_agent", {"task": "ping", "agent": "cron"})
    events = [e for e in printer.stop_recording() if e["type"] == "tool_call"]
    assert [e["name"] for e in events] == ["run_parallel", "run_commands_parallel", "run_agent"]
    assert events[0]["description"] == "Review the diff from two angles"
    assert events[0]["extras"]["tasks"] == '["a", "b"]'
    assert events[1]["description"] == "Lint the tree"
    assert "description" not in events[2]
