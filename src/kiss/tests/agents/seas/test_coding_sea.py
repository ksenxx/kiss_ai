# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the coding SEA (:mod:`kiss.agents.seas.coding_sea`).

The tests build a :class:`~kiss.agents.seas.coding_sea.ContainerHarness`
from a JSON config and drive its SEA getters and hooks directly: the
trajectory log, the human-only tool answers, the destructive-command
guard, the finish gate, the test-context notes of the Edit tool and the
shell notes.  Tests that need a container start one with the Docker SDK
and skip when Docker is not available.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pytest

MODEL = "gpt-5.6-luna"


def test_hooks_log_every_call_and_answer_interactive_tools(tmp_path: Path) -> None:
    """Every LLM call is counted (no cap); the tool hook logs calls and answers human-only tools."""
    import docker

    from kiss.agents.seas import coding_sea

    # The hook ends the trial when its container is gone, so the test needs a
    # live one; without a Docker daemon the liveness check is skipped.
    container_name = "c"
    live = None
    try:
        client = docker.from_env()
        client.ping()
        live = client.containers.run("python:3.11-slim", "sleep infinity", detach=True)
        container_name = live.id
    except Exception:
        pass
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "container": container_name,
                "workdir": "/app",
                "prompt": "p",
                "model": MODEL,
                "trajectory": str(tmp_path / "trajectory.jsonl"),
            }
        )
    )
    harness = coding_sea.ContainerHarness(str(config))
    try:
        for expected in range(1, 202):
            assert harness.on_llm_call([{"role": "user", "content": "x"}]) == [
                {"role": "user", "content": "x"}
            ]
            assert harness.turns == expected
        # no wall-clock limit: tool results carry no time note
        tool_result = {"role": "tool", "content": "out"}
        assert harness.on_llm_call([tool_result]) == [{"role": "tool", "content": "out"}]
    finally:
        if live is not None:
            live.remove(force=True)
    if live is not None:
        # the container is gone: the next model call ends the trial
        from kiss.core.kiss_error import BudgetExceededError

        with pytest.raises(BudgetExceededError):
            harness.on_llm_call([])
    assert harness.on_tool_call("Bash", {"command": "ls"}) == "OK"
    assert harness.on_tool_call("ask_user_question", {"question": "?"}) != "OK"
    assert harness.on_tool_call("talk", {"text": "hi", "language": "en"}) != "OK"
    assert harness.on_tool_call("run_agent", {"agent": "slack", "task": "x"}) != "OK"
    assert harness.docker_image() == f"container:{container_name}"
    assert harness.if_append_basic_tools()
    assert not harness.use_memory() and not harness.use_web_tools()
    assert "/app" in harness.system_prompt() and "wall-clock" not in harness.system_prompt()
    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events].count("llm_call") == 202 + (live is not None)
    tool_events = [e for e in events if e["event"] == "tool_call"]
    assert [e["blocked"] for e in tool_events] == [False, True, True, True]



def test_append_to_last_tool_result_handles_every_message_shape() -> None:
    """The note lands in tool results of every provider shape.

    It is never appended to prompts or assistant turns.
    """
    from kiss.agents.seas import coding_sea

    anthropic: dict[str, Any] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": "out"}],
    }
    anthropic_list: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t",
                "content": [{"type": "text", "text": "out"}],
            }
        ],
    }
    responses: dict[str, Any] = {"type": "function_call_output", "call_id": "c", "output": "out"}
    chat: dict[str, Any] = {"role": "tool", "tool_call_id": "c", "content": "out"}
    for message in (anthropic, anthropic_list, responses, chat):
        assert coding_sea.append_to_last_tool_result(message, "[note]")
    assert anthropic["content"][0]["content"] == "out\n\n[note]"
    assert anthropic_list["content"][0]["content"][-1]["text"] == "[note]"
    assert responses["output"] == "out\n\n[note]"
    assert chat["content"] == "out\n\n[note]"
    prompt: dict[str, Any] = {"role": "user", "content": "task"}
    assistant: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    for other in (prompt, assistant, "not a dict"):
        assert not coding_sea.append_to_last_tool_result(other, "[note]")
    assert prompt == {"role": "user", "content": "task"}
    assert assistant["content"] == [{"type": "text", "text": "hi"}]


def test_changed_definitions_and_test_paths() -> None:
    """The edit locator names the enclosing definitions of a change.

    The test-path heuristic covers common layouts; Python, Go receiver methods
    and exported JS functions are recognised.
    """
    from kiss.agents.seas import coding_test_context as test_context

    changed = test_context.changed_definitions
    old = (
        "class Field:\n    def check(self):\n        return 1\n\n"
        "    def other(self):\n        pass\n"
    )
    new = (
        "class Field:\n    def check(self):\n        if True:\n            return 2\n\n"
        "    def other(self):\n        pass\n"
    )
    assert changed(old, new) == ["check", "Field"]
    # a deleted line points at the definition that lost it; a new top-level function names itself
    assert changed(new, old) == ["check", "Field"]
    assert changed("x = 1\n", "x = 1\ndef helper():\n    return 3\n") == ["helper"]
    # generic names are dropped, a change outside any definition yields nothing
    assert changed("def main():\n    a\n", "def main():\n    b\n") == []
    assert changed("a = 1\n", "a = 2\n") == []
    # Go receiver methods and exported JS functions are recognised too
    go_old = "func (s *Server) Start() {\n\treturn\n}\n"
    assert changed(go_old, go_old.replace("return", "run()")) == ["Start"]
    js_old = "export async function load() {\n  a\n}\n"
    assert changed(js_old, js_old.replace("  a\n", "  b\n")) == ["load"]
    for path in (
        "tests/test_x.py",
        "pkg/x_test.go",
        "src/a.spec.ts",
        "spec/a_spec.rb",
        "src/__tests__/a.js",
        "FooTests.cs",
        "testing/util.py",
    ):
        assert test_context.is_test_path(path), path
    for path in ("django/db/models/fields.py", "src/contest.py", "latest/run.py"):
        assert not test_context.is_test_path(path), path
    assert test_context.find_referencing_tests("c", "/w", [], "/w/a.py") == []


def test_edit_tool_results_list_referencing_tests(tmp_path: Path) -> None:
    """Editing an existing source file appends the tests that mention the changed definitions."""
    import docker

    from kiss.agents.seas import coding_sea

    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        pytest.skip("Docker is not available")
    live = client.containers.run("python:3.11-slim", "sleep infinity", detach=True)
    try:
        setup = (
            "mkdir -p /repo/pkg /repo/tests && cd /repo && "
            "printf 'class Field:\\n    def check(self):\\n        return 1\\n' > pkg/fields.py && "
            "printf 'from pkg.fields import Field\\n\\ndef test_check():\\n"
            "    assert Field().check() == 1\\n' > tests/test_fields.py && "
            "printf 'def test_other():\\n    Field\\n    Field\\n    Field\\n' "
            "> tests/test_other.py && "
            "printf 'x = 1\\n' > tests/test_unrelated.py && printf 'Field\\n' > pkg/notes.txt"
        )
        assert live.exec_run(["sh", "-c", setup]).exit_code == 0
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "container": live.id,
                    "workdir": "/repo",
                    "prompt": "p",
                    "model": MODEL,
                    "trajectory": str(tmp_path / "trajectory.jsonl"),
                    "test_context": True,
                }
            )
        )
        harness = coding_sea.ContainerHarness(str(config))
        assert "tests that reference the changed definitions" in harness.system_prompt()
        # an Edit of a source file: snapshot, then the change lands in the container
        edit = {"file_path": "pkg/fields.py", "old_string": "1", "new_string": "2"}
        assert harness.on_tool_call("Edit", edit) == "OK"
        live.exec_run(["sh", "-c", "sed -i 's/return 1/return 2/' /repo/pkg/fields.py"])
        # edits of test files, missing files and non-string paths are ignored
        harness.on_tool_call("Write", {"file_path": "/repo/tests/test_new.py", "content": "x"})
        harness.on_tool_call("Write", {"file_path": "/repo/pkg/new_module.py", "content": "x"})
        harness.on_tool_call("Edit", {"file_path": 3})
        message = {"type": "function_call_output", "call_id": "c", "output": "Edited"}
        harness.on_llm_call([message])
        note = message["output"]
        assert "code you changed in pkg/fields.py (check, Field" in note
        assert "tests/test_other.py (3)" in note and "tests/test_fields.py (3)" in note
        assert "test_unrelated" not in note and "notes.txt" not in note
        # mentions both names: ranks first
        assert note.index("test_fields.py") < note.index("test_other.py")
        assert note.endswith("before you finish.]")  # nothing follows: no wall-clock note
        events = [
            json.loads(line)
            for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        ]
        assert [e["tests"] for e in events if e["event"] == "test_context"] == [
            [["tests/test_fields.py", 3], ["tests/test_other.py", 3]]
        ]
        # the same tests are not listed twice; an edit that changed nothing adds nothing
        fields = "/repo/pkg/fields.py"
        harness.on_tool_call("Edit", {"file_path": fields, "old_string": "2", "new_string": "3"})
        live.exec_run(["sh", "-c", "sed -i 's/return 2/return 3/' /repo/pkg/fields.py"])
        harness.on_tool_call("Edit", {"file_path": fields, "old_string": "q", "new_string": "r"})
        again = {"type": "function_call_output", "call_id": "c", "output": "Edited"}
        harness.on_llm_call([again])
        assert "Existing tests" not in again["output"]
        # a file deleted between snapshot and model call is skipped
        harness.on_tool_call("Edit", {"file_path": fields, "old_string": "3", "new_string": "4"})
        live.exec_run(["rm", "/repo/pkg/fields.py"])
        gone = {"type": "function_call_output", "call_id": "c", "output": "Edited"}
        harness.on_llm_call([gone])
        assert "Existing tests" not in gone["output"]
        # the feature is off unless the trial config enables it
        config.write_text(
            json.dumps(
                {
                    "container": live.id,
                    "workdir": "/repo",
                    "prompt": "p",
                    "model": MODEL,
                    "trajectory": str(tmp_path / "t2.jsonl"),
                }
            )
        )
        off = coding_sea.ContainerHarness(str(config))
        off.on_tool_call("Edit", edit)
        assert off.pending_edits == []
        # a path the shell cannot take, a huge file and a binary file never raise out of the hook
        nul = {"file_path": "pkg/\x00bad.py", "old_string": "x", "new_string": "y"}
        assert harness.on_tool_call("Edit", nul) == "OK"
        assert harness.read_container_file("/repo/pkg/\x00bad.py") is None
        live.exec_run(["sh", "-c", "head -c 500000 /dev/zero | tr '\0' 'a' > /repo/pkg/big.py; "
                      "printf 'a\0b' > /repo/pkg/bin.py"])
        assert harness.read_container_file("/repo/pkg/big.py") is None
        assert harness.read_container_file("/repo/pkg/bin.py") is None
        text = harness.read_container_file("/repo/tests/test_fields.py") or ""
        assert text.startswith("from pkg.fields")
    finally:
        live.remove(force=True)


def test_shell_guards_and_finish_gate(tmp_path: Path) -> None:
    """Destructive commands are blocked, install timeouts lifted, the gate answers one finish."""
    from kiss.agents.seas import coding_sea

    def harness(**extra: object) -> coding_sea.ContainerHarness:
        config = tmp_path / f"config-{len(extra)}.json"
        config.write_text(json.dumps({
            "container": "c", "workdir": "/app/", "prompt": "p", "model": MODEL,
            "trajectory": str(tmp_path / "trajectory.jsonl"), **extra,
        }))
        return coding_sea.ContainerHarness(str(config))

    plain = harness()
    blocked = coding_sea.DESTRUCTIVE_VERDICT
    for command in ("kill -9 -1", "sleep 1; kill -1", "pkill -9 -f .", "pkill -f '.' && ls",
                    "rm -rf /app", "cd /tmp && rm -rf /app/*", "rm -rf /", "(rm -r /app)",
                    "/bin/kill -9 -1", "kill -9 -1 >/dev/null", "kill -9 -1 # stop all",
                    "kill -s KILL -1", "/usr/bin/pkill -f .", "pkill -f . 2>/dev/null",
                    "pkill --full .", "pkill -f '.*'", "rm -rf '/app'", "/bin/rm -rf /app",
                    "rm -rf /app >/dev/null", "rm -rf /app /tmp/x", "sudo rm -rf /app/",
                    "FOO=1 kill -9 -1"):
        assert plain.on_tool_call("Bash", {"command": command}) == blocked, command
    parallel = {"commands": '["ls", "kill -9 -1"]'}
    assert plain.on_tool_call("run_commands_parallel", parallel) == blocked
    assert plain.on_tool_call("run_commands_parallel", {"commands": "kill -9 -1"}) == blocked
    for command in ("kill -9 1234", "kill -1 1234", "kill -1 $(cat /tmp/pid)", "kill -1 %1",
                    "pkill -f myserver", "pkill -f python3", "rm -rf /app/build", "rm -rf /app/*.o",
                    "rm -rf /tmp/x", "rm -rf /apps", "ls /app", "printf '%s\\n' 'kill -9 -1'",
                    "echo \"pkill -f .\"", "grep -F 'rm -rf /app' README.md",
                    "echo ok # rm -rf /app"):
        assert plain.on_tool_call("Bash", {"command": command}) == "OK", command
    # installs and builds get a long timeout in place; other commands keep theirs
    args: dict[str, object] = {"command": "apt-get install -y gcc"}
    assert plain.on_tool_call("Bash", args) == "OK" and args["timeout_seconds"] == 900
    args = {"command": "pip install numpy", "timeout_seconds": 1800}
    assert plain.on_tool_call("Bash", args) == "OK" and args["timeout_seconds"] == 1800
    args = {"command": "make -j4", "timeout_seconds": "bad"}
    assert plain.on_tool_call("Bash", args) == "OK" and args["timeout_seconds"] == 900
    for command in ("cd /x && make", "sudo apt-get update", "python3 -m pip install x",
                    "DEBIAN_FRONTEND=noninteractive apt-get -y install x", "cmake .. && make"):
        args = {"command": command, "timeout_seconds": 60}
        assert plain.on_tool_call("Bash", args) == "OK" and args["timeout_seconds"] == 900, command
    for command in ("ls -la", "grep -R make .", "python3 -c \"print('cmake')\"", "npm get registry",
                    "echo make", "cat Makefile"):
        args = {"command": command, "timeout_seconds": 30}
        assert plain.on_tool_call("Bash", args) == "OK" and args["timeout_seconds"] == 30, command
    args = {"commands": '["npm install", "cargo build"]', "timeout_seconds": 120}
    assert plain.on_tool_call("run_commands_parallel", args) == "OK"
    assert args["timeout_seconds"] == 900
    # The tool's own default (1800 s) is already long enough.
    args = {"commands": '["npm install"]'}
    assert plain.on_tool_call("run_commands_parallel", args) == "OK"
    assert "timeout_seconds" not in args
    # An unparsable list is treated as one command.
    args = {"commands": "not json; make", "timeout_seconds": 60}
    assert plain.on_tool_call("run_commands_parallel", args) == "OK"
    assert args["timeout_seconds"] == 900
    args = {"commands": '{"a": 1}'}
    assert plain.on_tool_call("run_commands_parallel", args) == "OK"
    assert "timeout_seconds" not in args
    assert plain.on_tool_call("Bash", {"command": 42}) == "OK"
    assert plain.on_tool_call("Bash", {}) == "OK"
    # no gate by default: every finish passes
    assert plain.on_tool_call("finish", {"success": True}) == "OK"
    assert plain.on_tool_call("finish", {}) == "OK"
    gated = harness(finish_gate=True)
    assert gated.on_tool_call("finish", {"success": False}) == "OK"
    # an implicit (text-only) finish is vetoed once, without spending the gate
    assert gated.on_tool_call("finish", {}) == coding_sea.FINISH_GATE_VERDICT
    assert gated.on_tool_call("finish", {}) == "OK"
    assert gated.on_tool_call("finish", {"success": "true"}) == coding_sea.FINISH_GATE_VERDICT
    assert gated.on_tool_call("finish", {"success": True}) == "OK"
    assert gated.on_tool_call("finish", {}) == "OK"
    gated2 = harness(finish_gate=True, workdir="/testbed")
    assert gated2.on_tool_call("finish", {"success": True}) == coding_sea.FINISH_GATE_VERDICT
    assert gated2.on_tool_call("finish", {"success": True}) == "OK"
    assert gated2.on_tool_call("Bash", {"command": "rm -rf /testbed"}) == blocked
    assert gated2.on_tool_call("Bash", {"command": "rm -rf /app"}) == "OK"
    events = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    gate_events = [e for e in events
                   if e.get("tool") == "finish" and e["args"] == {"success": True}]
    assert [e["blocked"] for e in gate_events] == [False, False, True, False]
    # the prompt carries the environment rule and no benchmark-specific wording; the trial
    # config takes the gate from the environment
    prompt = plain.system_prompt()
    assert "Leave the environment as the task expects to find it" in prompt
    assert "byte for byte" not in prompt
    assert "checker" not in prompt.lower()
    assert "no internet" not in prompt.lower()
    assert plain.model_config() is None
    tuned = harness(model_config={"output_config": {"effort": "medium"}})
    assert tuned.model_config() == {"output_config": {"effort": "medium"}}


def test_generated_trial_sea_binds_to_a_shared_harness(tmp_path: Path) -> None:
    """The generated ``sea.py`` imports this package and exposes the harness's getters and hooks."""
    from kiss.agents.seas import coding_sea

    trial = {"container": "c", "workdir": "/app", "prompt": "p", "model": MODEL}
    sea_path = coding_sea.write_trial_sea(tmp_path / "sea-trial", trial)
    config = json.loads((tmp_path / "sea-trial" / "config.json").read_text())
    assert config["trajectory"] == str(tmp_path / "sea-trial" / "trajectory.jsonl")
    assert config["work_dir"] == str(tmp_path / "sea-trial")
    spec = importlib.util.spec_from_file_location("generated_trial_sea", sea_path)
    assert spec is not None and spec.loader is not None
    sea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sea)
    assert sea.prompt() == "p"
    assert sea.model() == MODEL
    assert sea.docker_image() == "container:c"
    assert sea.work_dir() == str(tmp_path / "sea-trial")
    assert sea.model_config() is None
    assert sea.use_web_tools() is False and sea.use_memory() is False
    assert sea.tool_call_hook()("Bash", {"command": "ls"}) == "OK"
    harness = coding_sea.ContainerHarness.shared(str(tmp_path / "sea-trial" / "config.json"))
    assert sea._harness is harness
    assert sea.llm_call_hook() == harness.on_llm_call


def _live_container(image: str, setup: str) -> Any:
    """A running container of *image* after *setup* ran in it, or ``None`` without Docker."""
    import docker

    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        return None
    live = client.containers.run(image, "sleep infinity", detach=True)
    exit_code, out = live.exec_run(["sh", "-c", setup])
    assert exit_code == 0, out
    return live


def test_shell_notes_report_survivors_and_changed_inputs(tmp_path: Path) -> None:
    """Shell results name the processes a call left running and the pre-existing files it changed.

    The notes are facts the model cannot otherwise see: a ``nohup ... &``
    survivor, a kill that missed, an input file rewritten in place.  Files
    the agent edits with Edit/Write are its own and are not reported; each
    changed file is reported once.
    """
    from kiss.agents.seas import coding_sea

    live = _live_container(
        "python:3.11-slim",
        "mkdir -p /app/sub && printf 'a,b\\n1,2\\n' > /app/data.csv && echo x > /app/sub/keep.txt "
        "&& echo y > /app/gone.txt && echo z > /app/mine.py && echo r > /app/real.txt "
        "&& ln -s real.txt /app/link.txt && echo e > /app/extra.txt && echo l > /app/last.txt",
    )
    if live is None:
        pytest.skip("Docker is not available")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "container": live.id, "workdir": "/app", "prompt": "Do the task.", "model": MODEL,
        "trajectory": str(tmp_path / "trajectory.jsonl"),
    }))
    try:
        harness = coding_sea.ContainerHarness(str(config))
        # The task statement carries the workdir listing; the model cannot be switched.
        prompt = harness.prompt()
        assert prompt.startswith("Do the task.") and "data.csv" in prompt
        assert "ls -la /app" in prompt
        assert harness.on_tool_call("set_model", {"model_name": "x"}) != "OK"
        # First model call: baseline of the workdir; no shell ran, so no notes.
        user = {"role": "user", "content": "Do the task."}
        assert harness.on_llm_call([user]) == [user]
        assert set(harness.file_baseline or {}) == {
            "/app/data.csv", "/app/sub/keep.txt", "/app/gone.txt", "/app/mine.py",
            "/app/real.txt", "/app/extra.txt", "/app/last.txt"}

        def shell(command: str) -> dict[str, Any]:
            assert harness.on_tool_call("Bash", {"command": command}) == "OK"
            live.exec_run(["sh", "-c", command])
            result = {"role": "tool", "content": "ran"}
            harness.on_llm_call([result])
            return result

        # A background process that outlives its call is reported once, with its pid.
        result = shell("cd /app && nohup sleep 300 >/dev/null 2>&1 &")
        assert "started by your last shell command(s) are still running" in result["content"]
        assert "sleep 300" in result["content"]
        # An unrelated command: no repeat.
        assert shell("echo hi")["content"] == "ran"
        # A kill that misses keeps the survivor alive: the note comes back.
        result = shell("kill 999999 2>/dev/null; true")
        assert "still running after your last shell" in result["content"]
        assert "sleep 300" in result["content"]
        # A kill that lands: nothing left to report.
        result = shell("kill $(for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline 2>/dev/null "
                       "| grep -q '^sleep 300' && echo ${d#/proc/}; done); sleep 0.2")
        assert "still running" not in result["content"] and harness.tracked_pids == set()
        # Children a reported process spawns later (a build's compiler steps) are tracked
        # silently: no note per turn, but a missed kill lists them with their parent.
        result = shell("cd /app && nohup sh -c 'sleep 1; sleep 301' >/dev/null 2>&1 &")
        assert "started by your last shell command(s)" in result["content"]
        assert "sleep 1; sleep 301" in result["content"]
        assert shell("sleep 1.5")["content"] == "ran"
        assert any(cmd.startswith("sleep 301") for _pid, cmd in harness.process_snapshot().values())
        result = shell("kill 999999 2>/dev/null; true")
        assert "still running after your last shell" in result["content"]
        assert "sleep 301" in result["content"]
        result = shell("kill $(for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline 2>/dev/null "
                       "| grep -q 'sleep 301' && echo ${d#/proc/}; done) 2>/dev/null; sleep 0.2")
        assert "still running" not in result["content"] and harness.tracked_pids == set()
        # A child born between turns is adopted at the next call's start, so it stays
        # tracked when its parent exits during that call and is orphaned to pid 1; a
        # tracked shell that exec()s the program keeps its (pid, start time) identity.
        result = shell("cd /app && nohup sh -c 'sleep 0.5; exec sleep 302' >/dev/null 2>&1 & "
                       "cd /app && nohup sh -c 'sleep 0.5; sleep 303 & sleep 2' >/dev/null 2>&1 &")
        assert "started by your last shell command(s)" in result["content"]
        assert not [cmd for _pid, cmd in harness.process_snapshot().values()
                    if cmd.startswith("sleep 303")]
        time.sleep(1.0)
        assert shell("sleep 1.5")["content"] == "ran"
        assert [cmd for _pid, cmd in harness.process_snapshot().values()
                if cmd.startswith("sleep 303")]
        result = shell("kill 999999 2>/dev/null; true")
        assert "still running after your last shell" in result["content"]
        assert "sleep 303 (parent 1)" in result["content"]
        assert "sleep 302 (parent 1)" in result["content"]
        result = shell("kill $(for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline 2>/dev/null "
                       "| grep -q 'sleep 30[23]' && echo ${d#/proc/}; done) 2>/dev/null; sleep 0.2")
        assert "still running" not in result["content"] and harness.tracked_pids == set()
        # Pre-existing files changed or deleted by the shell are named once, with the size change.
        result = shell("printf '1,2\\n3,4\\n5,6\\n' > /app/data.csv && rm /app/gone.txt")
        assert "changed pre-existing files under /app" in result["content"]
        assert "/app/data.csv (size 8 -> 12)" in result["content"]
        assert "/app/gone.txt (deleted)" in result["content"]
        assert "keep.txt" not in result["content"]
        assert "changed pre-existing" not in shell("echo again >> /app/data.csv")["content"]
        # A file the agent edits itself is its own business.
        edit = {"file_path": "mine.py", "old_string": "z", "new_string": "w"}
        assert harness.on_tool_call("Edit", edit) == "OK"
        assert "changed pre-existing" not in shell("echo w > /app/mine.py")["content"]
        # New files are not inputs; a same-content rewrite counts as modified (mtime moved).
        result = shell("echo new > /app/new.txt && touch /app/sub/keep.txt")
        assert "new.txt" not in result["content"]
        assert "/app/sub/keep.txt (modified)" in result["content"]
        # An edit through a symlink owns the target too.
        assert harness.on_tool_call("Write", {"file_path": "link.txt", "content": "q"}) == "OK"
        assert {"/app/link.txt", "/app/real.txt"} <= harness.owned_files
        assert "changed pre-existing" not in shell("echo q > /app/link.txt")["content"]
        # A note that finds no tool result waits for the next one instead of vanishing.
        assert harness.on_tool_call("Bash", {"command": "rm /app/extra.txt"}) == "OK"
        live.exec_run(["sh", "-c", "rm /app/extra.txt"])
        text_only = {"role": "assistant", "content": "thinking"}
        harness.on_llm_call([text_only])
        assert text_only == {"role": "assistant", "content": "thinking"} and harness.pending_notes
        assert "/app/extra.txt (deleted)" in shell("true")["content"] and not harness.pending_notes
        # Deleting every tracked file is still a deletion, not "tracking unavailable".
        result = shell("rm /app/data.csv /app/mine.py /app/real.txt /app/new.txt "
                       "/app/sub/keep.txt /app/last.txt")
        assert "/app/last.txt (deleted)" in result["content"] and harness.file_snapshot() == {}
        lines = (tmp_path / "trajectory.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in lines]
        assert [e["event"] for e in events].count("shell_note") == 10
    finally:
        live.remove(force=True)


def test_shell_notes_off_without_container_or_workdir(tmp_path: Path) -> None:
    """A dead container or the root workdir disables the notes without disturbing the run."""
    import docker

    from kiss.agents.seas import coding_sea

    # Only a reachable daemon can report the container as gone; when the
    # daemon itself is down the liveness check treats that as a transient
    # hiccup and lets the trial go on (see ``ContainerHarness.container_alive``).
    try:
        docker.from_env().ping()
        daemon_up = True
    except Exception:
        daemon_up = False
    for workdir in ("/app", "/"):
        config = tmp_path / f"config-{len(workdir)}.json"
        config.write_text(json.dumps({
            "container": "no-such-container", "workdir": workdir, "prompt": "p", "model": MODEL,
            "trajectory": str(tmp_path / "trajectory.jsonl"),
        }))
        harness = coding_sea.ContainerHarness(str(config))
        assert harness.prompt() == "p"
        assert harness.on_tool_call("Bash", {"command": "nohup sleep 5 &"}) == "OK"
        result = {"role": "tool", "content": "ran"}
        # the liveness check is what ends the trial; the notes must not raise first
        from kiss.core.kiss_error import BudgetExceededError

        if daemon_up:
            with pytest.raises(BudgetExceededError):
                harness.on_llm_call([result])
        else:
            assert harness.on_llm_call([result]) == [result]
        assert result["content"] == "ran" and harness.file_baseline == {}
        assert harness.file_snapshot() is None


