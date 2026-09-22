# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``docker_image`` run parameter on the daemon.

A real :class:`kiss.server.web_server.RemoteAccessServer` on a temporary
Unix-domain socket (the :class:`DaemonRunApiHarness`) runs tasks whose
tools attach to a REAL Docker container started by the test.  The only
replaced boundary is the LLM: the executor's :meth:`KISSAgent.run` is
swapped for a stub that records the tools it was handed and *calls* them,
so the daemon's whole pipeline — wire ``dockerImage`` field →
``apply_agent_overrides`` (an SEA's ``docker_image()`` getter) →
``task_runner`` → ``SorcarAgent.run(docker_image=...)`` →
``DockerManager("container:<id>")`` → container-backed ``Bash`` /
``run_commands_parallel`` / ``Read`` / ``Write`` — executes for real.

Contract under test: ``container:<name-or-id>`` attaches the run's shell
and file tools to an existing container (commands run in the container's
own working directory, the container is left running afterwards);
``run_parallel`` children act in the parent's container; an SEA's
``docker_image()`` getter has the same effect as the parameter; a
malformed wire value runs the tools on the host.

The tests skip when no Docker daemon is reachable.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import docker
import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

IMAGE = "python:3.11-slim"


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = [
    requires_unix_sockets,
    pytest.mark.slow,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon is not running"),
]


class DockerImageRunParamTest(DaemonRunApiHarness):
    """``docker_image`` must put the executor's tools inside the named container."""

    def setUp(self) -> None:
        super().setUp()
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            IMAGE, command="sleep infinity", detach=True, working_dir="/srv",
            name=f"kiss-attach-test-{uuid.uuid4().hex[:8]}",
        )

    def tearDown(self) -> None:
        try:
            self.container.remove(force=True)
        finally:
            super().tearDown()

    def _install_tool_running_stub(
        self, calls: list[dict[str, Any]], run_parallel_task: str = "",
    ) -> None:
        """Swap the executor LLM loop for a stub that exercises the tools it gets.

        Args:
            calls: Receives, per task-executor session, the tool names,
                the ``Bash`` output of ``cat /etc/hostname; pwd``, the
                ``Read`` output of a file the stub wrote with ``Write``,
                and the container id the ``run_commands_parallel`` tool is
                bound to (``None`` on the host).
            run_parallel_task: When non-empty, the top-level session calls
                ``run_parallel`` with this one task, so a child session
                (also served by this stub) is recorded too.
        """

        host_dir = self.repo

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            arguments = dict(kwargs.get("arguments") or {})
            if "task_description" not in arguments:
                self_agent.step_count = 1
                return "result: prior progress\n"
            tools = {getattr(t, "__name__", "?"): t for t in kwargs.get("tools") or []}
            record: dict[str, Any] = {
                "tool_names": list(tools), "task": arguments["task_description"],
            }
            manager = getattr(tools["run_commands_parallel"], "__self__", None)
            record["container_id"] = getattr(getattr(manager, "container", None), "id", None)
            record["bash"] = tools["Bash"]("cat /etc/hostname; pwd", "where am I")
            note = "/srv/note.txt" if record["container_id"] else f"{host_dir}/note.txt"
            tools["Write"](note, "written by the stub\n")
            record["read"] = tools["Read"](note)
            if run_parallel_task and run_parallel_task not in record["task"]:
                record["run_parallel"] = tools["run_parallel"](f'["{run_parallel_task}"]', "1")
            calls.append(record)
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(raw, type="result", step_count=1, total_tokens=1, cost="$0.0001")
            return raw

        cast(Any, KISSAgent).run = stub_run

    @staticmethod
    def _parent_and_child(calls: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The recorded parent session (the one that fanned out) and its child."""
        parents = [c for c in calls if "run_parallel" in c]
        children = [c for c in calls if "run_parallel" not in c]
        assert len(parents) == 1 and len(children) == 1, calls
        return parents[0], children[0]

    def test_attach_runs_tools_in_the_named_container(self) -> None:
        """``container:<id>`` attaches Bash/Read/Write to the running container.

        The container's hostname is its id prefix and its working
        directory ``/srv`` (set at ``docker run``), so the Bash output
        proves both where the command ran and which directory it ran in.
        ``bash_job`` and the memory tools are absent (no job registry, no
        memory in a Docker run) and the container survives the task.
        """
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls)
        result = sorcar.run(
            "attach task",
            work_dir=self.repo,
            use_worktree=False,
            use_web_tools=False,
            docker_image=f"container:{self.container.id}",
            sock_path=self.sock_path,
            timeout=120,
        )
        assert result.success is True, result.text
        assert len(calls) == 1, calls
        call = calls[0]
        assert self.container.id[:12] in call["bash"], call["bash"]
        assert "/srv" in call["bash"].splitlines()[1], call["bash"]
        assert "written by the stub" in call["read"], call["read"]
        assert call["container_id"] == self.container.id
        assert "Bash" in call["tool_names"] and "run_parallel" in call["tool_names"]
        assert "bash_job" not in call["tool_names"]
        assert not any(name.startswith("memory_") for name in call["tool_names"])
        self.container.reload()
        assert self.container.status == "running"

    def test_run_parallel_children_share_the_parent_container(self) -> None:
        """A ``run_parallel`` child attaches to the parent's container, not the host."""
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls, run_parallel_task="child task")
        result = sorcar.run(
            "parent task",
            work_dir=self.repo,
            use_worktree=False,
            use_web_tools=False,
            docker_image=f"container:{self.container.id}",
            sock_path=self.sock_path,
            timeout=180,
        )
        assert result.success is True, result.text
        parent, child = self._parent_and_child(calls)
        assert "child task" in child["task"], child
        assert "agent ok" in parent["run_parallel"], parent["run_parallel"]
        assert child["container_id"] == self.container.id
        assert self.container.id[:12] in child["bash"], child["bash"]

    def test_sea_docker_image_getter_attaches(self) -> None:
        """An agent script's ``docker_image()`` getter selects the container."""
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls)
        script = self._write_py(
            "attach_sea.py",
            f'''
            """SEA attaching its tools to a test container."""


            def docker_image() -> str:
                """The container the run's tools execute in."""
                return "container:{self.container.id}"


            def use_worktree() -> bool:
                """No worktree."""
                return False
            ''',
        )
        result = sorcar.run(
            "sea attach task",
            work_dir=self.repo,
            use_web_tools=False,
            extension_agent_path=script,
            sock_path=self.sock_path,
            timeout=120,
        )
        assert result.success is True, result.text
        assert calls[0]["container_id"] == self.container.id

    def test_sea_docker_image_getter_must_return_a_string(self) -> None:
        """A non-string ``docker_image()`` fails the task with the loader's diagnostic."""
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls)
        script = self._write_py(
            "bad_attach_sea.py",
            '''
            """SEA with a mistyped docker_image getter."""


            def docker_image() -> int:
                """Wrong type."""
                return 42
            ''',
        )
        result = sorcar.run(
            "bad sea task",
            work_dir=self.repo,
            use_worktree=False,
            use_web_tools=False,
            extension_agent_path=script,
            sock_path=self.sock_path,
            timeout=120,
        )
        assert result.success is False
        assert "docker_image()" in result.text and "a string" in result.text, result.text
        assert calls == []

    def test_malformed_wire_value_runs_on_the_host(self) -> None:
        """A non-string ``dockerImage`` wire field means "host": ``bash_job`` is offered."""
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls)
        self._raw_daemon_run({"dockerImage": 123})
        assert len(calls) == 1, calls
        assert "bash_job" in calls[0]["tool_names"]
        assert calls[0]["container_id"] is None
        assert self.container.id[:12] not in calls[0]["bash"]
