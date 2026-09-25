# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a ``dockerImage`` run bind-mounts the work dir into its container.

A real :class:`kiss.server.web_server.RemoteAccessServer` on a temporary
Unix-domain socket (the :class:`DaemonRunApiHarness`) runs a task whose
``docker_image`` names an IMAGE, so ``RelentlessAgent.run`` starts a
REAL container through :class:`DockerManager`.  The only replaced
boundary is the LLM: the executor's :meth:`KISSAgent.run` is swapped for
a stub that records the prompt it was handed and *calls* the tools.

Contract under test: the run's ``work_dir`` (here a checkout-like tree the
test writes on the host, standing in for Agensea's index checkout) is
visible inside the container at its host path and is the container's
working directory, so a RELATIVE ``Read`` and a RELATIVE ``Bash`` inside
the container inspect the host checkout; a ``Write`` inside the container
lands on the host (a bind mount, not a copy); the system prompt names the
work dir; a ``run_parallel`` child attached to the parent's container sees
the same mount; the mounted directory survives the container's removal.

The tests skip when no Docker daemon is reachable.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, cast

import docker
import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

IMAGE = "python:3.11-slim"
NOTE = "index/notes/hello.md"
NOTE_TEXT = "# hello from the host checkout\n"


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


class DockerWorkDirMountTest(DaemonRunApiHarness):
    """An image run must see the host work dir inside its container."""

    def setUp(self) -> None:
        super().setUp()
        self.checkout = Path(self.repo).resolve()
        (self.checkout / NOTE).parent.mkdir(parents=True)
        (self.checkout / NOTE).write_text(NOTE_TEXT)

    def _install_tool_running_stub(
        self, calls: list[dict[str, Any]], run_parallel_task: str = "",
    ) -> None:
        """Swap the executor LLM loop for a stub that inspects the checkout with the tools.

        Args:
            calls: Receives, per task-executor session, the system prompt,
                the container the ``run_commands_parallel`` tool is bound
                to, the output of a relative ``Bash`` (``pwd``, hostname and
                ``cat`` of the note), a relative ``Read`` of the note and
                the container's bind mounts as the manager reports them.
                The top-level session also writes ``from_container.txt``
                next to the note through the container ``Write`` tool.
            run_parallel_task: When non-empty, the top-level session calls
                ``run_parallel`` with this one task so a child session
                (also served by this stub) is recorded too.
        """

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            arguments = dict(kwargs.get("arguments") or {})
            if "task_description" not in arguments:
                self_agent.step_count = 1
                return "result: prior progress\n"
            tools = {getattr(t, "__name__", "?"): t for t in kwargs.get("tools") or []}
            manager = getattr(tools["run_commands_parallel"], "__self__", None)
            record: dict[str, Any] = {
                "task": arguments["task_description"],
                "system_prompt": kwargs.get("system_prompt") or "",
                "container_id": getattr(getattr(manager, "container", None), "id", None),
                "volumes": dict(getattr(manager, "volumes", {}) or {}),
                "bash": tools["Bash"](f"pwd; cat /etc/hostname; cat {NOTE}", "inspect"),
                "read": tools["Read"](NOTE),
            }
            if run_parallel_task and run_parallel_task not in record["task"]:
                record["run_parallel"] = tools["run_parallel"](f'["{run_parallel_task}"]', "1")
            if "run_parallel" in record or not run_parallel_task:
                tools["Write"]("index/notes/from_container.txt", "written in the container\n")
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

    def _run(self, task: str, timeout: int) -> Any:
        return sorcar.run(
            task,
            work_dir=self.repo,
            use_worktree=False,
            use_web_tools=False,
            docker_image=IMAGE,
            sock_path=self.sock_path,
            timeout=timeout,
        )

    def _assert_inspects_the_checkout(self, call: dict[str, Any]) -> None:
        """The session's relative Bash and Read ran in the container, on the host checkout."""
        assert call["container_id"], call
        cwd, hostname, *note = call["bash"].splitlines()
        assert cwd == str(self.checkout), call["bash"]
        assert hostname == call["container_id"][:12], call["bash"]
        assert hostname != socket.gethostname()[:12]
        assert "\n".join(note).strip() == NOTE_TEXT.strip(), call["bash"]
        assert NOTE_TEXT.strip() in call["read"], call["read"]
        assert call["volumes"].get(str(self.checkout)) == str(self.checkout), call["volumes"]

    def test_relative_read_and_bash_inspect_the_mounted_checkout(self) -> None:
        """Inside the image's container, ``Read(NOTE)`` and ``cat NOTE`` show the host file.

        The container starts in the work dir (``pwd`` prints the host path),
        the hostname proves the command ran in the container, the prompt
        names the work dir, and a container ``Write`` reaches the host.
        The container is gone after the task but the checkout is intact.
        """
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls)
        result = self._run("inspect the checkout", timeout=180)
        assert result.success is True, result.text
        assert len(calls) == 1, calls
        call = calls[0]
        self._assert_inspects_the_checkout(call)
        assert f"- Work dir: {self.checkout}\n" in call["system_prompt"]
        written = self.checkout / "index/notes/from_container.txt"
        assert written.read_text() == "written in the container\n"
        assert (self.checkout / NOTE).read_text() == NOTE_TEXT
        containers = docker.from_env().containers.list(all=True)
        assert call["container_id"] not in {c.id for c in containers}

    def test_run_parallel_child_sees_the_same_mount(self) -> None:
        """A child attached to the parent's container inspects the checkout the same way.

        The child's ``DockerManager`` attaches with ``container:<id>``;
        it inherits the container's working directory (so relative paths
        still resolve in the checkout) and reports the parent's bind mount
        in ``volumes``, which keeps the work dir named in its prompt.
        """
        calls: list[dict[str, Any]] = []
        self._install_tool_running_stub(calls, run_parallel_task="child inspects")
        result = self._run("parent inspects", timeout=240)
        assert result.success is True, result.text
        parents = [c for c in calls if "run_parallel" in c]
        children = [c for c in calls if "run_parallel" not in c]
        assert len(parents) == 1 and len(children) == 1, calls
        parent, child = parents[0], children[0]
        assert "child inspects" in child["task"], child
        assert "agent ok" in parent["run_parallel"], parent["run_parallel"]
        assert child["container_id"] == parent["container_id"]
        self._assert_inspects_the_checkout(child)
        assert f"- Work dir: {self.checkout}\n" in child["system_prompt"]
