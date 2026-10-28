# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Harbor agent that runs the KISS Sorcar coding SEA against a Terminal-Bench trial.

Harbor starts the task container, calls :meth:`SeaHarborAgent.run` with the
task instruction, then runs the official verifier.  ``run`` resolves the
container id (its hostname), writes the per-trial SEA
(:func:`kiss.agents.seas.coding.coding_sea.write_trial_sea`) into Harbor's
agent log directory and drives the private benchmark daemon
(:mod:`kiss.benchmarks.terminal_bench.daemon`) through
:func:`kiss.agents.sorcar.daemon_client.run`.

The agent gets no browser, web-search or persistent-memory tools and no
pre-run task classification (``use_web_tools=False``, ``use_memory=False``,
``auto_classify=False`` on every run, on top of the daemon's persisted
settings).  Harbor's ``[agent] timeout_sec`` (set per task by the dataset)
is the wall-clock limit: when Harbor cancels ``run`` the daemon task is
stopped.  The USD cap per trial is the ``budget_usd`` agent option.

A trial whose environment is not a local Docker container (Harbor's
``modal`` environment, used for the GPU tasks) gets an exec bridge
(:class:`kiss.benchmarks.terminal_bench.exec_bridge_server.ExecBridgeServer`)
that serves ``environment.exec`` to the daemon over loopback HTTP; the SEA's
tools then run on the remote sandbox exactly as they would in a container.

Usage (from the repo root, with the benchmark daemon running)::

    uv run harbor run -d terminal-bench/terminal-bench \\
        -a kiss.benchmarks.terminal_bench.harbor_agent:SeaHarborAgent \\
        -m claude-opus-5-5 --ak budget_usd=100 --ak reasoning_effort=max \\
        -k 5 -n 16 -o jobs -y
"""

from __future__ import annotations

import asyncio
import json
import shlex
import threading
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent  # type: ignore[import-untyped]
from harbor.agents.capabilities import AgentCapabilities  # type: ignore[import-untyped]
from harbor.agents.options import AgentOptions  # type: ignore[import-untyped]
from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]
from harbor.environments.docker.docker import (  # type: ignore[import-untyped]
    _sanitize_docker_compose_project_name,
)
from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]

from kiss.agents.sorcar.exec_bridge import BRIDGE_PREFIX
from kiss.benchmarks.terminal_bench.daemon import DEFAULT_HOME, ENDPOINT_FILE
from kiss.benchmarks.terminal_bench.exec_bridge_server import ExecBridgeServer
from kiss.benchmarks.terminal_bench.trial import run_trial

AGENT_VERSION = "2026.10.28"

#: ``reviewer`` option values: the trial's own model, and no review at all.
SAME_MODEL = "same"
NO_REVIEW = "none"


class SeaOptions(AgentOptions):
    """``--ak`` options of :class:`SeaHarborAgent`."""

    #: USD cap of one trial (the daemon ends the task when it is spent).
    budget_usd: float = 100.0
    #: ``KISS_HOME`` of the benchmark daemon (its endpoint file lives there).
    home: str = str(DEFAULT_HOME)
    #: Gate ``finish(success=True)`` on evidence (the coding SEA's ``finish_gate``:
    #: the agent's own verifier, the blind verifier, no hedged summary, a review).
    finish_gate: bool = True
    #: Model of the gate's read-only reviews of the work: ``same`` (the
    #: trial's model), a model name, or ``none``.
    reviewer: str = SAME_MODEL
    #: Model that writes the gate's independent check from the task statement
    #: alone: ``same``, a model name, or ``none``.
    blind_verifier: str = SAME_MODEL
    #: Reasoning effort sent with every model request (``low`` .. ``max`` for
    #: Claude's ``output_config.effort``, ``minimal`` .. ``xhigh`` for OpenAI
    #: models); ``""`` sends the level of the model's catalog entry (``high``
    #: for a bare base name such as ``claude-opus-5-5``, the suffix level
    #: for a ``-{level}`` alias, nothing for a model without one).
    reasoning_effort: str = ""


class SeaHarborAgent(BaseAgent):
    """Bridge from Harbor's agent protocol to a KISS Sorcar coding-SEA run."""

    options_model = SeaOptions
    #: Task-supplied Agent Skills and MCP servers are listed in the instruction
    #: (:func:`skills_note`, :func:`mcp_note`), which is what Harbor's capability
    #: definitions ask for ("discoverable through its native configuration or prompt").
    capabilities = AgentCapabilities(skills=True, mcp_servers=True)

    @staticmethod
    def name() -> str:
        """Harbor agent name."""
        return "kiss-sorcar"

    def version(self) -> str | None:
        """Harbor agent version."""
        return AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the daemon attaches the agent's tools to the container."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the SEA on the trial container and record cost and tokens.

        Args:
            instruction: The task instruction (``instruction.md``).
            environment: Harbor's running Docker environment.
            context: Receives cost/token figures and trial metadata.
        """
        if not self.model_name:
            raise ValueError("pass the KISS model name with -m/--model")
        options: SeaOptions = self.options  # type: ignore[assignment]
        bridge: ExecBridgeServer | None = None
        if environment.type() == "docker":
            container = await main_container(environment)
        else:
            bridge = ExecBridgeServer(environment, asyncio.get_running_loop())
            container = BRIDGE_PREFIX + bridge.url
        try:
            metrics = await self.run_sea(instruction, environment, context, container, options)
        finally:
            # Every exit after the bridge started (a failed setup exec, Harbor's
            # cancellation, the normal end) stops its server thread.
            if bridge is not None:
                await asyncio.to_thread(bridge.close)
        self.record(context, metrics)

    async def run_sea(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
        container: str,
        options: SeaOptions,
    ) -> dict[str, Any]:
        """Build the trial's SEA config and run it on the daemon; return its metrics.

        Args:
            instruction: The task instruction (``instruction.md``).
            environment: Harbor's running environment (for ``pwd`` and the skills note).
            context: Receives the spend when Harbor cancels the run.
            container: The container name, or the ``exec:<url>`` bridge address.
            options: The agent's parsed ``--ak`` options.
        """
        workdir = await environment.exec("pwd")
        cwd = (workdir.stdout or "").strip() or "/app"
        prompt = (instruction + await skills_note(environment, self.skills_dir)
                  + mcp_note(self.mcp_servers))
        aliases = {SAME_MODEL: self.model_name, NO_REVIEW: "", "": ""}
        config = {
            "container": container, "workdir": cwd, "prompt": prompt,
            "model": self.model_name, "max_budget": options.budget_usd,
            "finish_gate": options.finish_gate,
            "reviewer": aliases.get(options.reviewer, options.reviewer),
            "blind_verifier": aliases.get(options.blind_verifier, options.blind_verifier),
            "model_config": ({"reasoning_effort": options.reasoning_effort}
                             if options.reasoning_effort else None),
        }
        cancel = threading.Event()
        job = asyncio.ensure_future(asyncio.to_thread(
            run_trial, Path(self.logs_dir), config,
            Path(options.home).expanduser() / ENDPOINT_FILE, cancel,
        ))
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            # Harbor's agent timeout: stop the daemon task and wait for the client
            # to confirm it (bounded by its stop grace), so the spend so far is
            # recorded before Harbor moves on to the verifier.
            cancel.set()
            self.record(context, await asyncio.shield(job))
            raise

    def record(self, context: AgentContext, metrics: dict[str, Any]) -> None:
        """Report the trial's spend to Harbor and write ``result.json`` next to the SEA.

        Only the cost is reported as a figure: the daemon's token count is a
        total, which Harbor's input/output token fields do not represent, so it
        stays in the metadata.

        Args:
            context: Harbor's agent context of the trial.
            metrics: The :func:`run_trial` metrics.
        """
        context.cost_usd = metrics["cost_usd"]
        context.metadata = metrics
        (Path(self.logs_dir) / "result.json").write_text(json.dumps(metrics, indent=1, default=str))


async def main_container(environment: BaseEnvironment) -> str:
    """The Docker container id of the trial's ``main`` service.

    Found through the Compose labels of the environment's project rather than
    the container's hostname: a task whose ``main`` service runs with
    ``network_mode: "service:<sidecar>"`` shares the sidecar's hostname, so
    ``cat /etc/hostname`` would name the sidecar.  Falls back to the hostname
    when the labels find nothing (a non-Compose environment).

    Args:
        environment: Harbor's running environment.
    """
    project = _sanitize_docker_compose_project_name(environment.session_id)
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-q", "--no-trunc",
        "--filter", f"label=com.docker.compose.project={project}",
        "--filter", "label=com.docker.compose.service=main",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    ids = out.decode().split()
    if len(ids) == 1:
        return ids[0]
    hostname = await environment.exec("cat /etc/hostname")
    return (hostname.stdout or "").strip()


def mcp_note(servers: list[Any]) -> str:
    """A note listing the task's MCP servers (name, transport, endpoint), or ``''``.

    The servers live in the task's Compose network, so the note gives their
    endpoints as seen from the trial container; the agent's tools carry no MCP
    client, so the model talks to them with a client library or plain HTTP
    from the container.

    Args:
        servers: Harbor's ``MCPServerConfig`` entries for the task.
    """
    if not servers:
        return ""
    lines = []
    for server in servers:
        endpoint = server.url or " ".join([server.command or "", *server.args]).strip()
        lines.append(f"- {server.name} (transport {server.transport}): {endpoint}")
    return ("\n\nMCP servers provided by the task, reachable from this container (no MCP "
            "client is built into your tools; use a client library or HTTP from the "
            "container):\n" + "\n".join(lines))


async def skills_note(environment: BaseEnvironment, skills_dir: str | None) -> str:
    """A note listing the task's Agent Skills (``<skills_dir>/*/SKILL.md``), or ``''``.

    Harbor hands the task's ``skills_dir`` to agents that declare the
    ``skills`` capability; the note makes the skills discoverable through the
    prompt (the model reads a SKILL.md with its file tools to use it), which is
    what the spec asks of an agent without a native skills directory.

    Args:
        environment: Harbor's running environment.
        skills_dir: Absolute path of the skills directory in the environment.
    """
    if not skills_dir:
        return ""
    found = await environment.exec(
        f"find {shlex.quote(skills_dir)} -mindepth 2 -maxdepth 2 -name SKILL.md -type f | sort",
        timeout_sec=10,
    )
    paths = (found.stdout or "").split()
    if found.return_code != 0 or not paths:
        return ""
    return ("\n\nAgent skills are available in this environment; read the SKILL.md of a "
            "relevant skill before using it:\n" + "\n".join(f"- {p}" for p in paths))
