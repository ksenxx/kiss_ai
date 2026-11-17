# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Harbor agent that runs the plain (vanilla) KISS Sorcar agent on a Terminal-Bench trial.

Unlike :class:`kiss.benchmarks.terminal_bench.harbor_agent.SeaHarborAgent`,
which drives the coding SEA (finish gate, reviewer, reading check), this agent
sends the task instruction unchanged to the default Sorcar agent of the
benchmark daemon (:mod:`kiss.benchmarks.terminal_bench.daemon`), with its file
and shell tools attached to the trial's Docker container
(``docker_image="container:<id>"``).

The agent runs without the browser and web-search tools, persistent memory,
MCP servers, the ``decide`` side model, questions to the user and pre-run task
classification (:data:`TOOL_PROFILE`, ``use_web_tools=False``,
``use_memory=False``, ``auto_classify=False``).  Harbor's per-task
``[agent] timeout_sec`` is the wall-clock limit; the USD cap per trial is the
``budget_usd`` agent option.

Usage (from the repo root, with the benchmark daemon running)::

    uv run harbor run -d terminal-bench@2.0 \\
        -a kiss.benchmarks.terminal_bench.vanilla_agent:VanillaSorcarAgent \\
        -m claude-fable-5-high --ak home=~/.kiss-tb2-vanilla --ak budget_usd=50 \\
        -k 5 -n 20 --cpus limit --memory limit -o jobs -y
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent  # type: ignore[import-untyped]
from harbor.agents.options import AgentOptions  # type: ignore[import-untyped]
from harbor.environments.base import BaseEnvironment  # type: ignore[import-untyped]
from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]

from kiss.agents.sorcar.docker_manager import ATTACH_PREFIX
from kiss.benchmarks.terminal_bench.daemon import DEFAULT_HOME, ENDPOINT_FILE
from kiss.benchmarks.terminal_bench.harbor_agent import main_container

AGENT_VERSION = "2026.11.14"

#: Built-in tool groups of the vanilla agent: commands, file edits, sub-agents
#: (pinned to the trial's model by the daemon) and run control.  Left out:
#: ``browser`` (web), ``memory``, ``mcp`` (remote servers), ``decide`` (a
#: second model), ``user`` (nobody answers) and ``skills`` (the host's skills).
TOOL_PROFILE = "shell+edit+agents+control"


class VanillaOptions(AgentOptions):
    """``--ak`` options of :class:`VanillaSorcarAgent`."""

    #: USD cap of one trial (the daemon ends the task when it is spent).
    budget_usd: float = 50.0
    #: ``KISS_HOME`` of the benchmark daemon (its endpoint file lives there).
    home: str = str(DEFAULT_HOME)


class VanillaSorcarAgent(BaseAgent):
    """Bridge from Harbor's agent protocol to one plain Sorcar daemon task."""

    options_model = VanillaOptions

    @staticmethod
    def name() -> str:
        """Harbor agent name."""
        return "kiss-sorcar-vanilla"

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
        """Run the vanilla Sorcar agent on the trial container and record its spend.

        Harbor's timeout cancels this coroutine; the daemon task is then
        stopped and its spend so far recorded before the cancellation
        propagates.

        Args:
            instruction: The task instruction (``instruction.md``), sent unchanged.
            environment: Harbor's running Docker environment.
            context: Receives the cost and the run's metrics.
        """
        if not self.model_name:
            raise ValueError("pass the KISS model name with -m/--model")
        options: VanillaOptions = self.options  # type: ignore[assignment]
        container = await main_container(environment)
        endpoint = Path(options.home).expanduser() / ENDPOINT_FILE
        cancel = threading.Event()
        job = asyncio.ensure_future(asyncio.to_thread(
            run_vanilla_trial, instruction, container, self.model_name,
            options.budget_usd, endpoint, cancel,
        ))
        try:
            metrics = await asyncio.shield(job)
        except asyncio.CancelledError:
            cancel.set()
            self.record(context, await job)
            raise
        self.record(context, metrics)

    def record(self, context: AgentContext, metrics: dict[str, Any]) -> None:
        """Report the trial's spend to Harbor and write ``result.json`` in the agent log dir.

        Args:
            context: Harbor's agent context of the trial.
            metrics: The :func:`run_vanilla_trial` metrics.
        """
        context.cost_usd = metrics["cost_usd"]
        context.metadata = metrics
        (Path(self.logs_dir) / "result.json").write_text(json.dumps(metrics, indent=1))


def run_vanilla_trial(
    instruction: str,
    container: str,
    model: str,
    budget_usd: float,
    endpoint_file: Path,
    cancel: threading.Event,
) -> dict[str, Any]:
    """Run the instruction as one plain Sorcar task in *container*; return its metrics.

    Args:
        instruction: The task instruction.
        container: Docker id of the trial's ``main`` container.
        model: KISS model name.
        budget_usd: USD cap of the task.
        endpoint_file: The benchmark daemon's ``sorcar-local.json``.
        cancel: Set by the caller to stop the task (Harbor's timeout).

    Returns:
        ``model, container, agent_success, cost_usd, tokens, steps, task_id,
        final_text, error, seconds``.  The spend is what the daemon reported,
        also for a run that was cancelled or whose connection was lost.
    """
    from kiss.agents.sorcar import daemon_client

    started = time.time()
    metrics: dict[str, Any] = {"model": model, "container": container, "error": "",
                               "agent_success": False, "final_text": ""}
    try:
        result = daemon_client.run(
            instruction,
            model=model,
            max_budget=budget_usd,
            docker_image=ATTACH_PREFIX + container,
            tool_profile=TOOL_PROFILE,
            use_web_tools=False,
            use_memory=False,
            auto_classify=False,
            timeout=None,
            endpoint_file=endpoint_file,
            cancel=cancel,
        )
        metrics["agent_success"] = bool(result.success)
        metrics["final_text"] = (result.text or "")[:4000]
    except daemon_client.CancelledError as exc:  # Harbor's timeout: keep the spend so far
        result = exc.result
        metrics["error"] = f"CancelledError: {exc}"[:2000]
    except Exception as exc:  # daemon failure: recorded, not fatal
        # An abort after the run was sent carries the spend reported so far
        # (``task_result``); a failure before it (no daemon) spent nothing.
        result = getattr(exc, "task_result", None) or daemon_client.TaskResult(
            text="", success=False, cost=0.0, tokens=0, steps=0)
        metrics["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    metrics.update({"cost_usd": float(result.cost), "tokens": int(result.tokens),
                    "steps": int(result.steps), "task_id": result.task_id,
                    "seconds": round(time.time() - started, 1)})
    return metrics
