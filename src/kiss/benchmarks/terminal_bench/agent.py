# Author: Koushik Sen (ksen@berkeley.edu)

"""Harbor agent adapter that delegates to KISS Sorcar.

This module implements a Harbor-compatible agent (BaseAgent subclass)
that translates harbor's exec interface into sorcar invocations.

Usage with harbor CLI:
    harbor run --dataset terminal-bench@2.0 \
        --agent-import-path kiss.benchmarks.terminal_bench.agent:SorcarHarborAgent \
        --model claude-opus-4-6 \
        --n-concurrent 8
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from kiss._version import __version__

# Phrases that uniquely identify tasks known to always timeout.
# These tasks have failed 0/6 across multiple evaluation runs due to
# fundamental constraints (compilation too slow, needs GUI, needs GPU,
# expert time estimate measured in hours/days).
_SKIP_PHRASES: tuple[str, ...] = (
    # compile-compcert: CompCert build needs hours with 2 CPUs/4GB
    "CompCert C verified compiler",
    # install-windows-3.11: requires VNC GUI interaction
    "Windows 3.11 for Workgroups",
    # extract-moves-from-video: YouTube download + video OCR
    "video of someone playing zork",
    # gpt2-codegolf: <5000 byte C for GPT-2 inference, expert ~40h
    "gpt-2 weights stored as a TF .ckpt",
    # train-fasttext: model training too slow for timeout
    "train a fasttext model on the yelp data",
    # caffe-cifar-10: Caffe source build + training, always times out
    "BVLC Caffe deep learning framework",
    # make-doom-for-mips: MIPS cross-compilation, expert ~8h
    "build the doomgeneric_mips ELF",
    # mteb-leaderboard: hardcoded expected answer from stale snapshot;
    # leaderboard changes, agent always finds different top models
    "Scandinavian MTEB leaderboard",
    # fix-ocaml-gc: OCaml GC debugging, consistently times out (1+ hour)
    "OCaml garbage collector",
)

logger = logging.getLogger(__name__)


_wheel_lock = threading.Lock()
_wheel_path: Path | None = None


def _get_wheel() -> Path:
    """Build a wheel from local source, cached for the process lifetime.

    Returns:
        Path to the built .whl file.
    """
    global _wheel_path
    with _wheel_lock:
        if _wheel_path is not None and _wheel_path.exists():
            return _wheel_path
        import kiss

        project_root = Path(kiss.__file__).resolve().parent.parent.parent
        dist = project_root / "dist"
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist)],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
        wheels = sorted(dist.glob("kiss_agent_framework-*.whl"))
        _wheel_path = wheels[-1]
        return _wheel_path


# API key environment variables to forward into the container.
_API_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
)


class SorcarHarborAgent(BaseAgent):
    """Harbor-compatible agent that uses KISS Sorcar as the backend.

    Receives task instructions from harbor, installs and invokes the
    sorcar CLI inside the container, and captures output.
    """

    @staticmethod
    def name() -> str:
        """Return the agent's name."""
        return "sorcar"

    def version(self) -> str | None:
        """Return the agent version string."""
        return __version__

    async def _exec_check(
        self,
        environment: BaseEnvironment,
        command: str,
        description: str,
    ) -> bool:
        """Run a command and log on failure.

        Args:
            environment: The harbor execution environment.
            command: Shell command to run.
            description: Human-readable label for log messages.

        Returns:
            True if the command succeeded (return_code == 0).
        """
        result = await environment.exec(command, user="root")
        if result.return_code != 0:
            logger.error(
                "%s failed (rc=%d): %s",
                description,
                result.return_code,
                (result.stderr or result.stdout or "")[:500],
            )
            return False
        return True

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install sorcar inside the harbor container.

        Installs uv, then kiss-agent-framework as a uv tool (which
        manages its own Python).  Each step is run separately so
        failures are logged clearly and do not silently abort the
        chain.

        Args:
            environment: The harbor execution environment.
        """
        # Step 1: Install system deps + uv
        if not await self._exec_check(
            environment,
            "apt-get update -qq && apt-get install -y -qq curl"
            " && curl -LsSf https://astral.sh/uv/install.sh | sh",
            "install uv",
        ):
            return

        # Step 2: Build a wheel from local source (avoids stale PyPI package),
        # upload it to the container, and install it.  Pin to Python 3.13
        # because transitive deps (e.g. pyiceberg) lack pre-built wheels for
        # 3.14, and minimal Docker images don't have a C compiler for source
        # builds.
        wheel = _get_wheel()
        container_wheel = f"/tmp/{wheel.name}"
        await environment.upload_file(wheel, container_wheel)
        await self._exec_check(
            environment,
            'export PATH="/root/.local/bin:$PATH"'
            f" && uv tool install --python 3.13 {container_wheel}",
            "install kiss-agent-framework",
        )

    async def _run_sorcar(
        self,
        environment: BaseEnvironment,
        task: str,
        env: dict[str, str],
        model_flag: str,
    ) -> ExecResult:
        """Run a single sorcar invocation inside the container.

        Args:
            environment: The harbor execution environment.
            task: Task instruction text.
            env: API key environment variables.
            model_flag: Model flag string for sorcar CLI.

        Returns:
            The exec result object.
        """
        escaped = shlex.quote(task)
        return await environment.exec(
            f'export PATH="/root/.local/bin:$PATH"'
            f" && sorcar -t {escaped} -w /app --no-web -n {model_flag}",
            user="root",
            env=env,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run sorcar with the task instruction inside the container.

        Args:
            instruction: Natural language task description from harbor.
            environment: The harbor execution environment.
            context: Agent context for storing token/cost metadata.
        """
        for phrase in _SKIP_PHRASES:
            if phrase in instruction:
                logger.info("Skipping impossible task (matched %r)", phrase)
                context.metadata = {"skipped": True, "reason": phrase}
                return

        # Verify sorcar is installed before attempting to run it.
        check = await environment.exec(
            'export PATH="/root/.local/bin:$PATH" && which sorcar',
            user="root",
        )
        if check.return_code != 0:
            logger.error("sorcar not found — setup likely failed")
            context.metadata = {"error": "sorcar not installed"}
            return

        model_flag = f"-m {self.model_name}" if self.model_name else ""
        env = {k: v for k in _API_KEY_VARS if (v := os.environ.get(k, ""))}

        result = await self._run_sorcar(environment, instruction, env, model_flag)

        context.metadata = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.return_code,
        }
