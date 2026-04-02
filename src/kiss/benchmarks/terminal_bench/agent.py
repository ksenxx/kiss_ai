# Author: Koushik Sen (ksen@berkeley.edu)

"""Harbor agent adapter that delegates to KISS Sorcar.

This module implements a Harbor-compatible agent (BaseAgent subclass)
that translates harbor's exec interface into sorcar invocations.

Usage with harbor CLI:
    harbor run --dataset terminal-bench@2.0 \
        --agent-import-path kiss.benchmarks.terminal_bench.agent:SorcarHarborAgent \
        --model anthropic/claude-opus-4-6 \
        --n-concurrent 4
"""

from __future__ import annotations

import os
import shlex

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


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
        return "0.2.75"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install sorcar inside the harbor container.

        Installs uv, then kiss-agent-framework as a uv tool (which
        manages its own Python), and downloads SYSTEM.md to the path
        where the installed package expects it.

        Args:
            environment: The harbor execution environment.
        """
        await environment.exec(
            "apt-get update && apt-get install -y curl"
            " && curl -LsSf https://astral.sh/uv/install.sh | sh"
            ' && export PATH="/root/.local/bin:$PATH"'
            " && uv tool install kiss-agent-framework"
            " && TOOL_PY=/root/.local/share/uv/tools"
            "/kiss-agent-framework/bin/python3"
            ' && SYSTEM_DIR=$($TOOL_PY -c "'
            "from pathlib import Path; import kiss;"
            'print(Path(kiss.__file__).parent.parent.parent)")'
            " && curl -fsSL -o ${SYSTEM_DIR}/SYSTEM.md"
            " https://raw.githubusercontent.com/ksenxx/kiss_ai/main/SYSTEM.md",
            user="root",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run sorcar with the task instruction inside the container.

        The agent executes sorcar which modifies the environment directly.
        Harbor evaluates the result by inspecting the environment state
        after this method returns.

        Args:
            instruction: Natural language task description from harbor.
            environment: The harbor execution environment.
            context: Agent context for storing token/cost metadata.
        """
        escaped = shlex.quote(instruction)
        model_flag = f"-m {self.model_name}" if self.model_name else ""
        env = {
            k: v
            for k, v in {
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            }.items()
            if v
        }
        result = await environment.exec(
            f'export PATH="/root/.local/bin:$PATH"'
            f" && sorcar -t {escaped} -w /app --headless true -n {model_flag}",
            user="root",
            env=env,
        )
        context.metadata = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.return_code,
        }
