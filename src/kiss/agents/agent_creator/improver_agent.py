# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""ImproverAgent - Improves existing agent code using a configurable coding agent.

The ImproverAgent takes an existing agent's source code folder and a report,
copies the folder to a new location, and uses a coding agent (Claude, Gemini,
or OpenAI Codex) to improve the code to reduce token usage and execution time.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anyio

import kiss.agents.agent_creator.config  # noqa: F401
from kiss.core.claude_coding_agent import ClaudeCodingAgent
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.gemini_cli_agent import GeminiCliAgent
from kiss.core.openai_codex_agent import OpenAICodexAgent
from kiss.core.utils import get_config_value


def create_coding_agent(
    coding_agent_type: Literal["claude code", "gemini cli", "openai codex"], name: str
) -> ClaudeCodingAgent | GeminiCliAgent | OpenAICodexAgent:
    """Create a Claude coding agent.

    Args:
        name: The name for the agent instance.

    Returns:
        An instance of ClaudeCodingAgent.
    """
    if coding_agent_type == "claude code":
        return ClaudeCodingAgent(name)
    elif coding_agent_type == "gemini cli":
        return GeminiCliAgent(name)
    elif coding_agent_type == "openai codex":
        return OpenAICodexAgent(name)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

@dataclass
class ImprovementReport:
    """Report documenting improvements made to an agent."""

    implemented_ideas: list[dict[str, str]] = field(default_factory=list)
    failed_ideas: list[dict[str, str]] = field(default_factory=list)
    generation: int = 0
    improved_tokens: int = 0
    improved_time: float = 0.0
    summary: str = ""

    def save(self, path: str) -> None:
        """Save the report to a JSON file."""
        report_dict = {
            "implemented_ideas": self.implemented_ideas,
            "failed_ideas": self.failed_ideas,
            "generation": self.generation,
            "improved_tokens": self.improved_tokens,
            "improved_time": self.improved_time,
            "summary": self.summary,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ImprovementReport":
        """Load a report from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            implemented_ideas=data.get("implemented_ideas", []),
            failed_ideas=data.get("failed_ideas", []),
            generation=data.get("generation", 0),
            improved_tokens=data.get("improved_tokens", 0),
            improved_time=data.get("improved_time", 0.0),
            summary=data.get("summary", ""),
        )


IMPROVE_AGENT_PROMPT = """You are an expert at optimizing AI agent code for efficiency.

## Your Task


You have been given an agent's source code in the folder: {source_folder}
and the KISSAgent API implementation in the folder: {kiss_folder}

Your goal is to improve this agent to:
1. **Reduce token usage** - Minimize tokens in prompts and responses
2. **Reduce execution time** - Make the agent run faster

## Important Constraints

- Do NOT create multiple variants or use evolutionary algorithms
- Do NOT use KISSEvolver, GEPA, mutation, or crossover techniques
- Make direct, targeted improvements to the existing code
- Preserve the agent's core functionality - it must still work correctly

## Optimization Strategies to Consider

### Token Reduction
- Shorten system prompts while preserving meaning
- Remove redundant instructions
- Use more concise variable names in prompts
- Consolidate repeated text
- Remove unnecessary examples from prompts
- Use structured output formats that require fewer tokens
- Search the web for token reduction techniques

### Time Reduction
- Add caching for repeated computations
- Run short-running commands in bash
- Batch operations where possible
- Use early termination when goals are achieved
- Reduce the number of LLM calls
- Optimize loops and data structures
- Search the web for time reduction techniques

### Code Quality
- Remove dead code
- Simplify complex logic
- Consolidate duplicate functions
- Use more efficient algorithms

### Agentic Patterns
- Search the web for information about various agentic patterns
- patterns that solve long-horizon tasks scalably, efficiently and accurately.
- patterns that makes Python code faster
- patterns that make bash commands faster
- implement these patterns in the agent's code if necessary

## Previous Improvement Report

{previous_report}

## Instructions

1. First, read and analyze all files in {target_folder}
2. Identify specific opportunities for token and time reduction
3. Make the improvements by editing the files in {target_folder}
4. Document what you changed and why

When complete, provide a summary of:
- What specific changes you made
- Expected token savings
- Expected time savings
- Any trade-offs or risks of the changes
"""


class ImproverAgent:
    """Agent that improves existing agent code using a configurable coding agent."""

    def __init__(
        self,
        model_name: str | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] | None = None,
    ):
        """Initialize the ImproverAgent.

        Args:
            model_name: LLM model to use for improvement.
            max_steps: Maximum steps for the coding agent.
            max_budget: Maximum budget in USD for the coding agent.
        """
        cfg = getattr(DEFAULT_CONFIG, "agent_creator", None)
        improver_cfg = cfg.improver if cfg else None

        self.model_name = get_config_value(model_name, improver_cfg, "model_name")
        self.max_steps = get_config_value(max_steps, improver_cfg, "max_steps")
        self.max_budget = get_config_value(max_budget, improver_cfg, "max_budget")
        self.coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] = (
            coding_agent_type or "claude code"
        )

    def _load_report(self, path: str | None) -> ImprovementReport | None:
        """Load a report from a path, returning None if it fails."""
        if path and Path(path).exists():
            try:
                return ImprovementReport.load(path)
            except Exception:
                pass
        return None

    def _format_report_for_prompt(self, report: ImprovementReport | None) -> str:
        """Format a report for inclusion in a prompt."""
        if report is None:
            return "No previous improvement report available."

        sections = ["Previous improvements made:"]

        if report.implemented_ideas:
            sections.append("\nSuccessful optimizations:")
            for idea in report.implemented_ideas:
                sections.append(
                    f"  - {idea.get('idea', 'Unknown')} (source: {idea.get('source', 'unknown')})"
                )

        if report.failed_ideas:
            sections.append("\nFailed optimizations (avoid these):")
            for idea in report.failed_ideas:
                sections.append(
                    f"  - {idea.get('idea', 'Unknown')} (reason: {idea.get('reason', 'unknown')})"
                )

        if report.summary:
            sections.append(f"\nSummary: {report.summary}")

        return "\n".join(sections)

    async def improve(
        self,
        source_folder: str,
        target_folder: str,
        report_path: str | None = None,
        base_dir: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]:
        """Improve an agent's code to reduce token usage and execution time.

        Args:
            source_folder: Path to the folder containing the agent's source code
            target_folder: Path where the improved agent will be written
            report_path: Optional path to a previous improvement report
            base_dir: Working directory for the Claude agent

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        if not Path(source_folder).exists():
            print(f"Source folder does not exist: {source_folder}")
            return False, None

        print(f"Copying {source_folder} to {target_folder}")
        shutil.copytree(source_folder, target_folder)

        previous_report = self._load_report(report_path)

        if base_dir is None:
            base_dir = str(Path(DEFAULT_CONFIG.agent.artifact_dir) / "improver_workdir")

        agent = create_coding_agent(self.coding_agent_type, "ImproverAgent")

        print(f"Running improvement on {target_folder}")
        start_time = time.time()

        try:
            result = await agent.run(
                model_name=self.model_name,
                prompt_template=IMPROVE_AGENT_PROMPT,
                arguments={
                    "source_folder": source_folder,
                    "target_folder": target_folder,
                    "previous_report": self._format_report_for_prompt(previous_report),
                    "kiss_folder": str(PROJECT_ROOT),
                },
                max_steps=self.max_steps,
                max_budget=self.max_budget,
                base_dir=base_dir,
                readable_paths=[target_folder, str(PROJECT_ROOT)],
                writable_paths=[target_folder],
            )
        except Exception as e:
            print(f"Error during improvement: {e}")
            return False, None

        if result is None:
            print("Improvement failed - no result from agent")
            return False, None

        # Create improvement report
        new_report = ImprovementReport(
            generation=(previous_report.generation + 1) if previous_report else 1,
            implemented_ideas=[
                {"idea": "Code optimization based on analysis", "source": "improver"}
            ],
            summary=result,
            improved_time=time.time() - start_time,
            improved_tokens=agent.total_tokens_used,
        )

        print(f"Improvement completed in {new_report.improved_time:.2f}s")
        print(f"Tokens used: {agent.total_tokens_used}")

        return True, new_report

    async def crossover_improve(
        self,
        primary_folder: str,
        primary_report_path: str,
        secondary_report_path: str,
        target_folder: str,
        base_dir: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]:
        """Improve an agent by combining ideas from two variants.

        Args:
            primary_folder: Path to the primary variant's source code
            primary_report_path: Path to the primary variant's improvement report
            secondary_report_path: Path to the secondary variant's improvement report
            target_folder: Path where the improved agent will be written
            base_dir: Working directory for the Claude agent

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        primary_report = self._load_report(primary_report_path)
        secondary_report = self._load_report(secondary_report_path)

        # Combine ideas from both reports
        merged_report = ImprovementReport(
            generation=max(
                primary_report.generation if primary_report else 0,
                secondary_report.generation if secondary_report else 0,
            ),
            implemented_ideas=(
                (primary_report.implemented_ideas if primary_report else [])
                + (secondary_report.implemented_ideas if secondary_report else [])
            ),
            failed_ideas=(
                (primary_report.failed_ideas if primary_report else [])
                + (secondary_report.failed_ideas if secondary_report else [])
            ),
            summary="Crossover of two variants",
        )

        # Save merged report temporarily
        temp_report_path = str(Path(target_folder).parent / "temp_crossover_report.json")
        Path(temp_report_path).parent.mkdir(parents=True, exist_ok=True)
        merged_report.save(temp_report_path)

        try:
            return await self.improve(
                source_folder=primary_folder,
                target_folder=target_folder,
                report_path=temp_report_path,
                base_dir=base_dir,
            )
        finally:
            if Path(temp_report_path).exists():
                Path(temp_report_path).unlink()


async def main() -> None:
    """Example usage of ImproverAgent."""
    improver = ImproverAgent()
    print(
        f"ImproverAgent: model={improver.model_name}, "
        f"max_steps={improver.max_steps}, max_budget=${improver.max_budget}"
    )


if __name__ == "__main__":
    anyio.run(main)
