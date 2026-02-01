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
from pathlib import Path
from typing import Any, Literal

from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.utils import get_config_value

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

AGENT_EVOLVER_PROMPT = """
You have to optimize an AI agent for long-running complex tasks.

## Agent Requirements

  - The agent must be designed for **long-running, complex tasks** using
    the Agent API available at {kiss_folder}.  Specifically, you should
    look at {kiss_folder}/API.md and {kiss_folder}/README.md first, and
    then look at code under the src folder as required.
    {kiss_folder}/src/kiss/core/models/model_info.py contains information
    about different LLM models and their context lengths, costs, etc.
    {kiss_folder}/src/kiss/agents/coding_agents/kiss_coding_agent.py has
    an example long-running complex task agent.
  - The agent **MUST** be tested for success on the given task description.
  - You **MUST not make the agent specific to any particular task, but
    rather make it a general purpose agent that can be used for any task**.
  - You MUST use KISSAgent, or KissCodingAgent, or ClaudeCodingAgent, or
    GeminiCliAgent, or OpenAICodexAgent or a mixture of them to implement
    the agent.
  - You MUST not use multithreading or multiprocessing or docker manager
    or 'anyio' or 'async' or 'await' in the agent implementation.
  - You may need to use the web to search on how to write such agents
  - Do NOT create multiple variants or use evolutionary algorithms
  - Do NOT use KISSEvolver, GEPA, mutation, or crossover techniques
  - Make direct, targeted improvements to the existing code
  - Preserve the agent's core functionality - it must still work correctly
  - Do NOT use caching mechanisms that avoid re-computation
  - You MUST run the following task to evaluate the agent.

## Task Description

{task_description}

## Agent Implementation Files

Create or modify the following files in {target_folder}:

1. `agent.py` - Main agent implementation that MUST include an
   `def agent_run(task: str) -> dict[str, Any]` function.
   This function is the entry point that will be called to run the agent on a task.
   It should accept a task description string and return a result.  The result must
   be a dictionary containing the following keys:
   - "feedback": str - Feedback from the agent on the task
   - "metrics": dict[str, Any] - Metrics from the agent on the task
     - "cost": float - Cost incurred by the agent
     - "tokens_used": int - Number of tokens used by the agent
     - "execution_time": float - Time taken to run the agent on the task in seconds
     - "success": int - 0 if the agent completed successfully, 1 otherwise
2. `config.py` - Agent configuration
3. `__init__.py` - Agent package initialization
4. `test_agent.py` - Tests for the agent
5. `requirements.txt` - Dependencies for the agent

The agent should collect fine-grained feedback on the task as it is executing.
When complete, provide a summary of the agent it created and evolved, and the
files that were written.

## Goals

Your goal is to improve this agent to:
1. **Reduce token usage** - Minimize tokens in prompts and responses
2. **Reduce execution time** - Make the agent run faster
3. **Maintain correctness** - Ensure the agent still completes the task correctly
4. **Reduce costs** - Lower overall cost of running the agent

## Optimization Strategies to Consider

### Token Reduction
- Shorten prompts while preserving meaning
- Remove redundant instructions
- Remove unnecessary examples from prompts
- Use structured output formats that require fewer tokens
- Search the web for token reduction techniques

### Time Reduction
- Run short-running commands in bash
- Batch operations where possible
- Use early termination when goals are achieved
- Reduce the number of LLM calls
- Optimize loops and data structures
- Search the web for time reduction techniques

### Agentic Patterns
- search the web for information about various agentic patterns
- patterns that solve long-horizon tasks scalably, efficiently and accurately
- patterns that makes Python code faster
- patterns that make bash commands faster
- try some of these patterns in the agent's source code based on your needs

## Previous Improvement Report (blank if none provided)

{previous_report}

## Feedback. (blank if none provided)

The agent has been given the following feedback on the task:
{feedback}

## Output Summary

When complete, provide a summary of:
- What specific changes you made
- Expected token savings
- Expected time savings
- Any trade-offs or risks of the changes

"""


class ImprovementReport:
    """Report documenting improvements made to an agent."""

    def __init__(
        self,
        metrics: dict[str, float],
        implemented_ideas: list[dict[str, str]],
        failed_ideas: list[dict[str, str]],
        generation: int = 0,
        summary: str = "",
    ):
        self.metrics = metrics
        self.implemented_ideas = implemented_ideas
        self.failed_ideas = failed_ideas
        self.generation = generation
        self.summary = summary

    def save(self, path: str) -> None:
        """Save the report to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "implemented_ideas": self.implemented_ideas,
                    "failed_ideas": self.failed_ideas,
                    "generation": self.generation,
                    "metrics": self.metrics,
                    "summary": self.summary,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "ImprovementReport":
        """Load a report from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            implemented_ideas=data.get("implemented_ideas", []),
            failed_ideas=data.get("failed_ideas", []),
            generation=data.get("generation", 0),
            metrics=data.get("metrics", {}),
            summary=data.get("summary", ""),
        )


class ImproverAgent:
    """Agent that improves existing agent code using a configurable coding agent."""

    def __init__(
        self,
        max_steps: int | None = None,
        max_budget: float | None = None,
        coding_agent_type: Literal["kiss code", "claude code", "gemini cli", "openai codex"]
        | None = None,
    ):
        """Initialize the ImproverAgent.

        Args:
            model_name: LLM model to use for improvement.
            max_steps: Maximum steps for the coding agent.
            max_budget: Maximum budget in USD for the coding agent.
        """
        cfg = getattr(DEFAULT_CONFIG, "create_and_optimize_agent", None)
        improver_cfg = getattr(cfg, "improver", None)
        self.max_steps = get_config_value(max_steps, improver_cfg, "max_steps")
        self.max_budget = get_config_value(max_budget, improver_cfg, "max_budget")

    def _load_report(self, path: str | None) -> ImprovementReport | None:
        """Load a report from a path, returning None if it fails."""
        if not path or not Path(path).exists():
            return None
        try:
            return ImprovementReport.load(path)
        except Exception:
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

    def create_initial(
        self,
        task_description: str,
        target_folder: str,
        feedback: str = "",
    ) -> tuple[bool, ImprovementReport | None]:
        """Create an initial agent from scratch.

        Args:
            task_description: Description of the task the agent should perform
            target_folder: Path where the new agent will be written
            feedback: Optional feedback to guide the initial creation

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        previous_report_text = "No previous report - initial implementation"

        return self._run_improvement(
            task_description=task_description,
            target_folder=target_folder,
            previous_report_text=previous_report_text,
            feedback=feedback,
            generation=0,
        )

    def improve(
        self,
        source_folder: str,
        target_folder: str,
        task_description: str,
        report_path: str | None = None,
        feedback: str = "",
    ) -> tuple[bool, ImprovementReport | None]:
        """Improve an agent's code to reduce token usage and execution time.

        Args:
            source_folder: Path to the folder containing the agent's source code
            target_folder: Path where the improved agent will be written
            task_description: Description of the task the agent should perform
            report_path: Optional path to a previous improvement report
            feedback: Optional feedback to guide the improvement

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        if not Path(source_folder).exists():
            print(f"Source folder does not exist: {source_folder}")
            return False, None

        print(f"Copying {source_folder} to {target_folder}")
        shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)

        previous_report = self._load_report(report_path)
        previous_report_text = self._format_report_for_prompt(previous_report)
        generation = (previous_report.generation + 1) if previous_report else 1

        return self._run_improvement(
            task_description=task_description,
            target_folder=target_folder,
            previous_report_text=previous_report_text,
            feedback=feedback,
            generation=generation,
        )

    def _run_improvement(
        self,
        task_description: str,
        target_folder: str,
        previous_report_text: str,
        feedback: str,
        generation: int,
    ) -> tuple[bool, ImprovementReport | None]:
        """Internal method to run the improvement process.

        Args:
            task_description: Description of the task the agent should perform
            target_folder: Path where the agent will be written
            previous_report_text: Formatted previous report text
            feedback: Feedback to guide the improvement
            generation: Generation number for the report

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """

        agent = KISSCodingAgent("Agent Improver")

        print(f"Running improvement on {target_folder}")
        start_time = time.time()

        try:
            result = agent.run(
                prompt_template=AGENT_EVOLVER_PROMPT,
                arguments={
                    "task_description": task_description,
                    "target_folder": target_folder,
                    "previous_report": previous_report_text,
                    "kiss_folder": str(PROJECT_ROOT),
                    "feedback": feedback,
                },
                max_steps=self.max_steps,
                max_budget=self.max_budget,
                readable_paths=[target_folder, str(PROJECT_ROOT)],
                writable_paths=[target_folder],
            )
        except Exception as e:
            print(f"Error during improvement: {e}")
            # Clean up partially failed target folder
            if Path(target_folder).exists():
                shutil.rmtree(target_folder)
            return False, None

        # Create improvement report
        new_report = ImprovementReport(
            metrics={
                "tokens_used": agent.total_tokens_used,
                "cost": agent.budget_used,
                "execution_time": time.time() - start_time,
            },
            implemented_ideas=[
                {"idea": "Code optimization based on analysis", "source": "improver"}
            ],
            failed_ideas=[],
            generation=generation,
            summary=result,
        )

        print(f"Improvement completed in {new_report.metrics['execution_time']:.2f}s")
        print(f"Tokens used: {agent.total_tokens_used}")
        print(f"Cost: ${agent.budget_used}")

        return True, new_report

    def crossover_improve(
        self,
        primary_folder: str,
        primary_report_path: str,
        secondary_report_path: str,
        primary_feedback: str,
        secondary_feedback: str,
        target_folder: str,
        task_description: str,
    ) -> tuple[bool, ImprovementReport | None]:
        """Improve an agent by combining ideas from two variants.

        Args:
            primary_folder: Path to the primary variant's source code
            primary_report_path: Path to the primary variant's improvement report
            secondary_report_path: Path to the secondary variant's improvement report
            primary_feedback: Feedback from the primary variant
            secondary_feedback: Feedback from the secondary variant
            target_folder: Path where the improved agent will be written
            task_description: Description of the task the agent should perform

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        p_report = self._load_report(primary_report_path)
        s_report = self._load_report(secondary_report_path)

        # Combine ideas from both reports
        merged_report = ImprovementReport(
            metrics={},
            implemented_ideas=(
                (p_report.implemented_ideas if p_report else [])
                + (s_report.implemented_ideas if s_report else [])
            ),
            failed_ideas=(
                (p_report.failed_ideas if p_report else [])
                + (s_report.failed_ideas if s_report else [])
            ),
            generation=max(
                p_report.generation if p_report else 0,
                s_report.generation if s_report else 0,
            ),
            summary="Crossover of two variants",
        )

        # Save merged report temporarily
        temp_report_path = str(Path(target_folder).parent / "temp_crossover_report.json")
        merged_report.save(temp_report_path)

        try:
            return self.improve(
                source_folder=primary_folder,
                target_folder=target_folder,
                task_description=task_description,
                report_path=temp_report_path,
                feedback=f"{primary_feedback}\n{secondary_feedback}",
            )
        finally:
            if Path(temp_report_path).exists():
                Path(temp_report_path).unlink()


def main() -> None:
    """Example usage of ImproverAgent."""
    improver = ImproverAgent()
    print(f"max_steps={improver.max_steps}, max_budget=${improver.max_budget}")


if __name__ == "__main__":
    main()
