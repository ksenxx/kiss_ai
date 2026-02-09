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

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent
from kiss.core import config as config_module
from kiss.core.compact_formatter import CompactFormatter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

AGENT_EVOLVER_PROMPT_PART1 = """
You have to optimize an AI agent for long-running complex tasks.

## Agent Requirements

  - The agent must be designed for **long-running, complex tasks** using
    the Agent API available at {kiss_folder}.  Specifically, you should
    look at {kiss_folder}/API.md and {kiss_folder}/README.md first, and
    then look at code under the src folder as required.
    {kiss_folder}/src/kiss/core/models/model_info.py contains information
    about different LLM models and their context lengths, costs, etc.
    Use {kiss_folder}/src/kiss/agents/coding_agents/relentless_coding_agent.py
    as the initial agent implementation.
  - The agent **MUST** be tested for success on the given task description.
"""

AGENT_EVOLVER_PROMPT_PART2 = """
  - You **MUST not make the agent specific to any particular task, but
    rather make it a general purpose agent that can be used for any task**.
"""

AGENT_EVOLVER_PROMPT_PART3 = """
  - You MUST use RelentlessCodingAgent, KISSAgent, or a mixture of them
    to implement the agent.
  - You MUST not use multithreading or multiprocessing or docker manager
    or 'anyio' or 'async' or 'await' in the agent implementation.
  - You may need to use the web to search on how to write such agents
  - Do NOT create multiple variants or use evolutionary algorithms
  - Do NOT use KISSEvolver, GEPA, mutation, or crossover techniques
  - Make direct, targeted improvements to the existing code
  - Preserve the agent's core functionality - it must still work correctly
  - Do NOT use caching mechanisms that avoid re-computation

## Task Description

{task_description}

## Agent Implementation Files

Create or modify the following files in {work_dir}:

1. `agent.py` - Main agent implementation that MUST include an
   `def agent_run(task: str) -> dict[str, Any]` function.
   This function is the entry point that will be called to run the agent on a task.
   It should accept a task description string and return a result.  The result must
   be a dictionary containing the following keys:
   - "metrics": dict[str, Any] - Metrics from the agent on the task
     - "cost": float - Cost incurred by the agent
     - "tokens_used": int - Number of tokens used by the agent
     - "execution_time": float - Time taken to run the agent on the task in seconds
     - "success": int - 0 if the agent completed successfully, 1 otherwise
2. `config.py` - Agent configuration
3. `__init__.py` - Agent package initialization
4. `test_agent.py` - Tests for the agent
5. `requirements.txt` - Dependencies for the agent

The agent should collect fine-grained metrics on the task as it is executing.
When complete, provide a summary of the agent it created and evolved, and the
files that were written.

## Goals

Your goal is to improve this agent to:
1. **Reduce LLM token usage** - Minimize tokens in prompts and responses
2. **Reduce execution time** - Make the agent run faster
3. **Maintain correctness** - Ensure the agent still completes the task correctly
4. **Reduce costs** - Lower overall cost of running the agent

## Optimization Strategies to Consider

### Token Reduction
- Shorten prompts while preserving meaning
- Remove redundant instructions
- Remove unnecessary examples from prompts
- Search the web for LLM token reduction techniques

### Time Reduction
- Run short-running commands in bash
- Batch operations where possible
- Use early termination when goals are achieved
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

## Output Summary

When complete, provide a summary of:
- What specific changes you made
- Token savings
- Time savings
- Cost savings
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
        """Initialize an ImprovementReport.

        Args:
            metrics: Dictionary of metric values (e.g., tokens_used, cost, execution_time).
            implemented_ideas: List of dicts describing successful optimizations,
                each with 'idea' and 'source' keys.
            failed_ideas: List of dicts describing failed optimizations,
                each with 'idea' and 'reason' keys.
            generation: The generation number this report represents.
            summary: A text summary of the improvements made.
        """
        self.metrics = metrics
        self.implemented_ideas = implemented_ideas
        self.failed_ideas = failed_ideas
        self.generation = generation
        self.summary = summary

    def save(self, path: str) -> None:
        """Save the report to a JSON file.

        Creates parent directories if they don't exist.

        Args:
            path: The file path where the report will be saved.

        Returns:
            None. Writes the report as JSON to the specified path.
        """
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
        """Load a report from a JSON file.

        Args:
            path: The file path to load the report from.

        Returns:
            An ImprovementReport instance populated with data from the file.
        """
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

    def _load_report(self, path: str | None) -> ImprovementReport | None:
        """Load a report from a path, returning None if it fails.

        Safely attempts to load an ImprovementReport, handling missing files
        and parsing errors gracefully.

        Args:
            path: The file path to load the report from, or None.

        Returns:
            The loaded ImprovementReport, or None if path is None, the file
            doesn't exist, or loading fails for any reason.
        """
        if not path or not Path(path).exists():
            return None
        try:
            return ImprovementReport.load(path)
        except Exception:
            return None

    def _format_report_for_prompt(self, report: ImprovementReport | None) -> str:
        """Format a report for inclusion in a prompt.

        Converts an ImprovementReport into a human-readable string suitable
        for including in an LLM prompt to inform the next improvement cycle.

        Args:
            report: The ImprovementReport to format, or None.

        Returns:
            A formatted string describing the report's implemented ideas,
            failed ideas, and summary. Returns a placeholder message if
            report is None.
        """
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
        work_dir: str,
    ) -> tuple[bool, ImprovementReport | None]:
        """Create an initial agent from scratch.

        Args:
            task_description: Description of the task the agent should perform
            work_dir: Path where the new agent will be written

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        Path(work_dir).mkdir(parents=True, exist_ok=True)

        previous_report_text = "No previous report - initial implementation"

        return self._run_improvement(
            task_description=task_description,
            work_dir=work_dir,
            previous_report_text=previous_report_text,
            generation=0,
        )

    def improve(
        self,
        source_folder: str,
        work_dir: str,
        task_description: str,
        report_path: str | None = None,
    ) -> tuple[bool, ImprovementReport | None]:
        """Improve an agent's code to reduce token usage and execution time.

        Args:
            source_folder: Path to the folder containing the agent's source code
            work_dir: Path where the improved agent will be written
            task_description: Description of the task the agent should perform
            report_path: Optional path to a previous improvement report

        Returns:
            Tuple of (success: bool, report: ImprovementReport | None)
        """
        if not Path(source_folder).exists():
            print(f"Source folder does not exist: {source_folder}")
            return False, None

        print(f"Copying {source_folder} to {work_dir}")
        shutil.copytree(source_folder, work_dir, dirs_exist_ok=True)

        previous_report = self._load_report(report_path)
        previous_report_text = self._format_report_for_prompt(previous_report)
        generation = (previous_report.generation + 1) if previous_report else 1

        return self._run_improvement(
            task_description=task_description,
            work_dir=work_dir,
            previous_report_text=previous_report_text,
            generation=generation,
        )

    def _run_improvement(
        self,
        task_description: str,
        work_dir: str,
        previous_report_text: str,
        generation: int,
    ) -> tuple[bool, ImprovementReport | None]:
        """Run the improvement process using a coding agent.

        Executes the RelentlessCodingAgent with the AGENT_EVOLVER_PROMPT to create
        or improve agent code. Tracks metrics including tokens used, cost,
        and execution time.

        Args:
            task_description: Description of the task the agent should perform.
            work_dir: Path where the agent code will be written.
            previous_report_text: Formatted text from previous improvement report
                to inform the coding agent of past optimizations.
            generation: Generation number to record in the new report.

        Returns:
            A tuple of (success, report) where success is True if improvement
            completed without errors, and report is the ImprovementReport
            documenting the changes. Returns (False, None) on failure.
        """

        agent = RelentlessCodingAgent("Agent Improver")

        print(f"Running improvement on {work_dir}")
        if not config_module.DEFAULT_CONFIG.create_and_optimize_agent.evolver.evolve_to_solve_task:
            agent_evolver_prompt = (
                AGENT_EVOLVER_PROMPT_PART1 + AGENT_EVOLVER_PROMPT_PART2 + AGENT_EVOLVER_PROMPT_PART3
            )
        else:
            agent_evolver_prompt = AGENT_EVOLVER_PROMPT_PART1 + AGENT_EVOLVER_PROMPT_PART3

        start_time = time.time()

        try:
            result = agent.run(
                prompt_template=agent_evolver_prompt,
                arguments={
                    "task_description": task_description,
                    "work_dir": work_dir,
                    "previous_report": previous_report_text,
                    "kiss_folder": str(PROJECT_ROOT),
                },
                readable_paths=[work_dir, str(PROJECT_ROOT)],
                writable_paths=[work_dir],
                formatter=CompactFormatter()
            )
        except Exception as e:
            print(f"Error during improvement: {e}")
            # Clean up partially failed target folder
            if Path(work_dir).exists():
                shutil.rmtree(work_dir)
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
        work_dir: str,
        task_description: str,
    ) -> tuple[bool, ImprovementReport | None]:
        """Improve an agent by combining ideas from two variants.

        Args:
            primary_folder: Path to the primary variant's source code
            primary_report_path: Path to the primary variant's improvement report
            secondary_report_path: Path to the secondary variant's improvement report
            work_dir: Path where the improved agent will be written
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
        temp_report_path = str(Path(work_dir).parent / "temp_crossover_report.json")
        merged_report.save(temp_report_path)

        try:
            return self.improve(
                source_folder=primary_folder,
                work_dir=work_dir,
                task_description=task_description,
                report_path=temp_report_path,
            )
        finally:
            if Path(temp_report_path).exists():
                Path(temp_report_path).unlink()


def main() -> None:
    """Example usage of ImproverAgent.

    Creates an ImproverAgent instance and prints it for demonstration.

    Returns:
        None.
    """
    improver = ImproverAgent()
    print(f"ImproverAgent created: {improver}")


if __name__ == "__main__":
    main()
