# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""AgentEvolver - Evolves AI agents using a Pareto frontier approach.

The AgentEvolver maintains a Pareto frontier of agent implementations,
optimizing for both token efficiency and execution time. It uses the
ImproverAgent to mutate and crossover agents, tracking improvements
across generations.

Key features:
1. Pareto frontier maintenance for multi-objective optimization
2. Mutation: Sample and improve a single variant
3. Crossover: Combine ideas from two variants
4. Configurable mutation vs crossover probability
5. Tracking of lineage via parent_ids

Pseudo-code for AgentEvolver Algorithm:

Inputs:
    - task_description: Description of the task the agent should perform
    - max_generations: Maximum number of improvement generations
    - max_frontier_size: Maximum size of the Pareto frontier
    - mutation_probability: Probability of mutation vs crossover (0.0 to 1.0)
    - coding_agent_type: Which coding agent to use (claude code, gemini cli, openai codex)

Data Structures:
    AgentVariant:
        - folder_path: Directory containing agent code
        - report: ImprovementReport tracking implemented/failed ideas
        - metrics: {success, tokens_used, execution_time}
        - id, generation, parent_ids (for lineage tracking)

    dominates(A, B):
        # A dominates B if A is at least as good in all metrics and strictly better in one
        # Minimizes: tokens_used, execution_time
        # Maximizes: success

    score(variant):
        # Combined ranking score (lower is better)
        # = success * (-1,000,000) + tokens_used * 1 + execution_time * 1000

Algorithm EVOLVE():
    1. INITIALIZE
       - Create temporary work_dir for variants
       - Set optimal_dir for storing best agent
       - Initialize empty pareto_frontier

    2. CREATE INITIAL AGENT
       - Use coding agent to generate agent files from task_description
       - Agent must implement agent_run(task) -> {success, tokens_used, execution_time}
       - Evaluate initial agent by calling agent_run(task_description)
       - Add to pareto_frontier
       - Copy to optimal_dir

    3. FOR generation = 1 TO max_generations:
       a. SELECT OPERATION
          IF random() < mutation_probability OR frontier_size < 2:
              # MUTATION
              parent = sample_uniform(pareto_frontier)
              new_variant = ImproverAgent.improve(parent)
          ELSE:
              # CROSSOVER
              v1, v2 = sample_two(pareto_frontier)
              primary, secondary = order_by_score(v1, v2)  # better score first
              new_variant = ImproverAgent.crossover_improve(primary, secondary)

       b. IF new_variant created successfully:
          - Evaluate: load agent.py, call agent_run(task_description)
          - Update pareto_frontier:
              - Reject if dominated by any existing variant
              - Remove variants dominated by new_variant
              - Add new_variant
              - If frontier > max_size: trim using crowding distance
          - Copy best variant (min score) to optimal_dir

    4. RETURN best variant from pareto_frontier

    5. CLEANUP work_dir
"""

import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import kiss.agents.agent_creator.config  # noqa: F401
from kiss.agents.agent_creator.improver_agent import (
    ImprovementReport,
    ImproverAgent,
    create_coding_agent,
)
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.utils import get_config_value

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class AgentVariant:
    """Represents an agent variant in the population."""

    folder_path: str
    report_path: str
    report: ImprovementReport
    metrics: dict[str, float] = field(default_factory=dict)  # Flexible metrics dictionary
    id: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    feedback: str = ""

    def dominates(self, other: "AgentVariant") -> bool:
        """Check if this variant Pareto-dominates another."""
        all_metrics = set(self.metrics.keys()) | set(other.metrics.keys())

        strictly_better = False

        for metric in all_metrics:
            self_val = self.metrics.get(metric, sys.maxsize)
            other_val = other.metrics.get(metric, sys.maxsize)

            if self_val > other_val:
                return False
            if self_val < other_val:
                strictly_better = True
        return strictly_better

    def score(self, weights: dict[str, float] | None = None) -> float:
        """Combined score for ranking (lower is better).

        Args:
            weights: Dict mapping metric names to weights. Positive weights mean
                    the metric should be minimized, negative weights mean maximized.
        """
        if weights is None:
            # Default weights: prioritize success (maximize), then minimize tokens and time
            # success is 0 or 1, tokens_used count, execution_time in seconds
            weights = {
                "success": 1000000,       # Minimize success (0 is best)
                "tokens_used": 1,         # Minimize token usage
                "execution_time": 1000,   # Minimize execution time
            }

        score = 0.0
        for metric, weight in weights.items():
            score += self.metrics.get(metric, 0) * weight
        return score

    def to_dict(self) -> dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "folder_path": self.folder_path,
            "report_path": self.report_path,
            "report": {
                "implemented_ideas": self.report.implemented_ideas,
                "failed_ideas": self.report.failed_ideas,
                "generation": self.report.generation,
                "metrics": self.report.metrics,
                "summary": self.report.summary,
            },
            "metrics": self.metrics,
            "id": self.id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "feedback": self.feedback,
        }


# Prompt for initial agent creation
INITIAL_AGENT_PROMPT = """You are an expert at building efficient, long-running AI agents.
Your task is to create an initial agent implementation based on the following requirements.

# Instructions
- You are going to run a very long-running task
- You may need to use the web to search on how to write such agents
- You may consider orchestrator agents and sub-agents to solve the task

## Task Description
{task_description}

## Agent Requirements

  - The agent must be designed for **long-running, complex tasks** using
    the Agent API available at {kiss_folder}.  Specifically, you should
    look at API.md and README.md first, and then look at code under the
    src folder as required. {kiss_folder}/src/kiss/core/models/model_info.py
    contains information about different LLM models and their context lengths.
  - The agent.py when executed as a file, **MUST** run the given task.
  - The agent **MUST** be tested for success on the given task description.
    **YOU MUST ABSOLUTELY WAIT FOR THE TEST TO FINISH.**
  - You **MUST not make the agent specific to any particular task, but
    rather make it a general purpose agent that can be used for any task**.
  - You MUST use KISSAgent, or ClaudeCodingAgent, or GeminiCliAgent, or
    OpenAICodexAgent or a mixture of them to implement the agent.
  - You MUST not use multithreading or multiprocessing or docker manager
    or 'anyio' or 'async' or 'await' in the agent implementation.

Create the following files in {target_folder}:

1. `agent.py` - Main agent implementation that MUST include an
   `def agent_run(task: str) -> dict[str, Any]` function.
   This function is the entry point that will be called to run the agent on a task.
   It should accept a task description string and return a result.  The result must
   be a dictionary containing the following keys:
   - "feedback": str - Feedback from the agent on the task
   - "metrics": dict[str, Any] - Metrics from the agent on the task
     - "tokens_used": int - Number of tokens used by the agent
     - "execution_time": float - Time taken to run the agent on the task in seconds
     - "success": int - 0 if the agent completed successfully, 1 otherwise
2. `config.py` - Agent configuration
3. `__init__.py` - Agent package initialization
4. `README.md` - Agent documentation
5. `test_agent.py` - Tests for the agent
6. `requirements.txt` - Dependencies for the agent
7. Any other files necessary for the agent to function properly

The agent should collect fine-grained feedback on the task as it is executing.
When complete, provide a summary of the agent created and the files that were written.
"""


class AgentEvolver:
    """Evolves AI agents using Pareto frontier optimization.

    Maintains a population of agent variants optimized for token efficiency
    and execution time. Uses mutation (improving one variant) or crossover
    (combining ideas from two variants) to create new variants.
    """

    def __init__(
        self,
        task_description: str,
        model_name: str | None = None,
        max_generations: int | None = None,
        initial_frontier_size: int | None = None,
        max_frontier_size: int | None = None,
        mutation_probability: float | None = None,
        coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] | None = None,
    ):
        """Initialize the AgentEvolver.

        Args:
            task_description: Description of the task the agent should perform.
            model_name: LLM model to use for agent creation and improvement.
            max_generations: Maximum number of improvement generations.
            max_frontier_size: Maximum size of the Pareto frontier.
            mutation_probability: Probability of mutation vs crossover.
            coding_agent_type: Which coding agent to use: 'claude code',
                'gemini cli', or 'openai codex'.
        """
        cfg = getattr(DEFAULT_CONFIG, "agent_creator", None)
        evolver_cfg = cfg.evolver if cfg else None

        self.task_description = task_description
        self.model_name = get_config_value(model_name, evolver_cfg, "model_name")
        self.max_generations = get_config_value(max_generations, evolver_cfg, "max_generations")
        self.initial_frontier_size = get_config_value(
            initial_frontier_size, evolver_cfg, "initial_frontier_size"
        )
        self.max_frontier_size = get_config_value(
            max_frontier_size, evolver_cfg, "max_frontier_size"
        )
        self.mutation_probability = get_config_value(
            mutation_probability, evolver_cfg, "mutation_probability"
        )
        self.initial_agent_max_steps: int = get_config_value(
            None, evolver_cfg, "initial_agent_max_steps"
        )
        self.initial_agent_max_budget: float = get_config_value(
            None, evolver_cfg, "initial_agent_max_budget"
        )
        self.coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] = (
            get_config_value(coding_agent_type, evolver_cfg, "coding_agent_type")
        )

        self.work_dir = Path(tempfile.mkdtemp())
        self.optimal_dir = Path(DEFAULT_CONFIG.agent.artifact_dir) / "optimal_agent"

        self.pareto_frontier: list[AgentVariant] = []
        self._variant_counter = 0
        self._generation = 0
        self.improver = ImproverAgent(
            model_name=self.model_name, coding_agent_type=self.coding_agent_type
        )

    def _next_variant_id(self) -> int:
        """Get the next unique variant ID."""
        self._variant_counter += 1
        return self._variant_counter

    def _get_variant_paths(self, variant_id: int) -> tuple[str, str]:
        """Get the folder and report paths for a variant."""
        folder = str(self.work_dir / f"variant_{variant_id}")
        report = str(self.work_dir / f"variant_{variant_id}" / "improvement_report.json")
        return folder, report

    def _create_initial_agent(self, variant_id: int) -> AgentVariant:
        """Create the initial agent from scratch."""
        target_folder, report_path = self._get_variant_paths(variant_id)
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        agent = create_coding_agent(self.coding_agent_type, "Initial Agent Creator")
        agent.run(
            model_name=self.model_name,
            prompt_template=INITIAL_AGENT_PROMPT,
            arguments={
                "task_description": self.task_description,
                "target_folder": target_folder,
                "kiss_folder": str(PROJECT_ROOT),
            },
            max_steps=self.initial_agent_max_steps,
            max_budget=self.initial_agent_max_budget,
            base_dir=str(self.work_dir / "creator_workdir"),
            writable_paths=[target_folder],
        )

        initial_report = ImprovementReport(
            implemented_ideas=[{"idea": "Initial implementation", "source": "initial"}],
            generation=0,
        )
        initial_report.save(report_path)

        return AgentVariant(
            folder_path=target_folder,
            report_path=report_path,
            report=initial_report,
            id=variant_id,
            generation=0,
        )

    def _load_module_from_path(self, module_name: str, file_path: str) -> Any:
        """Dynamically load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _evaluate_variant(
        self, variant: AgentVariant,
    ) -> dict[str, Any]:
        """Run the agent on the long-running task and collect metrics."""
        print(f"Evaluating variant {variant.id}...")

        # Create a temporary directory and copy the variant's code into it
        temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_variant_{variant.id}_"))
        old_cwd = os.getcwd()
        try:
            shutil.copytree(variant.folder_path, temp_dir / "agent_code", dirs_exist_ok=True)
            agent_dir = str(temp_dir / "agent_code")
            agent_file = temp_dir / "agent_code" / "agent.py"
            module_name = f"agent_variant_{id(self)}_{random.randint(0, 10000)}"

            try:
                sys.path.insert(0, agent_dir)
                os.chdir(temp_dir)
                agent_module = self._load_module_from_path(module_name, str(agent_file))
                if agent_module is None:
                    print(f"Failed to load module from {agent_file}")
                    return {
                        "feedback": "Failed to load module from agent.py",
                        "metrics": {"success": 1, "tokens_used": 0, "execution_time": 0.0},
                    }
                result: dict[str, Any] = agent_module.agent_run(self.task_description)
                return result
            except Exception:
                return {
                    "feedback": "Failed to run agent.py",
                    "metrics": {"success": 1, "tokens_used": 0, "execution_time": 0.0},
                }
            finally:
                os.chdir(old_cwd)
                if agent_dir in sys.path:
                    sys.path.remove(agent_dir)
                sys.modules.pop(module_name, None)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _update_pareto_frontier(self, new_variant: AgentVariant) -> bool:
        """Update the Pareto frontier with a new variant.

        Returns True if the variant was added to the frontier.
        """
        # Check if new variant is dominated by any in frontier
        for existing in self.pareto_frontier:
            if existing.dominates(new_variant):
                return False

        # Remove variants dominated by new variant
        self.pareto_frontier = [
            v for v in self.pareto_frontier if not new_variant.dominates(v)
        ]

        # Add new variant to frontier
        self.pareto_frontier.append(new_variant)

        # Trim frontier if too large (keep most diverse)
        if len(self.pareto_frontier) > self.max_frontier_size:
            self._trim_frontier()

        return True

    def _trim_frontier(self) -> None:
        """Trim the Pareto frontier to max size using crowding distance."""
        if len(self.pareto_frontier) <= self.max_frontier_size:
            return

        n = len(self.pareto_frontier)

        # Collect all metric names across all variants
        all_metrics: set[str] = set()
        for v in self.pareto_frontier:
            all_metrics.update(v.metrics.keys())

        # Calculate crowding distance
        crowding = [0.0] * n

        for metric in all_metrics:
            values = [v.metrics.get(metric, 0) for v in self.pareto_frontier]
            value_range = max(values) - min(values) or 1

            sorted_indices = sorted(range(n), key=lambda i: values[i])
            crowding[sorted_indices[0]] = crowding[sorted_indices[-1]] = float("inf")

            for i in range(1, n - 1):
                idx = sorted_indices[i]
                diff = values[sorted_indices[i + 1]] - values[sorted_indices[i - 1]]
                crowding[idx] += diff / value_range

        # Keep most diverse (highest crowding distance)
        sorted_indices = sorted(range(n), key=lambda i: crowding[i], reverse=True)
        kept_indices = sorted_indices[: self.max_frontier_size]
        self.pareto_frontier = [self.pareto_frontier[i] for i in kept_indices]

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        """Format metrics dictionary for display."""
        return ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )

    def _sample_from_frontier(self) -> AgentVariant:
        """Sample a variant uniformly from the Pareto frontier."""
        return random.choice(self.pareto_frontier)

    def _sample_two_from_frontier(self) -> tuple[AgentVariant, AgentVariant]:
        """Sample two different variants from the Pareto frontier, ordered by score."""
        v1, v2 = random.sample(self.pareto_frontier, 2)
        return (v1, v2) if v1.score() <= v2.score() else (v2, v1)

    def _mutate(self, variant: AgentVariant) -> AgentVariant | None:
        """Create a new variant by mutating an existing one."""
        new_id = self._next_variant_id()
        target_folder, report_path = self._get_variant_paths(new_id)

        success, new_report = self.improver.improve(
            source_folder=variant.folder_path,
            target_folder=target_folder,
            report_path=variant.report_path,
            feedback=variant.feedback,
            base_dir=str(self.work_dir / "improver_workdir"),
        )

        if not success or new_report is None:
            if Path(target_folder).exists():
                shutil.rmtree(target_folder)
            return None

        new_report.save(report_path)
        return AgentVariant(
            folder_path=target_folder,
            report_path=report_path,
            report=new_report,
            id=new_id,
            generation=self._generation,
            parent_ids=[variant.id],
        )

    def _crossover(
        self, primary: AgentVariant, secondary: AgentVariant
    ) -> AgentVariant | None:
        """Create a new variant by crossing over two variants."""
        new_id = self._next_variant_id()
        target_folder, report_path = self._get_variant_paths(new_id)

        success, new_report = self.improver.crossover_improve(
            primary_folder=primary.folder_path,
            primary_report_path=primary.report_path,
            secondary_report_path=secondary.report_path,
            primary_feedback=primary.feedback,
            secondary_feedback=secondary.feedback,
            target_folder=target_folder,
            base_dir=str(self.work_dir / "improver_workdir"),
        )

        if not success or new_report is None:
            if Path(target_folder).exists():
                shutil.rmtree(target_folder)
            return None

        new_report.save(report_path)
        return AgentVariant(
            folder_path=target_folder,
            report_path=report_path,
            report=new_report,
            id=new_id,
            generation=self._generation,
            parent_ids=[primary.id, secondary.id],
        )

    def evolve(self) -> AgentVariant:
        """Run the evolutionary optimization."""
        try:
            print(f"Starting AgentEvolver with {self.max_generations} generations")
            print(f"Max frontier size: {self.max_frontier_size}, Task: {self.task_description}")

            # Initialize with first agent
            # while pareto frontier size is less that self.initial_frontier_size:
            while len(self.pareto_frontier) < self.initial_frontier_size:
                variant_id = self._next_variant_id()
                print(f"\nInitializing variant_{variant_id} agent")
                initial = self._create_initial_agent(variant_id=variant_id)
                eval_result = self._evaluate_variant(initial)
                initial.metrics = eval_result["metrics"]
                self._update_pareto_frontier(initial)
                metrics_str = self._format_metrics(initial.metrics)
                print(f"Initial agent variant_{variant_id} metrics: {metrics_str}")
                self._copy_best_to_optimal(initial)

            # Evolution loop
            for gen in range(1, self.max_generations + 1):
                self._generation = gen
                print(f"\n=== Generation {gen}/{self.max_generations} ===")
                print(f"Pareto frontier size: {len(self.pareto_frontier)}")

                # Mutation or crossover
                if random.random() < self.mutation_probability or len(self.pareto_frontier) < 2:
                    print("Operation: Mutation")
                    parent = self._sample_from_frontier()
                    print(f"  Parent: variant_{parent.id} ({self._format_metrics(parent.metrics)})")
                    new_variant = self._mutate(parent)
                else:
                    print("Operation: Crossover")
                    primary, secondary = self._sample_two_from_frontier()
                    print(f"  Primary: variant_{primary.id}, Secondary: variant_{secondary.id}")
                    new_variant = self._crossover(primary, secondary)

                if new_variant is None:
                    print("  Failed to create new variant")
                    continue

                eval_result = self._evaluate_variant(new_variant)
                new_variant.metrics = eval_result["metrics"]
                new_variant.feedback = eval_result["feedback"]
                metrics_str = self._format_metrics(new_variant.metrics)
                print(f"  New variant_{new_variant.id}: {metrics_str}")

                added = self._update_pareto_frontier(new_variant)
                print(f"  {'Added to' if added else 'Not added to'} Pareto frontier")

                best = self.get_best_variant()
                print(f"  Best: variant_{best.id} ({self._format_metrics(best.metrics)})")
                self._copy_best_to_optimal(best)

            print("\n=== Evolution Complete ===")
            print(f"Final Pareto frontier size: {len(self.pareto_frontier)}")
            for v in self.pareto_frontier:
                print(f"  variant_{v.id}: {self._format_metrics(v.metrics)}")

            best = self.get_best_variant()
            return best
        finally:
            shutil.rmtree(self.work_dir)
            print(f"Cleaned up work directory: {self.work_dir}")

    def _copy_best_to_optimal(self, best: AgentVariant) -> None:
        """Copy the best variant to the optimal_agent folder.

        Uses a temporary directory and atomic rename to avoid race conditions
        where the optimal_dir might be read while being updated.
        """
        self.optimal_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy to a temporary location first (same parent ensures same filesystem)
        temp_dir = self.optimal_dir.parent / f"{self.optimal_dir.name}_{os.getpid()}_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(best.folder_path, temp_dir)

        # Atomic replace: remove old and rename in one step where possible
        if self.optimal_dir.exists():
            shutil.rmtree(self.optimal_dir)
        temp_dir.rename(self.optimal_dir)

        print(f"  Copied best variant to {self.optimal_dir}")

    def get_best_variant(self) -> AgentVariant:
        """Get the best variant by combined score."""
        if not self.pareto_frontier:
            raise RuntimeError("No variants available")
        return min(self.pareto_frontier, key=lambda v: v.score())

    def get_pareto_frontier(self) -> list[AgentVariant]:
        """Get all variants in the Pareto frontier."""
        return self.pareto_frontier.copy()

    def save_state(self, path: str) -> None:
        """Save the evolver state to a JSON file.

        Args:
            path: Path where to save the state JSON file. Required because
                  work_dir is cleaned up after evolve() completes.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "task_description": self.task_description,
            "generation": self._generation,
            "variant_counter": self._variant_counter,
            "pareto_frontier": [v.to_dict() for v in self.pareto_frontier],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"State saved to {path}")


# Very long task description for testing
LONG_RUNNING_TASK = """
> **Task:** Create a robust database engine using only Bash scripts.
>
> **Requirements:**
> 1.  Create a script named `db.sh` that interacts with a local data folder.
> 2.  **Basic Operations:** Implement `db.sh set <key> <value>`,
>     `db.sh get <key>`, and `db.sh delete <key>`.
> 3.  **Atomicity:** Implement transaction support.
>     *   `db.sh begin` starts a session where writes are cached but not visible to others.
>     *   `db.sh commit` atomically applies all cached changes.
>     *   `db.sh rollback` discards pending changes.
> 4.  **Concurrency:** Ensure that if two different terminal windows run `db.sh`
>     simultaneously, the data is never corrupted (use `mkdir`-based mutex locking).
> 5.  **Validation:** Write a test script `test_stress.sh` that launches 10
>     concurrent processes to spam the database, verifying no data is lost.
>
> **Constraints:**
> *   No external database tools (no sqlite3, no python).
> *   Standard Linux utilities only (sed, awk, grep, flock/mkdir).
> *   Safe: Operate entirely within a `./my_db` directory.
"""

def main() -> None:
    """Run the AgentEvolver on a long-running task."""
    evolver = AgentEvolver(
        task_description=LONG_RUNNING_TASK,
        max_generations=20,
        initial_frontier_size=4,
        max_frontier_size=6,
        mutation_probability=0.8,
    )

    best = evolver.evolve()

    print("\n=== Final Result ===")
    print(f"Best variant: {best.folder_path}")
    print(f"Metrics: {best.metrics}")
    print(f"Generation: {best.generation}")

    # Save state to the optimal directory (work_dir is cleaned up after evolve)
    state_path = str(evolver.optimal_dir / "evolver_state.json")
    evolver.save_state(state_path)


if __name__ == "__main__":
    main()
