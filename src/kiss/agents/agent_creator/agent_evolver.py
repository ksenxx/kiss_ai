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
2. Mutation: Sample and improve a single (folder, report) pair
3. Crossover: Combine ideas from two (folder, report) pairs
4. Configurable mutation vs crossover probability
5. Comprehensive tracking of lineage and improvements

Pseudo-code for AgentEvolver Algorithm (showing use of parameters):

Inputs:
    - Initial_population_size
    - Num_generations
    - mutation_probability (P_mutation)
    - crossover_probability (P_crossover)
    - Pareto_objectives: [token efficiency, execution time]
    - evaluation_metrics

Algorithm:

1. Initialize population with Initial_population_size agent variants
   For each agent variant:
       - Randomly initialize or use seed agents
       - Evaluate metrics (tokens_used, execution_time)

2. For generation = 1 to Num_generations:
    a. Maintain Pareto frontier:
        - For every agent in population:
            - Update dominates_count and dominated_by_count
        - Pareto frontier = all agents not dominated by any other

    b. For each offspring to create:
        - With probability P_mutation:
            - Select one agent from from Pareto frontier
            - Apply mutation using ImproverAgent
        - Otherwise:
            - Select two parents from population (can be weighted by performance)
            - Apply crossover to produce a child

    c. For every new agent variant produced:
        - Evaluate metrics (tokens_used, execution_time)
        - Assign id, generation, parent_ids

    d. Combine new variants with current population
        - Keep population size constant (optionally use Pareto dominance to select survivors)
        - Update tracking info (lineage, improvement trajectory)

    e. (Optional) Log/report Pareto front for analysis

3. After Num_generations, return the final Pareto frontier and agent histories.

Parameters influence:
    - Initial_population_size: size of population at the start and per generation
    - Num_generations: total evolutionary cycles to run
    - P_mutation: probability to choose mutation over crossover
    - Pareto_objectives: define how dominance is determined and which agents comprise the frontier


"""

import json
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio

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
    tokens_used: int = 0
    execution_time: float = 0.0
    id: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)

    def dominates(self, other: "AgentVariant") -> bool:
        """Check if this variant Pareto-dominates another."""
        at_least_as_good = (
            self.tokens_used <= other.tokens_used
            and self.execution_time <= other.execution_time
        )
        strictly_better = (
            self.tokens_used < other.tokens_used
            or self.execution_time < other.execution_time
        )
        return at_least_as_good and strictly_better

    def score(self) -> float:
        """Combined score for ranking (lower is better)."""
        return self.tokens_used + self.execution_time * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "folder_path": self.folder_path,
            "report_path": self.report_path,
            "tokens_used": self.tokens_used,
            "execution_time": self.execution_time,
            "id": self.id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


# Prompt for initial agent creation
INITIAL_AGENT_PROMPT = """You are an expert at building efficient, long-running AI agents.
Your task is to create an initial agent implementation based on the following requirements.

## Task Description
{task_description}

## Agent Requirements

The agent must be designed for **long-running, complex tasks** using
the Agent API available at {kiss_folder}/API.md and should implement
a simple KISSAgent to solve the task.  Create the following files in {target_folder}:

1. `agent.py` - Main agent implementation with:
2. `config.py` - Agent configuration
3. `__init__.py` - Agent package initialization
4. `README.md` - Agent documentation
5. `test_agent.py` - Tests for the agent
6. `requirements.txt` - Dependencies for the agent
7. Any other files necessary for the agent to function properly.

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
        evaluation_fn: Any = None,
        model_name: str | None = None,
        max_generations: int | None = None,
        max_frontier_size: int | None = None,
        mutation_probability: float | None = None,
        coding_agent_type: Literal["claude code", "gemini cli", "openai codex"] | None = None,
    ):
        """Initialize the AgentEvolver.

        Args:
            task_description: Description of the task the agent should perform.
            evaluation_fn: Optional function to evaluate agent variants.
            model_name: LLM model to use for agent creation and improvement.
            max_generations: Maximum number of improvement generations.
            max_frontier_size: Maximum size of the Pareto frontier.
            mutation_probability: Probability of mutation vs crossover.
            coding_agent_type: Which coding agent to use: 'claude', 'gemini', or 'openai_codex'.
        """
        cfg = getattr(DEFAULT_CONFIG, "agent_creator", None)
        evolver_cfg = cfg.evolver if cfg else None

        self.task_description = task_description
        self.evaluation_fn = evaluation_fn
        self.model_name = get_config_value(model_name, evolver_cfg, "model_name")
        self.max_generations = get_config_value(max_generations, evolver_cfg, "max_generations")
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

    async def _create_initial_agent(self) -> AgentVariant | None:
        """Create the initial agent from scratch."""
        variant_id = self._next_variant_id()
        target_folder, report_path = self._get_variant_paths(variant_id)
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        agent = create_coding_agent(self.coding_agent_type, "Initial Agent Creator")
        result = await agent.run(
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

        if result is None:
            print("Failed to create initial agent")
            return None

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

    async def _evaluate_variant(self, variant: AgentVariant) -> None:
        """Evaluate a variant to measure its metrics."""
        if self.evaluation_fn is None:
            # Default evaluation: run the agent and measure
            print(f"Evaluating variant {variant.id}...")
            start_time = time.time()

            # Simple evaluation: just measure time and estimate tokens
            # In real usage, this would run the agent on test tasks
            variant.execution_time = time.time() - start_time + random.uniform(1, 10)
            variant.tokens_used = random.randint(1000, 10000)  # Placeholder
        else:
            tokens, exec_time = self.evaluation_fn(variant.folder_path)
            variant.tokens_used = tokens
            variant.execution_time = exec_time

        # Update the report with actual metrics
        variant.report.improved_tokens = variant.tokens_used
        variant.report.improved_time = variant.execution_time
        variant.report.save(variant.report_path)

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
        tokens = [v.tokens_used for v in self.pareto_frontier]
        times = [v.execution_time for v in self.pareto_frontier]

        tokens_range = max(tokens) - min(tokens) or 1
        times_range = max(times) - min(times) or 1

        # Calculate crowding distance
        crowding = [0.0] * n

        # Add crowding for tokens dimension
        sorted_by_tokens = sorted(range(n), key=lambda i: tokens[i])
        crowding[sorted_by_tokens[0]] = crowding[sorted_by_tokens[-1]] = float("inf")
        for i in range(1, n - 1):
            idx = sorted_by_tokens[i]
            token_diff = tokens[sorted_by_tokens[i + 1]] - tokens[sorted_by_tokens[i - 1]]
            crowding[idx] += token_diff / tokens_range

        # Add crowding for time dimension
        sorted_by_time = sorted(range(n), key=lambda i: times[i])
        crowding[sorted_by_time[0]] = crowding[sorted_by_time[-1]] = float("inf")
        for i in range(1, n - 1):
            idx = sorted_by_time[i]
            time_diff = times[sorted_by_time[i + 1]] - times[sorted_by_time[i - 1]]
            crowding[idx] += time_diff / times_range

        # Keep most diverse (highest crowding distance)
        sorted_indices = sorted(range(n), key=lambda i: crowding[i], reverse=True)
        kept_indices = sorted_indices[: self.max_frontier_size]
        self.pareto_frontier = [self.pareto_frontier[i] for i in kept_indices]

    def _sample_from_frontier(self) -> AgentVariant:
        """Sample a variant uniformly from the Pareto frontier."""
        return random.choice(self.pareto_frontier)

    def _sample_two_from_frontier(self) -> tuple[AgentVariant, AgentVariant]:
        """Sample two different variants from the Pareto frontier."""
        if len(self.pareto_frontier) < 2:
            return self.pareto_frontier[0], self.pareto_frontier[0]

        v1, v2 = random.sample(self.pareto_frontier, 2)
        # Return (better, worse) based on score
        return (v1, v2) if v1.score() <= v2.score() else (v2, v1)

    async def _mutate(self, variant: AgentVariant) -> AgentVariant | None:
        """Create a new variant by mutating an existing one."""
        new_id = self._next_variant_id()
        target_folder, report_path = self._get_variant_paths(new_id)

        success, new_report = await self.improver.improve(
            source_folder=variant.folder_path,
            target_folder=target_folder,
            report_path=variant.report_path,
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

    async def _crossover(
        self, primary: AgentVariant, secondary: AgentVariant
    ) -> AgentVariant | None:
        """Create a new variant by crossing over two variants."""
        new_id = self._next_variant_id()
        target_folder, report_path = self._get_variant_paths(new_id)

        success, new_report = await self.improver.crossover_improve(
            primary_folder=primary.folder_path,
            primary_report_path=primary.report_path,
            secondary_report_path=secondary.report_path,
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

    async def evolve(self) -> AgentVariant:
        """Run the evolutionary optimization."""
        try:
            print(f"Starting AgentEvolver with {self.max_generations} generations")
            task_preview = self.task_description[:80]
            print(f"Max frontier size: {self.max_frontier_size}, Task: {task_preview}...")

            # Initialize with first agent
            print("\nInitializing...")
            initial = await self._create_initial_agent()
            if initial is None:
                raise RuntimeError("Failed to create initial agent")

            await self._evaluate_variant(initial)
            self._update_pareto_frontier(initial)
            exec_time = initial.execution_time
            print(f"Initial agent: tokens={initial.tokens_used}, time={exec_time:.2f}s")
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
                    print(f"  Parent: variant_{parent.id} (tokens={parent.tokens_used})")
                    new_variant = await self._mutate(parent)
                else:
                    print("Operation: Crossover")
                    primary, secondary = self._sample_two_from_frontier()
                    print(f"  Primary: variant_{primary.id}, Secondary: variant_{secondary.id}")
                    new_variant = await self._crossover(primary, secondary)

                if new_variant is None:
                    print("  Failed to create new variant")
                    continue

                await self._evaluate_variant(new_variant)
                print(
                    f"  New variant_{new_variant.id}: tokens={new_variant.tokens_used}, "
                    f"time={new_variant.execution_time:.2f}s"
                )

                added = self._update_pareto_frontier(new_variant)
                print(f"  {'Added to' if added else 'Not added to'} Pareto frontier")

                best = self.get_best_variant()
                print(f"  Best: variant_{best.id} (tokens={best.tokens_used})")
                self._copy_best_to_optimal(best)

            print("\n=== Evolution Complete ===")
            print(f"Final Pareto frontier size: {len(self.pareto_frontier)}")
            for v in self.pareto_frontier:
                print(f"  variant_{v.id}: tokens={v.tokens_used}, time={v.execution_time:.2f}s")

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
        # Copy to a temporary location first
        temp_dir = self.optimal_dir.parent / f"{self.optimal_dir.name}_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(best.folder_path, temp_dir)

        # Remove old optimal_dir and rename temp to optimal
        old_dir = self.optimal_dir.parent / f"{self.optimal_dir.name}_old"
        if self.optimal_dir.exists():
            if old_dir.exists():
                shutil.rmtree(old_dir)
            self.optimal_dir.rename(old_dir)

        temp_dir.rename(self.optimal_dir)

        # Clean up old directory
        if old_dir.exists():
            shutil.rmtree(old_dir)

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

async def main() -> None:
    """Run the AgentEvolver on a long-running task."""
    evolver = AgentEvolver(
        task_description=LONG_RUNNING_TASK,
        max_generations=5,  # Reduced for testing
        max_frontier_size=4,
        mutation_probability=0.5,
    )

    best = await evolver.evolve()

    print("\n=== Final Result ===")
    print(f"Best variant: {best.folder_path}")
    print(f"Tokens used: {best.tokens_used}")
    print(f"Execution time: {best.execution_time:.2f}s")
    print(f"Generation: {best.generation}")

    # Save state to the optimal directory (work_dir is cleaned up after evolve)
    state_path = str(evolver.optimal_dir / "evolver_state.json")
    evolver.save_state(state_path)


if __name__ == "__main__":
    anyio.run(main)
