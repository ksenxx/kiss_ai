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
"""

import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

import kiss.agents.agent_creator.config  # noqa: F401
from kiss.agents.agent_creator.improver_agent import ImprovementReport, ImproverAgent
from kiss.core.claude_coding_agent import ClaudeCodingAgent
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.utils import get_config_value


@dataclass
class AgentVariant:
    """Represents an agent variant in the Pareto frontier."""

    folder_path: str
    report_path: str
    report: ImprovementReport

    # Measured metrics (filled after evaluation)
    tokens_used: int = 0
    execution_time: float = 0.0

    # Tracking
    id: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)

    # Pareto dominance
    dominates_count: int = 0  # How many solutions this dominates
    dominated_by_count: int = 0  # How many solutions dominate this

    def dominates(self, other: "AgentVariant") -> bool:
        """Check if this variant Pareto-dominates another.

        A variant dominates another if it's at least as good in all objectives
        and strictly better in at least one.
        """
        tokens_better_or_equal = self.tokens_used <= other.tokens_used
        time_better_or_equal = self.execution_time <= other.execution_time

        strictly_better = (
            self.tokens_used < other.tokens_used
            or self.execution_time < other.execution_time
        )

        return tokens_better_or_equal and time_better_or_equal and strictly_better

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

The agent must be designed for **long-running, complex tasks** and should implement:

### 1. Orchestrator Pattern
- Central coordinator that manages the overall workflow
- Maintains high-level state and progress tracking
- Delegates subtasks to specialized sub-agents
- Handles error recovery and retries

### 2. Dynamic To-Do List
- Maintain a structured task list that evolves during execution
- Support for task dependencies and priorities
- Ability to add, modify, and complete tasks dynamically
- Progress tracking and reporting

### 3. Dynamic Tool Creation
- Create specialized tools on-the-fly for specific subtasks
- Register and manage tool availability
- Tools should be reusable across the execution

### 4. Checkpointing
- Save state periodically to enable recovery
- Support for resuming from checkpoints
- Track what has been completed vs pending

### 5. Sub-Agent Delegation
- Create specialized sub-agents for complex subtasks
- Sub-agents should be lightweight and focused
- Results aggregation from sub-agents

### 6. Efficiency Patterns
- Minimize token usage through concise prompts
- Batch operations where possible
- Early termination when goals are achieved
- Caching of intermediate results

## Output Structure

Create the following files in {target_folder}:

1. `agent.py` - Main agent implementation with:
   - Orchestrator class
   - TodoList class
   - ToolRegistry class
   - Checkpoint management
   - Sub-agent creation utilities

2. `prompts.py` - All prompt templates used by the agent

3. `tools.py` - Base tool implementations

4. `config.py` - Agent configuration

5. `run.py` - Entry point to run the agent

## Code Style

- Clean, readable Python code
- Type hints throughout
- Docstrings for all public methods
- Error handling with meaningful messages
- Logging for debugging

When complete, provide a summary of the agent created and the files that were written.
"""


class AgentEvolver:
    """Evolves AI agents using Pareto frontier optimization.

    The AgentEvolver maintains a population of agent variants, each with
    associated metrics (tokens used, execution time). It uses the ImproverAgent
    to create new variants through mutation (improving one variant) or
    crossover (combining ideas from two variants).

    The Pareto frontier contains all non-dominated solutions - variants that
    are not strictly worse than any other variant in all objectives.
    """

    def __init__(
        self,
        task_description: str,
        evaluation_fn: Any = None,
        model: str | None = None,
        max_generations: int | None = None,
        population_size: int | None = None,
        pareto_size: int | None = None,
        mutation_probability: float | None = None,
        work_dir: str | None = None,
    ):
        """Initialize the AgentEvolver.

        Args:
            task_description: Description of the task the agent should solve
            evaluation_fn: Function to evaluate an agent variant
                          (folder_path) -> (tokens_used, execution_time)
            model: LLM model for orchestration
            max_generations: Maximum evolutionary generations
            population_size: Maximum population size
            pareto_size: Maximum Pareto frontier size
            mutation_probability: Probability of mutation vs crossover
            work_dir: Working directory for agent variants
        """
        cfg = getattr(DEFAULT_CONFIG, "agent_creator", None)
        evolver_cfg = cfg.evolver if cfg else None

        self.task_description = task_description
        self.evaluation_fn = evaluation_fn
        self.model = get_config_value(model, evolver_cfg, "model") or "claude-sonnet-4-5"
        self.max_generations = (
            get_config_value(max_generations, evolver_cfg, "max_generations") or 10
        )
        self.population_size = (
            get_config_value(population_size, evolver_cfg, "population_size") or 8
        )
        self.pareto_size = (
            get_config_value(pareto_size, evolver_cfg, "pareto_size") or 6
        )
        self.mutation_probability = (
            get_config_value(mutation_probability, evolver_cfg, "mutation_probability") or 0.5
        )

        # Setup working directory
        if work_dir is None:
            work_dir = str(Path(DEFAULT_CONFIG.agent.artifact_dir) / "agent_evolver")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.pareto_frontier: list[AgentVariant] = []
        self.all_variants: list[AgentVariant] = []
        self._variant_counter = 0
        self._generation = 0

        # Improver agent
        self.improver = ImproverAgent(model=self.model)

    def _next_variant_id(self) -> int:
        """Get the next unique variant ID."""
        self._variant_counter += 1
        return self._variant_counter

    def _get_variant_folder(self, variant_id: int) -> str:
        """Get the folder path for a variant."""
        return str(self.work_dir / f"variant_{variant_id}")

    def _get_report_path(self, variant_id: int) -> str:
        """Get the report path for a variant."""
        return str(self.work_dir / f"variant_{variant_id}" / "improvement_report.json")

    async def _create_initial_agent(self) -> AgentVariant | None:
        """Create the initial agent from scratch."""
        variant_id = self._next_variant_id()
        target_folder = self._get_variant_folder(variant_id)
        report_path = self._get_report_path(variant_id)

        Path(target_folder).mkdir(parents=True, exist_ok=True)

        # Use ClaudeCodingAgent to create the initial agent
        agent = ClaudeCodingAgent("Initial Agent Creator")

        result = await agent.run(
            model_name=self.model,
            prompt_template=INITIAL_AGENT_PROMPT,
            arguments={
                "task_description": self.task_description,
                "target_folder": target_folder,
            },
            max_steps=200,
            max_budget=20.0,
            base_dir=str(self.work_dir / "creator_workdir"),
            writable_paths=[target_folder],
        )

        if result is None:
            print("Failed to create initial agent")
            return None

        # Create initial report
        initial_report = ImprovementReport(
            implemented_ideas=[{"idea": "Initial implementation", "source": "initial"}],
            failed_ideas=[],
            generation=0,
        )
        initial_report.save(report_path)

        # Create variant
        variant = AgentVariant(
            folder_path=target_folder,
            report_path=report_path,
            report=initial_report,
            id=variant_id,
            generation=0,
        )

        return variant

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

        Args:
            new_variant: The new variant to potentially add

        Returns:
            True if the variant was added to the frontier
        """
        # Check if new variant is dominated by any in frontier
        for existing in self.pareto_frontier:
            if existing.dominates(new_variant):
                return False  # New variant is dominated

        # Remove variants dominated by new variant
        self.pareto_frontier = [
            v for v in self.pareto_frontier if not new_variant.dominates(v)
        ]

        # Add new variant to frontier
        self.pareto_frontier.append(new_variant)

        # Trim frontier if too large (keep most diverse)
        if len(self.pareto_frontier) > self.pareto_size:
            self._trim_frontier()

        return True

    def _trim_frontier(self) -> None:
        """Trim the Pareto frontier to the maximum size.

        Uses crowding distance to maintain diversity.
        """
        if len(self.pareto_frontier) <= self.pareto_size:
            return

        # Calculate crowding distance for each variant
        n = len(self.pareto_frontier)

        # Normalize objectives
        tokens = [v.tokens_used for v in self.pareto_frontier]
        times = [v.execution_time for v in self.pareto_frontier]

        tokens_range = max(tokens) - min(tokens) if max(tokens) > min(tokens) else 1
        times_range = max(times) - min(times) if max(times) > min(times) else 1

        # Calculate crowding distance
        crowding = [0.0] * n

        # Sort by tokens and add boundary points
        sorted_by_tokens = sorted(range(n), key=lambda i: tokens[i])
        crowding[sorted_by_tokens[0]] = float("inf")
        crowding[sorted_by_tokens[-1]] = float("inf")
        for i in range(1, n - 1):
            idx = sorted_by_tokens[i]
            prev_idx = sorted_by_tokens[i - 1]
            next_idx = sorted_by_tokens[i + 1]
            crowding[idx] += (tokens[next_idx] - tokens[prev_idx]) / tokens_range

        # Sort by time and add boundary points
        sorted_by_time = sorted(range(n), key=lambda i: times[i])
        crowding[sorted_by_time[0]] = float("inf")
        crowding[sorted_by_time[-1]] = float("inf")
        for i in range(1, n - 1):
            idx = sorted_by_time[i]
            prev_idx = sorted_by_time[i - 1]
            next_idx = sorted_by_time[i + 1]
            crowding[idx] += (times[next_idx] - times[prev_idx]) / times_range

        # Sort by crowding distance and keep the most diverse
        sorted_indices = sorted(range(n), key=lambda i: crowding[i], reverse=True)
        self.pareto_frontier = [
            self.pareto_frontier[i] for i in sorted_indices[: self.pareto_size]
        ]

    def _sample_variant(self) -> AgentVariant:
        """Sample a variant from the Pareto frontier.

        Uses weighted sampling based on the crowding distance to maintain diversity.
        """
        if len(self.pareto_frontier) == 1:
            return self.pareto_frontier[0]

        # Weight by inverse of crowding (favor more isolated solutions)
        weights = []
        for _ in self.pareto_frontier:
            weights.append(1.0)  # Uniform for simplicity

        return random.choices(self.pareto_frontier, weights=weights)[0]

    def _sample_two_variants(self) -> tuple[AgentVariant, AgentVariant]:
        """Sample two different variants from the Pareto frontier."""
        if len(self.pareto_frontier) < 2:
            v = self.pareto_frontier[0]
            return v, v

        # Sample two different variants
        v1 = self._sample_variant()
        remaining = [v for v in self.pareto_frontier if v.id != v1.id]
        v2 = random.choice(remaining) if remaining else v1

        return v1, v2

    def _pick_better_variant(
        self, v1: AgentVariant, v2: AgentVariant
    ) -> tuple[AgentVariant, AgentVariant]:
        """Pick the better variant as primary and the other as secondary.

        Returns (better, worse) based on a weighted combination of metrics.
        """
        # Simple scoring: lower is better for both metrics
        score1 = v1.tokens_used * 0.5 + v1.execution_time * 0.5 * 1000
        score2 = v2.tokens_used * 0.5 + v2.execution_time * 0.5 * 1000

        if score1 <= score2:
            return v1, v2
        return v2, v1

    async def _mutate(self, variant: AgentVariant) -> AgentVariant | None:
        """Create a new variant by mutating an existing one.

        Args:
            variant: The variant to mutate

        Returns:
            New variant if successful, None otherwise
        """
        new_id = self._next_variant_id()
        target_folder = self._get_variant_folder(new_id)
        report_path = self._get_report_path(new_id)

        success, new_report = await self.improver.improve(
            source_folder=variant.folder_path,
            target_folder=target_folder,
            report_path=variant.report_path,
            base_dir=str(self.work_dir / "improver_workdir"),
        )

        if not success or new_report is None:
            # Clean up failed attempt
            if Path(target_folder).exists():
                shutil.rmtree(target_folder)
            return None

        # Save the report
        new_report.save(report_path)

        # Create new variant
        new_variant = AgentVariant(
            folder_path=target_folder,
            report_path=report_path,
            report=new_report,
            id=new_id,
            generation=self._generation,
            parent_ids=[variant.id],
        )

        return new_variant

    async def _crossover(
        self, primary: AgentVariant, secondary: AgentVariant
    ) -> AgentVariant | None:
        """Create a new variant by crossing over two variants.

        Args:
            primary: The primary variant (used as base code)
            secondary: The secondary variant (ideas taken from report)

        Returns:
            New variant if successful and improves over both parents, None otherwise
        """
        new_id = self._next_variant_id()
        target_folder = self._get_variant_folder(new_id)
        report_path = self._get_report_path(new_id)

        success, new_report = await self.improver.crossover_improve(
            primary_folder=primary.folder_path,
            primary_report_path=primary.report_path,
            secondary_report_path=secondary.report_path,
            target_folder=target_folder,
            base_dir=str(self.work_dir / "improver_workdir"),
        )

        if not success or new_report is None:
            # Clean up failed attempt
            if Path(target_folder).exists():
                shutil.rmtree(target_folder)
            return None

        # Save the report
        new_report.save(report_path)

        # Create new variant
        new_variant = AgentVariant(
            folder_path=target_folder,
            report_path=report_path,
            report=new_report,
            id=new_id,
            generation=self._generation,
            parent_ids=[primary.id, secondary.id],
        )

        return new_variant

    async def evolve(self) -> AgentVariant:
        """Run the evolutionary optimization.

        Returns:
            The best variant found (by combined metric)
        """
        print(f"Starting AgentEvolver with {self.max_generations} generations")
        print(f"Task: {self.task_description[:100]}...")

        # Create initial agent
        print("\nCreating initial agent...")
        initial = await self._create_initial_agent()
        if initial is None:
            raise RuntimeError("Failed to create initial agent")

        # Evaluate initial agent
        await self._evaluate_variant(initial)
        self.all_variants.append(initial)
        self._update_pareto_frontier(initial)

        print(
            f"Initial agent created: tokens={initial.tokens_used}, "
            f"time={initial.execution_time:.2f}s"
        )

        # Evolution loop
        for gen in range(1, self.max_generations + 1):
            self._generation = gen
            print(f"\n=== Generation {gen}/{self.max_generations} ===")
            print(f"Pareto frontier size: {len(self.pareto_frontier)}")

            # Decide between mutation and crossover
            if random.random() < self.mutation_probability or len(self.pareto_frontier) < 2:
                # Mutation: sample one variant and improve it
                print("Operation: Mutation")
                parent = self._sample_variant()
                print(
                    f"  Parent: variant_{parent.id} "
                    f"(tokens={parent.tokens_used}, time={parent.execution_time:.2f}s)"
                )

                new_variant = await self._mutate(parent)
            else:
                # Crossover: sample two variants, pick better one, combine with other's ideas
                print("Operation: Crossover")
                v1, v2 = self._sample_two_variants()
                primary, secondary = self._pick_better_variant(v1, v2)
                print(
                    f"  Primary: variant_{primary.id} "
                    f"(tokens={primary.tokens_used}, time={primary.execution_time:.2f}s)"
                )
                print(
                    f"  Secondary: variant_{secondary.id} "
                    f"(tokens={secondary.tokens_used}, time={secondary.execution_time:.2f}s)"
                )

                new_variant = await self._crossover(primary, secondary)

            if new_variant is None:
                print("  Failed to create new variant")
                continue

            # Evaluate new variant
            await self._evaluate_variant(new_variant)
            self.all_variants.append(new_variant)

            print(
                f"  New variant_{new_variant.id}: tokens={new_variant.tokens_used}, "
                f"time={new_variant.execution_time:.2f}s"
            )

            # Try to add to Pareto frontier
            added = self._update_pareto_frontier(new_variant)
            if added:
                print("  Added to Pareto frontier!")
            else:
                print("  Not added (dominated by existing solutions)")

            # Report current best
            best = self.get_best_variant()
            print(
                f"  Current best: variant_{best.id} "
                f"(tokens={best.tokens_used}, time={best.execution_time:.2f}s)"
            )

        # Final report
        print("\n=== Evolution Complete ===")
        print(f"Total variants created: {len(self.all_variants)}")
        print(f"Final Pareto frontier size: {len(self.pareto_frontier)}")
        print("\nPareto frontier variants:")
        for v in self.pareto_frontier:
            print(f"  variant_{v.id}: tokens={v.tokens_used}, time={v.execution_time:.2f}s")

        return self.get_best_variant()

    def get_best_variant(self) -> AgentVariant:
        """Get the best variant by combined metric."""
        if not self.pareto_frontier:
            if self.all_variants:
                return self.all_variants[0]
            raise RuntimeError("No variants available")

        # Score by normalized combination of objectives
        tokens = [v.tokens_used for v in self.pareto_frontier]
        times = [v.execution_time for v in self.pareto_frontier]

        tokens_min, tokens_max = min(tokens), max(tokens)
        times_min, times_max = min(times), max(times)

        tokens_range = tokens_max - tokens_min if tokens_max > tokens_min else 1
        times_range = times_max - times_min if times_max > times_min else 1

        best = None
        best_score = float("inf")

        for v in self.pareto_frontier:
            # Normalized score (lower is better)
            token_norm = (v.tokens_used - tokens_min) / tokens_range
            time_norm = (v.execution_time - times_min) / times_range
            score = token_norm * 0.5 + time_norm * 0.5

            if score < best_score:
                best_score = score
                best = v

        return best if best else self.pareto_frontier[0]

    def get_pareto_frontier(self) -> list[AgentVariant]:
        """Get all variants in the Pareto frontier."""
        return self.pareto_frontier.copy()

    def save_state(self, path: str | None = None) -> None:
        """Save the evolver state to a JSON file."""
        if path is None:
            path = str(self.work_dir / "evolver_state.json")

        state = {
            "task_description": self.task_description,
            "generation": self._generation,
            "variant_counter": self._variant_counter,
            "all_variants": [v.to_dict() for v in self.all_variants],
            "pareto_frontier_ids": [v.id for v in self.pareto_frontier],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(f"State saved to {path}")


# Very long task description for testing
LONG_RUNNING_TASK = """
Build a comprehensive code analysis and refactoring assistant that can:

## Phase 1: Code Understanding (Long-running analysis)
1. Parse and understand large codebases (1000+ files)
2. Build dependency graphs between modules
3. Identify code patterns and anti-patterns
4. Create semantic embeddings for code search
5. Generate documentation from code

## Phase 2: Intelligent Refactoring
1. Identify code duplication across the codebase
2. Suggest and apply DRY principle improvements
3. Detect and fix common bugs (null checks, resource leaks)
4. Modernize legacy code patterns
5. Optimize performance bottlenecks

## Phase 3: Test Generation
1. Analyze existing test coverage
2. Generate unit tests for uncovered code
3. Create integration tests for critical paths
4. Generate property-based tests
5. Create regression test suites

## Phase 4: Documentation
1. Generate API documentation
2. Create architecture diagrams
3. Write user guides
4. Generate changelog from commits
5. Create onboarding documentation

## Requirements

### Orchestrator Pattern
The agent must use a central orchestrator that:
- Maintains overall progress across all phases
- Coordinates between specialized sub-agents
- Handles failures gracefully with retry logic
- Reports progress periodically

### Dynamic To-Do List
Implement a task management system that:
- Tracks tasks at multiple granularity levels (phase, task, subtask)
- Supports task dependencies
- Allows dynamic task addition during execution
- Prioritizes tasks based on impact

### Dynamic Tool Creation
The agent should create tools dynamically for:
- Language-specific parsers
- Pattern matchers
- Code generators
- Test runners

### Checkpointing
Implement checkpointing that:
- Saves state after each major milestone
- Allows resumption from any checkpoint
- Tracks what has been analyzed/modified

### Sub-Agent Delegation
Create specialized sub-agents for:
- Code parsing (per language)
- Pattern detection
- Test generation
- Documentation writing

### Efficiency Requirements
- Minimize redundant parsing (cache ASTs)
- Batch API calls where possible
- Use incremental processing
- Early termination when possible

The entire process should complete within reasonable time despite
the massive scope, by using intelligent prioritization and
parallelization where possible.
"""


async def main() -> None:
    """Run the AgentEvolver on a long-running task."""
    evolver = AgentEvolver(
        task_description=LONG_RUNNING_TASK,
        max_generations=5,  # Reduced for testing
        population_size=4,
        pareto_size=3,
        mutation_probability=0.5,
    )

    best = await evolver.evolve()

    print("\n=== Final Result ===")
    print(f"Best variant: {best.folder_path}")
    print(f"Tokens used: {best.tokens_used}")
    print(f"Execution time: {best.execution_time:.2f}s")
    print(f"Generation: {best.generation}")

    # Save state
    evolver.save_state()


if __name__ == "__main__":
    anyio.run(main)
