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
    - initial_frontier_size: Number of initial agents to create
    - max_frontier_size: Maximum size of the Pareto frontier
    - mutation_probability: Probability of mutation vs crossover (0.0 to 1.0)

Data Structures:
    AgentVariant:
        - folder_path: Directory containing agent code
        - report_path: Path to improvement report JSON file
        - report: ImprovementReport tracking implemented/failed ideas
        - metrics: {success, tokens_used, execution_time, ...}
        - id, generation, parent_ids (for lineage tracking)

    dominates(A, B):
        # A dominates B if A is at least as good in all metrics and strictly better in one
        # ALL metrics are minimized (lower is better)
        # Note: success is 0 for success, 1 for failure (so we minimize it)

    score(variant, weights=None):
        # Combined ranking score (lower is better)
        # Default weights: success * 1,000,000 + tokens_used * 1 + execution_time * 1000
        # (success=0 is best, success=1 is worst, so this prioritizes successful agents)

Algorithm EVOLVE():
    1. INITIALIZE
       - Create temporary work_dir for variants
       - Set optimal_dir for storing best agent
       - Initialize empty pareto_frontier

    2. CREATE INITIAL AGENTS
       - WHILE len(pareto_frontier) < initial_frontier_size:
           - Use coding agent to generate agent files from task_description
           - Agent must implement agent_run(task) -> {metrics: {...}}
           - Evaluate agent by calling agent_run(task_description)
           - Update pareto_frontier (may reject if dominated)
           - Copy current best variant (min score) to optimal_dir

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
          - Evaluate: load agent.py from isolated temp dir, call agent_run(task_description)
          - Store metrics from evaluation result
          - Update pareto_frontier:
              - Reject if dominated by any existing variant
              - Remove variants dominated by new_variant
              - Add new_variant
              - If frontier > max_size: trim using crowding distance
          - Copy best variant (min score) to optimal_dir

    4. RETURN best variant from pareto_frontier (min score)

    5. CLEANUP work_dir
"""

import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kiss.agents.create_and_optimize_agent.config  # type: ignore # noqa: F401
from kiss.agents.create_and_optimize_agent.improver_agent import (
    ImprovementReport,
    ImproverAgent,
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
    metrics: dict[str, float]
    parent_ids: list[int]
    id: int = 0
    generation: int = 0

    def dominates(self, other: "AgentVariant") -> bool:
        """Check if this variant Pareto-dominates another.

        A variant dominates another if it is at least as good in all metrics
        and strictly better in at least one metric. All metrics are minimized
        (lower is better).

        Args:
            other: The other AgentVariant to compare against.

        Returns:
            True if this variant dominates the other, False otherwise.
        """
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
        """Calculate a combined score for ranking variants.

        Computes a weighted sum of metrics for ranking purposes. Lower scores
        are better. Default weights prioritize success, then minimize tokens
        and execution time.

        Args:
            weights: Dict mapping metric names to weights. Positive weights mean
                the metric should be minimized, negative weights mean maximized.
                If None, uses default weights: success=1000000, tokens_used=1,
                execution_time=1000.

        Returns:
            Combined weighted score as a float. Lower values indicate better variants.
        """
        if weights is None:
            # Default weights: prioritize success (maximize), then minimize tokens and time
            # success is 0 or 1, tokens_used count, execution_time in seconds
            weights = {
                "success": 1000000,  # Minimize success (0 is best)
                "tokens_used": 1,  # Minimize token usage
                "execution_time": 1000,  # Minimize execution time
            }

        score = 0.0
        for metric, weight in weights.items():
            score += self.metrics.get(metric, 0) * weight
        return score

    def to_dict(self) -> dict[str, Any]:
        """Convert the variant to a dictionary representation.

        Serializes all variant attributes including the nested ImprovementReport
        for JSON storage or state persistence.

        Returns:
            Dictionary containing all variant attributes: folder_path, report_path,
            report (as dict), metrics, id, generation, and parent_ids.
        """
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
        }


class AgentEvolver:
    """Evolves AI agents using Pareto frontier optimization.

    Maintains a population of agent variants optimized for token efficiency
    and execution time. Uses mutation (improving one variant) or crossover
    (combining ideas from two variants) to create new variants.
    """

    def __reset__(
        self,
        task_description: str,
        max_generations: int | None = None,
        initial_frontier_size: int | None = None,
        max_frontier_size: int | None = None,
        mutation_probability: float | None = None,
    ) -> None:
        """Initialize or reset the AgentEvolver state.

        Sets up the evolver with configuration values, creates working directories,
        and initializes an empty Pareto frontier. Called by evolve() before
        starting evolution.

        Args:
            task_description: Description of the task the agent should perform.
            max_generations: Maximum number of improvement generations. If None,
                uses config default.
            initial_frontier_size: Number of initial agents to create. If None,
                uses config default.
            max_frontier_size: Maximum size of the Pareto frontier. If None,
                uses config default.
            mutation_probability: Probability of mutation vs crossover (0.0 to 1.0).
                If None, uses config default.

        Returns:
            None. Initializes instance attributes.
        """
        cfg = getattr(DEFAULT_CONFIG, "create_and_optimize_agent", None)
        evolver_cfg = getattr(cfg, "evolver", None)

        self.task_description = task_description
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

        self.work_dir = Path(tempfile.mkdtemp())
        self.optimal_dir = Path(DEFAULT_CONFIG.agent.artifact_dir) / "optimal_agent"

        self.pareto_frontier: list[AgentVariant] = []
        self._variant_counter = 0
        self._generation = 0
        self.improver = ImproverAgent()

    def _next_variant_id(self) -> int:
        """Get the next unique variant ID.

        Increments the internal counter and returns the new value to ensure
        each variant has a unique identifier.

        Returns:
            The next unique integer ID for a variant.
        """
        self._variant_counter += 1
        return self._variant_counter

    def _get_variant_paths(self, variant_id: int) -> tuple[str, str]:
        """Get the folder and report paths for a variant.

        Constructs standardized paths for storing variant code and its
        improvement report based on the variant ID.

        Args:
            variant_id: The unique identifier for the variant.

        Returns:
            A tuple of (folder_path, report_path) where folder_path is the
            directory for the variant's code and report_path is the path
            to the improvement report JSON file.
        """
        folder = str(self.work_dir / f"variant_{variant_id}")
        report = str(self.work_dir / f"variant_{variant_id}" / "improvement_report.json")
        return folder, report

    def _create_initial_agent(self, variant_id: int) -> AgentVariant:
        """Create the initial agent from scratch.

        Uses the ImproverAgent to generate a new agent implementation based
        on the task description. If creation fails, returns a variant with
        a default report.

        Args:
            variant_id: The unique identifier for the new variant.

        Returns:
            A new AgentVariant with the created agent code, report, and
            empty metrics (to be filled by evaluation).
        """
        work_dir, report_path = self._get_variant_paths(variant_id)

        # Create initial agent using ImproverAgent
        success, initial_report = self.improver.create_initial(
            task_description=self.task_description,
            work_dir=work_dir,
        )

        if not success or initial_report is None:
            # Create a default report if creation failed
            initial_report = ImprovementReport(
                metrics={},
                implemented_ideas=[{"idea": "Initial implementation", "source": "initial"}],
                failed_ideas=[],
                generation=0,
            )

        initial_report.save(report_path)

        return AgentVariant(
            folder_path=work_dir,
            report_path=report_path,
            report=initial_report,
            metrics={},
            id=variant_id,
            generation=0,
            parent_ids=[],
        )

    def _load_module_from_path(self, module_name: str, file_path: str) -> Any:
        """Dynamically load a Python module from a file path.

        Loads a Python file as a module, making its contents available for
        execution. Used to load agent.py files from variant directories.

        Args:
            module_name: The name to assign to the loaded module in sys.modules.
            file_path: The absolute path to the Python file to load.

        Returns:
            The loaded module object, or None if loading fails.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _evaluate_variant(
        self,
        variant: AgentVariant,
    ) -> dict[str, Any]:
        """Run the agent on the long-running task and collect metrics.

        Copies the variant's code to an isolated temporary directory, loads
        the agent module, and executes agent_run() with the task description.
        Cleans up the temporary directory after evaluation.

        Args:
            variant: The AgentVariant to evaluate.

        Returns:
            A dictionary containing a 'metrics' key with success, tokens_used,
            and execution_time values. Returns failure metrics if evaluation
            fails for any reason.
        """
        print(f"Evaluating variant {variant.id}...")

        temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_variant_{variant.id}_"))
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        module_name: str | None = None
        try:
            shutil.copytree(variant.folder_path, temp_dir / "agent_code", dirs_exist_ok=True)
            agent_dir = str(temp_dir / "agent_code")
            agent_file = temp_dir / "agent_code" / "agent.py"
            module_name = f"agent_variant_{id(self)}_{random.randint(0, 10000)}"

            try:
                sys.path.insert(0, agent_dir)
                agent_module = self._load_module_from_path(module_name, str(agent_file))
                if agent_module is None:
                    print(f"Failed to load module from {agent_file}")
                    return {
                        "metrics": {"success": 1, "tokens_used": 0, "execution_time": 0.0},
                    }
                result: dict[str, Any] = agent_module.agent_run(self.task_description)
                return result
            except Exception:
                return {
                    "metrics": {"success": 1, "tokens_used": 0, "execution_time": 0.0},
                }
            finally:
                if agent_dir in sys.path:
                    sys.path.remove(agent_dir)
                if module_name:
                    sys.modules.pop(module_name, None)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _update_pareto_frontier(self, new_variant: AgentVariant) -> bool:
        """Update the Pareto frontier with a new variant.

        Adds the variant to the frontier if it is not dominated by any existing
        variant. Removes any existing variants that are dominated by the new
        variant. Trims the frontier if it exceeds max_frontier_size.

        Args:
            new_variant: The AgentVariant to potentially add to the frontier.

        Returns:
            True if the variant was added to the frontier, False if it was
            dominated and rejected.
        """
        # Check if new variant is dominated by any in frontier
        for existing in self.pareto_frontier:
            if existing.dominates(new_variant):
                return False

        # Remove variants dominated by new variant
        self.pareto_frontier = [v for v in self.pareto_frontier if not new_variant.dominates(v)]

        # Add new variant to frontier
        self.pareto_frontier.append(new_variant)

        # Trim frontier if too large (keep most diverse)
        if len(self.pareto_frontier) > self.max_frontier_size:
            self._trim_frontier()

        return True

    def _trim_frontier(self) -> None:
        """Trim the Pareto frontier to max size using crowding distance.

        Uses crowding distance to maintain diversity in the frontier when
        it exceeds max_frontier_size. Variants with higher crowding distance
        (more isolated in metric space) are kept preferentially.

        Returns:
            None. Modifies self.pareto_frontier in place.
        """
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
        """Format metrics dictionary for display.

        Converts a metrics dictionary to a human-readable string with
        appropriate formatting for floats.

        Args:
            metrics: Dictionary mapping metric names to their values.

        Returns:
            A comma-separated string of metric=value pairs, with floats
            formatted to 2 decimal places.
        """
        return ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
        )

    def _sample_from_frontier(self) -> AgentVariant:
        """Sample a variant uniformly from the Pareto frontier.

        Randomly selects one variant from the current Pareto frontier
        with uniform probability.

        Returns:
            A randomly selected AgentVariant from the frontier.
        """
        return random.choice(self.pareto_frontier)

    def _sample_two_from_frontier(self) -> tuple[AgentVariant, AgentVariant]:
        """Sample two different variants from the Pareto frontier, ordered by score.

        Randomly selects two distinct variants and returns them ordered
        with the better-scoring variant (lower score) first.

        Returns:
            A tuple of (primary, secondary) AgentVariants where primary has
            a lower (better) score than secondary.
        """
        v1, v2 = random.sample(self.pareto_frontier, 2)
        return (v1, v2) if v1.score() <= v2.score() else (v2, v1)

    def _mutate(self, variant: AgentVariant) -> AgentVariant | None:
        """Create a new variant by mutating an existing one.

        Uses the ImproverAgent to create an improved version of the given
        variant. The new variant inherits the parent's lineage.

        Args:
            variant: The parent AgentVariant to mutate.

        Returns:
            A new AgentVariant with improved code, or None if mutation fails.
            The new variant has empty metrics (to be filled by evaluation).
        """
        new_id = self._next_variant_id()
        work_dir, report_path = self._get_variant_paths(new_id)

        success, new_report = self.improver.improve(
            source_folder=variant.folder_path,
            work_dir=work_dir,
            task_description=self.task_description,
            report_path=variant.report_path,
        )

        if not success or new_report is None:
            if Path(work_dir).exists():
                shutil.rmtree(work_dir)
            return None

        new_report.save(report_path)
        return AgentVariant(
            folder_path=work_dir,
            report_path=report_path,
            report=new_report,
            metrics={},
            id=new_id,
            generation=self._generation,
            parent_ids=[variant.id],
        )

    def _crossover(self, primary: AgentVariant, secondary: AgentVariant) -> AgentVariant | None:
        """Create a new variant by crossing over two variants.

        Uses the ImproverAgent to combine ideas from two parent variants,
        using the primary variant's code as the base and incorporating
        ideas from both variants' improvement reports.

        Args:
            primary: The primary parent variant (provides base code).
            secondary: The secondary parent variant (provides additional ideas).

        Returns:
            A new AgentVariant combining ideas from both parents, or None if
            crossover fails. The new variant has empty metrics (to be filled
            by evaluation).
        """
        new_id = self._next_variant_id()
        work_dir, report_path = self._get_variant_paths(new_id)

        success, new_report = self.improver.crossover_improve(
            primary_folder=primary.folder_path,
            primary_report_path=primary.report_path,
            secondary_report_path=secondary.report_path,
            work_dir=work_dir,
            task_description=self.task_description,
        )

        if not success or new_report is None:
            if Path(work_dir).exists():
                shutil.rmtree(work_dir)
            return None

        new_report.save(report_path)
        return AgentVariant(
            folder_path=work_dir,
            report_path=report_path,
            report=new_report,
            metrics={},
            id=new_id,
            generation=self._generation,
            parent_ids=[primary.id, secondary.id],
        )

    def evolve(self,
        task_description: str,
        max_generations: int | None = None,
        initial_frontier_size: int | None = None,
        max_frontier_size: int | None = None,
        mutation_probability: float | None = None,
    ) -> AgentVariant:
        """Run the evolutionary optimization.

        Initializes the evolver, creates initial agents, and runs the
        evolution loop for max_generations. Each generation either mutates
        a single variant or crosses over two variants based on mutation_probability.

        Args:
            task_description: Description of the task the agent should perform.
            max_generations: Maximum number of improvement generations. If None,
                uses config default.
            initial_frontier_size: Number of initial agents to create. If None,
                uses config default.
            max_frontier_size: Maximum size of the Pareto frontier. If None,
                uses config default.
            mutation_probability: Probability of mutation vs crossover (0.0 to 1.0).
                If None, uses config default.

        Returns:
            The best AgentVariant from the final Pareto frontier (lowest score).
        """
        self.__reset__(
            task_description=task_description,
            max_generations=max_generations,
            initial_frontier_size=initial_frontier_size,
            max_frontier_size=max_frontier_size,
            mutation_probability=mutation_probability,
        )
        print(f"Starting AgentEvolver with {self.max_generations} generations")
        print(f"Max frontier size: {self.max_frontier_size}, Task: {self.task_description}")
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

    def _copy_best_to_optimal(self, best: AgentVariant) -> None:
        """Copy the best variant to the optimal_agent folder.

        Uses a temporary directory and atomic rename to avoid race conditions
        where the optimal_dir might be read while being updated.

        Args:
            best: The AgentVariant to copy to the optimal directory.

        Returns:
            None. Copies files to self.optimal_dir.
        """
        self.optimal_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy to a temporary location first (same parent ensures same filesystem)
        temp_dir = self.optimal_dir.parent / f"{self.optimal_dir.name}_{os.getpid()}_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(best.folder_path, temp_dir)

        if self.optimal_dir.exists():
            shutil.rmtree(self.optimal_dir)
        temp_dir.rename(self.optimal_dir)

        print(f"  Copied best variant to {self.optimal_dir}")

    def get_best_variant(self) -> AgentVariant:
        """Get the best variant by combined score.

        Finds the variant with the lowest score in the Pareto frontier.

        Returns:
            The AgentVariant with the minimum score.

        Raises:
            RuntimeError: If the Pareto frontier is empty.
        """
        if not self.pareto_frontier:
            raise RuntimeError("No variants available")
        return min(self.pareto_frontier, key=lambda v: v.score())

    def get_pareto_frontier(self) -> list[AgentVariant]:
        """Get all variants in the Pareto frontier.

        Returns a copy of the frontier to prevent external modification.

        Returns:
            A list of all AgentVariants currently in the Pareto frontier.
        """
        return self.pareto_frontier.copy()

    def save_state(self, path: str) -> None:
        """Save the evolver state to a JSON file.

        Serializes the current state including task description, generation,
        variant counter, and all variants in the Pareto frontier.

        Args:
            path: Path where to save the state JSON file. Required because
                work_dir is cleaned up after evolve() completes.

        Returns:
            None. Writes state to the specified file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
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
    """Run the AgentEvolver on a long-running task.

    Creates an AgentEvolver instance and runs evolution on the LONG_RUNNING_TASK
    example. Saves the final state to the optimal directory.

    Returns:
        None.
    """
    evolver = AgentEvolver()

    best = evolver.evolve(
        task_description=LONG_RUNNING_TASK,
        max_generations=20,
        initial_frontier_size=4,
        max_frontier_size=6,
        mutation_probability=0.8,
    )

    print("\n=== Final Result ===")
    print(f"Best variant: {best.folder_path}")
    print(f"Metrics: {best.metrics}")
    print(f"Generation: {best.generation}")

    # Save state to the optimal directory (work_dir is cleaned up after evolve)
    state_path = str(evolver.optimal_dir / "evolver_state.json")
    evolver.save_state(state_path)


if __name__ == "__main__":
    main()
