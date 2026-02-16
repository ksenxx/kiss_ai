# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# Cursor AI (cursor@cursor.com) for vibe coding GEPA
# add your name here

"""GEPA (Genetic-Pareto): Reflective Prompt Evolution for Compound AI Systems.

Based on: "GEPA: Reflective Prompt Evolution can Outperform Reinforcement Learning"
https://arxiv.org/pdf/2507.19457

Algorithm:
    Input: train set, AI system (parametrized by ≥1 prompts), and metric
    Split train set into dev & val sets
    Track a pool of candidates, including the best on each val item (Pareto front)
    Repeatedly:
        Select a prompt to try to improve (weighted by instance wins)
        Run system on a minibatch of dev examples, noting intermediate feedback
        Call a LM to propose alternatives for the prompt based on scores and feedback
        Gate mutations - only accept if they don't degrade on minibatch
        Update pool based on how candidates score on val set (instance-level)
"""

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kiss.agents.gepa.config import GEPAConfig  # type: ignore # noqa: F401
from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import get_config_value, get_template_field_names


class GEPAPhase(Enum):
    """Enum representing the current phase of GEPA optimization."""

    DEV_EVALUATION = "dev_evaluation"
    VAL_EVALUATION = "val_evaluation"
    REFLECTION = "reflection"
    MUTATION_GATING = "mutation_gating"
    MERGE = "merge"
    PARETO_UPDATE = "pareto_update"


@dataclass
class GEPAProgress:
    """Progress information for GEPA optimization callbacks.

    Provides visibility into GEPA's current state during optimization,
    including generation number, current phase, validation accuracy,
    and candidate information.
    """

    generation: int
    """Current generation number (0-indexed)."""

    max_generations: int
    """Total number of generations to run."""

    phase: GEPAPhase
    """Current phase of the optimization (dev/val evaluation, reflection, etc.)."""

    candidate_id: int | None = None
    """ID of the candidate currently being processed (if applicable)."""

    candidate_index: int | None = None
    """Index of current candidate in the population (0-indexed, if applicable)."""

    population_size: int = 0
    """Current population size."""

    best_val_accuracy: float | None = None
    """Best validation accuracy seen so far (average across all metrics)."""

    current_val_accuracy: float | None = None
    """Validation accuracy of the current candidate (if applicable)."""

    pareto_frontier_size: int = 0
    """Current size of the Pareto frontier."""

    num_candidates_evaluated: int = 0
    """Number of candidates evaluated in current generation."""

    message: str = ""
    """Optional message describing the current activity."""


def create_progress_callback(
    verbose: bool = False,
) -> "Callable[[GEPAProgress], None]":
    """Create a standard progress callback for GEPA optimization.

    Args:
        verbose: If True, prints all phases. If False, only prints val evaluation
            completion messages (when a candidate has been fully evaluated).

    Returns:
        A callback function that prints progress updates during optimization.

    Example:
        >>> gepa = GEPA(
        ...     agent_wrapper=my_wrapper,
        ...     initial_prompt_template="Task: {task}",
        ...     progress_callback=create_progress_callback(verbose=True),
        ... )
        >>> best = gepa.optimize(train_examples)
        Gen 1/3 | dev_evaluation     | Best: N/A    | Dev evaluation: candidate 0
        Gen 1/3 | val_evaluation     | Best: N/A    | Val evaluation: candidate 0
        Gen 1/3 | val_evaluation     | Best: 75.00% | Evaluated candidate 0: val_accuracy=0.7500
    """
    import sys

    def progress_callback(progress: GEPAProgress) -> None:
        """Report GEPA optimization progress."""
        is_val_complete = (
            progress.phase == GEPAPhase.VAL_EVALUATION
            and progress.message.startswith("Evaluated")
        )
        is_pareto_update = progress.phase == GEPAPhase.PARETO_UPDATE
        if verbose or is_val_complete or is_pareto_update:
            best_str = (
                f"{progress.best_val_accuracy:.2%}"
                if progress.best_val_accuracy is not None
                else "N/A"
            )
            print(
                f"  Gen {progress.generation + 1}/{progress.max_generations} | "
                f"{progress.phase.value:18} | Best: {best_str:>6} | {progress.message}",
                flush=True,
            )
            sys.stdout.flush()

    return progress_callback


@dataclass
class PromptCandidate:
    """Represents a prompt candidate with its performance metrics."""

    prompt_template: str
    dev_scores: dict[str, float] = field(default_factory=lambda: {})
    val_scores: dict[str, float] = field(default_factory=lambda: {})
    per_item_val_scores: list[dict[str, float]] = field(default_factory=lambda: [])
    val_instance_wins: set[int] = field(default_factory=lambda: set())
    evaluated_val_ids: set[int] = field(default_factory=lambda: set())
    parents: list[int] = field(default_factory=lambda: [])
    id: int = 0


class GEPA:
    """GEPA (Genetic-Pareto) prompt optimizer.

    Optimizes prompts by:
    1. Splitting training data into dev (feedback) and val (selection) sets
    2. Running on dev minibatches to collect trajectories
    3. Reflecting on trajectories to propose improvements
    4. Gating mutations - only accepting if they don't degrade
    5. Maintaining instance-level Pareto frontier (best per val item)
    6. Combining lessons from frontier through structural merge (no crossover)
    """

    def __init__(
        self,
        agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[Any]]],
        initial_prompt_template: str,
        evaluation_fn: Callable[[str], dict[str, float]] | None = None,
        max_generations: int | None = None,
        population_size: int | None = None,
        pareto_size: int | None = None,
        mutation_rate: float | None = None,
        reflection_model: str | None = None,
        dev_val_split: float | None = None,
        perfect_score: float = 1.0,
        use_merge: bool = True,
        max_merge_invocations: int = 5,
        merge_val_overlap_floor: int = 2,
        progress_callback: Callable[[GEPAProgress], None] | None = None,
        batched_agent_wrapper: Callable[
            [str, list[dict[str, str]]], list[tuple[str, list[Any]]]
        ] | None = None,
    ):
        """Initialize GEPA optimizer.

        Args:
            agent_wrapper: Function (prompt_template, arguments) -> (result, trajectory).
                Used when batched_agent_wrapper is not provided, or as fallback.
            initial_prompt_template: The initial prompt template to optimize
            evaluation_fn: Function to evaluate result -> {metric: score}
            max_generations: Maximum evolutionary generations
            population_size: Number of candidates per generation
            pareto_size: Maximum Pareto frontier size
            mutation_rate: Probability of mutation (default: 0.5)
            reflection_model: Model for reflection
            dev_val_split: Fraction for dev set (default: 0.5)
            perfect_score: Score threshold to skip mutation (default: 1.0)
            use_merge: Whether to enable structural merge (default: True)
            max_merge_invocations: Maximum merge operations to attempt (default: 5)
            merge_val_overlap_floor: Minimum validation overlap for merge (default: 2)
            progress_callback: Optional callback function called with GEPAProgress
                during optimization. Use this to track progress, display progress bars,
                or log intermediate results.
            batched_agent_wrapper: Optional batched version of agent_wrapper.
                Function (prompt_template, [arguments]) -> [(result, trajectory)].
                When provided, GEPA calls this with all examples in a minibatch at once
                instead of calling agent_wrapper one at a time. This enables prompt merging
                (combining multiple prompts into a single API call) for significantly
                higher throughput.
        """
        self.agent_wrapper = agent_wrapper
        self.batched_agent_wrapper = batched_agent_wrapper
        self.evaluation_fn = evaluation_fn or (
            lambda r: {"success": 1.0 if "success" in r.lower() else 0.0}
        )
        self.perfect_score = perfect_score

        cfg = config_module.DEFAULT_CONFIG.gepa  # type: ignore[attr-defined]
        self.max_generations = get_config_value(max_generations, cfg, "max_generations")
        self.population_size = get_config_value(population_size, cfg, "population_size")
        self.pareto_size = get_config_value(pareto_size, cfg, "pareto_size")
        self.mutation_rate = get_config_value(mutation_rate, cfg, "mutation_rate")
        self.reflection_model = get_config_value(reflection_model, cfg, "reflection_model")
        self.dev_val_split = dev_val_split if dev_val_split is not None else 0.5

        # Merge configuration
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.merge_val_overlap_floor = merge_val_overlap_floor
        self._merge_invocations = 0
        self._attempted_merges: set[tuple[int, int]] = set()

        # Progress callback
        self.progress_callback = progress_callback
        self._best_val_accuracy: float | None = None

        # State
        self.candidates: list[PromptCandidate] = []
        self.pareto_frontier: list[PromptCandidate] = []
        self.best_per_val_instance: dict[int, PromptCandidate] = {}
        self.dev_examples: list[dict[str, str]] = []
        self.val_examples: list[dict[str, str]] = []
        self._candidate_id = 0

        # Ancestry tracking for structural merge
        self._historical_prompts: dict[int, str] = {}
        self._ancestry: dict[int, list[int]] = {}

        # Valid placeholders from initial template
        self.initial_prompt_template = initial_prompt_template
        self.valid_placeholders = set(get_template_field_names(initial_prompt_template))

        # Prompt template for reflection
        # fmt: off
        self.reflection_prompt = (
            "I provided an assistant with the following instructions to perform a task "
            "for me:\n\n"
            "```\n{prompt_template}\n```\n\n"
            "The following are examples of different task inputs provided to the "
            "assistant along with the assistant's response for each of them. For each "
            "example, you will see:\n"
            "- The inputs given to the assistant\n"
            "- The assistant's final response\n"
            "- The agent trajectory (if available) showing the assistant's reasoning "
            "process, tool calls, and intermediate steps\n"
            "- Feedback on how the response could be better\n\n"
            "{inputs_outputs_feedback}\n\n"
            "Your task is to write a new instruction for the assistant.\n\n"
            "Read the inputs carefully and identify the input format and infer a "
            "detailed task description about the task I wish to solve with the "
            "assistant.\n\n"
            "Carefully examine the agent trajectories to understand HOW the assistant "
            "is approaching the task. Look at:\n"
            "- What tools the assistant is calling and with what arguments\n"
            "- The reasoning steps the assistant takes\n"
            "- Where the assistant makes mistakes or suboptimal choices\n"
            "- What information the assistant is missing or misinterpreting\n\n"
            "Read all the assistant responses and the corresponding feedback. Identify "
            "all niche and domain-specific factual information about the task and "
            "include it in the instruction, as a lot of it may not be available to the "
            "assistant in the future. The assistant may have utilized a generalizable "
            "strategy to solve the task; if so, include that in the instruction as "
            "well.\n\n"
            "Based on the feedback AND the agent trajectories, identify what the "
            "assistant is doing wrong or could do better, and incorporate specific "
            "guidance to address these issues in the new instruction.\n\n"
            "Important constraints:\n"
            "- The instruction must keep these exact placeholders intact: {placeholders}\n"
            "- Do not add new placeholders or remove existing ones\n"
            "- Focus on improving clarity, specificity, and actionable guidance\n\n"
            "Provide the new instruction by calling the 'finish' tool with the "
            "instruction as the 'result' argument."
        )
        # fmt: on

        # Initialize with seed candidate
        self.candidates.append(self._new_candidate(initial_prompt_template))

    def _get_val_accuracy(self, candidate: PromptCandidate) -> float:
        """Get the validation accuracy for a candidate."""
        if candidate.val_scores:
            return sum(candidate.val_scores.values()) / len(candidate.val_scores)
        return 0.0

    def _report_progress(
        self,
        generation: int,
        phase: GEPAPhase,
        candidate: PromptCandidate | None = None,
        candidate_index: int | None = None,
        num_candidates_evaluated: int = 0,
        message: str = "",
    ) -> None:
        """Report progress via callback if one is registered."""
        if not self.progress_callback:
            return

        self.progress_callback(GEPAProgress(
            generation=generation,
            max_generations=self.max_generations,
            phase=phase,
            candidate_id=candidate.id if candidate else None,
            candidate_index=candidate_index,
            population_size=len(self.candidates),
            best_val_accuracy=self._best_val_accuracy,
            current_val_accuracy=(
                self._get_val_accuracy(candidate)
                if candidate and candidate.val_scores
                else None
            ),
            pareto_frontier_size=len(self.pareto_frontier),
            num_candidates_evaluated=num_candidates_evaluated,
            message=message,
        ))

    def _update_best_val_accuracy(self, candidate: PromptCandidate) -> None:
        """Update best validation accuracy tracking."""
        if candidate.val_scores:
            val_accuracy = self._get_val_accuracy(candidate)
            if self._best_val_accuracy is None or val_accuracy > self._best_val_accuracy:
                self._best_val_accuracy = val_accuracy

    def _new_candidate(
        self, prompt_template: str, parents: list[int] | None = None
    ) -> PromptCandidate:
        """Create a new candidate with unique ID.

        Args:
            prompt_template: The prompt template string for the candidate.
            parents: Optional list of parent candidate IDs for ancestry tracking.

        Returns:
            A new PromptCandidate instance with assigned unique ID.
        """
        candidate = PromptCandidate(
            prompt_template=prompt_template,
            id=self._candidate_id,
            parents=parents or [],
        )
        self._historical_prompts[self._candidate_id] = prompt_template
        self._ancestry[self._candidate_id] = parents or []
        self._candidate_id += 1
        return candidate

    def _weighted_choice(self, candidates: list[PromptCandidate]) -> PromptCandidate:
        """Select candidate weighted by number of instance wins.

        Args:
            candidates: List of PromptCandidate instances to choose from.

        Returns:
            A randomly selected candidate, weighted by validation instance wins.

        Raises:
            ValueError: If candidates list is empty.
        """
        if not candidates:
            raise ValueError("No candidates to choose from")
        weights = [max(1, len(c.val_instance_wins)) for c in candidates]
        return random.choices(candidates, weights=weights)[0]

    def _run_minibatch(
        self,
        prompt: str,
        examples: list[dict[str, str]],
        capture_results: bool = False,
        phase: GEPAPhase | None = None,
        generation: int = 0,
        candidate_id: int = 0,
    ) -> tuple[dict[str, float], list[dict[str, float]], list[str], list[Any]]:
        """Run prompt on a minibatch of examples and collect scores.

        When a batched_agent_wrapper is provided, all examples are sent in a
        single call (enabling prompt merging for higher throughput). Otherwise
        falls back to calling agent_wrapper one example at a time.

        Args:
            prompt: The prompt template to evaluate.
            examples: List of example dictionaries containing input arguments.
            capture_results: If True, capture and return results and trajectories.
            phase: Current GEPA phase for progress reporting.
            generation: Current generation number for progress reporting.
            candidate_id: Current candidate ID for progress reporting.

        Returns:
            A tuple of (avg_scores, per_item_scores, results, trajectories) where:
            - avg_scores: Average scores across all examples.
            - per_item_scores: List of score dictionaries for each example.
            - results: List of result strings (empty if capture_results=False).
            - trajectories: List of trajectory data (empty if capture_results=False).
        """
        if self.batched_agent_wrapper is not None:
            return self._run_minibatch_batched(
                prompt, examples, capture_results, phase, generation, candidate_id,
            )
        return self._run_minibatch_sequential(
            prompt, examples, capture_results, phase, generation, candidate_id,
        )

    def _run_minibatch_sequential(
        self,
        prompt: str,
        examples: list[dict[str, str]],
        capture_results: bool,
        phase: GEPAPhase | None,
        generation: int,
        candidate_id: int,
    ) -> tuple[dict[str, float], list[dict[str, float]], list[str], list[Any]]:
        all_scores: list[dict[str, float]] = []
        results: list[str] = []
        trajectories: list[list[Any]] = []

        for i, args in enumerate(examples):
            result, trajectory = self.agent_wrapper(prompt, args)
            scores = self.evaluation_fn(result)
            all_scores.append(scores)
            if capture_results:
                results.append(result)
                trajectories.append(trajectory)

            self._report_example_progress(
                phase, generation, candidate_id, i, len(examples), scores,
            )

        return self._aggregate_scores(all_scores), all_scores, results, trajectories

    def _run_minibatch_batched(
        self,
        prompt: str,
        examples: list[dict[str, str]],
        capture_results: bool,
        phase: GEPAPhase | None,
        generation: int,
        candidate_id: int,
    ) -> tuple[dict[str, float], list[dict[str, float]], list[str], list[Any]]:
        assert self.batched_agent_wrapper is not None
        batch_results = self.batched_agent_wrapper(prompt, examples)

        all_scores: list[dict[str, float]] = []
        results: list[str] = []
        trajectories: list[list[Any]] = []

        for i, (result, trajectory) in enumerate(batch_results):
            scores = self.evaluation_fn(result)
            all_scores.append(scores)
            if capture_results:
                results.append(result)
                trajectories.append(trajectory)

            self._report_example_progress(
                phase, generation, candidate_id, i, len(examples), scores,
            )

        return self._aggregate_scores(all_scores), all_scores, results, trajectories

    def _report_example_progress(
        self,
        phase: GEPAPhase | None,
        generation: int,
        candidate_id: int,
        index: int,
        total: int,
        scores: dict[str, float],
    ) -> None:
        if not self.progress_callback or not phase:
            return
        if scores:
            metric_name = next(iter(scores.keys()))
            metric_score = scores[metric_name]
        else:
            metric_name = "score"
            metric_score = 0.0
        self._report_progress(
            generation=generation,
            phase=phase,
            message=(
                f"Candidate {candidate_id}: example {index + 1}/{total} "
                f"({metric_name}={metric_score:.2f})"
            ),
        )

    @staticmethod
    def _aggregate_scores(all_scores: list[dict[str, float]]) -> dict[str, float]:
        if not all_scores:
            return {}
        avg: dict[str, float] = {}
        for key in all_scores[0]:
            avg[key] = sum(s.get(key, 0.0) for s in all_scores) / len(all_scores)
        return avg

    def _format_inputs_outputs_feedback(
        self,
        examples: list[dict[str, str]],
        results: list[str],
        scores: list[dict[str, float]],
        trajectories: list[list[dict[str, Any]]] | None = None,
    ) -> str:
        """Format examples with inputs, outputs, trajectories, and feedback for reflection.

        Args:
            examples: List of input example dictionaries.
            results: List of agent result strings for each example.
            scores: List of score dictionaries for each example.
            trajectories: Optional list of agent trajectories (tool calls, reasoning, etc.).

        Returns:
            A formatted string containing all examples with their inputs, outputs,
            trajectories, and feedback suitable for the reflection prompt.
        """
        formatted_parts = []

        for i, (example, result, score) in enumerate(zip(examples, results, scores)):
            inputs_str = json.dumps(example, indent=2)
            score_details = ", ".join(f"{k}: {v:.2f}" for k, v in score.items())

            # Format feedback based on average score ratio
            avg_score = sum(score.values()) / len(score) if score else 0.0
            if avg_score >= self.perfect_score:
                feedback = f"Good response. Scores: {score_details}"
            elif avg_score >= self.perfect_score * 0.5:
                feedback = (
                    f"Partial success. Scores: {score_details}. "
                    "Consider how to improve the weaker aspects."
                )
            else:
                feedback = (
                    f"Needs improvement. Scores: {score_details}. "
                    "The response did not fully address the task requirements."
                )

            truncated = result[:1000] + "..." if len(result) > 1000 else result

            # Format trajectory if available
            trajectory_str = ""
            if trajectories and i < len(trajectories) and trajectories[i]:
                traj_parts = []
                for step in trajectories[i]:
                    if isinstance(step, dict):
                        content = str(step.get("content", ""))[:500]
                        traj_parts.append(f"[{step.get('role', 'unknown')}]: {content}...")
                    else:
                        traj_parts.append(str(step)[:500])
                if traj_parts:
                    trajectory_str = (
                        f"\n\n**Agent Trajectory (reasoning & tool calls):**\n"
                        f"```\n{chr(10).join(traj_parts)}\n```"
                    )

            formatted_parts.append(
                f"### Example {i + 1} ###\n"
                f"**Inputs:**\n```\n{inputs_str}\n```\n\n"
                f"**Assistant's Response:**\n```\n{truncated}\n```"
                f"{trajectory_str}\n\n"
                f"**Feedback:** {feedback}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    def _reflect(
        self,
        prompt: str,
        examples: list[dict[str, str]],
        results: list[str],
        scores: list[dict[str, float]],
        trajectories: list[list] | None = None,
    ) -> str:
        """Generate improved prompt via reflection using an LLM.

        Args:
            prompt: Current prompt template to improve.
            examples: Input examples used for evaluation.
            results: Agent results for each example.
            scores: Scores for each example.
            trajectories: Agent trajectories showing reasoning and tool calls.

        Returns:
            A new improved prompt template generated by the reflection model.
        """
        inputs_outputs_feedback = self._format_inputs_outputs_feedback(
            examples, results, scores, trajectories
        )

        agent = KISSAgent("GEPA Reflection")
        result = agent.run(
            model_name=self.reflection_model,
            prompt_template=self.reflection_prompt,
            arguments={
                "prompt_template": prompt,
                "inputs_outputs_feedback": inputs_outputs_feedback,
                "placeholders": ", ".join(self.valid_placeholders),
            },
        )
        return self._sanitize_prompt_template(result, fallback=prompt)

    def _sanitize_prompt_template(self, prompt: str, fallback: str) -> str:
        """Normalize and validate placeholders in a prompt template."""
        normalized = re.sub(r"{\s*([\"'])([^{}]+?)\1\s*}", r"{\2}", prompt)
        try:
            placeholders = set(get_template_field_names(normalized))
        except ValueError:
            return fallback
        if placeholders != self.valid_placeholders:
            return fallback
        return normalized

    def _compute_val_overlap(
        self, c1: PromptCandidate, c2: PromptCandidate
    ) -> tuple[set[int], dict[str, float], dict[str, float]]:
        """Compute validation instance overlap and per-parent averages on overlap.

        Args:
            c1: First prompt candidate.
            c2: Second prompt candidate.

        Returns:
            A tuple of (overlap_ids, c1_overlap_scores, c2_overlap_scores) where:
            - overlap_ids: Set of validation instance indices evaluated by both candidates.
            - c1_overlap_scores: Average scores for c1 on the overlap instances.
            - c2_overlap_scores: Average scores for c2 on the overlap instances.
        """
        overlap_ids = c1.evaluated_val_ids & c2.evaluated_val_ids
        c1_overlap_scores: dict[str, float] = {}
        c2_overlap_scores: dict[str, float] = {}

        if overlap_ids and c1.per_item_val_scores and c2.per_item_val_scores:
            for idx in overlap_ids:
                if idx < len(c1.per_item_val_scores) and idx < len(c2.per_item_val_scores):
                    for key, val in c1.per_item_val_scores[idx].items():
                        c1_overlap_scores[key] = c1_overlap_scores.get(key, 0.0) + val
                    for key, val in c2.per_item_val_scores[idx].items():
                        c2_overlap_scores[key] = c2_overlap_scores.get(key, 0.0) + val
            # Average the accumulated scores
            for key in c1_overlap_scores:
                c1_overlap_scores[key] /= len(overlap_ids)
            for key in c2_overlap_scores:
                c2_overlap_scores[key] /= len(overlap_ids)

        return overlap_ids, c1_overlap_scores, c2_overlap_scores

    def _get_ancestors(self, candidate_id: int) -> set[int]:
        """Get all ancestors of a candidate (including itself).

        Args:
            candidate_id: The ID of the candidate to trace ancestry for.

        Returns:
            Set of all ancestor candidate IDs, including the candidate itself.
        """
        ancestors = set()
        stack = [candidate_id]
        while stack:
            node = stack.pop()
            if node not in ancestors:
                ancestors.add(node)
                stack.extend(self._ancestry.get(node, []))
        return ancestors

    def _find_common_ancestor(self, id1: int, id2: int) -> int | None:
        """Find the nearest common ancestor of two candidates.

        Args:
            id1: ID of the first candidate.
            id2: ID of the second candidate.

        Returns:
            The ID of the nearest common ancestor, or None if no common ancestor exists.
        """
        common = self._get_ancestors(id1) & self._get_ancestors(id2)
        return max(common) if common else None

    def _find_merge_candidates(self) -> list[tuple[PromptCandidate, PromptCandidate]]:
        """Find pairs of Pareto frontier candidates suitable for merging.

        Identifies candidate pairs that share a common ancestor, have sufficient
        validation overlap, and haven't been merged before.

        Returns:
            List of (candidate1, candidate2) pairs sorted by merge potential
            (complementarity and coverage scores).
        """
        if len(self.pareto_frontier) < 2:
            return []

        merge_pairs: list[tuple[PromptCandidate, PromptCandidate, float]] = []

        for i, c1 in enumerate(self.pareto_frontier):
            for c2 in self.pareto_frontier[i + 1 :]:
                pair_key = (min(c1.id, c2.id), max(c1.id, c2.id))
                if pair_key in self._attempted_merges:
                    continue

                overlap_ids, _, _ = self._compute_val_overlap(c1, c2)
                if len(overlap_ids) < self.merge_val_overlap_floor:
                    continue

                if self._find_common_ancestor(c1.id, c2.id) is None:
                    continue

                # Score by complementarity (different strengths) and coverage
                union = c1.val_instance_wins | c2.val_instance_wins
                if not union:
                    continue
                symmetric_diff = c1.val_instance_wins ^ c2.val_instance_wins
                merge_score = (
                    len(symmetric_diff) / len(union) * 0.6
                    + len(union) / max(1, len(self.val_examples)) * 0.4
                )
                merge_pairs.append((c1, c2, merge_score))

        merge_pairs.sort(key=lambda x: x[2], reverse=True)
        return [(c1, c2) for c1, c2, _ in merge_pairs]

    def _merge_structural(self, c1: PromptCandidate, c2: PromptCandidate) -> PromptCandidate | None:
        """Perform structural 3-way merge of two candidates.

        Uses the common ancestor to perform a 3-way merge, resolving conflicts
        by selecting the prompt from the candidate with higher validation scores.

        Args:
            c1: First prompt candidate to merge.
            c2: Second prompt candidate to merge.

        Returns:
            A new merged PromptCandidate if the merge passes the quality gate,
            or None if the merge is rejected or not possible.
        """
        if self._merge_invocations >= self.max_merge_invocations:
            return None

        pair_key = (min(c1.id, c2.id), max(c1.id, c2.id))
        self._attempted_merges.add(pair_key)
        self._merge_invocations += 1

        ancestor_id = self._find_common_ancestor(c1.id, c2.id)
        ancestor_prompt = self._historical_prompts.get(ancestor_id) if ancestor_id else None
        if ancestor_prompt is None:
            return None

        # 3-way merge: prefer changed prompts, resolve conflicts by score
        p1, p2 = c1.prompt_template, c2.prompt_template
        c1_changed, c2_changed = p1 != ancestor_prompt, p2 != ancestor_prompt

        if c1_changed and c2_changed and p1 != p2:
            # Conflict: pick by validation score, tie-break randomly
            s1 = sum(c1.val_scores.values()) if c1.val_scores else 0
            s2 = sum(c2.val_scores.values()) if c2.val_scores else 0
            merged_prompt = p1 if s1 > s2 else p2 if s2 > s1 else random.choice([p1, p2])
        elif c2_changed:
            merged_prompt = p2
        elif c1_changed:
            merged_prompt = p1
        else:
            merged_prompt = ancestor_prompt

        # Gate: Evaluate on overlap subset
        overlap_ids, c1_scores, c2_scores = self._compute_val_overlap(c1, c2)
        overlap_examples = [
            self.val_examples[idx] for idx in overlap_ids if idx < len(self.val_examples)
        ]
        if len(overlap_examples) < self.merge_val_overlap_floor:
            return None

        merged = self._new_candidate(merged_prompt, parents=[c1.id, c2.id])
        _, overlap_scores, _, _ = self._run_minibatch(merged.prompt_template, overlap_examples)

        merged_avg = (
            sum(sum(s.values()) for s in overlap_scores) / len(overlap_scores)
            if overlap_scores
            else 0
        )
        parent_avg = max(sum(c1_scores.values()), sum(c2_scores.values()))

        return merged if merged_avg >= parent_avg * 0.95 else None

    def _try_merge_from_frontier(self) -> PromptCandidate | None:
        """Attempt to create a merged candidate from Pareto frontier.

        Finds the best merge candidate pair from the Pareto frontier and
        attempts to perform a structural merge.

        Returns:
            A new merged PromptCandidate if successful, or None if no valid
            merge candidates exist or the merge fails.
        """
        if not self.use_merge:
            return None

        merge_pairs = self._find_merge_candidates()
        if not merge_pairs:
            return None

        # Try the best merge candidate
        c1, c2 = merge_pairs[0]
        return self._merge_structural(c1, c2)

    def _update_pareto(
        self, candidate: PromptCandidate, generation: int = 0
    ) -> None:
        """Update instance-level Pareto frontier with a new candidate.

        Updates the best-per-instance tracking and rebuilds the Pareto frontier
        based on which candidates are best for each validation instance.

        Args:
            candidate: The PromptCandidate to consider for the Pareto frontier.
            generation: Current generation number for progress reporting.

        Returns:
            None. Updates self.pareto_frontier and self.best_per_val_instance in place.
        """
        # Track previous frontier IDs to detect new additions
        prev_frontier_ids = {c.id for c in self.pareto_frontier}

        candidate.val_instance_wins = set()
        candidate.evaluated_val_ids = set(range(len(candidate.per_item_val_scores)))

        # Update best per validation instance
        for idx, scores in enumerate(candidate.per_item_val_scores):
            score = sum(scores.values())
            current_best = self.best_per_val_instance.get(idx)

            if current_best is None:
                self.best_per_val_instance[idx] = candidate
                candidate.val_instance_wins.add(idx)
            else:
                best_score = sum(current_best.per_item_val_scores[idx].values())
                if score > best_score:
                    current_best.val_instance_wins.discard(idx)
                    self.best_per_val_instance[idx] = candidate
                    candidate.val_instance_wins.add(idx)
                elif score == best_score:
                    candidate.val_instance_wins.add(idx)

        # Rebuild frontier from unique best candidates
        frontier_candidates = {c.id: c for c in self.best_per_val_instance.values()}
        if candidate.val_instance_wins:
            frontier_candidates[candidate.id] = candidate

        # Limit size by instance wins
        new_frontier = sorted(
            frontier_candidates.values(),
            key=lambda c: len(c.val_instance_wins),
            reverse=True,
        )
        self.pareto_frontier = new_frontier[: self.pareto_size]

        # Report if this candidate was added to the Pareto frontier
        new_frontier_ids = {c.id for c in self.pareto_frontier}
        if candidate.id in new_frontier_ids and candidate.id not in prev_frontier_ids:
            self._report_progress(
                generation=generation,
                phase=GEPAPhase.PARETO_UPDATE,
                candidate=candidate,
                message=(
                    f"Added candidate {candidate.id} to Pareto frontier "
                    f"(wins={len(candidate.val_instance_wins)}, "
                    f"val_acc={self._get_val_accuracy(candidate):.4f})\n"
                    f"Prompt:\n{candidate.prompt_template}"
                ),
            )

    def _is_perfect(self, scores: dict[str, float]) -> bool:
        """Check if all scores meet the perfect threshold.

        Args:
            scores: Dictionary of metric names to score values.

        Returns:
            True if all scores are at or above the perfect_score threshold,
            False otherwise or if scores is empty.
        """
        return bool(scores) and all(v >= self.perfect_score for v in scores.values())

    def _should_accept(
        self, parent_scores: dict[str, float], child_scores: dict[str, float]
    ) -> bool:
        """Determine whether to accept a mutation based on score comparison.

        Args:
            parent_scores: Score dictionary for the parent candidate.
            child_scores: Score dictionary for the mutated child candidate.

        Returns:
            True if the child should be accepted (not worse than parent),
            False otherwise.
        """
        if not parent_scores or not child_scores:
            return True
        return sum(child_scores.values()) >= sum(parent_scores.values())

    def optimize(
        self,
        train_examples: list[dict[str, str]],
        dev_minibatch_size: int | None = None,
    ) -> PromptCandidate:
        """Run GEPA optimization.

        Args:
            train_examples: Training examples (will be split into dev/val)
            dev_minibatch_size: Dev examples per evaluation (default: all)

        Returns:
            Best PromptCandidate found
        """
        # Split into dev/val
        shuffled = train_examples.copy()
        random.shuffle(shuffled)
        split = max(1, int(len(shuffled) * self.dev_val_split))
        self.dev_examples = shuffled[:split] or shuffled[:1]
        self.val_examples = shuffled[split:] or shuffled[-1:]

        batch_size = dev_minibatch_size or len(self.dev_examples)

        # Reset best validation accuracy tracking
        self._best_val_accuracy = None

        for gen in range(self.max_generations):
            # Evaluate candidates and store reflection data
            # Type: dict[int, tuple[dev_batch, dev_results, dev_item_scores, trajectories]]
            candidate_reflection_data: dict[
                int,
                tuple[list[dict[str, str]], list[str], list[dict[str, float]], list[list]],
            ] = {}

            for idx, candidate in enumerate(self.candidates):
                # Report dev evaluation progress
                self._report_progress(
                    generation=gen,
                    phase=GEPAPhase.DEV_EVALUATION,
                    candidate=candidate,
                    candidate_index=idx,
                    num_candidates_evaluated=idx,
                    message=f"Dev evaluation: candidate {candidate.id}",
                )

                # Dev evaluation for feedback (capture results for reflection)
                dev_batch = (
                    self.dev_examples
                    if len(self.dev_examples) <= batch_size
                    else random.sample(self.dev_examples, batch_size)
                )
                (
                    candidate.dev_scores,
                    dev_item_scores,
                    dev_results,
                    dev_trajectories,
                ) = self._run_minibatch(
                    candidate.prompt_template,
                    dev_batch,
                    capture_results=True,
                    phase=GEPAPhase.DEV_EVALUATION,
                    generation=gen,
                    candidate_id=candidate.id,
                )
                # Store reflection data for this candidate (including trajectories)
                candidate_reflection_data[candidate.id] = (
                    dev_batch,
                    dev_results,
                    dev_item_scores,
                    dev_trajectories,
                )

                # Report val evaluation progress
                self._report_progress(
                    generation=gen,
                    phase=GEPAPhase.VAL_EVALUATION,
                    candidate=candidate,
                    candidate_index=idx,
                    num_candidates_evaluated=idx,
                    message=f"Val evaluation: candidate {candidate.id}",
                )

                # Val evaluation for selection
                candidate.val_scores, candidate.per_item_val_scores, _, _ = self._run_minibatch(
                    candidate.prompt_template,
                    self.val_examples,
                    phase=GEPAPhase.VAL_EVALUATION,
                    generation=gen,
                    candidate_id=candidate.id,
                )
                self._update_pareto(candidate, generation=gen)
                self._update_best_val_accuracy(candidate)

                # Report progress after val evaluation (now with updated scores)
                self._report_progress(
                    generation=gen,
                    phase=GEPAPhase.VAL_EVALUATION,
                    candidate=candidate,
                    candidate_index=idx,
                    num_candidates_evaluated=idx + 1,
                    message=(
                        f"Evaluated candidate {candidate.id}: "
                        f"val_accuracy={self._get_val_accuracy(candidate):.4f}"
                    ),
                )

            # Generate next generation (skip last)
            if gen < self.max_generations - 1:
                new_candidates: list[PromptCandidate] = []

                # Phase 1: Reflective mutation
                while len(new_candidates) < self.population_size:
                    if random.random() < self.mutation_rate and self.pareto_frontier:
                        parent = self._weighted_choice(self.pareto_frontier)

                        if self._is_perfect(parent.dev_scores):
                            new_candidates.append(parent)
                            continue

                        # Report reflection progress
                        self._report_progress(
                            generation=gen,
                            phase=GEPAPhase.REFLECTION,
                            candidate=parent,
                            num_candidates_evaluated=len(new_candidates),
                            message=f"Reflecting on candidate {parent.id} to generate mutation",
                        )

                        # Get or compute reflection data for parent
                        if parent.id in candidate_reflection_data:
                            dev_batch, dev_results, dev_item_scores, trajectories = (
                                candidate_reflection_data[parent.id]
                            )
                            parent_scores = parent.dev_scores
                        else:
                            dev_batch = (
                                self.dev_examples
                                if len(self.dev_examples) <= batch_size
                                else random.sample(self.dev_examples, batch_size)
                            )
                            parent_scores, dev_item_scores, dev_results, trajectories = (
                                self._run_minibatch(
                                    parent.prompt_template, dev_batch, capture_results=True
                                )
                            )

                        new_prompt = self._reflect(
                            parent.prompt_template,
                            dev_batch,
                            dev_results,
                            dev_item_scores,
                            trajectories,
                        )
                        child = self._new_candidate(new_prompt, parents=[parent.id])

                        # Report mutation gating progress
                        self._report_progress(
                            generation=gen,
                            phase=GEPAPhase.MUTATION_GATING,
                            candidate=child,
                            num_candidates_evaluated=len(new_candidates),
                            message=f"Gating mutation: evaluating child {child.id}",
                        )

                        child.dev_scores, _, _, _ = self._run_minibatch(
                            child.prompt_template,
                            dev_batch,
                            phase=GEPAPhase.MUTATION_GATING,
                            generation=gen,
                            candidate_id=child.id,
                        )
                        if self._should_accept(parent_scores, child.dev_scores):
                            new_candidates.append(child)
                    elif self.candidates:
                        new_candidates.append(random.choice(self.candidates))

                # Phase 2: Merge from Pareto frontier
                if self.use_merge and len(self.pareto_frontier) >= 2:
                    # Report merge progress
                    self._report_progress(
                        generation=gen,
                        phase=GEPAPhase.MERGE,
                        num_candidates_evaluated=len(new_candidates),
                        message="Attempting structural merge from Pareto frontier",
                    )

                    merged = self._try_merge_from_frontier()
                    if merged is not None:
                        merged.val_scores, merged.per_item_val_scores, _, _ = self._run_minibatch(
                            merged.prompt_template,
                            self.val_examples,
                            phase=GEPAPhase.MERGE,
                            generation=gen,
                            candidate_id=merged.id,
                        )
                        self._update_pareto(merged, generation=gen)
                        self._update_best_val_accuracy(merged)
                        if merged.val_instance_wins:
                            new_candidates.append(merged)

                            # Report successful merge
                            self._report_progress(
                                generation=gen,
                                phase=GEPAPhase.MERGE,
                                candidate=merged,
                                num_candidates_evaluated=len(new_candidates),
                                message=(
                                    f"Merge successful: created candidate {merged.id} "
                                    f"with val_accuracy={self._get_val_accuracy(merged):.4f}"
                                ),
                            )

                self.candidates = new_candidates

        return self._get_best_candidate()

    def _get_best_candidate(self) -> PromptCandidate:
        """Get the best candidate by instance wins and validation score.

        Selects from the Pareto frontier (or all candidates if frontier is empty),
        prioritizing by number of instance wins, then by total validation score.

        Returns:
            The best PromptCandidate, or a new candidate with the initial template
            if no candidates exist.
        """
        candidates = self.pareto_frontier or self.candidates
        if candidates:
            return max(
                candidates,
                key=lambda c: (len(c.val_instance_wins), sum(c.val_scores.values())),
            )
        return self._new_candidate(self.initial_prompt_template)

    def get_pareto_frontier(self) -> list[PromptCandidate]:
        """Get a copy of the current Pareto frontier.

        Returns:
            A list of PromptCandidate instances representing the current
            Pareto frontier (best candidates per validation instance).
        """
        return self.pareto_frontier.copy()

    def get_best_prompt(self) -> str:
        """Get the best prompt template found during optimization.

        Returns:
            The prompt template string from the best candidate.
        """
        return self._get_best_candidate().prompt_template
