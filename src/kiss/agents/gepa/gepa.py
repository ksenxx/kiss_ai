# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# Cursor AI (cursor@cursor.com) for vibe coding GEPA
# add your name here

"""GEPA (Genetic-Pareto): Reflective Prompt Evolution for Compound AI Systems.

Based on: "GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPERFORM REINFORCEMENT LEARNING"
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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kiss.agents.gepa.config import GEPAConfig  # type: ignore # noqa: F401
from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import get_config_value, get_template_field_names


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
    ):
        """Initialize GEPA optimizer.

        Args:
            agent_wrapper: Function (prompt_template, arguments) -> (result, trajectory)
            initial_prompt_template: The initial prompt template to optimize
            evaluation_fn: Function to evaluate result -> {metric: score}
            max_generations: Maximum evolutionary generations
            population_size: Number of candidates per generation
            pareto_size: Maximum Pareto frontier size
            mutation_rate: Probability of mutation (default: 0.5)
            reflection_model: Model for reflection
            dev_val_split: Fraction for dev set (default: 0.5)
            perfect_score: Score threshold to skip mutation (default: 1.0)
        """
        self.agent_wrapper = agent_wrapper
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

    def _new_candidate(
        self, prompt_template: str, parents: list[int] | None = None
    ) -> PromptCandidate:
        """Create a new candidate with unique ID."""
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
        """Select candidate weighted by number of instance wins."""
        if not candidates:
            raise ValueError("No candidates to choose from")
        weights = [max(1, len(c.val_instance_wins)) for c in candidates]
        return random.choices(candidates, weights=weights)[0]

    def _run_minibatch(
        self,
        prompt: str,
        examples: list[dict[str, str]],
        capture_results: bool = False,
    ) -> tuple[dict[str, float], list[dict[str, float]], list[str], list[Any]]:
        """Run prompt on examples.

        Returns:
            (avg_scores, per_item_scores, results, trajectories)
            results and trajectories are only populated if capture_results=True
        """
        all_scores: list[dict[str, float]] = []
        results: list[str] = []
        trajectories: list[list[Any]] = []

        for args in examples:
            result, trajectory = self.agent_wrapper(prompt, args)
            all_scores.append(self.evaluation_fn(result))
            if capture_results:
                results.append(result)
                trajectories.append(trajectory)

        # Average scores
        avg: dict[str, float] = {}
        if all_scores:
            for key in all_scores[0]:
                avg[key] = sum(s.get(key, 0.0) for s in all_scores) / len(all_scores)

        return avg, all_scores, results, trajectories

    def _format_inputs_outputs_feedback(
        self,
        examples: list[dict[str, str]],
        results: list[str],
        scores: list[dict[str, float]],
        trajectories: list[list[dict[str, Any]]] | None = None,
    ) -> str:
        """Format examples with inputs, outputs, trajectories, and feedback for reflection.

        Args:
            examples: List of input examples
            results: List of agent results
            scores: List of score dictionaries
            trajectories: Optional list of agent trajectories (tool calls, reasoning, etc.)
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
        """Generate improved prompt via reflection.

        Args:
            prompt: Current prompt template
            examples: Input examples used for evaluation
            results: Agent results for each example
            scores: Scores for each example
            trajectories: Agent trajectories showing reasoning and tool calls
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
        return result

    def _compute_val_overlap(
        self, c1: PromptCandidate, c2: PromptCandidate
    ) -> tuple[set[int], dict[str, float], dict[str, float]]:
        """Compute validation instance overlap and per-parent averages on overlap.

        Returns:
            (overlap_ids, c1_overlap_scores, c2_overlap_scores)
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
        """Get all ancestors of a candidate (including itself)."""
        ancestors = set()
        stack = [candidate_id]
        while stack:
            node = stack.pop()
            if node not in ancestors:
                ancestors.add(node)
                stack.extend(self._ancestry.get(node, []))
        return ancestors

    def _find_common_ancestor(self, id1: int, id2: int) -> int | None:
        """Find the nearest common ancestor of two candidates."""
        common = self._get_ancestors(id1) & self._get_ancestors(id2)
        return max(common) if common else None

    def _find_merge_candidates(self) -> list[tuple[PromptCandidate, PromptCandidate]]:
        """Find pairs of Pareto frontier candidates suitable for merging.

        Returns:
            List of (candidate1, candidate2) pairs sorted by merge potential
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

        Returns:
            Merged candidate if successful, None if merge rejected
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

        Returns:
            Merged candidate if successful, None otherwise
        """
        if not self.use_merge:
            return None

        merge_pairs = self._find_merge_candidates()
        if not merge_pairs:
            return None

        # Try the best merge candidate
        c1, c2 = merge_pairs[0]
        return self._merge_structural(c1, c2)

    def _update_pareto(self, candidate: PromptCandidate) -> None:
        """Update instance-level Pareto frontier."""
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

    def _is_perfect(self, scores: dict[str, float]) -> bool:
        """Check if all scores meet perfect threshold."""
        return bool(scores) and all(v >= self.perfect_score for v in scores.values())

    def _should_accept(
        self, parent_scores: dict[str, float], child_scores: dict[str, float]
    ) -> bool:
        """Accept mutation if child is not worse than parent."""
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

        for gen in range(self.max_generations):
            # Evaluate candidates and store reflection data
            # Type: dict[int, tuple[dev_batch, dev_results, dev_item_scores, trajectories]]
            candidate_reflection_data: dict[
                int,
                tuple[list[dict[str, str]], list[str], list[dict[str, float]], list[list]],
            ] = {}

            for candidate in self.candidates:
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
                ) = self._run_minibatch(candidate.prompt_template, dev_batch, capture_results=True)
                # Store reflection data for this candidate (including trajectories)
                candidate_reflection_data[candidate.id] = (
                    dev_batch,
                    dev_results,
                    dev_item_scores,
                    dev_trajectories,
                )

                # Val evaluation for selection
                candidate.val_scores, candidate.per_item_val_scores, _, _ = self._run_minibatch(
                    candidate.prompt_template, self.val_examples
                )
                self._update_pareto(candidate)

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
                        child.dev_scores, _, _, _ = self._run_minibatch(
                            child.prompt_template, dev_batch
                        )
                        if self._should_accept(parent_scores, child.dev_scores):
                            new_candidates.append(child)
                    elif self.candidates:
                        new_candidates.append(random.choice(self.candidates))

                # Phase 2: Merge from Pareto frontier
                if self.use_merge and len(self.pareto_frontier) >= 2:
                    merged = self._try_merge_from_frontier()
                    if merged is not None:
                        merged.val_scores, merged.per_item_val_scores, _, _ = self._run_minibatch(
                            merged.prompt_template, self.val_examples
                        )
                        self._update_pareto(merged)
                        if merged.val_instance_wins:
                            new_candidates.append(merged)

                self.candidates = new_candidates

        return self._get_best_candidate()

    def _get_best_candidate(self) -> PromptCandidate:
        """Get best candidate by (instance_wins, val_score)."""
        candidates = self.pareto_frontier or self.candidates
        if candidates:
            return max(
                candidates,
                key=lambda c: (len(c.val_instance_wins), sum(c.val_scores.values())),
            )
        return self._new_candidate(self.initial_prompt_template)

    def get_pareto_frontier(self) -> list[PromptCandidate]:
        """Get current Pareto frontier."""
        return self.pareto_frontier.copy()

    def get_best_prompt(self) -> str:
        """Get best prompt template."""
        return self._get_best_candidate().prompt_template
