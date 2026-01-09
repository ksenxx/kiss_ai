# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""KISSEvolve: Evolutionary Algorithm Discovery using LLMs."""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import kiss.agents.kiss_evolve.config  # noqa: F401
from kiss.agents.kiss_evolve.novelty_prompts import INNOVATION_INSTRUCTIONS
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.model import Model
from kiss.core.utils import get_config_value
from kiss.rag.simple_rag import SimpleRAG


@dataclass
class CodeVariant:
    """Represents a code variant in the evolutionary population."""

    code: str
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    parent_id: int | None = None
    generation: int = 0
    id: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    evaluation_error: str | None = None
    offspring_count: int = 0  # Track number of offspring for novelty-based sampling


class KISSEvolve:
    """KISSEvolve: Evolutionary algorithm discovery using LLMs.

    KISSEvolve evolves code variants through:
    1. Initial population generation
    2. Evaluation of each variant
    3. Selection of promising variants
    4. Mutation/crossover to create new variants
    5. Iteration until convergence or max generations
    # Pseudocode for KISSEvolve showing use of all parameters

    class KISSEvolve:
        - Input parameters:
            - code_agent_wrapper: function to generate new code from prompts and instructions
            - initial_code: starting code for the population
            - evaluation_fn: function that evaluates code's fitness/metrics
            - model_names: list of (model_name, probability) tuples for LLM selection
            - extra_coding_instructions: extra prompt instructions for LLM
            - population_size: number of code variants per generation
            - max_generations: how many evolutionary generations to run
            - mutation_rate: probability of applying mutation each step
            - elite_size: number of top performers to keep ("elitism")
            - num_islands: number of isolated subpopulations (for island-model evolution)
            - migration_frequency: how often to migrate code between islands
            - migration_size: how many individuals migrate at each event
            - migration_topology: migration pattern (e.g. ring, fully_connected)
            - enable_novelty_rejection: whether to use code-novelty sampling
            - novelty_threshold: similarity cutoff for considering code novel
            - max_rejection_attempts: max tries to generate novel code before accepting
            - novelty_rag_model: model for code similarity/RAG
            - parent_sampling_method: how parents are picked (uniform, probabilistic, etc.)
            - power_law_alpha: parameter for power-law parent selection
            - performance_novelty_lambda: weight for combining performance and novelty
        - Algorithm outline:
            1. Initialize population:
                - Create initial population of code variants, using `initial_code` as seed
            2. Evaluate all code variants with `evaluation_fn` to determine fitness/metrics
            3. For each generation (up to `max_generations`):
                a. (If using islands) For each island:
                    - Maintain separate subpopulations
                    - Proceed through selection/mutation/crossover independently
                    - Every `migration_frequency` gens, migrate variants per `migration_topology`
                b. Selection:
                    - Select `elite_size` best variants to carry over unchanged
                    - Select parents using `parent_sampling_method` (uniform, fitness, etc.)
                c. Mutation:
                    - With probability `mutation_rate`, generate mutated code for new variants
                    - Use `code_agent_wrapper` with prompts and `extra_coding_instructions`
                    - If `enable_novelty_rejection`, check novelty via `novelty_rag_model`
                        - Try up to `max_rejection_attempts` for novel code
                d. Crossover:
                    - Occasionally combine two parents to create new code variant
                e. Evaluate all new code variants with `evaluation_fn`
                f. Update population, repeat
            4. Return the best code variant found
    """


    def __init__(
        self,
        code_agent_wrapper: Callable[..., str],
        initial_code: str,
        evaluation_fn: Callable[[str], dict[str, Any]],
        model_names: list[tuple[str, float]],
        extra_coding_instructions: str = "",
        population_size: int | None = None,
        max_generations: int | None = None,
        mutation_rate: float | None = None,
        elite_size: int | None = None,
        num_islands: int | None = None,
        migration_frequency: int | None = None,
        migration_size: int | None = None,
        migration_topology: str | None = None,
        enable_novelty_rejection: bool | None = None,
        novelty_threshold: float | None = None,
        max_rejection_attempts: int | None = None,
        novelty_rag_model: Model | None = None,
        parent_sampling_method: str | None = None,
        power_law_alpha: float | None = None,
        performance_novelty_lambda: float | None = None,
    ):
        """Initialize KISSEvolve optimizer.

        Args:
            code_agent_wrapper (Callable[..., str]): The code generation agent
                wrapper. Should accept keyword arguments: model (Model),
                prompt_template (str), and arguments (dict[str, str]).
            initial_code (str): The initial code to evolve.
            evaluation_fn (Callable[[str], dict[str, Any]]): Function that takes
                code string and returns dict with:
                - 'fitness': float (higher is better)
                - 'metrics': dict[str, float] (optional additional metrics)
                - 'artifacts': dict[str, Any] (optional execution artifacts)
                - 'error': str (optional error message if evaluation failed)
            model_names (list[tuple[str, float]]): List of tuples containing
                (model_name, probability). Probabilities will be normalized to sum to 1.0.
            extra_coding_instructions (str): Extra instructions to add to the code
                generation prompt.
            population_size (int | None): Number of variants to maintain in population.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.population_size.
            max_generations (int | None): Maximum number of evolutionary generations.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.max_generations.
            mutation_rate (float | None): Probability of mutating a variant.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.mutation_rate.
            elite_size (int | None): Number of best variants to preserve each
                generation. If None, uses value from DEFAULT_CONFIG.kiss_evolve.elite_size.
            num_islands (int | None): Number of islands for island-based evolution.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.num_islands.
            migration_frequency (int | None): Number of generations between migrations.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.migration_frequency.
            migration_size (int | None): Number of individuals to migrate between islands.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.migration_size.
            migration_topology (str | None): Migration topology
                ('ring', 'fully_connected', 'random').
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.migration_topology.
            enable_novelty_rejection (bool | None): Enable code novelty rejection sampling.
                If None, uses value from DEFAULT_CONFIG.kiss_evolve.enable_novelty_rejection.
            novelty_threshold (float | None): Cosine similarity threshold for rejecting code
                (0.0-1.0, higher = more strict). If None, uses value from
                DEFAULT_CONFIG.kiss_evolve.novelty_threshold.
            max_rejection_attempts (int | None): Maximum number of rejection attempts before
                accepting a variant anyway. If None, uses value from
                DEFAULT_CONFIG.kiss_evolve.max_rejection_attempts.
            novelty_rag_model (Model | None): Model to use for generating code embeddings.
                If None and novelty rejection is enabled, uses the first model from models list.
            parent_sampling_method (str | None): Parent sampling method ('tournament', 'power_law',
                or 'performance_novelty'). If None, uses value from
                DEFAULT_CONFIG.kiss_evolve.parent_sampling_method.
            power_law_alpha (float | None): Power-law sampling parameter (α) for rank-based
                sampling. Lower = more exploration, higher = more exploitation. If None, uses value
                from DEFAULT_CONFIG.kiss_evolve.power_law_alpha.
            performance_novelty_lambda (float | None): Performance-novelty sampling parameter (λ)
                controlling selection pressure. If None, uses value from
                DEFAULT_CONFIG.kiss_evolve.performance_novelty_lambda.
        """
        self.initial_code = initial_code
        self.evaluation_fn = evaluation_fn
        self.code_agent_wrapper = code_agent_wrapper
        self.extra_coding_instructions = extra_coding_instructions

        # Validate and normalize model probabilities
        self.model_names = self._validate_and_normalize_models(model_names)

        # Get config with fallback defaults
        cfg = getattr(DEFAULT_CONFIG, "kiss_evolve", None)
        defaults = {
            "population_size": 8, "max_generations": 10, "mutation_rate": 0.7,
            "elite_size": 2, "num_islands": 1, "migration_frequency": 5,
            "migration_size": 1, "migration_topology": "ring",
            "enable_novelty_rejection": False, "novelty_threshold": 0.95,
            "max_rejection_attempts": 5, "parent_sampling_method": "tournament",
            "power_law_alpha": 1.0, "performance_novelty_lambda": 1.0,
        }

        def get_val(name: str, value: Any) -> Any:
            if cfg:
                return get_config_value(value, cfg, name, defaults[name])
            return value if value is not None else defaults[name]

        self.population_size = get_val("population_size", population_size)
        self.max_generations = get_val("max_generations", max_generations)
        self.mutation_rate = get_val("mutation_rate", mutation_rate)
        self.elite_size = get_val("elite_size", elite_size)
        self.num_islands = get_val("num_islands", num_islands)
        self.migration_frequency = get_val("migration_frequency", migration_frequency)
        self.migration_size = get_val("migration_size", migration_size)
        self.migration_topology = get_val("migration_topology", migration_topology)
        self.enable_novelty_rejection = get_val(
            "enable_novelty_rejection", enable_novelty_rejection
        )
        self.novelty_threshold = get_val("novelty_threshold", novelty_threshold)
        self.max_rejection_attempts = get_val("max_rejection_attempts", max_rejection_attempts)
        self.parent_sampling_method = get_val("parent_sampling_method", parent_sampling_method)
        self.power_law_alpha = get_val("power_law_alpha", power_law_alpha)
        self.performance_novelty_lambda = get_val(
            "performance_novelty_lambda", performance_novelty_lambda
        )

        self._validate_parameters()
        self._initialize_tracking(novelty_rag_model)
        self._setup_prompt_templates()

    @staticmethod
    def _validate_and_normalize_models(
        model_names: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Validate and normalize model probabilities."""
        if not model_names:
            raise ValueError("models list cannot be empty")
        if not all(isinstance(p, (int, float)) and p >= 0 for _, p in model_names):
            raise ValueError("all probabilities must be non-negative numbers")
        total_prob = sum(prob for _, prob in model_names)
        if total_prob == 0:
            raise ValueError("sum of probabilities cannot be zero")
        return [(name, prob / total_prob) for name, prob in model_names]

    def _validate_parameters(self) -> None:
        """Validate all configuration parameters."""
        if self.elite_size >= self.population_size:
            error_msg = (
                f"elite_size ({self.elite_size}) must be less than "
                f"population_size ({self.population_size})"
            )
            raise ValueError(error_msg)
        if not 0.0 <= self.mutation_rate <= 1.0:
            error_msg = f"mutation_rate must be between 0.0 and 1.0, got {self.mutation_rate}"
            raise ValueError(error_msg)
        if self.num_islands < 1:
            raise ValueError(f"num_islands must be at least 1, got {self.num_islands}")
        if self.migration_frequency < 1:
            raise ValueError(
                f"migration_frequency must be at least 1, got {self.migration_frequency}"
            )
        if self.migration_size < 1:
            raise ValueError(f"migration_size must be at least 1, got {self.migration_size}")
        if self.migration_topology not in ["ring", "fully_connected", "random"]:
            raise ValueError(
                f"migration_topology must be 'ring', 'fully_connected', or 'random', "
                f"got {self.migration_topology}"
            )
        if not 0.0 <= self.novelty_threshold <= 1.0:
            raise ValueError(
                f"novelty_threshold must be between 0.0 and 1.0, got {self.novelty_threshold}"
            )
        if self.max_rejection_attempts < 1:
            raise ValueError(
                f"max_rejection_attempts must be at least 1, got {self.max_rejection_attempts}"
            )
        if self.parent_sampling_method not in ["tournament", "power_law", "performance_novelty"]:
            raise ValueError(
                f"parent_sampling_method must be 'tournament', 'power_law', or "
                f"'performance_novelty', got {self.parent_sampling_method}"
            )
        if self.power_law_alpha < 0:
            raise ValueError(f"power_law_alpha must be non-negative, got {self.power_law_alpha}")
        if self.performance_novelty_lambda <= 0:
            raise ValueError(
                f"performance_novelty_lambda must be positive, "
                f"got {self.performance_novelty_lambda}"
            )

    def _initialize_tracking(self, novelty_rag_model: Model | None) -> None:
        """Initialize population tracking and novelty RAG."""
        # Population tracking
        self.population: list[CodeVariant] = []
        self.variant_counter = 0
        self.generation_history: list[list[CodeVariant]] = []

        # Island-based evolution tracking
        self.islands: list[list[CodeVariant]] = []
        self.island_histories: list[list[list[CodeVariant]]] = []

        # Code novelty rejection sampling using RAG
        self.novelty_rag: SimpleRAG | None = None
        if self.enable_novelty_rejection:
            # Use provided model or first model from models list
            rag_model_name: str
            if novelty_rag_model is not None:
                rag_model_name = novelty_rag_model.model_name
            else:
                rag_model_name = self.model_names[0][0]
            self.novelty_rag = SimpleRAG(model_name=rag_model_name, metric="cosine")
            print(
                f"Code novelty rejection sampling enabled: "
                f"threshold={self.novelty_threshold}, "
                f"max_attempts={self.max_rejection_attempts}"
            )

    def _setup_prompt_templates(self) -> None:
        """Initialize prompt templates for mutation and crossover."""
        self.mutation_prompt_template = """You are an expert code optimizer. Your task is to \
improve the given code by optimizing it for better performance while maintaining correctness.

## Instructions ##
- Analyze the given code and identify optimization opportunities
- Focus on performance improvements (speed, memory efficiency, algorithmic
  improvements)
- Maintain correctness - the optimized code must produce the same results
- You can optimize: algorithms, data structures, memory access patterns,
  computation order
- You MUST return ONLY the optimized code by calling the 'finish' tool with
  the code as the 'result' argument.
- The code should be a complete, runnable function or module

## Original Code ##
{original_code}

## Evaluation Feedback ##
{feedback}

## Optimization Constraints ##
{constraints}

"""

        self.crossover_prompt_template = """You are an expert code optimizer. Your task is to \
combine the best aspects of two code variants to create an improved version.

## Instructions ##
- Analyze both code variants and identify their strengths
- Combine the best features from both variants
- Ensure the combined code is correct and performs well
- You MUST return ONLY the combined code by calling the 'finish' tool with
  the code as the 'result' argument.
- The code should be a complete, runnable function or module

## Code Variant 1 (Fitness: {fitness1}) ##
{code1}

## Code Variant 2 (Fitness: {fitness2}) ##
{code2}

"""

        self.constraints = (
            "- MUST NOT change function signatures or external API\n"
            "- MUST maintain correctness (same input -> same output)\n"
            "- ALLOWED: Internal optimizations, algorithm improvements, "
            "data structure changes\n"
        )

    def _select_model(self) -> str:
        """Randomly select a model name based on probabilities."""
        model_names, weights = zip(*self.model_names)
        selected: str = random.choices(model_names, weights=weights, k=1)[0]
        return selected

    def _is_code_novel(self, code: str) -> bool:
        """Check if code is novel (sufficiently different from existing code in RAG).

        Args:
            code: Code string to check for novelty.

        Returns:
            True if code is novel (similarity below threshold), False otherwise.
        """
        if not self.enable_novelty_rejection or self.novelty_rag is None:
            return True

        # If RAG is empty, all code is novel
        stats = self.novelty_rag.get_collection_stats()
        if stats["num_documents"] == 0:
            return True

        # Query RAG for similar code
        results = self.novelty_rag.query(code, top_k=1)
        if not results:
            return True

        # Check if highest similarity is below threshold
        # For cosine similarity, higher score = more similar
        max_similarity = float(results[0]["score"])
        is_novel: bool = max_similarity < self.novelty_threshold

        if not is_novel:
            print(
                f"  Rejected code variant (similarity: {max_similarity:.4f} >= "
                f"threshold: {self.novelty_threshold:.4f})"
            )

        return is_novel

    def _add_code_to_rag(self, code: str, variant_id: int) -> None:
        """Add code variant to RAG for novelty tracking.

        Args:
            code: Code string to add.
            variant_id: Unique identifier for the code variant.
        """
        if not self.enable_novelty_rejection or self.novelty_rag is None:
            return

        document = {
            "id": str(variant_id),
            "text": code,
            "metadata": {"variant_id": variant_id},
        }
        self.novelty_rag.add_documents([document])

    def __pick_random_innovation_prompt(self) -> str:
        """Pick a random innovation prompt from the list of innovation prompts."""
        if random.random() < 0.5:
            return ""
        rand_index = random.randint(0, len(INNOVATION_INSTRUCTIONS) - 1)
        return INNOVATION_INSTRUCTIONS[rand_index]["prompt"]

    def _mutate_code(self, variant: CodeVariant, feedback: str = "") -> str:
        """Mutate a code variant using LLM with novelty rejection sampling.

        Args:
            variant: Code variant to mutate.
            feedback: Evaluation feedback for the variant.

        Returns:
            Mutated code string that passes novelty check (or after max attempts).
        """
        model_name = self._select_model()
        prompt = (
            self.mutation_prompt_template
            + self.extra_coding_instructions
            + self.__pick_random_innovation_prompt()
        )

        # Try generating novel code up to max_rejection_attempts times
        for attempt in range(self.max_rejection_attempts):
            result = self.code_agent_wrapper(
                model_name=model_name,
                prompt_template=prompt,
                arguments={
                    "original_code": variant.code,
                    "feedback": feedback or "No specific feedback available.",
                    "constraints": self.constraints,
                },
            )

            # Check novelty if enabled
            if self._is_code_novel(result):
                return result

            # If not novel and not last attempt, try again
            if attempt < self.max_rejection_attempts - 1:
                print(
                    f"  Retrying mutation (attempt {attempt + 2}/{self.max_rejection_attempts})..."
                )

        # If all attempts failed novelty check, return the last generated code anyway
        print(
            f"  Accepted code variant after {self.max_rejection_attempts} attempts "
            "(may not be novel)"
        )
        return result

    def _crossover_code(self, variant1: CodeVariant, variant2: CodeVariant) -> str:
        """Create a new variant by combining two parent variants with novelty rejection sampling.

        Args:
            variant1: First parent code variant.
            variant2: Second parent code variant.

        Returns:
            Combined code string that passes novelty check (or after max attempts).
        """
        model_name = self._select_model()
        prompt = (
            self.crossover_prompt_template
            + self.extra_coding_instructions
            + self.__pick_random_innovation_prompt()
        )

        # Try generating novel code up to max_rejection_attempts times
        for attempt in range(self.max_rejection_attempts):
            result = self.code_agent_wrapper(
                model_name=model_name,
                prompt_template=prompt,
                arguments={
                    "code1": variant1.code,
                    "fitness1": str(variant1.fitness),
                    "code2": variant2.code,
                    "fitness2": str(variant2.fitness),
                },
            )

            # Check novelty if enabled
            if self._is_code_novel(result):
                return result

            # If not novel and not last attempt, try again
            if attempt < self.max_rejection_attempts - 1:
                print(
                    f"  Retrying crossover (attempt {attempt + 2}/{self.max_rejection_attempts})..."
                )

        # If all attempts failed novelty check, return the last generated code anyway
        print(
            f"  Accepted code variant after {self.max_rejection_attempts} attempts "
            "(may not be novel)"
        )
        return result

    def _evaluate_variant(self, variant: CodeVariant) -> None:
        """Evaluate a code variant and update its fitness."""
        try:
            result = self.evaluation_fn(variant.code)
            variant.fitness = result.get("fitness", 0.0)
            variant.metrics = result.get("metrics", {})
            variant.artifacts = result.get("artifacts", {})
            variant.evaluation_error = result.get("error")
        except Exception as e:
            # Catch all exceptions since evaluation_fn may execute arbitrary
            # code that could raise any exception type. Set fitness to 0.0 to
            # mark as failed.
            variant.fitness = 0.0
            variant.evaluation_error = str(e)
            variant.metrics = {}
            variant.artifacts = {}

    def _initialize_population(self) -> None:
        """Initialize the population with the initial code."""
        initial_variant = CodeVariant(
            code=self.initial_code,
            id=self.variant_counter,
            generation=0,
        )
        self.variant_counter += 1
        self._evaluate_variant(initial_variant)
        self.population = [initial_variant]
        # Add initial code to RAG for novelty tracking
        self._add_code_to_rag(self.initial_code, initial_variant.id)

    def _initialize_islands(self) -> None:
        """Initialize islands for island-based evolution."""
        self.islands = []
        self.island_histories = []

        # Create initial variant for each island
        for island_id in range(self.num_islands):
            initial_variant = CodeVariant(
                code=self.initial_code,
                id=self.variant_counter,
                generation=0,
            )
            self.variant_counter += 1
            self._evaluate_variant(initial_variant)

            # Initialize island with initial variant
            island_population = [initial_variant]
            self.islands.append(island_population)
            self.island_histories.append([island_population.copy()])
            # Add initial code to RAG for novelty tracking
            self._add_code_to_rag(self.initial_code, initial_variant.id)

    def _power_law_sample(self, population: list[CodeVariant], n: int = 1) -> list[CodeVariant]:
        """Select parents using power-law (rank-based) sampling.

        Programs are ranked by fitness, and selection probabilities follow a power-law
        distribution: p_i = r_i^(-α) / Σ r_j^(-α), where r_i is the rank.

        Args:
            population: List of code variants to sample from.
            n: Number of parents to select.

        Returns:
            List of selected parent variants.
        """
        if len(population) < n:
            return population.copy()

        # Sort by fitness (descending) and assign ranks
        sorted_pop = sorted(population, key=lambda v: v.fitness, reverse=True)
        num_variants = len(sorted_pop)

        # Calculate selection probabilities using power-law
        # Rank 1 (best) has r=1, rank 2 has r=2, etc.
        ranks = list(range(1, num_variants + 1))
        # Use rank^(-alpha) as weights
        weights = [r ** (-self.power_law_alpha) for r in ranks]
        total_weight = sum(weights)

        # Normalize to probabilities
        probabilities = [w / total_weight for w in weights]

        # Sample n parents (with replacement)
        selected = random.choices(sorted_pop, weights=probabilities, k=n)
        return selected

    def _performance_novelty_sample(
        self, population: list[CodeVariant], n: int = 1
    ) -> list[CodeVariant]:
        """Select parents using weighted performance-novelty sampling.

        Combines fitness (via sigmoid) with novelty (based on offspring count) to
        determine selection probability:
        - Performance: s_i = σ(λ · (F(P_i) - α₀)), where α₀ is median fitness
        - Novelty: h_i = 1 / (1 + N(P_i)), where N(P_i) is offspring count
        - Combined: p_i = (s_i · h_i) / Σ(s_j · h_j)

        Args:
            population: List of code variants to sample from.
            n: Number of parents to select.

        Returns:
            List of selected parent variants.
        """
        if len(population) < n:
            return population.copy()

        if len(population) == 1:
            return population * n

        # Calculate median fitness (α₀)
        fitnesses = [v.fitness for v in population]
        fitnesses_sorted = sorted(fitnesses)
        median_idx = len(fitnesses_sorted) // 2
        if len(fitnesses_sorted) % 2 == 0:
            alpha_0 = (fitnesses_sorted[median_idx - 1] + fitnesses_sorted[median_idx]) / 2.0
        else:
            alpha_0 = fitnesses_sorted[median_idx]

        # Calculate selection weights for each variant
        weights = []
        for variant in population:
            # Performance component: sigmoid of (fitness - median)
            # σ(x) = 1 / (1 + exp(-x))
            sigmoid_input = self.performance_novelty_lambda * (variant.fitness - alpha_0)
            # Clamp to avoid overflow
            sigmoid_input = max(-500, min(500, sigmoid_input))
            performance_score = 1.0 / (1.0 + math.exp(-sigmoid_input))

            # Novelty component: discount based on offspring count
            novelty_discount = 1.0 / (1.0 + variant.offspring_count)

            # Combined weight
            weight = performance_score * novelty_discount
            weights.append(weight)

        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to uniform if all weights are zero
            probabilities = [1.0 / len(population)] * len(population)
        else:
            probabilities = [w / total_weight for w in weights]

        # Sample n parents (with replacement)
        selected = random.choices(population, weights=probabilities, k=n)
        return selected

    def _select_parents(
        self, n: int = 2, population: list[CodeVariant] | None = None
    ) -> list[CodeVariant]:
        """Select parent variants using the configured sampling method.

        Args:
            n: Number of parents to select.
            population: Population to select from. If None, uses self.population.

        Returns:
            List of selected parent variants.
        """
        pop = population if population is not None else self.population
        if len(pop) < n:
            return pop.copy()

        if self.parent_sampling_method == "power_law":
            return self._power_law_sample(pop, n)
        elif self.parent_sampling_method == "performance_novelty":
            return self._performance_novelty_sample(pop, n)
        else:
            # Default: tournament selection (also handles "tournament" and unknown methods)
            tournament_size = min(3, len(pop))
            selected = []
            for _ in range(n):
                tournament = random.sample(pop, tournament_size)
                best = max(tournament, key=lambda v: v.fitness)
                selected.append(best)
            return selected

    def _evolve_population(
        self, population: list[CodeVariant], generation: int
    ) -> list[CodeVariant]:
        """Evolve one generation of a population.

        Args:
            population: The population to evolve.
            generation: The current generation number.

        Returns:
            The new population after evolution.
        """
        # Sort population by fitness
        population.sort(key=lambda v: v.fitness, reverse=True)

        # Keep elite variants
        elite = population[: self.elite_size].copy()
        new_population = []

        # Create copies of elite variants
        for variant in elite:
            elite_copy = CodeVariant(
                code=variant.code,
                fitness=variant.fitness,
                metrics=variant.metrics.copy(),
                parent_id=variant.id,
                generation=generation,
                id=self.variant_counter,
            )
            self.variant_counter += 1
            new_population.append(elite_copy)

        # Generate new variants through mutation and crossover
        while len(new_population) < self.population_size:
            should_mutate = random.random() < self.mutation_rate and len(population) > 0
            if should_mutate:
                # Mutation: select a parent using the configured sampling method
                parents = self._select_parents(1, population)
                parent = parents[0] if parents else random.choice(population)
                feedback = ""
                if parent.artifacts:
                    feedback = f"Previous performance: {parent.fitness:.4f}. "
                    if parent.evaluation_error:
                        feedback += f"Error: {parent.evaluation_error}"

                mutated_code = self._mutate_code(parent, feedback)
                new_variant = CodeVariant(
                    code=mutated_code,
                    parent_id=parent.id,
                    generation=generation,
                    id=self.variant_counter,
                )
                self.variant_counter += 1
                # Increment parent's offspring count
                parent.offspring_count += 1
                self._evaluate_variant(new_variant)
                # Add accepted variant to RAG for novelty tracking
                self._add_code_to_rag(mutated_code, new_variant.id)
                new_population.append(new_variant)
            else:
                # Crossover: combine two parents
                parents = self._select_parents(2, population)
                if len(parents) >= 2:
                    combined_code = self._crossover_code(parents[0], parents[1])
                    new_variant = CodeVariant(
                        code=combined_code,
                        parent_id=parents[0].id,
                        generation=generation,
                        id=self.variant_counter,
                    )
                    self.variant_counter += 1
                    # Increment both parents' offspring counts
                    parents[0].offspring_count += 1
                    parents[1].offspring_count += 1
                    self._evaluate_variant(new_variant)
                    # Add accepted variant to RAG for novelty tracking
                    self._add_code_to_rag(combined_code, new_variant.id)
                    new_population.append(new_variant)
                else:
                    # Fallback: just copy a variant
                    if population:
                        parent = random.choice(population)
                        new_variant = CodeVariant(
                            code=parent.code,
                            fitness=parent.fitness,
                            metrics=parent.metrics.copy(),
                            parent_id=parent.id,
                            generation=generation,
                            id=self.variant_counter,
                        )
                        self.variant_counter += 1
                        new_population.append(new_variant)

        return new_population

    def _evolve_generation(self, generation: int) -> None:
        """Evolve one generation of the population."""
        self.population = self._evolve_population(self.population, generation)
        self.generation_history.append(self.population.copy())

    def evolve(self) -> CodeVariant:
        """Run the evolutionary algorithm.

        Returns:
            CodeVariant: The best code variant found during evolution.
        """
        if self.num_islands > 1:
            return self._evolve_with_islands()
        else:
            return self._evolve_single_population()

    def _evolve_single_population(self) -> CodeVariant:
        """Run the evolutionary algorithm with a single population."""
        pop_size = self.population_size
        print(f"Initializing KISSEvolve with population size {pop_size}")
        self._initialize_population()

        # Sort population by fitness for reporting
        self.population.sort(key=lambda v: v.fitness, reverse=True)
        best_fitness = self.population[0].fitness
        print(f"Generation 0: Best fitness = {best_fitness:.4f}")

        for generation in range(1, self.max_generations + 1):
            print(f"\nEvolving generation {generation}...")
            self._evolve_generation(generation)

            # Sort and report
            self.population.sort(key=lambda v: v.fitness, reverse=True)
            best = self.population[0]
            print(f"Generation {generation}: Best fitness = {best.fitness:.4f}")
            if best.metrics:
                print(f"  Metrics: {best.metrics}")

        # Return best variant
        self.population.sort(key=lambda v: v.fitness, reverse=True)
        return self.population[0]

    def _evolve_with_islands(self) -> CodeVariant:
        """Run the evolutionary algorithm with island-based evolution."""
        pop_size = self.population_size
        print(
            f"Initializing KISSEvolve with {self.num_islands} islands, "
            f"population size {pop_size} per island"
        )
        print(
            f"Migration: every {self.migration_frequency} generations, "
            f"{self.migration_size} individuals, topology: {self.migration_topology}"
        )
        self._initialize_islands()

        # Report initial state
        all_variants = [v for island in self.islands for v in island]
        all_variants.sort(key=lambda v: v.fitness, reverse=True)
        best_fitness = all_variants[0].fitness
        print(f"Generation 0: Best fitness across all islands = {best_fitness:.4f}")
        for island_id, island_pop in enumerate(self.islands):
            if island_pop:
                island_best = max(island_pop, key=lambda v: v.fitness)
                print(
                    f"  Island {island_id}: Best fitness = {island_best.fitness:.4f}, "
                    f"Population size = {len(island_pop)}"
                )

        for generation in range(1, self.max_generations + 1):
            print(f"\nEvolving generation {generation}...")

            # Evolve each island
            for island_id in range(self.num_islands):
                new_population = self._evolve_island_generation(island_id, generation)
                self.islands[island_id] = new_population
                self.island_histories[island_id].append(new_population.copy())

            # Perform migration if it's time
            if generation % self.migration_frequency == 0:
                print(f"  Performing migration at generation {generation}...")
                self._migrate_between_islands()

            # Sort and report
            all_variants = [v for island in self.islands for v in island]
            all_variants.sort(key=lambda v: v.fitness, reverse=True)
            best = all_variants[0]
            print(f"Generation {generation}: Best fitness across all islands = {best.fitness:.4f}")
            if best.metrics:
                print(f"  Metrics: {best.metrics}")

            # Report per-island statistics
            for island_id, island_pop in enumerate(self.islands):
                if island_pop:
                    island_best = max(island_pop, key=lambda v: v.fitness)
                    island_avg = sum(v.fitness for v in island_pop) / len(island_pop)
                    print(
                        f"  Island {island_id}: Best = {island_best.fitness:.4f}, "
                        f"Avg = {island_avg:.4f}, Size = {len(island_pop)}"
                    )

        # Return best variant across all islands
        all_variants = [v for island in self.islands for v in island]
        all_variants.sort(key=lambda v: v.fitness, reverse=True)
        return all_variants[0]

    def get_best_variant(self) -> CodeVariant:
        """Get the best variant from the current population or islands."""
        if self.num_islands > 1 and self.islands:
            # Get best from all islands
            all_variants = [v for island in self.islands for v in island]
            if all_variants:
                return max(all_variants, key=lambda v: v.fitness)
        elif self.population:
            return max(self.population, key=lambda v: v.fitness)
        return CodeVariant(code=self.initial_code, id=0)

    def get_population_stats(self) -> dict[str, Any]:
        """Get statistics about the current population."""
        if not self.population:
            return {"size": 0, "avg_fitness": 0.0, "best_fitness": 0.0}

        fitnesses = [v.fitness for v in self.population]
        return {
            "size": len(self.population),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
        }

    def _get_migration_targets(self, island_id: int) -> list[int]:
        """Get target islands for migration from a given island."""
        if self.migration_topology == "ring":
            # Ring topology: each island migrates to the next one
            return [(island_id + 1) % self.num_islands]
        elif self.migration_topology == "fully_connected":
            # Fully connected: migrate to all other islands
            return [i for i in range(self.num_islands) if i != island_id]
        elif self.migration_topology == "random":
            # Random: migrate to a random other island
            other_islands = [i for i in range(self.num_islands) if i != island_id]
            if other_islands:
                return [random.choice(other_islands)]
            return []
        else:
            return []

    def _migrate_between_islands(self) -> None:
        """Perform migration between islands."""
        if self.num_islands <= 1:
            return

        # Collect migrants from each island
        migrants_by_target: dict[int, list[CodeVariant]] = {i: [] for i in range(self.num_islands)}

        for island_id in range(self.num_islands):
            island_pop = self.islands[island_id]
            if not island_pop:
                continue

            # Sort by fitness and select top individuals for migration
            sorted_pop = sorted(island_pop, key=lambda v: v.fitness, reverse=True)
            num_migrants = min(self.migration_size, len(sorted_pop))
            migrants = sorted_pop[:num_migrants]

            # Determine target islands
            target_islands = self._get_migration_targets(island_id)

            # Distribute migrants to target islands
            for target_id in target_islands:
                # Create copies of migrants for the target island
                for migrant in migrants:
                    migrant_copy = CodeVariant(
                        code=migrant.code,
                        fitness=migrant.fitness,
                        metrics=migrant.metrics.copy(),
                        parent_id=migrant.id,
                        generation=migrant.generation,
                        id=self.variant_counter,
                        artifacts=migrant.artifacts.copy(),
                        evaluation_error=migrant.evaluation_error,
                    )
                    self.variant_counter += 1
                    migrants_by_target[target_id].append(migrant_copy)

        # Add migrants to target islands
        for target_id, migrants in migrants_by_target.items():
            if migrants:
                # Remove worst individuals to make room (if needed)
                target_island = self.islands[target_id]
                if len(target_island) + len(migrants) > self.population_size:
                    # Sort and keep the best
                    target_island.sort(key=lambda v: v.fitness, reverse=True)
                    target_island = target_island[: self.population_size - len(migrants)]
                    self.islands[target_id] = target_island

                # Add migrants
                self.islands[target_id].extend(migrants)
                print(f"  Migrated {len(migrants)} individuals to island {target_id}")

    def _evolve_island_generation(self, island_id: int, generation: int) -> list[CodeVariant]:
        """Evolve one generation for a specific island."""
        return self._evolve_population(self.islands[island_id], generation)
