# KISSEvolve: Evolutionary Algorithm Discovery

KISSEvolve is an evolutionary algorithm discovery framework that uses LLM-guided mutation and crossover to evolve code variants. It enables automatic discovery of improved algorithms through evolutionary search with multiple advanced features.

## Overview

KISSEvolve evolves code variants through:

1. **Initial population generation** from starting code
1. **Evaluation** of each variant using a fitness function
1. **Selection** of promising variants using various sampling methods
1. **Mutation/crossover** to create new variants using LLM guidance
1. **Iteration** until convergence or max generations

## Key Features

- **LLM-Guided Evolution**: Uses language models to intelligently mutate and combine code variants
- **Multiple Sampling Methods**: Supports tournament selection, power-law sampling, and performance-novelty sampling
- **Island-Based Evolution**: Optional island model with configurable migration topologies
- **Novelty Rejection**: Optional code novelty rejection sampling to filter redundant variants
- **Elite Preservation**: Preserves best variants across generations
- **Multi-Model Support**: Can use multiple models with different probabilities
- **Rich Evaluation**: Supports fitness, metrics, artifacts, and error tracking

## Installation

KISSEvolve is part of the KISS Agent Framework. See the main [README.md](../../../../README.md) for installation instructions.

## Quick Start

```python
from typing import Any

from kiss.agents.kiss_evolve.kiss_evolve import KISSEvolve
from kiss.core.kiss_agent import KISSAgent

def evaluate_code(code: str) -> dict[str, Any]:
    """Evaluate code variant and return fitness and metrics."""
    try:
        namespace = {}
        exec(code, namespace)
        func = namespace.get('sort_array')
        
        if not func:
            return {"fitness": 0.0, "error": "Function not found"}
        
        import time
        times = []
        for size in [100, 500, 1000, 2000]:
            arr = list(range(size, 0, -1))
            start = time.perf_counter()
            result = func(arr)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            
            if result != sorted(arr):
                return {"fitness": 0.0, "error": "Incorrect output"}
        
        avg_time = sum(times) / len(times)
        fitness = 1000.0 / (avg_time + 1.0)
        
        return {
            "fitness": fitness,
            "metrics": {"avg_time_ms": avg_time, "times": times}
        }
    except Exception as e:
        return {"fitness": 0.0, "error": str(e)}

initial_code = """
def sort_array(arr):
    # Bubble sort implementation
    n = len(arr)
    arr = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

# Create a code agent wrapper function
def code_agent_wrapper(model_name: str, prompt_template: str, arguments: dict[str, str]) -> str:
    """Wrapper function for KISSEvolve that creates and runs a KISSAgent."""
    agent = KISSAgent(name="Code Optimizer")
    return agent.run(
        model_name=model_name,
        prompt_template=prompt_template,
        arguments=arguments,
        is_agentic=True  # Agentic mode uses finish tool to return code
    )

optimizer = KISSEvolve(
    code_agent_wrapper=code_agent_wrapper,
    initial_code=initial_code,
    evaluation_fn=evaluate_code,
    model_names=[("gemini-3-pro-preview", 1.0)],  # List of (model_name, probability) tuples
    population_size=8,
    max_generations=10,
    mutation_rate=0.7,
    elite_size=2
)

best_variant = optimizer.evolve()
print(f"Best fitness: {best_variant.fitness:.2f}")
print(f"Best code:\n{best_variant.code}")
```

## API Reference

### `KISSEvolve` Class

#### `__init__`

```python
KISSEvolve(
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
)
```

**Parameters:**

- `code_agent_wrapper`: Function that accepts `model_name`, `prompt_template`, and `arguments` and returns generated code as a string
- `initial_code`: The initial code to evolve
- `evaluation_fn`: Function that takes code string and returns dict with:
  - `fitness`: float (higher is better)
  - `metrics`: dict[str, float] (optional additional metrics)
  - `artifacts`: dict[str, Any] (optional execution artifacts)
  - `error`: str (optional error message if evaluation failed)
- `model_names`: List of tuples containing `(model_name, probability)`. Probabilities are normalized to sum to 1.0
- `extra_coding_instructions`: Extra instructions to add to the code generation prompt
- `population_size`: Number of variants to maintain in population (default: 8)
- `max_generations`: Maximum number of evolutionary generations (default: 10)
- `mutation_rate`: Probability of mutating a variant (default: 0.7)
- `elite_size`: Number of best variants to preserve each generation (default: 2)
- `num_islands`: Number of islands for island-based evolution (1 = disabled, default: 2)
- `migration_frequency`: Number of generations between migrations (default: 5)
- `migration_size`: Number of individuals to migrate between islands (default: 1)
- `migration_topology`: Migration topology - 'ring', 'fully_connected', or 'random' (default: 'ring')
- `enable_novelty_rejection`: Enable code novelty rejection sampling (default: False)
- `novelty_threshold`: Cosine similarity threshold for rejecting code (0.0-1.0, higher = more strict, default: 0.95)
- `max_rejection_attempts`: Maximum number of rejection attempts before accepting a variant (default: 5)
- `novelty_rag_model`: Model to use for generating code embeddings (default: first model from models list)
- `parent_sampling_method`: Parent sampling method - 'tournament', 'power_law', or 'performance_novelty' (default: 'power_law')
- `power_law_alpha`: Power-law sampling parameter (α) - lower = more exploration, higher = more exploitation (default: 1.0)
- `performance_novelty_lambda`: Performance-novelty sampling parameter (λ) controlling selection pressure (default: 1.0)

#### `evolve`

```python
evolve() -> CodeVariant
```

Runs the evolutionary algorithm and returns the best code variant found.

### `CodeVariant` Dataclass

```python
@dataclass
class CodeVariant:
    code: str
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    parent_id: int | None = None
    generation: int = 0
    id: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    evaluation_error: str | None = None
    offspring_count: int = 0
```

## Configuration

Default values for all parameters can be configured via `DEFAULT_CONFIG.kiss_evolve` in `src/kiss/agents/kiss_evolve/config.py`. See the API Reference above for parameter descriptions and defaults.

## Advanced Features

### Island-Based Evolution

Island-based evolution maintains multiple subpopulations (islands) that evolve independently, with periodic migration between islands. This helps maintain diversity and can prevent premature convergence.

```python
optimizer = KISSEvolve(
    # ... other parameters ...
    num_islands=4,  # Create 4 islands
    migration_frequency=5,  # Migrate every 5 generations
    migration_size=2,  # Migrate 2 individuals
    migration_topology="ring"  # Ring topology: island 0 -> 1 -> 2 -> 3 -> 0
)
```

### Novelty Rejection Sampling

Novelty rejection filters out code variants that are too similar to existing variants, encouraging exploration of diverse solutions.

```python
optimizer = KISSEvolve(
    # ... other parameters ...
    enable_novelty_rejection=True,
    novelty_threshold=0.95,  # Reject if cosine similarity > 0.95
    max_rejection_attempts=5  # Try up to 5 times before accepting
)
```

### Parent Sampling Methods

KISSEvolve supports three parent sampling methods:

1. **Tournament Selection** (default): Randomly selects parents from top performers
1. **Power-Law Sampling**: Uses rank-based power-law distribution (α parameter controls exploration vs exploitation)
1. **Performance-Novelty Sampling**: Balances performance and novelty using a sigmoid function (λ parameter controls selection pressure)

```python
optimizer = KISSEvolve(
    # ... other parameters ...
    parent_sampling_method="power_law",
    power_law_alpha=0.5  # Lower = more exploration
)
```

### Multi-Model Support

You can use multiple models with different probabilities:

```python
optimizer = KISSEvolve(
    # ... other parameters ...
    model_names=[
        ("gemini-3-pro-preview", 0.7),  # 70% probability
        ("gpt-4o", 0.3),  # 30% probability
    ]
)
```

## Examples

See `src/kiss/scripts/kissevolve_bubblesort.py` for a complete example that evolves bubble sort to discover faster sorting algorithms.

## Authors

- Koushik Sen (ksen@berkeley.edu)
