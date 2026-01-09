# GEPA: Genetic-Pareto Prompt Evolution

GEPA (Genetic-Pareto) is a prompt optimization framework that uses natural language reflection to evolve prompts for compound AI systems. It maintains a Pareto frontier of top-performing prompts and combines complementary lessons through evolutionary search.

**Paper**: [GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPERFORM REINFORCEMENT LEARNING](https://arxiv.org/pdf/2507.19457)

## Overview

GEPA optimizes prompts by:
1. **Sampling trajectories** from the agent system
2. **Reflecting on trajectories** in natural language using an LLM
3. **Mutating prompts** based on reflection insights
4. **Maintaining a Pareto frontier** of top-performing prompts
5. **Combining complementary lessons** from the frontier through evolutionary search

## Key Features

- **Natural Language Reflection**: Uses LLM-based reflection to analyze agent trajectories and propose prompt improvements
- **Pareto Frontier**: Maintains a multi-objective Pareto frontier of top-performing prompts
- **Placeholder Preservation**: Automatically sanitizes prompts to preserve valid placeholders from the initial template
- **Multi-Objective Optimization**: Supports multiple evaluation metrics simultaneously
- **Configurable Rollouts**: Allows multiple rollouts per generation for more robust evaluation

## Installation

GEPA is part of the KISS Agent Framework. See the main [README.md](../../../../README.md) for installation instructions.

## Quick Start

```python
from kiss.agents.gepa import GEPA
from kiss.core.kiss_agent import KISSAgent

def evaluate_result(result: str) -> dict[str, float]:
    """Evaluate agent result and return scores (higher is better)."""
    scores = {}
    if "success" in result.lower():
        scores["success"] = 1.0
    else:
        scores["success"] = 0.0
    # Add more metrics as needed
    return scores

def my_tool(input: str) -> str:
    """Process the input and return a result."""
    return f"Processed: {input}"

# Create agent wrapper function for GEPA
def agent_wrapper(
    prompt_template: str, arguments: dict[str, str]
) -> tuple[str, list[dict[str, str]]]:
    """Wrapper function that runs the agent and returns result and trajectory.
    
    Note: get_trajectory() returns a JSON string, so we parse it to a list.
    """
    import json
    agent = KISSAgent(name="My Agent")
    result = agent.run(
        model_name="gemini-3-pro-preview",
        prompt_template=prompt_template,
        arguments=arguments,
        tools=[my_tool]  # Optional: tools are available to the agent
    )
    trajectory = json.loads(agent.get_trajectory())
    return result, trajectory

prompt_template = """
You are a helpful assistant. Use the available tools to help the user.
Task: {task}
"""

gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template=prompt_template,
    evaluation_fn=evaluate_result,
    max_generations=10,
    population_size=8,
    pareto_size=4,
    mutation_rate=0.5  # Optional: probability of mutating a prompt (default: 0.5)
)

# Run optimization with optional rollouts_per_generation parameter
best_candidate = gepa.optimize(
    arguments={"task": "Complete the task"},
    rollouts_per_generation=1  # Optional: number of rollouts per generation (default: 1)
)
print(f"Best prompt: {best_candidate.prompt_template}")
print(f"Scores: {best_candidate.scores}")

# Access the Pareto frontier for multi-objective analysis
pareto_frontier = gepa.get_pareto_frontier()
print(f"Pareto frontier size: {len(pareto_frontier)}")
```

## API Reference

### `GEPA` Class

#### `__init__`

```python
GEPA(
    agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list[dict[str, str]]]],
    initial_prompt_template: str,
    evaluation_fn: Callable | None = None,
    max_generations: int | None = None,
    population_size: int | None = None,
    pareto_size: int | None = None,
    mutation_rate: float | None = None,
    crossover_probability: float | None = None,
    reflection_model: str | None = None,
)
```

**Parameters:**
- `agent_wrapper`: Function that takes a prompt template and arguments, runs the agent, and returns `(result, trajectory)`. Note: `agent.get_trajectory()` returns a JSON string, so you need to parse it with `json.loads()` to get the list.
- `initial_prompt_template`: The initial prompt template to optimize (may contain placeholders like `{task}`)
- `evaluation_fn`: Function to evaluate a rollout result and return scores as `dict[str, float]` (higher is better). If `None`, uses default evaluation based on "success" keyword in result.
- `max_generations`: Maximum number of evolutionary generations (default: from config)
- `population_size`: Number of candidates to maintain in population (default: 8)
- `pareto_size`: Maximum size of Pareto frontier (default: 4)
- `mutation_rate`: Probability of mutating a prompt template in each generation (default: 0.5)
- `crossover_probability`: Probability of combining with lessons from Pareto frontier using crossover (default: 0.3)
- `reflection_model`: Model to use for reflection (default: "gemini-3-flash-preview")

#### `optimize`

```python
optimize(
    arguments: dict[str, str],
    rollouts_per_generation: int | None = None,
) -> PromptCandidate
```

**Parameters:**
- `arguments`: Arguments to pass to the agent wrapper (fills placeholders in prompt template)
- `rollouts_per_generation`: Number of rollouts per generation (default: 1)

**Returns:** `PromptCandidate` with the best prompt template and scores

#### `get_pareto_frontier`

```python
get_pareto_frontier() -> list[PromptCandidate]
```

Returns a copy of the current Pareto frontier (non-dominated candidates).

#### `get_best_prompt`

```python
get_best_prompt() -> str
```

Returns the best prompt template from available candidates.

### `PromptCandidate` Dataclass

```python
@dataclass
class PromptCandidate:
    prompt_template: str
    ancestor_id: int | None = None
    reflection: str | None = None
    scores: dict[str, float] | None = None  # Multi-objective scores
    trajectory: list[dict[str, str]] | None = None
    id: int | None = None
```

## Configuration

Default values for all parameters can be configured via `DEFAULT_CONFIG.gepa` in `src/kiss/agents/gepa/config.py`. See the API Reference above for parameter descriptions and defaults.

## How It Works

1. **Initialization**: GEPA starts with an initial prompt template and creates the first candidate.

2. **Evaluation**: For each generation:
   - Each candidate in the population is evaluated by running the agent wrapper
   - Multiple rollouts can be performed per generation (configurable)
   - The best result from rollouts is kept for each candidate

3. **Reflection**: After evaluation, GEPA uses an LLM to reflect on the agent trajectory and generate an improved prompt template.

4. **Pareto Frontier Update**: Candidates are added to the Pareto frontier if they are not dominated by existing candidates. A candidate dominates another if it is at least as good in all metrics and better in at least one.

5. **Mutation**: New candidates are created by:
   - Using the reflection from the parent candidate
   - Optionally combining with lessons from other candidates in the Pareto frontier (30% chance)
   - Sanitizing to preserve only valid placeholders from the initial template

6. **Iteration**: The process repeats for the specified number of generations.

## Placeholder Handling

GEPA automatically preserves valid placeholders from the initial prompt template. Any placeholders added during reflection that weren't in the original template are removed to ensure compatibility with the agent wrapper.

## Multi-Objective Optimization

GEPA supports multiple evaluation metrics simultaneously. The Pareto frontier maintains candidates that are not dominated by any other candidate, allowing you to explore trade-offs between different objectives.

## Examples

See the main [README.md](../../../../README.md) for more examples and usage patterns.

## Authors

- Koushik Sen (ksen@berkeley.edu)
- Cursor AI (cursor@cursor.com) for vibe coding GEPA

