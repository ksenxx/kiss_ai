# GEPA: Genetic-Pareto Prompt Evolution

GEPA (Genetic-Pareto) is a prompt optimization framework that uses natural language reflection to evolve prompts for compound AI systems. It maintains an instance-level Pareto frontier of top-performing prompts and combines complementary lessons through structural merge.

**Paper**: [GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPERFORM REINFORCEMENT LEARNING](https://arxiv.org/pdf/2507.19457)

**Official Implementation**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)

## Algorithm

```
Input: train set, AI system (parametrized by ≥1 prompts), and metric
Split train set into dev & val sets
Track a pool of candidates, including the best on each val item (Pareto front)
Repeatedly:
    Select a prompt to try to improve (weighted by instance wins)
    Run system on a minibatch of dev examples, noting intermediate feedback
    Skip mutation if candidate achieves perfect score on minibatch
    Call a LM to propose alternatives for the prompt based on scores and feedback
    Gate mutations - only accept if they don't degrade on minibatch
    Update pool based on how candidates score on val set (instance-level)
```

## Quick Start

```python
from kiss.agents.gepa import GEPA, GEPAProgress
from kiss.core.kiss_agent import KISSAgent
import json

def agent_wrapper(prompt_template: str, arguments: dict[str, str]):
    """Run agent and return (result, trajectory)."""
    agent = KISSAgent(name="My Agent")
    result = agent.run(
        model_name="gpt-4o-mini",
        prompt_template=prompt_template,
        arguments=arguments,
    )
    return result, json.loads(agent.get_trajectory())

def evaluate(result: str) -> dict[str, float]:
    return {"success": 1.0 if "success" in result.lower() else 0.0}

def on_progress(progress: GEPAProgress) -> None:
    """Optional callback to track optimization progress."""
    print(f"Gen {progress.generation}/{progress.max_generations} | "
          f"{progress.phase.value} | Best: {progress.best_val_accuracy}")

# Create optimizer
gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template="You are helpful. Task: {task}",
    evaluation_fn=evaluate,
    max_generations=5,
    population_size=4,
    progress_callback=on_progress,  # Optional progress tracking
)

# Run optimization
best = gepa.optimize(
    train_examples=[
        {"task": "Write a poem"},
        {"task": "Explain physics"},
        {"task": "Create a recipe"},
        {"task": "Describe ML"},
    ],
    dev_minibatch_size=2,
)

print(f"Best prompt: {best.prompt_template}")
print(f"Val scores: {best.val_scores}")
```

## API Reference

### `GEPA.__init__`

```python
GEPA(
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
)
```

**Parameters:**

- `agent_wrapper`: Function `(prompt_template, arguments) -> (result, trajectory)`
- `initial_prompt_template`: Initial prompt to optimize
- `evaluation_fn`: Function `result -> {metric: score}` (higher is better)
- `max_generations`: Evolutionary generations (default from config)
- `population_size`: Candidates per generation (default from config)
- `pareto_size`: Max Pareto frontier size (default from config)
- `mutation_rate`: Mutation probability (default: 0.5)
- `reflection_model`: Model for reflection
- `dev_val_split`: Fraction for dev set (default: 0.5)
- `perfect_score`: Score threshold to skip mutation (default: 1.0)
- `use_merge`: Enable structural merge from Pareto frontier (default: True)
- `max_merge_invocations`: Maximum merge attempts per optimization run (default: 5)
- `merge_val_overlap_floor`: Minimum shared validation instances for merge (default: 2)
- `progress_callback`: Optional callback function called with `GEPAProgress` during optimization
- `batched_agent_wrapper`: Optional function `(prompt_template, examples) -> [(result, trajectory)]`.
  When provided, GEPA evaluates a minibatch through one batched call instead of one call per example.
  This is useful for prompt-merging inference pipelines that combine multiple examples into one API request.

### `GEPA.optimize`

```python
optimize(
    train_examples: list[dict[str, str]],
    dev_minibatch_size: int | None = None,
) -> PromptCandidate
```

**Parameters:**

- `train_examples`: Training examples (split into dev/val)
- `dev_minibatch_size`: Dev examples per evaluation (default: all)

### `GEPAPhase`

Enum representing the current phase of GEPA optimization:

```python
class GEPAPhase(Enum):
    DEV_EVALUATION = "dev_evaluation"    # Evaluating on dev set
    VAL_EVALUATION = "val_evaluation"    # Evaluating on validation set
    REFLECTION = "reflection"            # LLM reflecting to generate mutations
    MUTATION_GATING = "mutation_gating"  # Testing if mutation should be accepted
    MERGE = "merge"                      # Structural merge from Pareto frontier
    PARETO_UPDATE = "pareto_update"      # New candidate added to Pareto frontier
```

### `GEPAProgress`

Progress information passed to the callback during optimization:

```python
@dataclass
class GEPAProgress:
    generation: int              # Current generation number (0-indexed)
    max_generations: int         # Total number of generations
    phase: GEPAPhase             # Current optimization phase
    candidate_id: int | None     # ID of current candidate (if applicable)
    candidate_index: int | None  # Index in population (if applicable)
    population_size: int         # Current population size
    best_val_accuracy: float | None     # Best validation accuracy so far
    current_val_accuracy: float | None  # Current candidate's validation accuracy
    pareto_frontier_size: int    # Size of Pareto frontier
    num_candidates_evaluated: int  # Candidates evaluated this generation
    message: str                 # Description of current activity
```

### `PromptCandidate`

```python
@dataclass
class PromptCandidate:
    prompt_template: str
    dev_scores: dict[str, float] = field(default_factory=dict)
    val_scores: dict[str, float] = field(default_factory=dict)
    per_item_val_scores: list[dict[str, float]] = field(default_factory=list)
    val_instance_wins: set[int] = field(default_factory=set)
    evaluated_val_ids: set[int] = field(default_factory=set)
    parents: list[int] = field(default_factory=list)
    id: int = 0
```

## Key Features

- **Dev/Val Split**: Separates feedback from selection to prevent overfitting
- **Instance-Level Pareto**: Tracks best candidate per validation instance
- **Mutation Gating**: Only accepts mutations that don't degrade
- **Weighted Selection**: Parents selected by number of instance wins
- **Optional Batched Inference**: Supports `batched_agent_wrapper` for prompt-merging and higher throughput
- **Trajectory-Based Reflection**: Uses agent trajectories (tool calls, reasoning steps) to guide prompt improvements
- **Structural 3-Way Merge**: Combines complementary candidates using ancestry tracking and conflict resolution
- **Progress Callbacks**: Real-time visibility into optimization progress for UI integration

## Optional Batched Wrapper

If your inference stack can merge prompts before the API call, pass `batched_agent_wrapper`:

```python
def batched_agent_wrapper(
    prompt_template: str,
    examples: list[dict[str, str]],
) -> list[tuple[str, list]]:
    # Merge all example prompts into one API call, then split outputs back
    # into one (result, trajectory) tuple per example.
    ...

gepa = GEPA(
    agent_wrapper=agent_wrapper,  # fallback for non-batched mode
    batched_agent_wrapper=batched_agent_wrapper,
    initial_prompt_template=initial_prompt,
)
```

## Progress Callback

### Built-in Progress Callback

Use the built-in `create_progress_callback()` for simple console output:

```python
from kiss.agents.gepa import GEPA, create_progress_callback

# Simple usage - prints only val evaluation completions
gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template=initial_prompt,
    progress_callback=create_progress_callback(),
)

# Verbose mode - prints all phases (dev, val, reflection, mutation, merge)
gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template=initial_prompt,
    progress_callback=create_progress_callback(verbose=True),
)
```

Output example:

```
  Gen 1/3 | val_evaluation     | Best:    N/A | Evaluated candidate 0: val_accuracy=0.7500
  Gen 1/3 | pareto_update      | Best: 75.00% | Added candidate 0 to Pareto frontier (wins=2, val_acc=0.7500)
Prompt:
You are a helpful assistant. Task: {task}
  Gen 2/3 | val_evaluation     | Best: 75.00% | Evaluated candidate 1: val_accuracy=0.8000
  Gen 2/3 | pareto_update      | Best: 80.00% | Added candidate 1 to Pareto frontier (wins=3, val_acc=0.8000)
Prompt:
You are a precise and helpful assistant. Task: {task}
```

The `pareto_update` phase is always printed (even when `verbose=False`) to show when new candidates join the Pareto frontier, including the full prompt template.

### Custom Progress Callback with Rich

For more advanced UI integration:

```python
from rich.progress import Progress, SpinnerColumn, TextColumn
from kiss.agents.gepa import GEPA, GEPAProgress

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
) as progress:
    task = progress.add_task("Optimizing...", total=None)

    def on_progress(p: GEPAProgress) -> None:
        best = f"{p.best_val_accuracy:.2%}" if p.best_val_accuracy else "..."
        progress.update(
            task,
            description=f"Gen {p.generation+1}/{p.max_generations} | {p.phase.value} | Best: {best}",
        )

    gepa = GEPA(
        agent_wrapper=agent_wrapper,
        initial_prompt_template=initial_prompt,
        progress_callback=on_progress,
    )
    best = gepa.optimize(train_examples)
```

## How It Works

### Phase 1: Reflective Mutation

1. **Split** training examples into dev (feedback) and val (selection) sets
1. **Evaluate** candidates on dev minibatch, collect trajectories
1. **Skip** mutation if candidate achieves perfect score
1. **Reflect** using LLM to propose improved prompt based on trajectories and feedback
1. **Gate**: accept only if not worse than parent on dev
1. **Evaluate** on val set for selection
1. **Update** instance-level Pareto frontier

### Phase 2: Structural Merge (per generation)

8. **Find merge candidates**: Pareto frontier pairs with common ancestor and sufficient validation overlap
1. **Score complementarity**: Prioritize pairs excelling on different instances
1. **3-way merge**: Use ancestry to determine merged prompt (prefer changed prompts, resolve conflicts by score)
1. **Gate on overlap**: Evaluate merged prompt on shared validation instances
1. **Accept if improved**: Add to frontier if merge doesn't degrade (within 5% tolerance)
1. **Repeat** for specified generations

## Configuration

Default values in `src/kiss/agents/gepa/config.py`.

## Authors

- Koushik Sen (ksen@berkeley.edu)
