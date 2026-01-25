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
from kiss.agents.gepa import GEPA
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

# Create optimizer
gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template="You are helpful. Task: {task}",
    evaluation_fn=evaluate,
    max_generations=5,
    population_size=4,
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
    agent_wrapper: Callable[[str, dict[str, str]], tuple[str, list]],
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
    feedback_hints_fn: Callable[[dict, str, dict, list | None, str | None], str] | None = None,
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
- `feedback_hints_fn`: Optional function to generate module-specific feedback hints.
    Signature: `(example, result, score, trajectory, module_context) -> str`
    Useful for composite systems where different modules need different feedback guidance.
    The returned string will be included in the reflection prompt to provide context-specific
    guidance to the reflection LLM.

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

### `PromptCandidate`

```python
@dataclass
class PromptCandidate:
    prompt_template: str
    id: int = 0
    dev_scores: dict[str, float]                  # Scores on dev set
    val_scores: dict[str, float]                  # Scores on val set
    per_item_val_scores: list[dict[str, float]]   # Per-instance val scores
    val_instance_wins: set[int]                   # Val instances this is best on
    evaluated_val_ids: set[int]                   # Val instances evaluated on
    parents: list[int]                            # Parent IDs for ancestry tracking
```

## Key Features

- **Dev/Val Split**: Separates feedback from selection to prevent overfitting
- **Instance-Level Pareto**: Tracks best candidate per validation instance
- **Mutation Gating**: Only accepts mutations that don't degrade
- **Weighted Selection**: Parents selected by number of instance wins
- **Trajectory-Based Reflection**: Uses agent trajectories (tool calls, reasoning steps) to guide prompt improvements
- **Structural 3-Way Merge**: Combines complementary candidates using ancestry tracking and conflict resolution
- **Module-Specific Feedback**: Supports custom feedback hints for composite systems (e.g., multi-agent architectures)

## How It Works

### Phase 1: Reflective Mutation
1. **Split** training examples into dev (feedback) and val (selection) sets
2. **Evaluate** candidates on dev minibatch, collect trajectories
3. **Skip** mutation if candidate achieves perfect score
4. **Reflect** using LLM to propose improved prompt based on trajectories and feedback
5. **Gate**: accept only if not worse than parent on dev
6. **Evaluate** on val set for selection
7. **Update** instance-level Pareto frontier

### Phase 2: Structural Merge (per generation)
8. **Find merge candidates**: Pareto frontier pairs with common ancestor and sufficient validation overlap
9. **Score complementarity**: Prioritize pairs excelling on different instances
10. **3-way merge**: Use ancestry to determine merged prompt (prefer changed prompts, resolve conflicts by score)
11. **Gate on overlap**: Evaluate merged prompt on shared validation instances
12. **Accept if improved**: Add to frontier if merge doesn't degrade (within 5% tolerance)
13. **Repeat** for specified generations

## Module-Specific Feedback for Composite Systems

When optimizing composite systems (e.g., multi-agent architectures with multiple modules), you can provide module-specific feedback hints to guide the reflection process. This is particularly useful when different modules have different responsibilities and require different optimization guidance.

**Example: Module-Specific Feedback Function**

```python
def create_module_feedback_hints(example, result, score, trajectory, module_context):
    """Generate module-specific feedback based on context."""
    if module_context == "context_module":
        # Provide context-specific guidance
        if "no_lookup" not in result.lower():
            return (
                "This module should determine if context lookup is needed. "
                "Review the user query and past context to decide the appropriate lookup action."
            )
    elif module_context == "functional_module":
        # Provide functional module guidance
        return (
            "This module should extract entities and select the appropriate tool. "
            "Ensure tool arguments are correctly formatted and match the expected schema."
        )
    return ""  # No additional hints for other modules

# Use with GEPA
gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template="...",
    evaluation_fn=evaluate,
    feedback_hints_fn=create_module_feedback_hints,
)

# When optimizing a specific module, you can pass module_context
# (Note: module_context is currently passed internally during optimization)
```

This feature is inspired by best practices for optimizing composite systems, as described in [experiences building enterprise agents with GEPA](https://slavozard.bearblog.dev/experiences-from-building-enterprise-agents-with-dspy-and-gepa/).

## Configuration

Default values in `src/kiss/agents/gepa/config.py`.

## Authors

- Koushik Sen (ksen@berkeley.edu)
- Cursor AI (cursor@cursor.com)
