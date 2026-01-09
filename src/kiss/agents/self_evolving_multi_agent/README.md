# Self-Evolving Multi-Agent

An advanced coding agent with planning, error recovery, dynamic tool creation, and the ability to evolve itself for better efficiency and accuracy.

## Overview

The Self-Evolving Multi-Agent is a sophisticated orchestration system that:

- **Plans and tracks tasks** using a todo list with status tracking
- **Delegates to sub-agents** for focused task execution
- **Creates tools dynamically** when it detects repetitive patterns
- **Recovers from errors** automatically with configurable retry logic
- **Runs in Docker isolation** for safe code execution
- **Evolves itself** using KISSEvolve to optimize for efficiency and accuracy

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SelfEvolvingMultiAgent                │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Orchestrator Agent                  │   │
│  │  - Creates and manages todo list                 │   │
│  │  - Delegates tasks to sub-agents                 │   │
│  │  - Creates dynamic tools                         │   │
│  │  - Handles error recovery                        │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│         ┌────────────────┼────────────────┐            │
│         ▼                ▼                ▼            │
│  ┌────────────┐  ┌────────────┐   ┌────────────┐       │
│  │ SubAgent-1 │  │ SubAgent-2 │   │ SubAgent-N │       │
│  │  (Todo 1)  │  │  (Todo 2)  │   │  (Todo N)  │       │
│  └────────────┘  └────────────┘   └────────────┘       │
│                          │                              │
│                          ▼                              │
│              ┌─────────────────────┐                   │
│              │   Docker Container  │                   │
│              │   (python:3.12-slim)│                   │
│              └─────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from kiss.agents.self_evolving_multi_agent import (
    SelfEvolvingMultiAgent,
    run_self_evolving_multi_agent_task,
)

# Option 1: Using the convenience function
result = run_self_evolving_multi_agent_task(
    task="""
    Create a Python script that:
    1. Generates the first 20 Fibonacci numbers
    2. Saves them to a file called 'fibonacci.txt'
    3. Reads the file back and prints the sum
    """,
    model_name="gemini-3-flash-preview",
    max_steps=30,
    max_budget=1.0,
)

print(f"Status: {result['status']}")
print(f"Result: {result['result']}")
print(f"Stats: {result['stats']}")

# Option 2: Using the class directly
agent = SelfEvolvingMultiAgent(
    model_name="gemini-3-flash-preview",
    docker_image="python:3.12-slim",
    max_steps=50,
    max_budget=2.0,
    enable_planning=True,
    enable_error_recovery=True,
    enable_dynamic_tools=True,
)

result = agent.run("Create a calculator module with tests")
print(result)

# Access execution statistics
stats = agent.get_stats()
print(f"Completed todos: {stats['completed']}/{stats['total_todos']}")
print(f"Dynamic tools created: {stats['dynamic_tools_created']}")
```

### Running from Command Line

```bash
# Run the example task
uv run python -m kiss.agents.self_evolving_multi_agent.multi_agent
```

## Available Tools

The orchestrator agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `plan_task` | Create a plan by adding todo items (newline-separated) |
| `execute_todo` | Execute a specific todo item using a sub-agent |
| `run_bash` | Execute a bash command in the Docker container |
| `create_tool` | Create a new reusable tool dynamically |
| `read_file` | Read a file from the workspace |
| `write_file` | Write content to a file |
| `finish` | Complete the task with the final result |

### Dynamic Tool Creation

The agent can create reusable tools at runtime:

```python
# The agent might call:
create_tool(
    name="run_tests",
    description="Run pytest on a specific file",
    bash_command_template="python -m pytest {arg} -v"
)

# Then use it later:
run_tests("test_calculator.py")
```

## Agent Evolution

The `AgentEvolver` uses KISSEvolve to optimize the multi-agent system for:

1. **Fewer LLM calls** - Reduce API costs and latency
2. **Lower budget consumption** - Efficient resource usage
3. **Accurate completion** - Maintain correctness on long-horizon tasks

### Running the Evolver

```python
from kiss.agents.self_evolving_multi_agent import AgentEvolver, EVALUATION_TASKS

# Create evolver
evolver = AgentEvolver(
    model_name="gemini-3-flash-preview",
    population_size=4,
    max_generations=3,
    focus_on_efficiency=True,
)

# Run baseline evaluation first
baseline = evolver.run_baseline_evaluation()
print(f"Baseline fitness: {baseline['fitness']:.4f}")

# Evolve the agent
best = evolver.evolve()
print(f"Evolved fitness: {best.fitness:.4f}")

# Save the best variant
evolver.save_best(best, "evolved_agent.py")
```

### From Command Line

```bash
# Run evolution
uv run python -m kiss.agents.self_evolving_multi_agent.agent_evolver

# Test baseline only (without evolution)
uv run python -m kiss.agents.self_evolving_multi_agent.agent_evolver \
    --self_evolving_multi_agent.evolver_test_only true
```

### Evaluation Tasks

The evolver uses a suite of tasks with varying complexity:

| Task | Complexity | Description |
|------|------------|-------------|
| `fibonacci` | Simple | Generate Fibonacci numbers and save to file |
| `data_pipeline` | Medium | Multi-file data processing pipeline |
| `calculator_project` | Long-horizon | Complete calculator with tests |
| `text_analyzer_suite` | Long-horizon | Text analysis suite with multiple modules |

## Configuration

All settings can be configured via the `SelfEvolvingMultiAgentConfig` class or CLI arguments:

### Agent Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gemini-3-flash-preview` | LLM model for the agent |
| `max_steps` | `50` | Maximum orchestrator steps |
| `max_budget` | `2.0` | Maximum budget in USD |
| `max_retries` | `3` | Maximum retries on error |
| `verbose` | `True` | Enable verbose output |
| `save_trajectories` | `True` | Save agent trajectories |

### Sub-Agent Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sub_agent_max_steps` | `15` | Maximum steps for sub-agents |
| `sub_agent_max_budget` | `0.5` | Maximum budget for sub-agents |

### Docker Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `docker_image` | `python:3.12-slim` | Docker image for execution |
| `workdir` | `/workspace` | Working directory in container |

### Feature Toggles

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_planning` | `True` | Enable planning capabilities |
| `enable_error_recovery` | `True` | Enable error recovery |
| `enable_dynamic_tools` | `True` | Enable dynamic tool creation |
| `max_dynamic_tools` | `5` | Maximum number of dynamic tools |
| `max_plan_items` | `10` | Maximum items in a plan |

### Evolver Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evolver_model` | `gemini-3-flash-preview` | Model for evolution |
| `evolver_population_size` | `4` | Population size |
| `evolver_max_generations` | `3` | Maximum generations |
| `evolver_mutation_rate` | `0.7` | Mutation rate |
| `evolver_elite_size` | `1` | Elite size |
| `evolver_output` | `evolved_agent.py` | Output file for best agent |
| `evolver_test_only` | `False` | Only test without evolution |
| `test_task_timeout` | `300` | Timeout per task in seconds |

### CLI Configuration

```bash
# Override settings via CLI
uv run python -m kiss.agents.self_evolving_multi_agent.multi_agent \
    --self_evolving_multi_agent.model gpt-4o \
    --self_evolving_multi_agent.max_steps 30 \
    --self_evolving_multi_agent.enable_dynamic_tools false
```

## API Reference

### SelfEvolvingMultiAgent

```python
class SelfEvolvingMultiAgent:
    def __init__(
        self,
        model_name: str | None = None,
        docker_image: str | None = None,
        workdir: str | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        enable_planning: bool | None = None,
        enable_error_recovery: bool | None = None,
        enable_dynamic_tools: bool | None = None,
    ): ...

    def run(self, task: str) -> str:
        """Run the agent on a task. Returns the final result."""

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Get the agent's execution trajectory."""

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
```

### run_self_evolving_multi_agent_task

```python
def run_self_evolving_multi_agent_task(
    task: str,
    model_name: str | None = None,
    docker_image: str | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run a task.

    Returns:
        Dictionary with keys: status, result, trajectory, stats
        On failure: status, error, traceback, trajectory, stats
    """
```

### AgentEvolver

```python
class AgentEvolver:
    def __init__(
        self,
        model_name: str | None = None,
        population_size: int | None = None,
        max_generations: int | None = None,
        mutation_rate: float | None = None,
        elite_size: int | None = None,
        tasks: list[EvaluationTask] | None = None,
        focus_on_efficiency: bool = True,
    ): ...

    def evolve(self) -> CodeVariant:
        """Run evolutionary optimization. Returns best variant."""

    def save_best(self, variant: CodeVariant, path: str) -> None:
        """Save the best variant to a file."""

    def run_baseline_evaluation(self) -> dict[str, Any]:
        """Evaluate the base agent to establish baseline."""
```

## How It Works

### Planning Phase

1. The orchestrator receives a task description
2. It uses `plan_task` to break down the task into todo items
3. Each todo item is tracked with status: pending → in_progress → completed/failed

### Execution Phase

1. The orchestrator calls `execute_todo` for each pending item
2. A sub-agent is spawned to handle each todo
3. Sub-agents have access to `run_bash`, `read_file`, and `write_file`
4. Results are captured and status is updated

### Error Recovery

1. When a sub-agent fails, the error is recorded
2. If error recovery is enabled and retries remain, the todo is reset to pending
3. The orchestrator can adjust its approach based on the error message

### Dynamic Tool Creation

1. The orchestrator can detect repetitive patterns
2. It creates reusable tools with `create_tool`
3. New tools are added to its available tool set
4. Maximum of 5 dynamic tools by default

## Files

| File | Description |
|------|-------------|
| `multi_agent.py` | Main `SelfEvolvingMultiAgent` implementation |
| `agent_evolver.py` | `AgentEvolver` for evolving the agent |
| `config.py` | Configuration with Pydantic models |
| `__init__.py` | Package exports |

## License

Apache-2.0
