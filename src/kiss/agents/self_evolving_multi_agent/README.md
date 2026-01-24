# Self-Evolving Multi-Agent

An advanced coding agent with planning, error recovery, dynamic tool creation, and the ability to evolve itself for better efficiency and accuracy.

## Overview

The Self-Evolving Multi-Agent is a sophisticated orchestration system that:

- **Plans and tracks tasks** using a todo list with status tracking
- **Delegates to sub-agents** for focused task execution
- **Creates tools dynamically** when prompted by the orchestrator
- **Recovers from errors** automatically with retry logic
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
from kiss.agents.self_evolving_multi_agent import SelfEvolvingMultiAgent, run_task

# Option 1: Using the class directly
agent = SelfEvolvingMultiAgent()
result = agent.run("""
    Create a Python script that:
    1. Generates the first 20 Fibonacci numbers
    2. Saves them to a file called 'fibonacci.txt'
    3. Reads the file back and prints the sum
""")
print(result)

# Access execution statistics
stats = agent.get_stats()
print(f"Completed todos: {stats['completed']}/{stats['total_todos']}")
print(f"Dynamic tools created: {stats['dynamic_tools']}")

# Option 2: Using run_task (for evolver integration)
result = run_task("""
    Create a calculator module with tests
""")
print(f"Result: {result['result']}")
print(f"Metrics: {result['metrics']}")
print(f"Stats: {result['stats']}")
```

### Running from Command Line

```bash
# Run the example complex task (E-Commerce backend)
uv run python -m kiss.agents.self_evolving_multi_agent.multi_agent
```

## Available Tools

The orchestrator agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `plan_task` | Create a plan by adding todo items (newline-separated) |
| `execute_todo` | Execute a specific todo item using a sub-agent |
| `complete_todo` | Mark a task finished after manual work |
| `run_bash` | Execute a bash command in the Docker container |
| `create_tool` | Create a new reusable tool dynamically |
| `read_file` | Read a file from the workspace |
| `write_file` | Write content to a file |

### Dynamic Tool Creation

The agent can create reusable tools at runtime (explicitly via `create_tool`):

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
from kiss.agents.self_evolving_multi_agent.agent_evolver import AgentEvolver, EVALUATION_TASKS

# Create evolver
evolver = AgentEvolver(
    package_name="kiss.agents.self_evolving_multi_agent",
    agent_file_path="multi_agent.py",
    model_name="gemini-3-flash-preview",
    focus_on_efficiency=True,
)

# Run baseline evaluation first
baseline = evolver.run_baseline_evaluation()
print(f"Baseline fitness: {baseline['fitness']:.4f}")

# Evolve the agent
best = evolver.evolve()
print(f"Evolved fitness: {best.fitness:.4f}")

# Save the best variant
evolver.save_best(best)
```

### From Command Line

```bash
# Run evolution
uv run python -m kiss.agents.self_evolving_multi_agent.agent_evolver
```

### Evaluation Tasks

The evolver uses a suite of tasks with varying complexity:

| Task | Complexity | Description |
|------|------------|-------------|
| `fibonacci` | Simple | Generate Fibonacci numbers and save to file |
| `data_pipeline` | Medium | Multi-file data processing pipeline |
| `calculator_project` | Long-horizon | Complete calculator with tests |
| `text_analyzer_suite` | Long-horizon | Text analysis suite with multiple modules |
| `ecommerce_backend` | Long-horizon | Full E-Commerce backend with FastAPI |
| `blog_platform` | Long-horizon | Complete blog platform with auth |
| `task_scheduler` | Long-horizon | Distributed task scheduler system |
| `ml_pipeline` | Long-horizon | Machine learning pipeline system |

## Configuration

All settings can be configured via the `SelfEvolvingMultiAgentConfig` class or CLI arguments:

### Agent Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gemini-3-flash-preview` | LLM model for the orchestrator |
| `sub_agent_model` | `gemini-3-flash-preview` | LLM model for sub-agents |
| `max_steps` | `100` | Maximum orchestrator steps |
| `max_budget` | `10.0` | Maximum budget in USD |
| `max_retries` | `3` | Maximum retries on error |

### Sub-Agent Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sub_agent_max_steps` | `50` | Maximum steps for sub-agents |
| `sub_agent_max_budget` | `2.0` | Maximum budget for sub-agents in USD |

### Docker Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `docker_image` | `python:3.12-slim` | Docker image for execution |
| `workdir` | `/workspace` | Working directory in container |

### Evolver Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evolver_model` | `gemini-3-flash-preview` | Model for evolution |

Evolution parameters are inherited from `DEFAULT_CONFIG.kiss_evolve`:
- `population_size`, `max_generations`, `mutation_rate`, `elite_size`

### CLI Configuration

```bash
# Override settings via CLI
uv run python -m kiss.agents.self_evolving_multi_agent.multi_agent \
    --self_evolving_multi_agent.model gpt-4o \
    --self_evolving_multi_agent.max_steps 50
```

## API Reference

### SelfEvolvingMultiAgent

```python
class SelfEvolvingMultiAgent:
    def __init__(self) -> None:
        """Initialize agent with settings from DEFAULT_CONFIG.self_evolving_multi_agent."""

    def run(self, task: str) -> str:
        """Run the agent on a task. Returns the final result."""

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics.
        
        Returns dict with: total_todos, completed, failed, error_count, dynamic_tools
        """
```

### run_task

```python
def run_task(task: str) -> dict:
    """Run task and return result with metrics for the evolver.

    Returns:
        Dictionary with keys:
        - result: The task result
        - metrics: {"llm_calls": int, "steps": int}
        - stats: Agent statistics (present on success)
        - error: Error message (if failed)
    """
```

### AgentEvolver

```python
class AgentEvolver:
    def __init__(
        self,
        package_name: str,
        agent_file_path: str,
        model_name: str | None = None,
        tasks: list[EvaluationTask] | None = None,
        focus_on_efficiency: bool = True,
    ): ...

    def evolve(self) -> CodeVariant:
        """Run evolutionary optimization. Returns best variant."""

    def save_best(self, variant: CodeVariant, path: str | None = None) -> None:
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
2. If retries remain, the todo is reset to pending
3. The orchestrator can adjust its approach based on the error message

### Dynamic Tool Creation

1. The orchestrator can choose to create reusable tools with `create_tool`
2. New tools are added to its available tool set

## Files

| File | Description |
|------|-------------|
| `multi_agent.py` | Main `SelfEvolvingMultiAgent` implementation with complex task example |
| `agent_evolver.py` | `AgentEvolver` for evolving the agent with evaluation tasks |
| `config.py` | Configuration with Pydantic models |
| `__init__.py` | Package exports |

## License

Apache-2.0
