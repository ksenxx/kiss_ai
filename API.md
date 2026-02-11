# KISS Framework API Reference

**Keep It Simple, Stupid** - Comprehensive API documentation for the KISS AI agent framework.

## Introduction

The KISS framework provides a clean, simple API for building AI agents with native function calling, multi-agent orchestration, and evolutionary optimization. This document covers all public classes, methods, and utilities available in the framework.

For a high-level overview and quick start guide, see [README.md](README.md).

## Table of Contents

- [KISSAgent](#kissagent) - Core agent class with function calling
- [RelentlessCodingAgent](#relentlesscodingagent) - Single-agent coding system with auto-continuation
- [KISSCodingAgent](#kisscodingagent) - Multi-agent coding system with planning and orchestration
- [ClaudeCodingAgent](#claudecodingagent) - Claude Agent SDK-based coding agent
- [GeminiCliAgent](#geminicliagent) - Google ADK-based coding agent
- [OpenAICodexAgent](#openaicodexagent) - OpenAI Agents SDK-based coding agent
- [DockerManager](#dockermanager) - Docker container management
- [Multiprocessing](#multiprocessing) - Parallel execution utilities
- [SimpleRAG](#simplerag) - Simple RAG system for document retrieval
- [AgentEvolver](#agentevolver) - Evolutionary agent optimization
- [GEPA](#gepa) - Genetic-Pareto prompt optimizer
- [KISSEvolve](#kissevolve) - Evolutionary algorithm discovery
- [UsefulTools](#usefultools) - File operations and bash execution with path-based access control
- [Utility Functions](#utility-functions) - Helper functions
- [Printer](#printer) - Abstract printer base class and MultiPrinter
- [BrowserPrinter](#browserprinter) - Browser SSE streaming output
- [ConsolePrinter](#consoleprinter) - Rich terminal output
- [Pre-built Agents](#pre-built-agents) - Ready-to-use agents
- [Configuration System](#configuration-system) - Config management

______________________________________________________________________

## KISSAgent

The core agent class implementing a ReAct agent using native function calling of LLMs.

### Constructor

```python
KISSAgent(name: str)
```

**Parameters:**

- `name` (str): The name of the agent. Used for identification and artifact naming.

### Methods

#### `run()`

```python
def run(
    self,
    model_name: str,
    prompt_template: str,
    arguments: dict[str, str] | None = None,
    tools: list[Callable[..., Any]] | None = None,
    is_agentic: bool = True,
    max_steps: int | None = None,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    printer: Printer | None = None,
) -> str
```

Runs the agent's main ReAct loop to solve the task.

**Parameters:**

- `model_name` (str): The name of the model to use (e.g., "gpt-4o", "claude-sonnet-4-5", "gemini-2.5-flash", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "openrouter/anthropic/claude-3.5-sonnet")
- `prompt_template` (str): The prompt template for the agent. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `tools` (list\[Callable[..., Any]\] | None): List of callable functions the agent can use. The `finish` tool is automatically added if it is not provided in `tools`. If `use_web` is enabled in config, `search_web` and `fetch_url` is also automatically added. Default is None.
- `is_agentic` (bool): If True, runs in agentic mode with tools. If False, returns raw LLM response. Default is True.
- `max_steps` (int | None): Maximum number of ReAct loop iterations. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`, which is 100).
- `max_budget` (float | None): Maximum budget in USD for this agent run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`, which is 10.0).
- `model_config` (dict[str, Any] | None): Optional model configuration to pass to the model. Default is None.
- `printer` (Printer | None): Optional printer for output formatting and streaming. When `verbose=True` (default) and no printer is provided, a `MultiPrinter` is created automatically based on config flags: `print_to_browser` (default: True) adds a `BrowserPrinter`, `print_to_console` (default: True) adds a `ConsolePrinter`. The printer's `token_callback` is used for real-time token streaming. Default is None.

**Returns:**

- `str`: The result returned by the agent's `finish()` call, or the raw LLM response in non-agentic mode.

**Raises:**

- `KISSError`: If budget exceeded, max steps exceeded, or tools provided in non-agentic mode.

#### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the agent's conversation trajectory as a JSON string.

**Returns:**

- `str`: JSON-formatted string containing the list of messages with roles and content.

#### `finish()`

```python
def finish(self, result: str) -> str
```

Built-in method that serves as the default finish tool for the agent. This method is automatically added as a tool for all KISSAgent instances.

**Parameters:**

- `result` (str): The final result/answer from the agent.

**Returns:**

- `str`: The same result string passed in.

**Note:** This is a simpler version than the utility function `kiss.core.utils.finish()`. If you want structured output with status and analysis, you can provide the utility version as a custom tool instead.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model_name` (str): The name of the model being used.
- `model`: The model instance being used.
- `function_map` (dict[str, Callable]): Dictionary mapping tool names to callable functions.
- `messages` (list\[dict[str, Any]\]): List of messages in the trajectory.
- `step_count` (int): Current step number.
- `total_tokens_used` (int): Total tokens used in this run.
- `budget_used` (float): Budget used in this run.
- `run_start_timestamp` (int): Unix timestamp when the run started.
- `is_agentic` (bool): Whether the agent is running in agentic mode.
- `max_steps` (int): Maximum number of steps allowed.
- `max_budget` (float): Maximum budget allowed for this run.

### Tool Definition

Tools are defined as regular Python functions with type hints and docstrings:

```python
def my_tool(param1: str, param2: int = 10) -> str:
    """Description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
    # Tool implementation
    return f"Result: {param1}, {param2}"
```

The framework automatically extracts the function signature, type hints, and docstring to generate the tool schema for the LLM.

______________________________________________________________________

## RelentlessCodingAgent

A single-agent coding system with smart auto-continuation for long-running tasks. It uses a single agent that executes the task across multiple trials, automatically continuing where it left off via structured JSON progress tracking (done/next items) and work directory scanning.

The agent continues attempting tasks through multiple trials until success or exhaustion of retry attempts.

### Constructor

```python
RelentlessCodingAgent(name: str)
```

**Parameters:**

- `name` (str): Name of the agent. Used for identification and artifact naming.

### Methods

#### `run()`

```python
def run(
    self,
    model_name: str | None = None,
    prompt_template: str = "",
    arguments: dict[str, str] | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    work_dir: str | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    trials: int | None = None,
    docker_image: str | None = None,
    printer: Printer | None = None,
) -> str
```

Run the single-agent coding system with auto-continuation.

**Parameters:**

- `model_name` (str | None): Model for task execution. Default is None (uses config default: "claude-opus-4-6").
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution. Default is "".
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int | None): Maximum number of steps per trial. Default is None (uses config default: 200).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses config default: 200.0).
- `work_dir` (str | None): The working directory for the agent's operations. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no paths are allowed for read access (except work_dir which is always added).
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no paths are allowed for write access (except work_dir which is always added).
- `trials` (int | None): Number of continuation attempts. Default is None (uses config default: 200).
- `docker_image` (str | None): Optional Docker image name to run bash commands in a container. If provided, all bash commands will run inside the Docker container instead of on the host. Example: "ubuntu:latest", "python:3.11-slim". Default is None (local execution).
- `printer` (Printer | None): Optional printer for output formatting. Default is None.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

#### `perform_task()`

```python
def perform_task(self) -> str
```

Execute the main task using multiple trials with auto-continuation. Each trial runs a KISSAgent with an adaptive step limit; on failure, structured progress (done/next items) and a scan of existing files are passed to the next trial.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

**Raises:**

- `KISSError`: If the task fails after all continuation trials.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model_name` (str): Model name for task execution.
- `task_description` (str): The formatted task description.
- `total_tokens_used` (int): Total tokens used across all trials in this run.
- `budget_used` (float): Total budget used across all trials in this run.
- `work_dir` (str): The working directory for the agent's operations.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps per trial.
- `max_budget` (float): Maximum total budget allowed for this run.
- `trials` (int): Number of continuation attempts.
- `max_tokens` (int): Maximum context length for the model used.
- `docker_image` (str | None): The Docker image name if Docker execution is enabled.
- `docker_manager` (DockerManager | None): The active Docker manager instance during execution (None when not using Docker or outside of `run()`).
- `useful_tools` (UsefulTools): The UsefulTools instance used for bash/read/edit/write operations.

### Key Features

- **Single-Agent with Auto-Continuation**: A single agent executes the task across multiple trials, automatically continuing where it left off via structured JSON progress tracking (done/next items)
- **Structured Progress Tracking**: Each trial reports completed and remaining tasks in JSON format, which is deduplicated, validated, and passed to subsequent trials along with a scan of existing files in the work directory
- **Adaptive Step Thresholds**: Step limits per trial scale based on trial number and progress, with conservative early trials and more steps for trials showing good progress
- **Efficiency Rules**: Built-in prompt instructions enforce step minimization, batching, and immediate completion when tests pass
- **Output Truncation**: Long tool outputs are automatically truncated to keep context manageable
- **Relentless Retries**: Continues attempting tasks through multiple continuation trials until success
- **Bash Command Parsing**: Automatically extracts readable/writable paths from commands using `parse_bash_command_paths()`
- **Path Access Control**: Enforces read/write permissions on file system paths before command execution
- **Docker Support**: Optional Docker container execution for bash commands via the `docker_image` parameter. When enabled, all bash commands run inside an isolated Docker container.
- **Built-in Tools**: Each trial agent has access to `finish()`, `Bash` (or Docker bash when `docker_image` is set), `Read`, and `Edit`

### Example

```python
from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

agent = RelentlessCodingAgent("My Relentless Agent")

result = agent.run(
    prompt_template="""
        Write, test, and optimize a fibonacci function in Python
        that is efficient and correct. Save it to fibonacci.py.
    """,
    model_name="claude-sonnet-4-5",
    readable_paths=["src/"],
    writable_paths=["output/"],
    base_dir="workdir",
    max_steps=50,
    trials=3
)
print(f"Result: {result}")

# Result is YAML with 'success' and 'summary' keys
import yaml
result_dict = yaml.safe_load(result)
print(f"Success: {result_dict['success']}")
print(f"Summary: {result_dict['summary']}")
```

### Example with Docker

```python
from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

agent = RelentlessCodingAgent("Docker Relentless Agent")

# Run with Docker - bash commands execute inside the container
result = agent.run(
    prompt_template="""
        Install numpy and write a Python script that creates 
        a random matrix and computes its eigenvalues.
    """,
    docker_image="python:3.11-slim",  # Commands run in Docker
    max_steps=50,
    trials=2
)
print(f"Result: {result}")
```

______________________________________________________________________

## KISSCodingAgent

A multi-agent coding system with orchestration and sub-agents using KISSAgent. It efficiently breaks down complex coding tasks into manageable sub-tasks through a multi-agent architecture with:

- **Orchestrator Agent**: Manages overall task execution and delegates to sub-tasks
- **Executor Agents**: Handle specific sub-tasks independently
- **Prompt Refinement**: Automatically refines prompts on failures using trajectory analysis for improved retry attempts

The system supports recursive sub-task delegation where any agent can call `perform_subtask()` to further decompose work.

### Constructor

```python
KISSCodingAgent(name: str)
```

**Parameters:**

- `name` (str): Name of the agent. Used for identification and artifact naming.

### SubTask Class

```python
class SubTask:
    def __init__(self, name: str, description: str) -> None
```

Represents a sub-task in the multi-agent coding system.

**Attributes:**

- `id` (int): Unique identifier for this sub-task (auto-incremented)
- `name` (str): Name of the sub-task
- `description` (str): Detailed description of what needs to be done

### Methods

#### `run()`

```python
def run(
    self,
    prompt_template: str,
    arguments: dict[str, str] | None = None,
    orchestrator_model_name: str | None = None,
    subtasker_model_name: str | None = None,
    refiner_model_name: str | None = None,
    trials: int | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    work_dir: str | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    docker_image: str | None = None,
    printer: Printer | None = None,
) -> str
```

Run the multi-agent coding system with orchestration and sub-task delegation.

**Parameters:**

- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `orchestrator_model_name` (str | None): Model for the main orchestrator agent. Default is None (uses config default: "claude-opus-4-6").
- `subtasker_model_name` (str | None): Model for executor agents handling sub-tasks. Default is None (uses config default: "claude-opus-4-6").
- `refiner_model_name` (str | None): Model for dynamic prompt refinement when tasks fail. Default is None (uses config default: "claude-sonnet-4-5").
- `trials` (int | None): Number of retry attempts for each task/subtask. Default is None (uses config default: 200).
- `max_steps` (int | None): Maximum number of steps per agent. Default is None (uses `DEFAULT_CONFIG.coding_agent.kiss_coding_agent.max_steps`, which is 200).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses `DEFAULT_CONFIG.coding_agent.kiss_coding_agent.max_budget`, which is 100.0).
- `work_dir` (str | None): The working directory for the agent's operations. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no paths are allowed for read access (except work_dir which is always added).
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no paths are allowed for write access (except work_dir which is always added).
- `docker_image` (str | None): Optional Docker image name to run bash commands in a container. If provided, all bash commands executed by sub-agents will run inside the Docker container instead of on the host. Example: "ubuntu:latest", "python:3.11-slim". Default is None (local execution).
- `printer` (Printer | None): Optional printer for output formatting. Default is None.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

#### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the agent's conversation trajectory as a JSON string.

**Returns:**

- `str`: JSON-formatted string containing the list of messages.

#### `perform_task()`

```python
def perform_task(self) -> str
```

Execute the main task using the orchestrator agent. The orchestrator can delegate work by calling `perform_subtask()` as needed.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

#### `perform_subtask()`

```python
def perform_subtask(
    self,
    subtask_name: str,
    description: str,
) -> str
```

Execute a sub-task using a dedicated executor agent (using `subtasker_model_name`). Can be called by the orchestrator.

**Parameters:**

- `subtask_name` (str): Name of the sub-task for identification.
- `description` (str): Detailed description of what needs to be done.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `orchestrator_model_name` (str): Model name for orchestrator agent.
- `subtasker_model_name` (str): Model name for executor agents handling sub-tasks.
- `refiner_model_name` (str): Model name for dynamic prompt refinement.
- `task_description` (str): The formatted task description.
- `total_tokens_used` (int): Total tokens used across all agents in this run.
- `budget_used` (float): Total budget used across all agents in this run.
- `work_dir` (str): The working directory for the agent's operations.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps per agent.
- `max_budget` (float): Maximum total budget allowed for this run.
- `trials` (int): Number of retry attempts for each task/subtask.
- `max_tokens` (int): Maximum context length across all models used.
- `docker_image` (str | None): The Docker image name if Docker execution is enabled.
- `docker_manager` (DockerManager | None): The active Docker manager instance during execution (None when not using Docker or outside of `run()`).
- `useful_tools` (UsefulTools): The UsefulTools instance used for bash/edit operations.

### Key Features

- **Multi-Agent Architecture**: Orchestrator (using `orchestrator_model_name`) delegates to executor agents (using `subtasker_model_name`);
- **Prompt Refinement**:
  - Automatically refines prompts when tasks fail using trajectory analysis
  - Uses refiner_model_name (default: claude-sonnet-4-5) for non-agentic prompt improvement
  - Analyzes original prompt, previous prompt, and agent trajectory to generate refined prompts
  - Retries tasks with refined prompts up to `trials` times
- **Efficient Orchestration**: Manages execution to stay within configured step limits through smart delegation
- **Bash Command Parsing**: Automatically extracts readable/writable paths from commands using `parse_bash_command_paths()`
- **Path Access Control**: Enforces read/write permissions on file system paths before command execution
- **Docker Support**: Optional Docker container execution for bash commands via the `docker_image` parameter. When enabled, all bash commands from sub-agents run inside an isolated Docker container.
- **Built-in Tools**:
  - Orchestrator agent has access to `finish()` and `perform_subtask()`
  - Executor agents have access to `finish()` and `Bash` (or Docker bash when `docker_image` is set), `Edit`, and `MultiEdit`

### Example

```python
from kiss.agents.coding_agents import KISSCodingAgent

agent = KISSCodingAgent("My Coding Agent")

result = agent.run(
    prompt_template="""
        Write, test, and optimize a fibonacci function in Python
        that is efficient and correct. Save it to fibonacci.py.
    """,
    orchestrator_model_name="claude-sonnet-4-5",
    refiner_model_name="claude-sonnet-4-5",
    readable_paths=["src/"],
    writable_paths=["output/"],
    base_dir="workdir",
    max_steps=50,
    trials=3
)
print(f"Result: {result}")

# Result is YAML with 'success' and 'summary' keys
import yaml
result_dict = yaml.safe_load(result)
print(f"Success: {result_dict['success']}")
print(f"Summary: {result_dict['summary']}")
```

### Example with Docker

```python
from kiss.agents.coding_agents import KISSCodingAgent

agent = KISSCodingAgent("Docker Coding Agent")

# Run with Docker - bash commands execute inside the container
result = agent.run(
    prompt_template="""
        Install numpy and write a Python script that creates 
        a random matrix and computes its eigenvalues.
    """,
    docker_image="python:3.11-slim",  # Commands run in Docker
    max_steps=50,
    trials=2
)
print(f"Result: {result}")
```

______________________________________________________________________

## ClaudeCodingAgent

> **Requires:** `claude-agent-sdk` and `anthropic` packages. Install with `uv sync --group claude-coding-agent`. This class is `None` if the SDK is not installed.

A coding agent that uses the Claude Agent SDK to generate tested Python programs with file system access controls.

### Constructor

```python
ClaudeCodingAgent(name: str)
```

**Parameters:**

- `name` (str): Name of the agent. Used for identification and artifact naming.

### Methods

#### `run()`

```python
def run(
    self,
    model_name: str | None = None,
    prompt_template: str = "",
    arguments: dict[str, str] | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    work_dir: str | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    use_browser: bool = True,
    max_thinking_tokens: int = 1024,
) -> str
```

Run the Claude coding agent for a given task.

**Parameters:**

- `model_name` (str | None): The name of the model to use (e.g., "claude-sonnet-4-5"). Default is None (uses "claude-sonnet-4-5").
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution. Default is "".
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int | None): Maximum number of steps. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`).
- `work_dir` (str | None): The working directory for the agent's operations. Default is None (uses `{artifact_dir}/claude_workdir`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None.
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no paths are allowed for Read/Grep/Glob.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no paths are allowed for Write/Edit/MultiEdit.
- `use_browser` (bool): If True, starts a local uvicorn server and opens a browser window for real-time streaming output. If False, uses `ConsolePrinter` for terminal output. Default is True.
- `max_thinking_tokens` (int): Maximum tokens for Claude's extended thinking. Default is 1024.

**Returns:**

- `str`: The result of the task, or an empty string if no result.

#### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the agent's conversation trajectory as a JSON string.

**Returns:**

- `str`: JSON-formatted string containing the list of messages.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model_name` (str): The name of the model being used.
- `function_map` (list[str]): List of built-in tools available (Read, Write, Edit, etc.).
- `messages` (list\[dict[str, Any]\]): List of messages in the trajectory.
- `step_count` (int): Current step number.
- `total_tokens_used` (int): Total tokens used in this run.
- `budget_used` (float): Budget used in this run.
- `run_start_timestamp` (int): Unix timestamp when the run started.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps allowed.
- `max_budget` (float): Maximum budget allowed for this run.

### Key Features

- **Real-time Streaming**: Uses `include_partial_messages=True` to receive `StreamEvent` messages, enabling live streaming of assistant text, thinking content, and tool call inputs as they are generated.
- **Extended Thinking**: Supports Claude's extended thinking via `max_thinking_tokens` (default: 1024), allowing the model to reason through complex problems before responding. `ThinkingBlock` content is captured in the trajectory.
- **Rich Console Output**: Uses a dedicated `ConsolePrinter` class (`print_to_console.py`) for formatted terminal output with Rich panels for tool calls (syntax-highlighted file content, diffs, commands), thinking block delimiters, streaming text deltas, and result summaries with cost/token stats.
- **Browser Streaming Output**: When `use_browser=True`, uses `BrowserPrinter` class (`print_to_browser.py`) which starts a local uvicorn/starlette SSE server and opens a browser window for real-time streaming with a modern dark-themed UI featuring scrollable panels for thinking, text, tool calls with syntax highlighting, tool results, and a final result card with stats.
- **Message Processing**: Handles five message types from the Claude Agent SDK query stream:
  - `StreamEvent`: Partial streaming deltas (text, thinking, tool input JSON) — printed in real-time
  - `SystemMessage`: System-level messages (e.g., tool output) — printed to console
  - `AssistantMessage`: Complete assistant turns with `TextBlock`, `ThinkingBlock`, and `ToolUseBlock` content — recorded in trajectory
  - `UserMessage`: Tool result blocks — printed to console
  - `ResultMessage`: Final result with cost tracking — printed with stats panel

### Available Built-in Tools

The agent has access to these built-in tools from Claude Agent SDK:

- `Read`: Read files from the working directory
- `Write`: Create or overwrite files
- `Edit`: Make precise string-based edits to files
- `MultiEdit`: Make multiple precise string-based edits to files
- `Glob`: Find files by glob pattern
- `Grep`: Search file contents with regex
- `Bash`: Run shell commands
- `WebSearch`: Search the web for information
- `WebFetch`: Fetch and process content from a URL

### Example

```python
from kiss.agents.coding_agents import ClaudeCodingAgent

agent = ClaudeCodingAgent("My Agent")
result = agent.run(
    model_name="claude-sonnet-4-5",
    prompt_template="Write a fibonacci function with tests"
)
if result:
    print(f"Result: {result}")
```

### Example with Browser Output

```python
from kiss.agents.coding_agents import ClaudeCodingAgent

# Browser window opens automatically with live streaming output
agent = ClaudeCodingAgent("My Agent")
result = agent.run(
    model_name="claude-sonnet-4-5",
    prompt_template="Write a fibonacci function with tests",
    use_browser=True,
)
if result:
    print(f"Result: {result}")
```

______________________________________________________________________

## Printer

Abstract base class for KISS agent output and the `MultiPrinter` composite.

**Location:** `kiss.core.printer`

### Printer (ABC)

```python
class Printer(ABC):
    @abstractmethod
    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str: ...

    @abstractmethod
    async def token_callback(self, token: str) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...
```

All concrete printers (`ConsolePrinter`, `BrowserPrinter`) implement this interface.

### MultiPrinter

```python
class MultiPrinter(Printer):
    def __init__(self, printers: list[Printer]) -> None: ...
```

A composite printer that delegates to multiple printers. Used by `KISSAgent` to stream output to both the terminal and browser simultaneously.

______________________________________________________________________

## BrowserPrinter

> **Requires:** `uvicorn` and `starlette` packages. Included in `uv sync --group claude-coding-agent`.

**Location:** `kiss.core.print_to_browser`

Browser output streaming handler. Starts a local uvicorn server with SSE (Server-Sent Events) and opens a browser window to display agent output in real-time with a modern dark-themed UI.

### Constructor

```python
BrowserPrinter()
```

### Methods

#### `start()`

```python
def start(self, open_browser: bool = True) -> None
```

Start the uvicorn server on a random free port and optionally open the browser.

**Parameters:**

- `open_browser` (bool): If True (default), opens the browser automatically.

#### `stop()`

```python
def stop(self) -> None
```

Send a "done" event to all connected browsers and shut down the server.

#### `reset()`

```python
def reset(self) -> None
```

Reset internal state (block type, tool name, JSON buffer).

#### `print()`

```python
def print(self, content: Any, type: str = "text", **kwargs: Any) -> str
```

Unified output method that handles all content types and sends to browser via SSE.

**Parameters:**

- `content` (Any): The content to display.
- `type` (str): The content type. Supported types:
  - `"text"`: General text output
  - `"prompt"`: Agent prompt display
  - `"stream_event"`: Streaming event from Claude Agent SDK (handles `content_block_start`, `content_block_delta`, `content_block_stop`)
  - `"message"`: Complete message (SystemMessage, ResultMessage, UserMessage)
  - `"usage_info"`: Usage information (step count, token counts, etc.)
  - `"tool_call"`: Tool call display (pass `tool_input` dict in kwargs)
  - `"tool_result"`: Tool result display (pass `is_error` bool in kwargs)
  - `"result"`: Final result display (pass `step_count`, `total_tokens`, `cost` in kwargs)
- `**kwargs`: Additional keyword arguments specific to the content type.

**Returns:**

- `str`: Extracted text content (for `stream_event` type), empty string otherwise.

### Browser UI Features

- **Dark theme** with modern, professional design
- **Scrollable panels** for thinking blocks, tool call details, tool results, and final results
- **Syntax highlighting** via Highlight.js for code in tool calls
- **Collapsible thinking blocks** that auto-collapse after completion
- **Expandable tool cards** with file paths, commands, diffs, and extra parameters
- **Auto-scroll** with smart detection (pauses when user scrolls up)
- **Live status indicator** with event count and elapsed time
- **SSE-based streaming** for instant updates without polling

______________________________________________________________________

## ConsolePrinter

**Location:** `kiss.core.print_to_console`

Rich terminal output handler. Provides formatted console output with syntax-highlighted tool calls, thinking block delimiters, streaming text deltas, and result summaries.

### Constructor

```python
ConsolePrinter(file: Any = None)
```

**Parameters:**

- `file` (Any): Optional file object for output. Default is None (uses `sys.stdout`).

### Methods

#### `reset()`

```python
def reset(self) -> None
```

Reset internal state (block type, tool name, JSON buffer, mid-line flag).

#### `print()`

```python
def print(self, content: Any, type: str = "text", **kwargs: Any) -> str
```

Unified output method that handles all content types and prints to terminal with Rich formatting.

**Parameters:**

- `content` (Any): The content to display.
- `type` (str): The content type. Supported types:
  - `"text"`: General text output
  - `"prompt"`: Agent prompt display (Rich Panel)
  - `"stream_event"`: Streaming event from Claude Agent SDK (handles `content_block_start`, `content_block_delta`, `content_block_stop`)
  - `"message"`: Complete message (SystemMessage, ResultMessage, UserMessage)
  - `"usage_info"`: Usage information as a Rich Panel
  - `"tool_call"`: Tool call display with syntax highlighting (pass `tool_input` dict in kwargs)
  - `"tool_result"`: Tool result display (pass `is_error` bool in kwargs)
  - `"result"`: Final result display as a Rich Panel with stats (pass `step_count`, `total_tokens`, `cost` in kwargs)
- `**kwargs`: Additional keyword arguments specific to the content type.

**Returns:**

- `str`: Extracted text content (for `stream_event` type), empty string otherwise.

______________________________________________________________________

## GeminiCliAgent

> **Requires:** `google-adk` and `google-genai` packages. Install with `uv sync --group gemini`. This class is `None` if the SDK is not installed.

A coding agent that uses the Google ADK (Agent Development Kit) to generate tested Python programs with file system access controls.

### Constructor

```python
GeminiCliAgent(name: str)
```

**Parameters:**

- `name` (str): Name of the agent. Used for identification and artifact naming.

### Methods

#### `run()`

```python
def run(
    self,
    model_name: str = DEFAULT_GEMINI_MODEL,
    prompt_template: str = "",
    arguments: dict[str, str] | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    printer: Printer | None = None,
) -> str | None
```

Run the Gemini CLI agent for a given task.

**Parameters:**

- `model_name` (str): The name of the model to use. Default is "gemini-2.5-flash".
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int | None): Maximum number of steps. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/gemini_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, only base_dir is readable.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, only base_dir is writable.
- `printer` (Printer | None): Optional printer for output formatting. Default is None.

**Returns:**

- `str | None`: The result of the task, or None if no result.

#### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the agent's conversation trajectory as a JSON string.

**Returns:**

- `str`: JSON-formatted string containing the list of messages.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model_name` (str): The name of the model being used.
- `function_map` (list[str]): List of built-in tools available (read_file, write_file, etc.).
- `messages` (list\[dict[str, Any]\]): List of messages in the trajectory.
- `step_count` (int): Current step number.
- `total_tokens_used` (int): Total tokens used in this run.
- `budget_used` (float): Budget used in this run.
- `run_start_timestamp` (int): Unix timestamp when the run started.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps allowed.
- `max_budget` (float): Maximum budget allowed for this run.

### Available Built-in Tools

The agent has access to these built-in tools:

- `read_file`: Read file contents from the working directory
- `write_file`: Create or overwrite files
- `list_dir`: List directory contents
- `run_shell`: Execute shell commands with timeout
- `web_search`: Search the web for information (placeholder)

### Example

```python
from kiss.agents.coding_agents import GeminiCliAgent

agent = GeminiCliAgent("my_agent")
result = agent.run(
    model_name="gemini-2.5-flash",
    prompt_template="Write a fibonacci function with tests"
)
if result:
    print(f"Result: {result}")
```

______________________________________________________________________

## OpenAICodexAgent

> **Requires:** `openai-agents` package. Install with `uv sync --group openai`. This class is `None` if the SDK is not installed.

A coding agent that uses the OpenAI Agents SDK to generate tested Python programs with file system access controls.

### Constructor

```python
OpenAICodexAgent(name: str)
```

**Parameters:**

- `name` (str): Name of the agent. Used for identification and artifact naming.

### Methods

#### `run()`

```python
def run(
    self,
    model_name: str = DEFAULT_CODEX_MODEL,
    prompt_template: str = "",
    arguments: dict[str, str] | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    printer: Printer | None = None,
) -> str | None
```

Run the OpenAI Codex agent for a given task.

**Parameters:**

- `model_name` (str): The name of the model to use. Default is "gpt-5.3-codex".
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int | None): Maximum number of steps. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/codex_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, only base_dir is readable.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, only base_dir is writable.
- `printer` (Printer | None): Optional printer for output formatting. Default is None.

**Returns:**

- `str | None`: The result of the task, or None if no result.

#### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the agent's conversation trajectory as a JSON string.

**Returns:**

- `str`: JSON-formatted string containing the list of messages.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model_name` (str): The name of the model being used.
- `function_map` (list[str]): List of built-in tools available (read_file, write_file, etc.).
- `messages` (list\[dict[str, Any]\]): List of messages in the trajectory.
- `step_count` (int): Current step number.
- `total_tokens_used` (int): Total tokens used in this run.
- `budget_used` (float): Budget used in this run.
- `run_start_timestamp` (int): Unix timestamp when the run started.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps allowed.
- `max_budget` (float): Maximum budget allowed for this run.

### Available Built-in Tools

The agent has access to these built-in tools:

- `read_file`: Read file contents from the working directory
- `write_file`: Create or overwrite files
- `list_dir`: List directory contents
- `run_shell`: Execute shell commands with timeout
- `web_search`: Search the web for information (via OpenAI SDK WebSearchTool)

### Example

```python
from kiss.agents.coding_agents import OpenAICodexAgent

agent = OpenAICodexAgent("My Agent")
result = agent.run(
    model_name="gpt-5.3-codex",
    prompt_template="Write a fibonacci function with tests"
)
if result:
    print(f"Result: {result}")
```

______________________________________________________________________

## DockerManager

Manages Docker container lifecycle and command execution.

### Constructor

```python
DockerManager(
    image_name: str,
    tag: str = "latest",
    workdir: str = "/",
    mount_shared_volume: bool = True,
    ports: dict[int, int] | None = None,
)
```

**Parameters:**

- `image_name` (str): The name of the Docker image (e.g., 'ubuntu', 'python'). Can include tag like 'ubuntu:22.04'.
- `tag` (str): The tag/version of the image. Default is 'latest'.
- `workdir` (str): The working directory inside the container. Default is '/'.
- `mount_shared_volume` (bool): Whether to mount a shared volume. Set to False for images that already have content in the workdir (e.g., SWE-bench). Default is True.
- `ports` (dict[int, int] | None): Port mapping from container port to host port. Example: `{8080: 8080}` maps container port 8080 to host port 8080. Default is None.

### Methods

#### `open()`

```python
def open(self) -> None
```

Pull and load a Docker image, then create and start a container.

#### `run_bash_command()`

```python
def run_bash_command(self, command: str, description: str) -> str
```

Execute a bash command in the running Docker container.

**Parameters:**

- `command` (str): The bash command to execute.
- `description` (str): A short description of the command in natural language.

**Returns:**

- `str`: The output of the command, including stdout, stderr, and exit code.

**Raises:**

- `KISSError`: If no container is open.

#### `get_host_port()`

```python
def get_host_port(self, container_port: int) -> int | None
```

Get the host port mapped to a container port.

**Parameters:**

- `container_port` (int): The container port to look up.

**Returns:**

- `int | None`: The host port mapped to the container port, or None if not mapped.

**Raises:**

- `KISSError`: If no container is open.

#### `close()`

```python
def close(self) -> None
```

Stop and remove the Docker container. Also cleans up temporary directories.

### Context Manager

DockerManager supports the context manager protocol:

```python
with DockerManager("ubuntu:latest", ports={80: 8080}) as env:
    output = env.run_bash_command("echo 'Hello'", "Echo test")
    host_port = env.get_host_port(80)
```

### Instance Attributes

- `container`: The Docker container instance (after `open()`).
- `host_shared_path` (str | None): Path to the host-side shared directory (auto-generated temp directory).
- `client_shared_path` (str): Path to the container-side shared directory (from config, default: `/testbed`).
- `image` (str): The Docker image name.
- `tag` (str): The Docker image tag.
- `workdir` (str): The working directory inside the container.
- `ports` (dict[int, int] | None): The port mappings.

______________________________________________________________________

## Multiprocessing

Parallel execution utilities using Python's `multiprocessing` module.

### `run_functions_in_parallel()`

```python
def run_functions_in_parallel(
    tasks: list[tuple[Callable[..., Any], list[Any]]],
) -> list[Any]
```

Run a list of functions in parallel using multiprocessing.

**Parameters:**

- `tasks` (list\[tuple[Callable, list]\]): List of tuples, where each tuple contains (function, arguments). Each function is a callable, and arguments is a list that can be unpacked with \*args.

**Returns:**

- `list[Any]`: List of results from each function, in the same order as the input tasks.

**Raises:**

- `Exception`: Any exception raised by the functions will be propagated with context.

**Example:**

```python
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

tasks = [(add, [1, 2]), (multiply, [3, 4])]
results = run_functions_in_parallel(tasks)
print(results)  # [3, 12]
```

### `run_functions_in_parallel_with_kwargs()`

```python
def run_functions_in_parallel_with_kwargs(
    functions: list[Callable[..., Any]],
    args_list: list[list[Any]] | None = None,
    kwargs_list: list[dict[str, Any]] | None = None,
) -> list[Any]
```

Run a list of functions in parallel with support for keyword arguments.

**Parameters:**

- `functions` (list[Callable]): List of callable functions to execute.
- `args_list` (list[list] | None): Optional list of argument lists for positional arguments. If None, an empty list is used for each function.
- `kwargs_list` (list[dict] | None): Optional list of keyword argument dictionaries. If None, an empty dict is used for each function.

**Returns:**

- `list[Any]`: List of results from each function, in the same order as the input functions.

**Raises:**

- `ValueError`: If the number of functions doesn't match the number of argument lists.
- `Exception`: Any exception raised by the functions will be propagated.

**Example:**

```python
def greet(name, title="Mr."):
    return f"Hello, {title} {name}!"

functions = [greet, greet]
args_list = [["Alice"], ["Bob"]]
kwargs_list = [{"title": "Dr."}, {}]
results = run_functions_in_parallel_with_kwargs(functions, args_list, kwargs_list)
print(results)  # ["Hello, Dr. Alice!", "Hello, Mr. Bob!"]
```

### `get_available_cores()`

```python
def get_available_cores() -> int
```

Get the number of available CPU cores.

**Returns:**

- `int`: Number of CPU cores available on the system.

______________________________________________________________________

## SimpleRAG

Simple and elegant RAG (Retrieval-Augmented Generation) system for document storage and retrieval using an in-memory vector store.

### Constructor

```python
SimpleRAG(
    model_name: str,
    metric: str = "cosine",
    embedding_model_name: str | None = None,
)
```

**Parameters:**

- `model_name` (str): Model name to use for the LLM provider.
- `metric` (str): Distance metric to use - "cosine" or "l2". Default is "cosine".
- `embedding_model_name` (str | None): Optional specific model name for embeddings. If None, uses model_name or provider default.

### Methods

#### `add_documents()`

```python
def add_documents(
    self,
    documents: list[dict[str, Any]],
    batch_size: int = 100
) -> None
```

Add documents to the vector store.

**Parameters:**

- `documents` (list[dict]): List of document dictionaries. Each document should have:
  - `"id"` (str): Unique identifier
  - `"text"` (str): Document text content
  - `"metadata"` (dict, optional): Optional metadata dictionary
- `batch_size` (int): Number of documents to process in each batch. Default is 100.

#### `query()`

```python
def query(
    self,
    query_text: str,
    top_k: int = 5,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]
```

Query similar documents from the collection.

**Parameters:**

- `query_text` (str): Query text to search for.
- `top_k` (int): Number of top results to return. Default is 5.
- `filter_fn` (Callable | None): Optional filter function that takes a document dict and returns bool.

**Returns:**

- `list[dict]`: List of dictionaries containing:
  - `"id"`: Document ID
  - `"text"`: Document text
  - `"metadata"`: Document metadata
  - `"score"`: Similarity score (higher is better for cosine, lower for L2)

#### `delete_documents()`

```python
def delete_documents(self, document_ids: list[str]) -> None
```

Delete documents from the collection by their IDs.

**Parameters:**

- `document_ids` (list[str]): List of document IDs to delete.

#### `get_document()`

```python
def get_document(self, document_id: str) -> dict[str, Any] | None
```

Get a document by its ID.

**Parameters:**

- `document_id` (str): Document ID to retrieve.

**Returns:**

- `dict | None`: Document dictionary or None if not found.

#### `get_collection_stats()`

```python
def get_collection_stats(self) -> dict[str, Any]
```

Get statistics about the collection.

**Returns:**

- `dict`: Dictionary containing:
  - `"num_documents"`: Number of documents
  - `"embedding_dimension"`: Dimension of embeddings
  - `"metric"`: Distance metric used

#### `clear_collection()`

```python
def clear_collection(self) -> None
```

Clear all documents from the collection.

### Example

```python
from kiss.rag import SimpleRAG

rag = SimpleRAG(model_name="gpt-4o")

documents = [
    {"id": "1", "text": "Python is a programming language", "metadata": {"topic": "programming"}},
    {"id": "2", "text": "Machine learning uses algorithms", "metadata": {"topic": "ML"}},
]
rag.add_documents(documents)

results = rag.query("What is Python?", top_k=2)
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']}")
```

______________________________________________________________________

## AgentEvolver

AgentEvolver evolves AI agents using a Pareto frontier approach, optimizing for both token efficiency and execution time. It uses mutation and crossover operations to create new agent variants.

### Constructor

```python
AgentEvolver()
```

AgentEvolver is instantiated without parameters. All configuration is passed to the `evolve()` method.

**Note:** AgentEvolver uses RelentlessCodingAgent (a single-agent coding system with auto-continuation) internally for agent improvement. Evaluation is done internally by loading and running the generated `agent.py` which must implement `agent_run(task: str) -> dict[str, Any]`.

### Methods

#### `evolve()`

```python
def evolve(
    self,
    task_description: str,
    max_generations: int | None = None,
    initial_frontier_size: int | None = None,
    max_frontier_size: int | None = None,
    mutation_probability: float | None = None,
    progress_callback: Callable[[EvolverProgress], None] | None = None,
) -> AgentVariant
```

Run the evolutionary optimization process.

**Parameters:**

- `task_description` (str): Description of the task the agent should perform.
- `max_generations` (int | None): Maximum number of improvement generations. Uses config default if None.
- `initial_frontier_size` (int | None): Number of initial agents to create. Uses config default if None.
- `max_frontier_size` (int | None): Maximum size of the Pareto frontier. Uses config default if None.
- `mutation_probability` (float | None): Probability of mutation vs crossover (1.0 = always mutate). Uses config default if None.
- `progress_callback` (Callable\[[EvolverProgress], None\] | None): Optional callback function called with `EvolverProgress` during optimization. Use this to track progress, display progress bars, or log intermediate results.

**Returns:**

- `AgentVariant`: The best agent variant found during evolution.

#### `get_best_variant()`

```python
def get_best_variant(self) -> AgentVariant
```

Get the best variant from the Pareto frontier by combined score.

**Returns:**

- `AgentVariant`: The best agent variant (lowest combined score of tokens + time).

**Raises:**

- `RuntimeError`: If no variants are available.

#### `get_pareto_frontier()`

```python
def get_pareto_frontier(self) -> list[AgentVariant]
```

Get all variants in the Pareto frontier.

**Returns:**

- `list[AgentVariant]`: Copy of the current Pareto frontier.

#### `save_state()`

```python
def save_state(self, path: str) -> None
```

Save the evolver state to a JSON file.

**Parameters:**

- `path` (str): Path where to save the state JSON file.

### AgentVariant

```python
@dataclass
class AgentVariant:
    folder_path: str
    report_path: str
    report: ImprovementReport
    metrics: dict[str, float]
    parent_ids: list[int]
    id: int = 0
    generation: int = 0
```

**Attributes:**

- `folder_path` (str): Directory containing agent code.
- `report_path` (str): Path to improvement report JSON file.
- `report` (ImprovementReport): Improvement report tracking implemented/failed ideas.
- `metrics` (dict[str, float]): Metrics dictionary with keys like `success`, `tokens_used`, `execution_time`.
- `parent_ids` (list[int]): List of parent variant IDs (for lineage tracking).
- `id` (int): Unique identifier for this variant.
- `generation` (int): Generation number when this variant was created.

### ImprovementReport

```python
class ImprovementReport:
    def __init__(
        self,
        metrics: dict[str, float],
        implemented_ideas: list[dict[str, str]],
        failed_ideas: list[dict[str, str]],
        generation: int = 0,
        summary: str = "",
    )
```

**Attributes:**

- `metrics` (dict[str, float]): Metrics from the improvement run (e.g., `tokens_used`, `cost`, `execution_time`).
- `implemented_ideas` (list\[dict[str, str]\]): List of successfully implemented optimization ideas.
- `failed_ideas` (list\[dict[str, str]\]): List of failed optimization attempts.
- `generation` (int): Generation number.
- `summary` (str): Summary of the improvement.

### EvolverPhase

```python
class EvolverPhase(Enum):
    INITIALIZING = "initializing"    # Creating initial agent variants
    EVALUATING = "evaluating"        # Evaluating a variant
    MUTATION = "mutation"            # Mutating a variant
    CROSSOVER = "crossover"          # Crossing over two variants
    PARETO_UPDATE = "pareto_update"  # Updating the Pareto frontier
    COMPLETE = "complete"            # Evolution complete
```

Enum representing the current phase of AgentEvolver optimization, similar to `GEPAPhase` for GEPA.

### EvolverProgress

```python
@dataclass
class EvolverProgress:
    generation: int                           # Current generation (0 = init, 1+ = evolution)
    max_generations: int                      # Total generations to run
    phase: EvolverPhase                       # Current optimization phase
    variant_id: int | None = None             # ID of current variant (if applicable)
    parent_ids: list[int] = field(default_factory=list)  # Parent variant IDs
    frontier_size: int = 0                    # Current Pareto frontier size
    best_score: float | None = None           # Best combined score (lower is better)
    current_metrics: dict[str, float] = field(default_factory=dict)  # Current variant metrics
    added_to_frontier: bool | None = None     # Whether variant was added to frontier
    message: str = ""                         # Descriptive activity message
```

Progress information passed to the callback during optimization.

**Attributes:**

- `generation` (int): Current generation number (0 during initialization, 1-indexed during evolution).
- `max_generations` (int): Total number of generations to run.
- `phase` (EvolverPhase): Current phase of the optimization.
- `variant_id` (int | None): ID of the variant currently being processed.
- `parent_ids` (list[int]): Parent variant IDs for the current operation.
- `frontier_size` (int): Current size of the Pareto frontier.
- `best_score` (float | None): Best combined score seen so far (lower is better).
- `current_metrics` (dict[str, float]): Metrics of the current variant.
- `added_to_frontier` (bool | None): Whether the current variant was added to the Pareto frontier.
- `message` (str): Descriptive message about the current activity.

### create_progress_callback()

```python
def create_progress_callback(verbose: bool = False) -> Callable[[EvolverProgress], None]
```

Create a standard progress callback for console output.

**Parameters:**

- `verbose` (bool): If True, prints all phases. If False (default), only prints evaluation completions, Pareto updates, and completion messages.

**Returns:**

- `Callable[[EvolverProgress], None]`: A callback function for use with `evolve()`.

### Example

```python
from kiss.agents.create_and_optimize_agent import AgentEvolver, EvolverProgress, create_progress_callback

evolver = AgentEvolver()

# Using the built-in progress callback
best = evolver.evolve(
    task_description="Build a code analysis assistant that reviews Python files",
    max_generations=5,
    initial_frontier_size=2,
    max_frontier_size=4,
    mutation_probability=0.8,
    progress_callback=create_progress_callback(verbose=True),
)

print(f"Best variant: {best.folder_path}")
print(f"Metrics: {best.metrics}")
print(f"Generation: {best.generation}")

# Or use a custom callback
def my_callback(progress: EvolverProgress) -> None:
    if progress.phase.value == "complete":
        print(f"Done! Best score: {progress.best_score}")

best = evolver.evolve(
    task_description="Build a code analysis assistant",
    progress_callback=my_callback,
)

# Save state for later analysis
evolver.save_state("evolver_state.json")
```

______________________________________________________________________

## GEPA

GEPA (Genetic-Pareto) is a reflective prompt evolution optimizer based on the paper "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning".

### Constructor

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
    progress_callback: Callable[[GEPAProgress], None] | None = None,
)
```

**Parameters:**

- `agent_wrapper` (Callable): Function `(prompt_template, arguments) -> (result, trajectory)` that runs the agent.
- `initial_prompt_template` (str): The initial prompt template to optimize.
- `evaluation_fn` (Callable | None): Function to evaluate result -> `{metric: score}`. Default checks for "success" in result.
- `max_generations` (int | None): Maximum evolutionary generations. Uses config default (10) if None.
- `population_size` (int | None): Number of candidates per generation. Uses config default (8) if None.
- `pareto_size` (int | None): Maximum Pareto frontier size. Uses config default (4) if None.
- `mutation_rate` (float | None): Probability of mutation. Uses config default (0.5) if None.
- `reflection_model` (str | None): Model for reflection. Uses config default ("gemini-3-flash-preview") if None.
- `dev_val_split` (float | None): Fraction for dev set. Default is 0.5.
- `perfect_score` (float): Score threshold to skip mutation. Default is 1.0.
- `use_merge` (bool): Whether to use structural merge. Default is True.
- `max_merge_invocations` (int): Maximum merge attempts. Default is 5.
- `merge_val_overlap_floor` (int): Minimum validation overlap for merge. Default is 2.
- `progress_callback` (Callable | None): Optional callback function called with `GEPAProgress` during optimization.

### GEPAPhase

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

### GEPAProgress

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

### create_progress_callback

```python
def create_progress_callback(verbose: bool = False) -> Callable[[GEPAProgress], None]
```

Create a standard progress callback for GEPA optimization.

**Parameters:**

- `verbose` (bool): If True, prints all phases. If False, only prints val evaluation completion messages and Pareto frontier updates.

**Returns:**

- A callback function that prints progress updates during optimization.

**Note:** The `pareto_update` phase is always printed (even when `verbose=False`) to show when new candidates join the Pareto frontier, including the full prompt template.

### Methods

#### `optimize()`

```python
def optimize(
    self,
    train_examples: list[dict[str, str]],
    dev_minibatch_size: int | None = None,
) -> PromptCandidate
```

Run GEPA optimization.

**Parameters:**

- `train_examples` (list[dict]): Training examples (will be split into dev/val).
- `dev_minibatch_size` (int | None): Dev examples per evaluation. Default uses all.

**Returns:**

- `PromptCandidate`: Best PromptCandidate found.

#### `get_best_prompt()`

```python
def get_best_prompt(self) -> str
```

Get the best prompt template found.

**Returns:**

- `str`: The best prompt template.

#### `get_pareto_frontier()`

```python
def get_pareto_frontier(self) -> list[PromptCandidate]
```

Get current Pareto frontier.

**Returns:**

- `list[PromptCandidate]`: Copy of the Pareto frontier.

### PromptCandidate

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

______________________________________________________________________

## KISSEvolve

Evolutionary algorithm discovery using LLMs. Evolves code variants through selection, mutation, and crossover.

### Constructor

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

- `code_agent_wrapper` (Callable): The code generation agent wrapper. Should accept keyword arguments: `model_name` (str), `prompt_template` (str), and `arguments` (dict[str, str]).
- `initial_code` (str): The initial code to evolve.
- `evaluation_fn` (Callable): Function that takes code string and returns dict with:
  - `'fitness'`: float (higher is better)
  - `'metrics'`: dict[str, float] (optional)
  - `'artifacts'`: dict[str, Any] (optional)
  - `'error'`: str (optional error message)
- `model_names` (list\[tuple[str, float]\]): List of (model_name, probability) tuples. Probabilities are normalized.
- `extra_coding_instructions` (str): Extra instructions to add to the code generation prompt.
- `population_size` (int | None): Number of variants per generation.
- `max_generations` (int | None): Maximum number of evolutionary generations.
- `mutation_rate` (float | None): Probability of mutating a variant (0.0-1.0).
- `elite_size` (int | None): Number of best variants to preserve each generation.
- `num_islands` (int | None): Number of islands for island-based evolution.
- `migration_frequency` (int | None): Generations between migrations.
- `migration_size` (int | None): Number of individuals to migrate.
- `migration_topology` (str | None): Migration pattern ('ring', 'fully_connected', 'random').
- `enable_novelty_rejection` (bool | None): Enable code novelty rejection sampling.
- `novelty_threshold` (float | None): Cosine similarity threshold for rejecting code (0.0-1.0).
- `max_rejection_attempts` (int | None): Maximum rejection attempts before accepting.
- `novelty_rag_model` (Model | None): Model for generating code embeddings.
- `parent_sampling_method` (str | None): Parent sampling method ('tournament', 'power_law', 'performance_novelty').
- `power_law_alpha` (float | None): Power-law sampling parameter (α).
- `performance_novelty_lambda` (float | None): Performance-novelty sampling parameter (λ).

### Methods

#### `evolve()`

```python
def evolve(self) -> CodeVariant
```

Run the evolutionary algorithm.

**Returns:**

- `CodeVariant`: The best code variant found during evolution.

#### `get_best_variant()`

```python
def get_best_variant(self) -> CodeVariant
```

Get the best variant from the current population or islands.

**Returns:**

- `CodeVariant`: The best code variant.

#### `get_population_stats()`

```python
def get_population_stats(self) -> dict[str, Any]
```

Get statistics about the current population.

**Returns:**

- `dict`: Dictionary with keys: `size`, `avg_fitness`, `best_fitness`, `worst_fitness`.

### CodeVariant

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

______________________________________________________________________

## UsefulTools

A class that provides file operations and bash command execution with path-based access control and security checks.

### Constructor

```python
UsefulTools(
    base_dir: str,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
)
```

**Parameters:**

- `base_dir` (str): The base directory for operations (created if it doesn't exist).
- `readable_paths` (list[str] | None): List of paths the tools can read from. Default is None (no restrictions).
- `writable_paths` (list[str] | None): List of paths the tools can write to. Default is None (no restrictions).

### Methods

#### `Read()`

```python
def Read(self, file_path: str, max_lines: int = 2000) -> str
```

Read file contents with path access control.

**Parameters:**

- `file_path` (str): Absolute path to the file to read.
- `max_lines` (int): Maximum number of lines to return. Default is 2000.

**Returns:**

- `str`: The file contents, or an error message if access is denied or reading fails.

#### `Write()`

```python
def Write(self, file_path: str, content: str) -> str
```

Write content to a file, creating it if it doesn't exist or overwriting if it does. Respects writable_paths access control.

**Parameters:**

- `file_path` (str): Path to the file to write.
- `content` (str): The full content to write to the file.

**Returns:**

- `str`: A success message with bytes written, or an error message if access is denied or writing fails.

#### `Bash()`

```python
def Bash(self, command: str, description: str, timeout_seconds: float = 30) -> str
```

Execute a bash command with automatic path permission checks and security validation.

**Parameters:**

- `command` (str): The bash command to execute.
- `description` (str): A brief description of what the command does.
- `timeout_seconds` (float): Timeout in seconds for the command. Default is 30.

**Returns:**

- `str`: The command output (stdout), or an error message if permission denied, security violation, or execution failed.

**Security Features:**

- Automatically parses commands to detect file operations
- Enforces readable_paths and writable_paths restrictions
- Returns descriptive error messages for violations

#### `Edit()`

```python
def Edit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    timeout_seconds: float = 30,
) -> str
```

Performs precise string replacements in files with exact matching.

**Parameters:**

- `file_path` (str): Absolute path to the file to modify.
- `old_string` (str): Exact text to find and replace.
- `new_string` (str): Replacement text, must differ from old_string.
- `replace_all` (bool): If True, replace all occurrences. Default is False.
- `timeout_seconds` (float): Timeout in seconds for the edit command. Default is 30.

**Returns:**

- `str`: The output of the edit operation.

#### `MultiEdit()`

```python
def MultiEdit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    timeout_seconds: float = 30,
) -> str
```

Performs precise string replacements in files with exact matching. Has the same functionality as `Edit()` - both methods use the same underlying implementation for string replacement.

**Parameters:**

- `file_path` (str): Absolute path to the file to modify.
- `old_string` (str): Exact text to find and replace.
- `new_string` (str): Replacement text, must differ from old_string.
- `replace_all` (bool): If True, replace all occurrences. Default is False.
- `timeout_seconds` (float): Timeout in seconds for the edit command. Default is 30.

**Returns:**

- `str`: The output of the edit operation.

**Example:**

```python
from kiss.core.useful_tools import UsefulTools

tools = UsefulTools(
    base_dir="/workdir",
    readable_paths=["/workdir/src"],
    writable_paths=["/workdir/output"],
)

# This will work
output = tools.Bash("cat src/file.txt > output/result.txt", "Copy file")

# This will be denied
output = tools.Bash("cat /etc/passwd", "Read system file")
# Returns: "Error: Access denied for reading /etc/passwd"
```

### Module-level Functions

The `kiss.core.useful_tools` module also provides these standalone functions:

#### `fetch_url()`

```python
def fetch_url(
    url: str,
    headers: dict[str, str],
    max_characters: int = 10000,
    timeout_seconds: float = 10.0,
) -> str
```

Fetch and extract text content from a URL using BeautifulSoup.

**Parameters:**

- `url` (str): The URL to fetch.
- `headers` (dict[str, str]): HTTP headers to use for the request.
- `max_characters` (int): Maximum number of characters to return. Default is 10000.
- `timeout_seconds` (float): Request timeout in seconds. Default is 10.0.

**Returns:**

- `str`: Extracted text content from the page, or an error message if fetch failed.

#### `search_web()`

```python
def search_web(query: str, max_results: int = 10) -> str
```

Perform a web search and return the top search results with page contents.

Tries DuckDuckGo first (more reliable for automated access), then falls back to Startpage if needed. Uses Playwright headless browser with Safari/WebKit to render JavaScript and avoid bot detection.

**Parameters:**

- `query` (str): The search query.
- `max_results` (int): Maximum number of results to fetch content for. Default is 10.

**Returns:**

- `str`: A string containing titles, links, and page contents of the top search results.

#### `parse_bash_command_paths()`

```python
def parse_bash_command_paths(command: str) -> tuple[list[str], list[str]]
```

Parse a bash command to extract readable and writable directory paths.

This function analyzes bash commands to intelligently determine which directories are being read from and which are being written to. It handles:

- Common read commands: cat, grep, find, ls, python, gcc, rsync, etc.
- Common write commands: touch, mkdir, rm, mv, cp, rsync, etc.
- Output redirection: >, >>, &>, 2>, etc.
- Input redirection: \<, \<<, \<<\<
- Heredoc stripping: heredoc body text (between `<< DELIM` and `DELIM`) is stripped before parsing so that heredoc content is not misinterpreted as command arguments
- Command separators: `&&`, `||`, `;` chains
- Pipe chains with multiple commands
- Flags and arguments parsing
- Special commands like tee (reads stdin and writes to file)
- Path cleaning: trailing punctuation characters (e.g., `)`, `'`, `"`, `;`, `}`, `]`) are stripped from paths before safety checks

**Parameters:**

- `command` (str): The bash command to parse.

**Returns:**

- `tuple[list[str], list[str]]`: A tuple of (readable_dirs, writable_dirs) where each is a sorted list of directory paths.

**Example:**

```python
from kiss.core.useful_tools import parse_bash_command_paths

# Reading and writing
readable, writable = parse_bash_command_paths("cat input.txt > output.txt")
# readable: ['input.txt'], writable: ['output.txt']

# Complex command with pipes
readable, writable = parse_bash_command_paths("grep 'pattern' src/*.py | tee results.txt")
# readable: ['src/'], writable: ['results.txt']

# Copy operations
readable, writable = parse_bash_command_paths("cp -r src/ dest/")
# readable: ['src/'], writable: ['dest/']
```

**Note:**

This function is used internally by `UsefulTools.Bash()` and `KISSCodingAgent.run_bash_command()` to automatically determine which paths need read/write permissions before executing bash commands.

______________________________________________________________________

## Utility Functions

Helper functions from `kiss.core.utils`.

### `get_config_value()`

```python
def get_config_value(
    value: T | None,
    config_obj: Any,
    attr_name: str,
    default: T | None = None
) -> T
```

Get a config value, preferring explicit value over config default.

**Parameters:**

- `value`: The explicitly provided value (may be None).
- `config_obj`: The config object to read from if value is None.
- `attr_name` (str): The attribute name to read from config_obj.
- `default`: Fallback default if both value and config attribute are None.

**Returns:**

- The resolved value (explicit value > config value > default).

### `get_template_field_names()`

```python
def get_template_field_names(text: str) -> list[str]
```

Get the field names from the text template.

**Parameters:**

- `text` (str): The text containing template field placeholders.

**Returns:**

- `list[str]`: A list of field names found in the text.

### `add_prefix_to_each_line()`

```python
def add_prefix_to_each_line(text: str, prefix: str) -> str
```

Adds a prefix to each line of the text.

**Parameters:**

- `text` (str): The text to add prefix to.
- `prefix` (str): The prefix to add to each line.

**Returns:**

- `str`: The text with prefix added to each line.

### `config_to_dict()`

```python
def config_to_dict() -> dict[Any, Any]
```

Convert the default config to a dictionary (excludes API keys).

**Returns:**

- `dict`: A dictionary representation of the default config.

### `fc()`

```python
def fc(file_path: str) -> str
```

Reads a file and returns the content.

**Parameters:**

- `file_path` (str): The path to the file to read.

**Returns:**

- `str`: The content of the file.

### `resolve_path()`

```python
def resolve_path(p: str, base_dir: str) -> Path
```

Resolve a path relative to base_dir if not absolute.

**Parameters:**

- `p` (str): The path to resolve.
- `base_dir` (str): The base directory to resolve relative paths against.

**Returns:**

- `Path`: The resolved absolute path.

**Example:**

```python
from kiss.core.utils import resolve_path

# Relative path
path = resolve_path("output/file.txt", "/workdir")
# Returns: Path("/workdir/output/file.txt")

# Absolute path (returned as-is)
path = resolve_path("/tmp/file.txt", "/workdir")
# Returns: Path("/tmp/file.txt")
```

### `is_subpath()`

```python
def is_subpath(target: Path, whitelist: list[Path]) -> bool
```

Check if target has any prefix in whitelist.

**Parameters:**

- `target` (Path): The target path to check.
- `whitelist` (list[Path]): List of paths to check against.

**Returns:**

- `bool`: True if target is a subpath of any path in whitelist, False otherwise.

**Example:**

```python
from pathlib import Path
from kiss.core.utils import is_subpath

target = Path("/workdir/output/file.txt")
whitelist = [Path("/workdir/output"), Path("/workdir/tmp")]

if is_subpath(target, whitelist):
    print("Access allowed")
else:
    print("Access denied")
```

### `finish()` (for KISSAgent in utils.py)

```python
def finish(
    status: str = "success",
    analysis: str = "",
    result: str = "",
) -> str
```

A utility function from `kiss.core.utils` that can be used as a finish tool for KISSAgent instances. Returns a YAML-formatted result string.

**Location:** `kiss.core.utils`

**Parameters:**

- `status` (str): The status of the agent's task ('success' or 'failure'). Default is 'success'.
- `analysis` (str): The analysis of the agent's trajectory.
- `result` (str): The result generated by the agent.

**Returns:**

- `str`: A YAML string containing the status, analysis, and result.

**Note:** KISSAgent has its own built-in `finish(result: str) -> str` method that takes only a result parameter and returns it directly. The utility function version is optional and provides more structured output.

### `finish()` (for KISSCodingAgent / RelentlessCodingAgent)

```python
def finish(success: bool, summary: str) -> str
```

Used by KISSCodingAgent, RelentlessCodingAgent, and their sub-agents to complete task execution. This is defined separately in each coding agent module and has a different signature than the utility version.

**Location:** `kiss.agents.coding_agents.kiss_coding_agent` and `kiss.agents.coding_agents.relentless_coding_agent`

**Parameters:**

- `success` (bool): True if the task was successful, False otherwise.
- `summary` (str): Summary message describing the task outcome.

**Returns:**

- `str`: A YAML string containing the 'success' (boolean) and 'summary' (string) keys.

### `read_project_file()`

```python
def read_project_file(file_path_relative_to_project_root: str) -> str
```

Read a file from the project root. Compatible with installations packaged as .whl or source.

**Parameters:**

- `file_path_relative_to_project_root` (str): Path relative to the project root.

**Returns:**

- `str`: The file's contents.

### `read_project_file_from_package()`

```python
def read_project_file_from_package(file_name_as_python_package: str) -> str
```

Read a file from the project root using Python package conventions.

**Location:** `kiss.core.utils`

**Parameters:**

- `file_name_as_python_package` (str): File name as a Python package.

**Returns:**

- `str`: The file's contents.

______________________________________________________________________

## Token Streaming via Printer

The KISS framework supports real-time token streaming through the `Printer` abstraction. Each `Printer` implements a `token_callback` method that receives streamed tokens. When a printer is provided to `KISSAgent.run()` (or when `verbose=True` creates one automatically), model responses are streamed token-by-token in real-time.

### Streaming Behavior by Provider

- **Anthropic (Claude)**: Uses `messages.stream()` with `text_stream` for both `generate()` and `generate_and_process_with_tools()`.
- **OpenAI / Together AI / OpenRouter**: Uses `chat.completions.create(stream=True)` with `stream_options={"include_usage": True}` to preserve token counts. For tool calls, streaming accumulates tool-call deltas and reconstructs the full call.
- **Gemini**: Uses `generate_content_stream()` for `generate()` and `generate_content_stream()` with part-level parsing for `generate_and_process_with_tools()`.

When no printer is provided and `verbose=False`, all providers fall back to their original non-streaming code paths.

### KISSAgent Integration

When `verbose=True` (default), `KISSAgent` automatically creates a `MultiPrinter` based on config flags: `print_to_browser` (default: True) adds a `BrowserPrinter`, `print_to_console` (default: True) adds a `ConsolePrinter`. You can also pass a custom printer. The printer receives:

1. **Model response tokens** as they are generated.
1. **Tool execution output** after each tool call completes.

### Coding Agent Integration

All coding agents support the `printer` parameter in their `run()` methods:

- **KISSCodingAgent**: The printer is passed through to the underlying `KISSAgent.run()` calls for both orchestrator and executor sub-agents.
- **RelentlessCodingAgent**: The printer is passed through to the underlying `KISSAgent.run()` calls for each trial.
- **GeminiCliAgent**: The printer receives text content and tool response text from ADK events.
- **OpenAICodexAgent**: The printer receives message text and tool output text from the Agents SDK run result.
- **ClaudeCodingAgent**: Uses its own built-in streaming output via `ConsolePrinter` (terminal) or `BrowserPrinter` (browser UI) controlled by the `use_browser` parameter.

### Example

```python
from kiss.core.kiss_agent import KISSAgent
from kiss.core.print_to_console import ConsolePrinter

agent = KISSAgent("streaming-agent")
result = agent.run(
    model_name="gpt-4o",
    prompt_template="Write a haiku about programming.",
    is_agentic=False,
    printer=ConsolePrinter(),
)
```

______________________________________________________________________

## Configuration System

The KISS framework uses a Pydantic-based configuration system accessible through `DEFAULT_CONFIG`. Configuration is extensible via `add_config()` for sub-systems like GEPA, KISSEvolve, etc.

### Config Structure

```python
from kiss.core.config import DEFAULT_CONFIG

# Access configuration
DEFAULT_CONFIG.agent.api_keys.OPENAI_API_KEY = "your-key"
DEFAULT_CONFIG.agent.max_steps = 100
DEFAULT_CONFIG.agent.max_agent_budget = 10.0
DEFAULT_CONFIG.agent.global_max_budget = 200.0
DEFAULT_CONFIG.agent.verbose = True
DEFAULT_CONFIG.agent.use_web = True
DEFAULT_CONFIG.agent.print_to_console = True
DEFAULT_CONFIG.agent.print_to_browser = True
```

### Configuration Sections

#### `agent.api_keys`

- `OPENAI_API_KEY` (str): OpenAI API key
- `ANTHROPIC_API_KEY` (str): Anthropic API key
- `GEMINI_API_KEY` (str): Google Gemini API key
- `TOGETHER_API_KEY` (str): Together AI API key
- `OPENROUTER_API_KEY` (str): OpenRouter API key

#### `agent`

- `max_steps` (int): Maximum steps per agent run (default: 100)
- `max_agent_budget` (float): Maximum budget per agent in USD (default: 10.0)
- `global_max_budget` (float): Global budget limit in USD (default: 200.0)
- `verbose` (bool): Enable verbose output (default: True)
- `use_web` (bool): Enable web search tool (default: True)
- `debug` (bool): Enable debug mode (default: False)
- `print_to_console` (bool): Enable ConsolePrinter for Rich terminal output (default: True)
- `print_to_browser` (bool): Enable BrowserPrinter for live browser UI output (default: True)
- `artifact_dir` (str): Directory for agent artifacts (default: auto-generated with timestamp)

#### `coding_agent.relentless_coding_agent`

- `model_name` (str): Model for task execution (default: "claude-opus-4-6")
- `max_steps` (int): Maximum steps per trial for the Relentless Coding Agent (default: 200)
- `max_budget` (float): Maximum budget in USD for the Relentless Coding Agent (default: 200.0)
- `trials` (int): Number of continuation attempts (default: 200)

#### `coding_agent.kiss_coding_agent`

- `orchestrator_model_name` (str): Model for main orchestration (default: "claude-opus-4-6")
- `subtasker_model_name` (str): Model for subtask generation and execution (default: "claude-opus-4-6")
- `refiner_model_name` (str): Model for dynamic prompt refinement on failures (default: "claude-sonnet-4-5")
- `max_steps` (int): Maximum steps for the KISS Coding Agent (default: 200)
- `max_budget` (float): Maximum budget in USD for the KISS Coding Agent (default: 100.0)
- `trials` (int): Retry attempts per task/subtask (default: 200)

#### `docker`

- `client_shared_path` (str): Path inside Docker container for shared volume (default: "/testbed")

#### `gepa`

- `reflection_model` (str): Model to use for reflection (default: "gemini-3-flash-preview")
- `max_generations` (int): Maximum number of evolutionary generations (default: 10)
- `population_size` (int): Number of candidates to maintain in population (default: 8)
- `pareto_size` (int): Maximum size of Pareto frontier (default: 4)
- `mutation_rate` (float): Probability of mutating a prompt template in each generation (default: 0.5)

#### `kiss_evolve`, `create_and_optimize_agent`

Configuration sections for evolutionary optimization systems. See respective classes for details.

### `add_config()`

```python
def add_config(name: str, config_class: type[BaseModel]) -> None
```

Extend the KISS configuration with a new config section. Each call adds a new config field while preserving existing fields from previous calls. Also parses command-line arguments matching the config fields.

**Location:** `kiss.core.config_builder`

**Parameters:**

- `name` (str): Name of the new config section (e.g., "gepa", "kiss_evolve").
- `config_class` (type[BaseModel]): A Pydantic BaseModel class defining the config fields.

______________________________________________________________________

## Pre-built Agents

Ready-to-use agent functions from `kiss.agents.kiss`.

### `prompt_refiner_agent()`

```python
def prompt_refiner_agent(
    original_prompt_template: str,
    previous_prompt_template: str,
    agent_trajectory_summary: str,
    model_name: str,
) -> str
```

Refines the prompt template based on the agent's trajectory summary.

**Parameters:**

- `original_prompt_template` (str): The original prompt template.
- `previous_prompt_template` (str): The previous version of the prompt template that led to the given trajectory.
- `agent_trajectory_summary` (str): The agent's trajectory summary as a string.
- `model_name` (str): The name of the model to use for the agent.

**Returns:**

- `str`: The refined prompt template.

### `run_bash_task_in_sandboxed_ubuntu_latest()`

```python
def run_bash_task_in_sandboxed_ubuntu_latest(task: str, model_name: str) -> str
```

Run a bash task in a sandboxed Ubuntu latest container.

**Parameters:**

- `task` (str): The task to run.
- `model_name` (str): The name of the model to use for the agent.

**Returns:**

- `str`: The result of the task.

### `get_run_simple_coding_agent()`

```python
def get_run_simple_coding_agent(
    test_fn: Callable[[str], bool]
) -> Callable[..., str]
```

Return a function that runs a simple coding agent with a test function.

**Parameters:**

- `test_fn` (Callable\[[str], bool\]): The test function to use for the agent.

**Returns:**

- `Callable`: A function that runs a simple coding agent. Accepts keyword arguments: `model_name` (str), `prompt_template` (str), and `arguments` (dict[str, str]).
