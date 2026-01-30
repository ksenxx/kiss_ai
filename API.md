# KISS Framework API Reference

**Keep It Simple, Stupid** - Comprehensive API documentation for the KISS AI agent framework.

## Introduction

The KISS framework provides a clean, simple API for building AI agents with native function calling, multi-agent orchestration, and evolutionary optimization. This document covers all public classes, methods, and utilities available in the framework.

For a high-level overview and quick start guide, see [README.md](README.md).

## Table of Contents

- [KISSAgent](#kissagent) - Core agent class with function calling
- [KISSCodingAgent](#kisscodingagent) - Multi-agent coding system with planning and orchestration
- [ClaudeCodingAgent](#claudecodingagent) - Claude Agent SDK-based coding agent
- [GeminiCliAgent](#geminicliagent) - Google ADK-based coding agent
- [OpenAICodexAgent](#openaicodexagent) - OpenAI Agents SDK-based coding agent
- [DockerManager](#dockermanager) - Docker container management
- [Multiprocessing](#multiprocessing) - Parallel execution utilities
- [SimpleRAG](#simplerag) - Simple RAG system for document retrieval
- [AgentEvolver](#agentevolver) - Evolutionary agent optimization
- [ImproverAgent](#improveragent) - Multi agent creator and optimizer
- [GEPA](#gepa) - Genetic-Pareto prompt optimizer
- [KISSEvolve](#kissevolve) - Evolutionary algorithm discovery
- [Utility Functions](#utility-functions) - Helper functions
- [SimpleFormatter](#simpleformatter) - Terminal output formatting
- [Pre-built Agents](#pre-built-agents) - Ready-to-use agents

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
    formatter: Formatter | None = None,
    is_agentic: bool = True,
    max_steps: int = 100,
    max_budget: float = 10.0,
    model_config: dict[str, Any] | None = None,
) -> str
```

Runs the agent's main ReAct loop to solve the task.

**Parameters:**

- `model_name` (str): The name of the model to use (e.g., "gpt-4o", "claude-sonnet-4-5", "gemini-2.5-flash", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "openrouter/anthropic/claude-3.5-sonnet")
- `prompt_template` (str): The prompt template for the agent. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `tools` (list\[Callable[..., Any]\] | None): List of callable functions the agent can use. The `finish` tool is automatically added if it is not provided in `tools`. If `use_web_search` is enabled in config, `search_web` is also automatically added. Default is None.
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.
- `is_agentic` (bool): If True, runs in agentic mode with tools. If False, returns raw LLM response. Default is True.
- `max_steps` (int): Maximum number of ReAct loop iterations. Default is 100.
- `max_budget` (float): Maximum budget in USD for this agent run. Default is 10.0.
- `model_config` (dict[str, Any] | None): Optional model configuration to pass to the model. Default is None.

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

Built-in tool that the agent must call to complete its task and return the final answer.
The tool is added by default to any KISSAgent.

**Parameters:**

- `result` (str): The final result/answer from the agent.

**Returns:**

- `str`: The same result string passed in.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model_name` (str): The name of the model being used.
- `model`: The model instance being used.
- `function_map` (list[str]): List of function/tool names available to this agent.
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

## KISSCodingAgent

A multi-agent coding system with orchestration and sub-agents using KISSAgent. It efficiently breaks down complex coding tasks into manageable sub-tasks through a multi-agent architecture with:
- **Orchestrator Agent**: Manages overall task execution and delegates to sub-tasks
- **Executor Agents**: Handle specific sub-tasks independently
- **Refiner Agent**: Refines prompts on failures for improved retry attempts

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
    def __init__(self, name: str, context: str, description: str) -> None
```

Represents a sub-task in the multi-agent coding system.

**Attributes:**

- `id` (int): Unique identifier for this sub-task (auto-incremented)
- `name` (str): Name of the sub-task
- `context` (str): Relevant context information for this sub-task
- `description` (str): Detailed description of what needs to be done

### Methods

#### `run()`

```python
def run(
    self,
    prompt_template: str,
    arguments: dict[str, str] | None = None,
    orchestrator_model_name: str = DEFAULT_CONFIG.agent.kiss_coding_agent.orchestrator_model_name,
    subtasker_model_name: str = DEFAULT_CONFIG.agent.kiss_coding_agent.subtasker_model_name,
    refiner_model_name: str = DEFAULT_CONFIG.agent.kiss_coding_agent.refiner_model_name,
    trials: int = DEFAULT_CONFIG.agent.kiss_coding_agent.trials,
    max_steps: int = DEFAULT_CONFIG.agent.max_steps,
    max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
    base_dir: str = "<artifact_dir>/kiss_workdir",
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
) -> str
```

Run the multi-agent coding system with orchestration and sub-task delegation.

**Parameters:**

- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `orchestrator_model_name` (str): Model for the main orchestrator agent. Default is from config (claude-sonnet-4-5).
- `subtasker_model_name` (str): Model for executor sub-task agents. Default is from config (claude-opus-4-5).
- `refiner_model_name` (str): Model for prompt refinement on failures. Default is from config (claude-sonnet-4-5).
- `trials` (int): Number of retry attempts for each task/subtask. Default is 3.
- `max_steps` (int): Maximum number of steps per agent. Default is from config.
- `max_budget` (float): Maximum budget in USD for this run. Default is from config.
- `base_dir` (str): The base directory relative to which readable and writable paths are resolved if they are not absolute.
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no path restrictions.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no path restrictions.

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
    context: str,
    description: str,
) -> str
```

Execute a sub-task using a dedicated executor agent. Can be called by the orchestrator or recursively by other sub-task executors.

**Parameters:**

- `subtask_name` (str): Name of the sub-task for identification.
- `context` (str): Relevant context information for this sub-task.
- `description` (str): Detailed description of what needs to be done.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

#### `run_bash_command()`

```python
def run_bash_command(self, command: str, description: str) -> str
```

Run a bash command with automatic path permission checks. Uses `parse_bash_command_paths()` to extract readable and writable paths from the command, then validates permissions before execution.

**Parameters:**

- `command` (str): The bash command to execute.
- `description` (str): A brief description of what the command does.

**Returns:**

- `str`: The command output (stdout), or an error message if permission denied or execution failed.

**Notes:**

- Automatically parses commands to detect file operations
- Enforces readable_paths and writable_paths restrictions
- Returns descriptive error messages for permission violations

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `orchestrator_model_name` (str): Model name for orchestrator agent.
- `subtasker_model_name` (str): Model name for executor agents.
- `refiner_model_name` (str): Model name for prompt refinement.
- `task_description` (str): The formatted task description.
- `messages` (list\[dict[str, Any]\]): List of messages in the trajectory (aggregated from all sub-agents).
- `total_tokens_used` (int): Total tokens used across all agents in this run.
- `budget_used` (float): Total budget used across all agents in this run.
- `run_start_timestamp` (int): Unix timestamp when the run started.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps per agent.
- `max_budget` (float): Maximum total budget allowed for this run.
- `trials` (int): Number of retry attempts for each task/subtask.

### Key Features

- **Multi-Agent Architecture**: Orchestrator delegates to executor agents; supports recursive sub-task decomposition
- **Three-Model Strategy**:
  - Orchestrator (default: claude-sonnet-4-5) for high-level coordination
  - Subtasker (default: claude-opus-4-5) for complex sub-task execution
  - Refiner (default: claude-sonnet-4-5) for prompt improvement on failures
- **Efficient Orchestration**: Manages execution to stay within configured step limits through smart delegation
- **Bash Command Parsing**: Automatically extracts readable/writable paths from commands using `parse_bash_command_paths()`
- **Path Access Control**: Enforces read/write permissions on file system paths before command execution
- **Retry Logic with Refinement**: On failure, refines prompts using trajectory analysis for improved retry attempts (configurable trials)
- **Built-in Tools**: Each agent has access to `finish()`, `run_bash_command()`, and `perform_subtask()`
- **Recursive Delegation**: Sub-tasks can spawn further sub-tasks as needed

### Example

```python
from kiss.core.kiss_coding_agent import KISSCodingAgent

agent = KISSCodingAgent("My Coding Agent")

result = agent.run(
    prompt_template="""
        Write, test, and optimize a fibonacci function in Python
        that is efficient and correct. Save it to fibonacci.py.
    """,
    orchestrator_model_name="claude-sonnet-4-5",
    subtasker_model_name="claude-opus-4-5",
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

______________________________________________________________________

## ClaudeCodingAgent

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
async def run(
    self,
    model_name: str,
    prompt_template: str,
    arguments: dict[str, str] | None = None,
    max_steps: int = DEFAULT_CONFIG.agent.max_steps,
    max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
    base_dir: str = "<artifact_dir>/claude_workdir",
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
) -> str | None
```

Run the Claude coding agent for a given task.

**Parameters:**

- `model_name` (str): The name of the model to use (e.g., "claude-sonnet-4-5").
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int): Maximum number of steps. Default is from config.
- `max_budget` (float): Maximum budget in USD for this run. Default is from config.
- `base_dir` (str): The base directory relative to which readable and writable paths are resolved if they are not absolute.
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no path restrictions.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no path restrictions.

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
import anyio
from kiss.core.claude_coding_agent import ClaudeCodingAgent

async def main():
    agent = ClaudeCodingAgent("My Agent")
    result = await agent.run(
        model_name="claude-sonnet-4-5",
        prompt_template="Write a fibonacci function with tests"
    )
    if result:
        print(f"Result: {result}")

anyio.run(main)
```

______________________________________________________________________

## GeminiCliAgent

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
async def run(
    self,
    model_name: str = "gemini-2.5-flash",
    prompt_template: str = "",
    arguments: dict[str, str] | None = None,
    max_steps: int = DEFAULT_CONFIG.agent.max_steps,
    max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
    base_dir: str = "<artifact_dir>/gemini_workdir",
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    formatter: Formatter | None = None,
) -> str | None
```

Run the Gemini CLI agent for a given task.

**Parameters:**

- `model_name` (str): The name of the model to use. Default is "gemini-2.5-flash".
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int): Maximum number of steps. Default is from config.
- `max_budget` (float): Maximum budget in USD for this run. Default is from config.
- `base_dir` (str): The base directory relative to which readable and writable paths are resolved if they are not absolute.
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, only base_dir is readable.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, only base_dir is writable.
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.

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
import anyio
from kiss.core.gemini_cli_agent import GeminiCliAgent

async def main():
    agent = GeminiCliAgent("my_agent")
    result = await agent.run(
        model_name="gemini-2.5-flash",
        prompt_template="Write a fibonacci function with tests"
    )
    if result:
        print(f"Result: {result}")

anyio.run(main)
```

______________________________________________________________________

## OpenAICodexAgent

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
async def run(
    self,
    model_name: str = "gpt-5.2-codex",
    prompt_template: str = "",
    arguments: dict[str, str] | None = None,
    max_steps: int = DEFAULT_CONFIG.agent.max_steps,
    max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
    base_dir: str = "<artifact_dir>/codex_workdir",
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    formatter: Formatter | None = None,
) -> str | None
```

Run the OpenAI Codex agent for a given task.

**Parameters:**

- `model_name` (str): The name of the model to use. Default is "gpt-5.2-codex".
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int): Maximum number of steps. Default is from config.
- `max_budget` (float): Maximum budget in USD for this run. Default is from config.
- `base_dir` (str): The base directory relative to which readable and writable paths are resolved if they are not absolute.
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, only base_dir is readable.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, only base_dir is writable.
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.

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
import anyio
from kiss.core.openai_codex_agent import OpenAICodexAgent

async def main():
    agent = OpenAICodexAgent("My Agent")
    result = await agent.run(
        model_name="gpt-5.2-codex",
        prompt_template="Write a fibonacci function with tests"
    )
    if result:
        print(f"Result: {result}")

anyio.run(main)
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
- `host_shared_path` (str | None): Path to the host-side shared directory.
- `client_shared_path` (str): Path to the container-side shared directory.
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
AgentEvolver(
    task_description: str,
    evaluation_fn: Callable[[str], tuple[int, float]] | None = None,
    model_name: str | None = None,
    max_generations: int | None = None,
    max_frontier_size: int | None = None,
    mutation_probability: float | None = None,
    coding_agent_type: Literal["kiss code", "claude code", "gemini cli", "openai codex"] | None = None,
)
```

**Parameters:**

- `task_description` (str): Description of the task the agent should perform.
- `evaluation_fn` (Callable | None): Optional function to evaluate agent variants. Takes folder path, returns `(tokens_used, execution_time)`. If None, uses placeholder evaluation.
- `model_name` (str | None): LLM model to use for agent creation and improvement. Uses config default if None.
- `max_generations` (int | None): Maximum number of improvement generations. Uses config default if None.
- `max_frontier_size` (int | None): Maximum size of the Pareto frontier. Uses config default if None.
- `mutation_probability` (float | None): Probability of mutation vs crossover (1.0 = always mutate). Uses config default if None.
- `coding_agent_type` (Literal | None): Which coding agent to use: `"kiss code"`, `"claude code"`, `"gemini cli"`, or `"openai codex"`. Uses config default if None.

### Methods

#### `evolve()`

```python
async def evolve(self) -> AgentVariant
```

Run the evolutionary optimization process.

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
    tokens_used: int = 0
    execution_time: float = 0.0
    id: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
```

### ImprovementReport

```python
@dataclass
class ImprovementReport:
    implemented_ideas: list[dict[str, str]] = field(default_factory=list)
    failed_ideas: list[dict[str, str]] = field(default_factory=list)
    generation: int = 0
    improved_tokens: int = 0
    improved_time: float = 0.0
    summary: str = ""
```

### Example

```python
import anyio
from kiss.agents.agent_creator import AgentEvolver

async def main():
    evolver = AgentEvolver(
        task_description="Build a code analysis assistant that reviews Python files",
        max_generations=5,
        max_frontier_size=4,
        mutation_probability=0.8,
    )

    best = await evolver.evolve()

    print(f"Best variant: {best.folder_path}")
    print(f"Tokens used: {best.tokens_used}")
    print(f"Execution time: {best.execution_time:.2f}s")

    # Save state for later analysis
    evolver.save_state("evolver_state.json")

anyio.run(main)
```

______________________________________________________________________

## ImproverAgent

ImproverAgent optimizes existing agent code to reduce token usage and execution time using a configurable coding agent.

### Constructor

```python
ImproverAgent(
    model_name: str | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    coding_agent_type: Literal["kiss code", "claude code", "gemini cli", "openai codex"] | None = None,
)
```

**Parameters:**

- `model_name` (str | None): LLM model to use for improvement. Uses config default if None.
- `max_steps` (int | None): Maximum steps for the coding agent. Uses config default if None.
- `max_budget` (float | None): Maximum budget in USD for the coding agent. Uses config default if None.
- `coding_agent_type` (Literal | None): Which coding agent to use: `"kiss code"`, `"claude code"`, `"gemini cli"`, or `"openai codex"`. Defaults to `"claude code"` if None.

### Methods

#### `improve()`

```python
async def improve(
    self,
    source_folder: str,
    target_folder: str,
    report_path: str | None = None,
    base_dir: str | None = None,
) -> tuple[bool, ImprovementReport | None]
```

Improve an agent's code to reduce token usage and execution time.

**Parameters:**

- `source_folder` (str): Path to the folder containing the agent's source code.
- `target_folder` (str): Path where the improved agent will be written.
- `report_path` (str | None): Optional path to a previous improvement report.
- `base_dir` (str | None): Working directory for the coding agent.

**Returns:**

- `tuple[bool, ImprovementReport | None]`: Tuple of (success, report). Report is None if improvement failed.

#### `crossover_improve()`

```python
async def crossover_improve(
    self,
    primary_folder: str,
    primary_report_path: str,
    secondary_report_path: str,
    target_folder: str,
    base_dir: str | None = None,
) -> tuple[bool, ImprovementReport | None]
```

Improve an agent by combining ideas from two variants.

**Parameters:**

- `primary_folder` (str): Path to the primary variant's source code.
- `primary_report_path` (str): Path to the primary variant's improvement report.
- `secondary_report_path` (str): Path to the secondary variant's improvement report.
- `target_folder` (str): Path where the improved agent will be written.
- `base_dir` (str | None): Working directory for the coding agent.

**Returns:**

- `tuple[bool, ImprovementReport | None]`: Tuple of (success, report).

### Example

```python
import anyio
from kiss.agents.agent_creator import ImproverAgent

async def main():
    improver = ImproverAgent(model_name="claude-sonnet-4-5")

    success, report = await improver.improve(
        source_folder="/path/to/original/agent",
        target_folder="/path/to/improved/agent",
    )

    if success and report:
        print(f"Improvement completed in {report.improved_time:.2f}s")
        print(f"Tokens used: {report.improved_tokens}")

anyio.run(main)
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
)
```

**Parameters:**

- `agent_wrapper` (Callable): Function `(prompt_template, arguments) -> (result, trajectory)` that runs the agent.
- `initial_prompt_template` (str): The initial prompt template to optimize.
- `evaluation_fn` (Callable | None): Function to evaluate result -> `{metric: score}`. Default checks for "success" in result.
- `max_generations` (int | None): Maximum evolutionary generations. Uses config default if None.
- `population_size` (int | None): Number of candidates per generation. Uses config default if None.
- `pareto_size` (int | None): Maximum Pareto frontier size. Uses config default if None.
- `mutation_rate` (float | None): Probability of mutation. Default is 0.5.
- `reflection_model` (str | None): Model for reflection. Uses config default if None.
- `dev_val_split` (float | None): Fraction for dev set. Default is 0.5.
- `perfect_score` (float): Score threshold to skip mutation. Default is 1.0.
- `use_merge` (bool): Whether to use structural merge. Default is True.
- `max_merge_invocations` (int): Maximum merge attempts. Default is 5.
- `merge_val_overlap_floor` (int): Minimum validation overlap for merge. Default is 2.

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
    id: int = 0
    dev_scores: dict[str, float] = field(default_factory=dict)
    val_scores: dict[str, float] = field(default_factory=dict)
    per_item_val_scores: list[dict[str, float]] = field(default_factory=list)
    val_instance_wins: set[int] = field(default_factory=set)
    evaluated_val_ids: set[int] = field(default_factory=set)
    parents: list[int] = field(default_factory=list)
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

### `finish()` (for KISSAgent)

```python
def finish(
    status: str = "success",
    analysis: str = "",
    result: str = "",
) -> str
```

The agent must call this function with the final status, analysis, and result when it has solved the given task. This is automatically added as a tool for KISSAgent instances.

**Parameters:**

- `status` (str): The status of the agent's task ('success' or 'failure'). Default is 'success'.
- `analysis` (str): The analysis of the agent's trajectory.
- `result` (str): The result generated by the agent.

**Returns:**

- `str`: A YAML string containing the status, analysis, and result.

### `finish()` (for KISSCodingAgent)

```python
def finish(success: bool, summary: str) -> str
```

Used by KISSCodingAgent and its sub-agents to complete task execution. This is a different signature than the KISSAgent finish() function.

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

### `search_web()`

```python
def search_web(query: str, max_results: int = 5) -> str
```

Perform a web search and return the top search results with page contents.

**Parameters:**

- `query` (str): The search query.
- `max_results` (int): Maximum number of results to fetch content for. Default is 5.

**Returns:**

- `str`: A string containing titles, links, and page contents of the top search results.

### `parse_bash_command_paths()`

```python
def parse_bash_command_paths(command: str) -> tuple[list[str], list[str]]
```

Parse a bash command to extract readable and writable directory paths.

This function analyzes bash commands to intelligently determine which directories are being read from and which are being written to. It handles:
- Common read commands: cat, grep, find, ls, python, gcc, etc.
- Common write commands: touch, mkdir, rm, mv, cp, etc.
- Output redirection: >, >>, &>, 2>, etc.
- Pipe chains with multiple commands
- Flags and arguments parsing

**Parameters:**

- `command` (str): The bash command to parse.

**Returns:**

- `tuple[list[str], list[str]]`: A tuple of (readable_dirs, writable_dirs) where each is a sorted list of directory paths.

**Example:**

```python
from kiss.core.utils import parse_bash_command_paths

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

This function is used internally by KISSCodingAgent.run_bash_command() to automatically determine which paths need read/write permissions before executing bash commands.

______________________________________________________________________

## SimpleFormatter

Simple formatter implementation using Rich for terminal output. All output methods respect the `DEFAULT_CONFIG.agent.verbose` setting - when verbose is False, no output is produced.

### Constructor

```python
SimpleFormatter()
```

### Methods

#### `format_message()`

```python
def format_message(self, message: dict[str, Any]) -> str
```

Format a single message as a string. Returns empty string if verbose mode is disabled.

**Parameters:**

- `message` (dict): Message dictionary with 'role' and 'content' keys.

**Returns:**

- `str`: Formatted message string, or empty string if verbose is False.

#### `format_messages()`

```python
def format_messages(self, messages: list[dict[str, Any]]) -> str
```

Format a list of messages as a string. Returns empty string if verbose mode is disabled.

**Parameters:**

- `messages` (list[dict]): List of message dictionaries.

**Returns:**

- `str`: Formatted messages string, or empty string if verbose is False.

#### `print_message()`

```python
def print_message(self, message: dict[str, Any]) -> None
```

Print a single message to the console with Rich formatting. No output if verbose mode is disabled.

**Parameters:**

- `message` (dict): Message dictionary with 'role' and 'content' keys.

#### `print_messages()`

```python
def print_messages(self, messages: list[dict[str, Any]]) -> None
```

Print a list of messages to the console. No output if verbose mode is disabled.

**Parameters:**

- `messages` (list[dict]): List of message dictionaries.

#### `print_status()`

```python
def print_status(self, message: str) -> None
```

Print a status message in green. No output if verbose mode is disabled.

**Parameters:**

- `message` (str): The status message to print.

#### `print_error()`

```python
def print_error(self, message: str) -> None
```

Print an error message in red to stderr. No output if verbose mode is disabled.

**Parameters:**

- `message` (str): The error message to print.

#### `print_warning()`

```python
def print_warning(self, message: str) -> None
```

Print a warning message in yellow. No output if verbose mode is disabled.

**Parameters:**

- `message` (str): The warning message to print.

#### `print_label_and_value()`

```python
def print_label_and_value(self, label: str, value: str) -> None
```

Print a label and value pair with distinct colors. No output if verbose mode is disabled.

**Parameters:**

- `label` (str): The label to print (displayed in cyan).
- `value` (str): The value to print (displayed in bold white).

______________________________________________________________________

______________________________________________________________________

## Configuration System

The KISS framework uses a Pydantic-based configuration system accessible through `DEFAULT_CONFIG`.

### Config Structure

```python
from kiss.core.config import DEFAULT_CONFIG

# Access configuration
DEFAULT_CONFIG.api_keys.openai_api_key = "your-key"
DEFAULT_CONFIG.agent.max_steps = 100
DEFAULT_CONFIG.agent.max_agent_budget = 10.0
DEFAULT_CONFIG.agent.global_budget = 200.0
DEFAULT_CONFIG.agent.verbose = True
DEFAULT_CONFIG.agent.use_web_search = True
```

### Configuration Sections

#### `api_keys`

- `openai_api_key` (str): OpenAI API key
- `anthropic_api_key` (str): Anthropic API key
- `google_api_key` (str): Google API key
- `together_api_key` (str): Together AI API key
- `openrouter_api_key` (str): OpenRouter API key
- `serper_api_key` (str): Serper API key for web search

#### `agent`

- `max_steps` (int): Maximum steps per agent run (default: 100)
- `max_agent_budget` (float): Maximum budget per agent in USD (default: 10.0)
- `global_budget` (float): Global budget limit in USD (default: 200.0)
- `verbose` (bool): Enable verbose output (default: True)
- `use_web_search` (bool): Enable web search tool (default: False)
- `model_name` (str): Default model name (default: "gpt-4o")
- `artifact_dir` (str): Directory for agent artifacts (default: auto-generated with timestamp)

#### `agent.kiss_coding_agent`

- `orchestrator_model_name` (str): Model for main orchestration (default: "claude-sonnet-4-5")
- `subtasker_model_name` (str): Model for executor sub-agents (default: "claude-opus-4-5")
- `refiner_model_name` (str): Model for prompt refinement (default: "claude-sonnet-4-5")
- `trials` (int): Retry attempts per task/subtask (default: 3)
- `max_steps` (int): Maximum steps per agent (default: 50)
- `max_budget` (float): Maximum total budget in USD (default: 100.0)

#### `docker`

- `default_image` (str): Default Docker image (default: "ubuntu:latest")
- `default_workdir` (str): Default working directory in container (default: "/workspace")

#### `gepa`, `kiss_evolve`, `agent_creator`

Configuration sections for evolutionary optimization systems. See respective classes for details.

______________________________________________________________________

## Pre-built Agents

Ready-to-use agent functions from `kiss.agents.kiss`.

### `refine_prompt_template()`

```python
def refine_prompt_template(
    original_prompt_template: str,
    previous_prompt_template: str,
    agent_trajectory: str,
    model_name: str,
) -> str
```

Refine the prompt template based on the agent's trajectory.

**Parameters:**

- `original_prompt_template` (str): The original prompt template.
- `previous_prompt_template` (str): The previous version of the prompt template that led to the given trajectory.
- `agent_trajectory` (str): The agent's trajectory as a string.
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
