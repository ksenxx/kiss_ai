# KISS Framework API Reference

**Keep It Simple, Stupid** - Comprehensive API documentation for the KISS AI agent framework.

## Introduction

The KISS framework provides a clean, simple API for building AI agents with native function calling, multi-agent orchestration, and evolutionary optimization. This document covers all public classes, methods, and utilities available in the framework.

For a high-level overview and quick start guide, see [README.md](README.md).

## Table of Contents

- [KISSAgent](#kissagent) - Core agent class with function calling
- [RelentlessCodingAgent](#relentlesscodingagent) - Relentless multi-agent coding system
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
- [UsefulTools](#usefultools) - Bash execution with path-based access control
- [Utility Functions](#utility-functions) - Helper functions
- [SimpleFormatter](#simpleformatter) - Rich terminal output formatting
- [CompactFormatter](#compactformatter) - Compact single-line output formatting
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
    formatter: Formatter | None = None,
    is_agentic: bool = True,
    max_steps: int | None = None,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    token_callback: TokenCallback | None = None,
) -> str
```

Runs the agent's main ReAct loop to solve the task.

**Parameters:**

- `model_name` (str): The name of the model to use (e.g., "gpt-4o", "claude-sonnet-4-5", "gemini-2.5-flash", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "openrouter/anthropic/claude-3.5-sonnet")
- `prompt_template` (str): The prompt template for the agent. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `tools` (list\[Callable[..., Any]\] | None): List of callable functions the agent can use. The `finish` tool is automatically added if it is not provided in `tools`. If `use_web` is enabled in config, `search_web` and `fetch_url` is also automatically added. Default is None.
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.
- `is_agentic` (bool): If True, runs in agentic mode with tools. If False, returns raw LLM response. Default is True.
- `max_steps` (int | None): Maximum number of ReAct loop iterations. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`, which is 100).
- `max_budget` (float | None): Maximum budget in USD for this agent run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`, which is 10.0).
- `model_config` (dict[str, Any] | None): Optional model configuration to pass to the model. Default is None.
- `token_callback` (TokenCallback | None): Optional async callback invoked with each streamed text token. When set, model responses are streamed and each text delta is passed to this callback. Tool execution output is also streamed through this callback. The callback signature is `async def callback(token: str) -> None`. Default is None.

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

## RelentlessCodingAgent

A relentless coding agent that uses a multi-agent architecture to solve tasks. It implements a multi-agent architecture with:

- **Orchestrator Agent**: Manages overall task execution and keeps steps below configured limits
- **Executor Sub-agents**: Handle specific sub-tasks efficiently
- **Token Optimization**: Uses smaller models for simple tasks

The agent continues attempting tasks through multiple trials until success or exhaustion of retry attempts.

### Constructor

```python
RelentlessCodingAgent(name: str)
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
    trials: int | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    work_dir: str | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    docker_image: str | None = None,
    formatter: Formatter | None = None,
    token_callback: TokenCallback | None = None,
) -> str
```

Run the multi-agent coding system with orchestration and sub-task delegation.

**Parameters:**

- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `orchestrator_model_name` (str | None): Model for the main orchestrator agent. Default is None (uses config default: "claude-sonnet-4-5").
- `subtasker_model_name` (str | None): Model for executor agents handling sub-tasks. Default is None (uses config default: "claude-opus-4-6").
- `trials` (int | None): Number of continuation attempts for each task/subtask. Default is None (uses config default: 200).
- `max_steps` (int | None): Maximum number of steps per agent. Default is None (uses config default: 200).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses config default: 200.0).
- `work_dir` (str | None): The working directory for the agent's operations. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no paths are allowed for read access (except work_dir which is always added).
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no paths are allowed for write access (except work_dir which is always added).
- `docker_image` (str | None): Optional Docker image name to run bash commands in a container. If provided, all bash commands executed by sub-agents will run inside the Docker container instead of on the host. Example: "ubuntu:latest", "python:3.11-slim". Default is None (local execution).
- `formatter` (Formatter | None): Custom formatter for output. Default is `CompactFormatter`.
- `token_callback` (TokenCallback | None): Optional async callback invoked with each streamed text token. The callback is passed through to the underlying KISSAgent instances for both orchestrator and executor sub-agents. Default is None.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

#### `perform_task()`

```python
def perform_task(self) -> str
```

Execute the main task using the orchestrator agent. The orchestrator can delegate work by calling `perform_subtask()` as needed.

**Returns:**

- `str`: A YAML-encoded dictionary with keys 'success' (boolean) and 'summary' (string).

**Raises:**

- `KISSError`: If the task fails after all continuation trials.

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

**Raises:**

- `KISSError`: If the subtask fails after all retry trials.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `orchestrator_model_name` (str): Model name for orchestrator agent.
- `subtasker_model_name` (str): Model name for executor agents handling sub-tasks.
- `task_description` (str): The formatted task description.
- `total_tokens_used` (int): Total tokens used across all agents in this run.
- `budget_used` (float): Total budget used across all agents in this run.
- `work_dir` (str): The working directory for the agent's operations.
- `base_dir` (str): The base directory for the agent's working files.
- `readable_paths` (list[Path]): List of paths the agent can read from.
- `writable_paths` (list[Path]): List of paths the agent can write to.
- `max_steps` (int): Maximum number of steps per agent.
- `max_budget` (float): Maximum total budget allowed for this run.
- `trials` (int): Number of continuation attempts for each task/subtask.
- `max_tokens` (int): Maximum context length across all models used.
- `docker_image` (str | None): The Docker image name if Docker execution is enabled.
- `docker_manager` (DockerManager | None): The active Docker manager instance during execution (None when not using Docker or outside of `run()`).
- `useful_tools` (UsefulTools): The UsefulTools instance used for bash/edit operations.
- `token_callback` (TokenCallback | None): The token callback passed to `run()`.

### Key Features

- **Multi-Agent Architecture**: Orchestrator (using `orchestrator_model_name`) delegates to executor agents (using `subtasker_model_name`)
- **Token-Aware Continuation**: Agents signal when 50% of tokens are used, providing a structured summary with "Work Done" and "Work to Do Next" sections for seamless task handoff
- **Cost-Saving Guidance**: Orchestrator is instructed to avoid unnecessary `perform_subtask()` calls when it can handle work directly, reducing cost
- **Relentless Retries**: Continues attempting tasks through multiple continuation attempts until success
- **Efficient Orchestration**: Manages execution to stay within configured step limits through smart delegation
- **Bash Command Parsing**: Automatically extracts readable/writable paths from commands using `parse_bash_command_paths()`
- **Path Access Control**: Enforces read/write permissions on file system paths before command execution
- **Docker Support**: Optional Docker container execution for bash commands via the `docker_image` parameter. When enabled, all bash commands from sub-agents run inside an isolated Docker container.
- **Built-in Tools**:
  - Orchestrator agent has access to `finish()` and `perform_subtask()`
  - Executor agents have access to `finish()`, `Bash` (or Docker bash when `docker_image` is set), `Edit`, and `MultiEdit`

### Example

```python
from kiss.agents.coding_agents import RelentlessCodingAgent

agent = RelentlessCodingAgent("My Relentless Agent")

result = agent.run(
    prompt_template="""
        Write, test, and optimize a fibonacci function in Python
        that is efficient and correct. Save it to fibonacci.py.
    """,
    orchestrator_model_name="claude-sonnet-4-5",
    subtasker_model_name="claude-opus-4-5",
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
from kiss.agents.coding_agents import RelentlessCodingAgent

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
    formatter: Formatter | None = None,
    token_callback: TokenCallback | None = None,
) -> str
```

Run the multi-agent coding system with orchestration and sub-task delegation.

**Parameters:**

- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `orchestrator_model_name` (str | None): Model for the main orchestrator agent. Default is None (uses config default: "claude-sonnet-4-5").
- `subtasker_model_name` (str | None): Model for executor agents handling sub-tasks. Default is None (uses config default: "claude-opus-4-6").
- `refiner_model_name` (str | None): Model for dynamic prompt refinement when tasks fail. Default is None (uses config default: "claude-sonnet-4-5").
- `trials` (int | None): Number of retry attempts for each task/subtask. Default is None (uses config default: 200).
- `max_steps` (int | None): Maximum number of steps per agent. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`, which is 100).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`, which is 10.0).
- `work_dir` (str | None): The working directory for the agent's operations. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/kiss_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no paths are allowed for read access (except work_dir which is always added).
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no paths are allowed for write access (except work_dir which is always added).
- `docker_image` (str | None): Optional Docker image name to run bash commands in a container. If provided, all bash commands executed by sub-agents will run inside the Docker container instead of on the host. Example: "ubuntu:latest", "python:3.11-slim". Default is None (local execution).
- `formatter` (Formatter | None): Custom formatter for output. Default is `CompactFormatter`.
- `token_callback` (TokenCallback | None): Optional async callback invoked with each streamed text token. The callback is passed through to the underlying KISSAgent instances for both orchestrator and executor sub-agents. Default is None.

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
- `token_callback` (TokenCallback | None): The token callback passed to `run()`.

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
    model_name: str,
    prompt_template: str,
    arguments: dict[str, str] | None = None,
    max_steps: int | None = None,
    max_budget: float | None = None,
    base_dir: str | None = None,
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
    token_callback: TokenCallback | None = None,
) -> str | None
```

Run the Claude coding agent for a given task.

**Parameters:**

- `model_name` (str): The name of the model to use (e.g., "claude-sonnet-4-5").
- `prompt_template` (str): The prompt template for the task. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `max_steps` (int | None): Maximum number of steps. Default is None (uses `DEFAULT_CONFIG.agent.max_steps`).
- `max_budget` (float | None): Maximum budget in USD for this run. Default is None (uses `DEFAULT_CONFIG.agent.max_agent_budget`).
- `base_dir` (str | None): The base directory relative to which readable and writable paths are resolved if they are not absolute. Default is None (uses `{artifact_dir}/claude_workdir`).
- `readable_paths` (list[str] | None): The paths from which the agent is allowed to read. If None, no paths are allowed for Read/Grep/Glob.
- `writable_paths` (list[str] | None): The paths to which the agent is allowed to write. If None, no paths are allowed for Write/Edit/MultiEdit.
- `token_callback` (TokenCallback | None): Optional async callback invoked with assistant text, tool result text, and the final result. Default is None.

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
from kiss.agents.coding_agents import ClaudeCodingAgent

agent = ClaudeCodingAgent("My Agent")
result = agent.run(
    model_name="claude-sonnet-4-5",
    prompt_template="Write a fibonacci function with tests"
)
if result:
    print(f"Result: {result}")
```

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
    formatter: Formatter | None = None,
    token_callback: TokenCallback | None = None,
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
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.
- `token_callback` (TokenCallback | None): Optional async callback invoked with text content and tool response text from ADK events. Default is None.

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
    formatter: Formatter | None = None,
    token_callback: TokenCallback | None = None,
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
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.
- `token_callback` (TokenCallback | None): Optional async callback invoked with message text and tool output text from the Agents SDK result. Default is None.

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

**Note:** AgentEvolver uses RelentlessCodingAgent internally for agent improvement. Evaluation is done internally by loading and running the generated `agent.py` which must implement `agent_run(task: str) -> dict[str, Any]`.

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

A class that provides bash command execution with path-based access control and security checks.

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

## CompactFormatter

Compact formatter that parses model message content into structured parts (thought, tool action, usage info) and renders them concisely. This formatter is used by default in `RelentlessCodingAgent` and `KISSCodingAgent` for cleaner logs during multi-agent orchestration. All output methods respect the `DEFAULT_CONFIG.agent.verbose` setting.

**Key internal features:**

- **Markdown stripping**: Thought text is converted from markdown to plain text using `markdown-it-py`, collapsing multiple spaces into one via `_strip_markdown()`.
- **AST-based tool call parsing**: Tool calls in ```` ```python ```` fenced code blocks are parsed with Python's `ast` module via `_parse_tool_desc()`. If the call has a `description` keyword argument, that string is used as the action label; otherwise a compact `func(key=value, ...)` summary is generated with values truncated to `LINE_LENGTH` (160 characters).
- **Content splitting**: `_extract_parts()` splits message content into three parts: the thought (text before the tool call code block), the tool action description, and usage information (text matching `#### Usage Information`).
- **Consolidated `_print()` helper**: Status, error, and warning printing are unified through a single `_print(message, style, stderr)` method that dispatches to the appropriate Rich console or plain stdout/stderr.

### Constructor

```python
CompactFormatter()
```

### Methods

#### `format_message()`

```python
def format_message(self, message: dict[str, Any]) -> str
```

Parse and format a single message. The content is split into three parts: the thought (text before the tool call code block, with markdown stripped), the tool action description (from the `description` parameter of the tool call, or a generated summary), and usage information. Returns empty string if verbose mode is disabled.

**Parameters:**

- `message` (dict): Message dictionary with 'role' and 'content' keys.

**Returns:**

- `str`: Multi-line formatted output with `[role]: thought...`, `[action]:description`, and usage info. Returns `[role]: (empty)` if content is empty, or empty string if verbose is False.

#### `format_messages()`

```python
def format_messages(self, messages: list[dict[str, Any]]) -> str
```

Format a list of messages by applying `format_message()` to each. Returns empty string if verbose mode is disabled.

**Parameters:**

- `messages` (list[dict]): List of message dictionaries.

**Returns:**

- `str`: Formatted messages string with each message's output joined by newlines, or empty string if verbose is False.

#### `print_message()`

```python
def print_message(self, message: dict[str, Any]) -> None
```

Print a single formatted message. No output if verbose mode is disabled.

**Parameters:**

- `message` (dict): Message dictionary with 'role' and 'content' keys.

#### `print_messages()`

```python
def print_messages(self, messages: list[dict[str, Any]]) -> None
```

Print a list of formatted messages. No output if verbose mode is disabled.

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

- `label` (str): The label to print.
- `value` (str): The value to print.

______________________________________________________________________

## Token Streaming Callback

The KISS framework supports real-time token streaming through an async callback mechanism. When a `token_callback` is provided, model responses are streamed token-by-token instead of waiting for the full response. Tool execution output is also streamed through the same callback.

### TokenCallback Type

```python
from kiss.core.models.model import TokenCallback

# Type alias:
TokenCallback = Callable[[str], Coroutine[Any, Any, None]]
```

A `TokenCallback` is an async function that receives a single string token and returns nothing. It is invoked synchronously from the model layer using an internal event loop bridge.

### Streaming Behavior by Provider

- **Anthropic (Claude)**: Uses `messages.stream()` with `text_stream` for both `generate()` and `generate_and_process_with_tools()`.
- **OpenAI / Together AI / OpenRouter**: Uses `chat.completions.create(stream=True)` with `stream_options={"include_usage": True}` to preserve token counts. For tool calls, streaming accumulates tool-call deltas and reconstructs the full call.
- **Gemini**: Uses `generate_content_stream()` for `generate()` and `generate_content_stream()` with part-level parsing for `generate_and_process_with_tools()`.

When `token_callback` is `None`, all providers fall back to their original non-streaming code paths with no behavioral change.

### KISSAgent Integration

When passed to `KISSAgent.run()`, the callback receives:

1. **Model response tokens** as they are generated.
1. **Tool execution output** after each tool call completes (including error messages from failed tool calls).

### Coding Agent Integration

All coding agents support `token_callback` in their `run()` methods:

- **KISSCodingAgent / RelentlessCodingAgent**: The callback is passed through to the underlying `KISSAgent.run()` calls for both orchestrator and executor sub-agents. Streamed content includes model response tokens and tool output from all sub-agents.
- **ClaudeCodingAgent**: The callback receives assistant thought text, tool result text, and the final result message as they are processed from the Claude Agent SDK message stream.
- **GeminiCliAgent**: The callback receives text content and tool response text from ADK events after all events are collected and processed.
- **OpenAICodexAgent**: The callback receives message text and tool output text from the Agents SDK run result after processing.

### Example

```python
import asyncio
from kiss.core.kiss_agent import KISSAgent

# Collect tokens into a list
tokens: list[str] = []

async def my_callback(token: str) -> None:
    tokens.append(token)
    print(token, end="", flush=True)  # Print tokens as they arrive

agent = KISSAgent("streaming-agent")
result = agent.run(
    model_name="gpt-4o",
    prompt_template="Write a haiku about programming.",
    is_agentic=False,
    token_callback=my_callback,
)

# tokens now contains each streamed text delta
print(f"\nTotal tokens received: {len(tokens)}")
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
- `artifact_dir` (str): Directory for agent artifacts (default: auto-generated with timestamp)

#### `agent.relentless_coding_agent`

- `orchestrator_model_name` (str): Model for orchestration (default: "claude-sonnet-4-5")
- `subtasker_model_name` (str): Model for subtask execution (default: "claude-opus-4-6")
- `max_steps` (int): Maximum steps for the Relentless Coding Agent (default: 200)
- `max_budget` (float): Maximum budget in USD for the Relentless Coding Agent (default: 200.0)
- `trials` (int): Continuation attempts per task/subtask (default: 200)

#### `agent.kiss_coding_agent`

- `orchestrator_model_name` (str): Model for main orchestration (default: "claude-sonnet-4-5")
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
