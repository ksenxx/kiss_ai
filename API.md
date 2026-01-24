# KISS Framework API Reference

## Table of Contents

- [KISSAgent](#kissagent) - Core agent class
- [DockerManager](#dockermanager) - Docker container management
- [Multiprocessing](#multiprocessing) - Parallel execution utilities
- [SimpleRAG](#simplerag) - Simple RAG system for document retrieval
- [GEPA](#gepa) - Genetic-Pareto prompt optimizer
- [ClaudeCodingAgent](#claudecodingagent) - Claude Agent SDK-based coding agent
- [KISSEvolve](#kissevolve) - Evolutionary algorithm discovery
- [Utility Functions](#utility-functions) - Helper functions
- [SimpleFormatter](#simpleformatter) - Terminal output formatting
- [Pre-built Agents](#pre-built-agents) - Ready-to-use agents

---

## KISSAgent

The framework centers around the `KISSAgent` class, which implements a ReAct agent using native function calling of LLMs.

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
    max_budget: float = 1.0,
    model_config: dict[str, Any] | None = None,
) -> str
```

Runs the agent's main ReAct loop to solve the task.

**Parameters:**
- `model_name` (str): The name of the model to use (e.g., "gpt-4o", "claude-sonnet-4-5", "gemini-3-pro-preview", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "openrouter/anthropic/claude-3.5-sonnet")
- `prompt_template` (str): The prompt template for the agent. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `tools` (list[Callable[..., Any]] | None): List of callable functions the agent can use. The `finish` tool is automatically added if it is not provided in `tools`. If `use_google_search` is enabled in config, `search_web` is also automatically added. Default is None.
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

### Class Attributes

- `global_budget_used` (float): Total budget used across all agent instances.
- `agent_counter` (int): Counter for unique agent IDs.

### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model`: The model instance being used.
- `step_count` (int): Current step number in the ReAct loop.
- `total_tokens_used` (int): Total tokens used in this run.
- `budget_used` (float): Budget used in this run.
- `messages` (list[dict[str, Any]]): List of messages in the trajectory.

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

---

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

---

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
- `tasks` (list[tuple[Callable, list]]): List of tuples, where each tuple contains (function, arguments). Each function is a callable, and arguments is a list that can be unpacked with *args.

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

---

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

---

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

---

## ClaudeCodingAgent

A coding agent that uses the Claude Agent SDK to generate tested Python programs.

### Constructor

```python
ClaudeCodingAgent(name: str)
```

**Parameters:**
- `name` (str): Name of the agent.

### Methods

#### `run()`

```python
async def run(
    self,
    task: str,
    model_name: str = "claude-sonnet-4-5",
    base_dir: str = "<artifact_dir>/claude_workdir",
    readable_paths: list[str] | None = None,
    writable_paths: list[str] | None = None,
) -> dict[str, object] | None
```

Run the Claude coding agent for a given task.

**Parameters:**
- `task` (str): The task to run the Claude coding agent for.
- `model_name` (str): The name of the model to use. Default is "claude-sonnet-4-5".
- `base_dir` (str): The base directory to use for the agent.
- `readable_paths` (list[str] | None): The paths the agent can read from.
- `writable_paths` (list[str] | None): The paths the agent can write to.

**Returns:**
- `dict | None`: The result of the Claude coding agent's task containing:
  - `"success"` (bool): Whether the task was successful
  - `"result"` (str): YAML string with created/modified/deleted files and summary

#### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the trajectory of the agent in standard JSON format for visualization.

**Returns:**
- `str`: JSON-formatted trajectory string.

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
from kiss.agents.claudecodingagent import ClaudeCodingAgent

async def main():
    agent = ClaudeCodingAgent("My Agent")
    result = await agent.run(
        "Write a fibonacci function with tests",
        model_name="claude-sonnet-4-5"
    )
    if result:
        print(f"Success: {result['success']}")
        print(f"Result: {result['result']}")

anyio.run(main)
```

---

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
- `model_names` (list[tuple[str, float]]): List of (model_name, probability) tuples. Probabilities are normalized.
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

---

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

### `finish()`

```python
def finish(
    status: str = "success",
    analysis: str = "",
    result: str = "",
) -> str
```

The agent must call this function with the final status, analysis, and result when it has solved the given task.

**Parameters:**
- `status` (str): The status of the agent's task ('success' or 'failure'). Default is 'success'.
- `analysis` (str): The analysis of the agent's trajectory.
- `result` (str): The result generated by the agent.

**Returns:**
- `str`: A YAML string containing the status, analysis, and result.

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

---

## SimpleFormatter

Simple formatter implementation using Rich for terminal output. Implements the `Formatter` interface.

### Constructor

```python
SimpleFormatter()
```

### Methods

#### `format_message()`

```python
def format_message(self, message: dict[str, str]) -> str
```

Format a single message as a string.

**Parameters:**
- `message` (dict): Message dictionary with 'role' and 'content' keys.

**Returns:**
- `str`: Formatted message string.

#### `format_messages()`

```python
def format_messages(self, messages: list[dict[str, str]]) -> str
```

Format a list of messages as a string.

**Parameters:**
- `messages` (list[dict]): List of message dictionaries.

**Returns:**
- `str`: Formatted messages string.

#### `print_message()`

```python
def print_message(self, message: dict[str, str]) -> None
```

Print a single message to the console with Rich formatting.

**Parameters:**
- `message` (dict): Message dictionary with 'role' and 'content' keys.

#### `print_messages()`

```python
def print_messages(self, messages: list[dict[str, str]]) -> None
```

Print a list of messages to the console.

**Parameters:**
- `messages` (list[dict]): List of message dictionaries.

#### `print_status()`

```python
def print_status(self, message: str) -> None
```

Print a status message in green.

**Parameters:**
- `message` (str): The status message to print.

#### `print_error()`

```python
def print_error(self, message: str) -> None
```

Print an error message in red to stderr.

**Parameters:**
- `message` (str): The error message to print.

#### `print_warning()`

```python
def print_warning(self, message: str) -> None
```

Print a warning message in yellow.

**Parameters:**
- `message` (str): The warning message to print.

---

## Pre-built Agents

Ready-to-use agents from `kiss.agents.kiss`.

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
- `test_fn` (Callable[[str], bool]): The test function to use for the agent.

**Returns:**
- `Callable`: A function that runs a simple coding agent. Accepts keyword arguments: `model_name` (str), `prompt_template` (str), and `arguments` (dict[str, str]).
