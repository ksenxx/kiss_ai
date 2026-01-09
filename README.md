# KISS Agent Framework (Keep it Simple, Stupid Agent Framework)

A simple and portable AI agent framework for building and evolving LLM agents. The framework follows the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle) and provides native function calling for seamless tool integration.

**Version:** 0.1.1 (see `kiss.__version__` or `src/kiss/_version.py`)  
**Description:** KISS Agent Framework - A simple and portable agent framework for building and evolving AI agents  
**Python:** >=3.13

## Overview

KISS is a lightweight agent framework that implements a ReAct (Reasoning and Acting) loop for LLM agents. The framework provides:

- **Simple Architecture**: Clean, minimal core that's easy to understand and extend
- **GEPA Integration**: Genetic-Pareto prompt optimization for compound AI systems
- **KISSEvolve Integration**: Evolutionary algorithm discovery framework with LLM-guided mutation and crossover
- **Model Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Gemini, Together AI, OpenRouter)
- **Native Function Calling**: Seamless tool integration using native function calling APIs (OpenAI, Anthropic, Gemini, Together AI, and OpenRouter)
- **Docker Integration**: Built-in Docker manager for running agents in isolated environments
- **Trajectory Tracking**: Automatic saving of agent execution trajectories
- **Token Usage Tracking**: Built-in token usage tracking with automatic context length detection and step counting
- **Budget Tracking**: Automatic cost tracking and budget monitoring across all agent runs
- **Self-Evolution**: Framework for agents to evolve and refine their prompts
- **SWE-bench Dataset Support**: Built-in support for downloading and working with SWE-bench Verified dataset
- **RAG Support**: Simple retrieval-augmented generation system with in-memory vector store
- **Useful Agents**: Pre-built utility agents including prompt refinement and general bash execution agents
- **Multiprocessing Support**: Utilities for parallel execution of functions using multiprocessing
- **Trajectory Visualization**: Web-based visualizer for viewing agent execution trajectories with modern UI

## Architecture

The framework centers around the `KISSAgent` class, which implements a ReAct agent using native function calling of LLMs:

1. **Prompt Setup**: Agent receives a prompt template (no response format needed)
2. **Model Generation**: LLM generates a response with native function calls
3. **Function Call Extraction**: Framework extracts function calls directly from the API response
4. **Tool Execution**: Tools are executed and results are returned via the model's native function response format
5. **Loop**: Process repeats until the agent calls `finish()` or max steps are reached

### Key Components

- **`KISSAgent`**: Main agent class with native function calling support
  - Supports both agentic mode (with tools) and non-agentic mode (simple LLM calls)
  - Model name is passed to the `run()` method
  - Tracks token usage throughout the run and appends token information to messages
- **`Model`**: Abstract base class for LLM providers (located in `core/models/`)
  - **`Gemini3Model`**: Gemini model implementation with native function calling
  - **`OpenAIModel`**: OpenAI model implementation with native function calling
  - **`AnthropicModel`**: Anthropic Claude model implementation with native function calling
  - **`TogetherModel`**: Together AI model implementation with native function calling (supports Llama, Qwen, DeepSeek, Mistral, and more)
  - **`OpenRouterModel`**: OpenRouter model implementation with native function calling (unified API for 400+ models from multiple providers)
  - Automatic context length detection for supported models (OpenAI, Anthropic, Gemini, Together AI, OpenRouter)
  - Token count extraction from API responses
  - Embedding support: OpenAI and Together AI models support embeddings via `get_embedding()` method (Anthropic and Gemini do not support embeddings)
- **`Formatter`**: Abstract base class for output formatting and status messages
  - **`SimpleFormatter`**: Rich-formatted output with colored status messages
- **`DockerManager`**: Manages Docker containers for isolated execution
  - Supports optional shared volume mounting (configurable via `mount_shared_volume` parameter)
  - Set `mount_shared_volume=False` for images with pre-existing content (e.g., SWE-bench)
- **`Config`**: Centralized configuration management
- **`multiprocess`**: Utilities for parallel execution of functions using ProcessPoolExecutor
- **`SimpleRAG`**: Simple retrieval-augmented generation system with in-memory vector store and cosine/L2 similarity search

## Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone/download KISS and navigate to the directory
cd kiss

# Create virtual environment
uv venv --python 3.13

# Install dependencies (including dev tools)
uv sync --group dev

# (Optional) activate the venv for convenience (uv run works without activation)
source .venv/bin/activate

# Set up API keys (optional, for LLM providers)
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export TOGETHER_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"
```

## Quick Start

### KISSAgent API Reference

The `KISSAgent` class is the core component of the framework, providing native function calling support for LLM agents.

#### Constructor

```python
KISSAgent(name: str)
```

**Parameters:**
- `name` (str): The name of the agent. Used for identification and artifact naming.

#### Methods

##### `run()`

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
    max_budget: float = 5.0,
) -> str
```

Runs the agent's main ReAct loop to solve the task.

**Parameters:**
- `model_name` (str): The name of the model to use (e.g., "gpt-4o", "claude-sonnet-4-5", "gemini-3-pro-preview", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "openrouter/anthropic/claude-3.5-sonnet")
- `prompt_template` (str): The prompt template for the agent. Can include `{placeholder}` syntax for variable substitution.
- `arguments` (dict[str, str] | None): Arguments to substitute into the prompt template. Default is None.
- `tools` (list[Callable[..., Any]] | None): List of callable functions the agent can use. The `finish` tool is automatically added if it is not provided in `tools`. Default is None.
- `formatter` (Formatter | None): Custom formatter for output. Default is `SimpleFormatter`.
- `is_agentic` (bool): If True, runs in agentic mode with tools. If False, returns raw LLM response. Default is True.
- `max_steps` (int): Maximum number of ReAct loop iterations. Default is 100.
  - `max_budget` (float): Maximum budget in USD for this agent run. Default is 1.0.

**Returns:**
- `str`: The result returned by the agent's `finish()` call, or the raw LLM response in non-agentic mode.

**Raises:**
- `KISSError`: If budget exceeded, max steps exceeded, or tools provided in non-agentic mode.

##### `get_trajectory()`

```python
def get_trajectory(self) -> str
```

Returns the agent's conversation trajectory as a JSON string.

**Returns:**
- `str`: JSON-formatted string containing the list of messages with roles and content.

##### `finish()`

```python
def finish(self, result: str) -> str
```

Built-in tool that the agent must call to complete its task and return the final answer.

**Parameters:**
- `result` (str): The final result/answer from the agent.

**Returns:**
- `str`: The same result string passed in.

#### Class Attributes

- `global_budget_used` (float): Total budget used across all agent instances.
- `agent_counter` (int): Counter for unique agent IDs.
- `artifact_subdir` (str): Subdirectory name for saving artifacts.

#### Instance Attributes (after `run()`)

- `id` (int): Unique identifier for this agent instance.
- `name` (str): The agent's name.
- `model`: The model instance being used.
- `step_count` (int): Current step number in the ReAct loop.
- `total_tokens_used` (int): Total tokens used in this run.
- `budget_used` (float): Budget used in this run.
- `message_ids_of_trajectory` (list[int]): List of message IDs in the trajectory.

#### Tool Definition

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

#### Complete Example

```python
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError

# Define tools with type hints and docstrings
def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query string
    
    Returns:
        Search results as a string
    """
    # Simulated search
    return f"Results for '{query}': Found 10 relevant articles..."

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A Python math expression to evaluate
    
    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        raise KISSError(f"Calculation error: {e}") from e

# Create the agent
agent = KISSAgent(name="Research Assistant")

# Define the prompt with placeholders
prompt_template = """
You are a research assistant. Help the user with their question.

User Question: {question}

Use the available tools to find information and perform calculations as needed.
When you have the final answer, call the finish function with the result.
"""

# Run the agent
result = agent.run(
    model_name="gpt-4o",
    prompt_template=prompt_template,
    arguments={"question": "What is 15% of 847?"},
    tools=[search_web, calculate],
    max_steps=10,
    max_budget=0.5,
)

print(f"Result: {result}")

# Access trajectory for debugging/logging
trajectory = agent.get_trajectory()
print(f"Steps taken: {agent.step_count}")
print(f"Tokens used: {agent.total_tokens_used}")
print(f"Budget used: ${agent.budget_used:.4f}")
```

### Using GEPA for Prompt Optimization

> 📖 **For detailed GEPA documentation, see [GEPA README](src/kiss/agents/gepa/README.md)**

GEPA (Genetic-Pareto) is a prompt optimization framework that uses natural language reflection to evolve prompts. It maintains a Pareto frontier of top-performing prompts and combines complementary lessons through evolutionary search. GEPA is based on the paper ["GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPERFORM REINFORCEMENT LEARNING"](https://arxiv.org/pdf/2507.19457).

For usage examples, API reference, and configuration options, please see the [GEPA README](src/kiss/agents/gepa/README.md).

### Using KISSEvolve for Algorithm Discovery

> 📖 **For detailed KISSEvolve documentation, see [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md)**

KISSEvolve is an evolutionary algorithm discovery framework that uses LLM-guided mutation and crossover to evolve code variants. It supports advanced features including island-based evolution, novelty rejection sampling, and multiple parent sampling methods.

For usage examples, API reference, and configuration options, please see the [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md).

### Using Self-Evolving Multi-Agent

> 📖 **For detailed Self-Evolving Multi-Agent documentation, see [Self-Evolving Multi-Agent README](src/kiss/agents/self_evolving_multi_agent/README.md)**

The Self-Evolving Multi-Agent is an advanced agent with planning, error recovery, dynamic tool creation, and the ability to evolve itself for better efficiency and accuracy.

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
    2. Saves them to 'fibonacci.txt'
    3. Reads the file and prints the sum
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

For usage examples, API reference, and configuration options, please see the [Self-Evolving Multi-Agent README](src/kiss/agents/self_evolving_multi_agent/README.md).

### Running Agent Examples

**Vulnerability Detector Agent (ARVO):**
```bash
uv run python -m kiss.agents.arvo_agent.arvo_agent
```

The ARVO Vulnerability Detector agent uses the Arvo fuzzing framework to discover security vulnerabilities in C/C++ code:
- Runs in a Docker container with Arvo fuzzing framework
- Analyzes code to create hypotheses for potential vulnerabilities
- Generates Python scripts to create test inputs for fuzzing
- Detects ASAN crashes to identify security vulnerabilities
- Automatically refines prompts when vulnerabilities are not found

**Programmatic Usage:**
```python
from kiss.agents.arvo_agent.arvo_agent import find_vulnerability, get_all_arvo_tags

# Get available Arvo Docker image tags
tags = get_all_arvo_tags("n132/arvo")

# Find vulnerabilities in a specific Docker image
result = find_vulnerability(
    model_name="gemini-3-flash-preview",
    image_name="n132/arvo:tag-name",
    num_trials=10,  # Number of attempts to find a vulnerability
    location="/src"  # Location of the source code in the container
)

if result:
    print(f"Vulnerability found! POC script: {result}")
else:
    print("No vulnerability found after all trials")
```

**SWE-bench Verified Agent:**
```bash
uv run src/kiss/agents/swe_agent_verified/run_swebench.py --swebench_verified.model gemini-3-flash-preview --swebench_verified.instance_id "django__django-11099"
```

The SWE-bench Verified agent is a Software Engineering agent that:
- Runs in pre-built SWE-bench Docker containers with repositories pre-installed
- Executes bash commands to solve real-world GitHub issues
- Can read, edit, and create files in the `/testbed` directory
- Follows a structured workflow for issue resolution
- Automatically evaluates results using the official SWE-bench evaluation harness
- Supports command-line configuration for model, instance selection, budget, and more

See the [SWE-bench Verified README](src/kiss/agents/swe_agent_verified/README.md) for detailed documentation.

**KISSEvolve Bubble Sort Example:**
```bash
uv run python -m kiss.scripts.kissevolve_bubblesort
```

This script demonstrates KISSEvolve by evolving a bubble sort algorithm (O(n²)) to discover faster sorting algorithms like quicksort or mergesort (O(n log n)). It:
- Starts with a bubble sort implementation
- Uses KISSEvolve with LLM-guided mutation and crossover
- Evaluates correctness and performance of evolved variants
- Reports complexity analysis and performance improvements

**AlgoTune Benchmark Experiments:**
```bash
uv run python -m kiss.agents.kiss_evolve.algotune.run_algotune
```

This module runs KISSEvolve on AlgoTune benchmarks to optimize numerical programs. It:
- Automatically clones and installs the AlgoTune repository
- Implements multiple AlgoTune benchmark tasks (PCA, matrix multiplication, sorting, SVM, etc.)
- Uses KISSEvolve to evolve code for better performance
- Measures speedup compared to reference implementations
- Supports running all benchmarks or specific tasks via configuration
- Saves results in JSON format for analysis in the artifacts directory



### Using SimpleRAG for Retrieval-Augmented Generation

SimpleRAG provides a lightweight RAG system with in-memory vector storage and similarity search:

> **Note**: SimpleRAG requires a model with embedding support. Currently, OpenAI and Together AI models support embeddings. Anthropic and Gemini models do not provide embedding APIs.

```python
from kiss.rag import SimpleRAG

# Initialize RAG system with a model name that supports embeddings
rag = SimpleRAG(model_name="gpt-4o", metric="cosine")  # or "l2" for L2 distance

# Add documents
documents = [
    {
        "id": "1",
        "text": "Python is a programming language known for its simplicity.",
        "metadata": {"topic": "programming", "language": "Python"},
    },
    {
        "id": "2",
        "text": "Machine learning uses algorithms to learn from data.",
        "metadata": {"topic": "ML", "field": "AI"},
    },
    {
        "id": "3",
        "text": "Docker containers provide isolated execution environments.",
        "metadata": {"topic": "devops", "tool": "Docker"},
    },
]
rag.add_documents(documents)

# Query similar documents
results = rag.query("What is Python?", top_k=2)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Metadata: {result['metadata']}")
    print()

# Query with filter
def filter_by_topic(doc: dict) -> bool:
    return doc.get("metadata", {}).get("topic") == "programming"

filtered_results = rag.query("programming language", top_k=5, filter_fn=filter_by_topic)

# Get collection statistics
stats = rag.get_collection_stats()
print(f"Documents: {stats['num_documents']}, Embedding dim: {stats['embedding_dimension']}")

# Delete documents
rag.delete_documents(["1", "2"])

# Get a specific document
doc = rag.get_document("3")

# Clear all documents
rag.clear_collection()
```

### Using Useful Agents

The framework includes pre-built utility agents for common tasks:

**Prompt Refinement Agent:**
```python
import json

from kiss.agents.kiss import refine_prompt_template

refined_prompt = refine_prompt_template(
    original_prompt_template="Original prompt...",
    previous_prompt_template="Previous version...",
    agent_trajectory=json.dumps(agent.get_trajectory()),
    model_name="gemini-3-pro-preview"
)
```

**General Bash Agent:**
```python
from kiss.agents.kiss import run_bash_task_in_sandboxed_ubuntu_latest

result = run_bash_task_in_sandboxed_ubuntu_latest(
    task="Install and configure nginx",
    model_name="gemini-3-pro-preview"
)
```

**Simple Coding Agent:**
```python
from kiss.agents.kiss import get_run_simple_coding_agent

def test_fn(code: str) -> bool:
    """Test if the generated code is correct."""
    try:
        namespace = {}
        exec(code, namespace)
        func = namespace.get('my_function')
        if not func:
            return False
        # Add your test logic here
        return func(42) == 84  # Example test
    except Exception:
        return False

prompt_template = """
Write a Python function called 'my_function' that doubles its input.
The function should be: def my_function(x): return x * 2
"""

# get_run_simple_coding_agent returns a function that can be used to run the agent
run_simple_coding_agent = get_run_simple_coding_agent(test_fn)
result = run_simple_coding_agent(
    prompt_template=prompt_template,
    arguments={},
    model_name="gemini-3-pro-preview"
)
print(result)
```

## Project Structure

```
kiss/
├── src/kiss/
│   ├── agents/          # Example agents
│   │   ├── gepa/                   # GEPA (Genetic-Pareto) prompt optimizer
│   │   │   ├── gepa.py
│   │   │   ├── config.py           # GEPA configuration
│   │   │   └── README.md           # GEPA documentation
│   │   ├── kiss_evolve/            # KISSEvolve evolutionary algorithm discovery
│   │   │   ├── kiss_evolve.py
│   │   │   ├── novelty_prompts.py  # Prompts for novelty-based evolution
│   │   │   ├── config.py           # KISSEvolve configuration
│   │   │   ├── README.md           # KISSEvolve documentation
│   │   │   └── algotune/           # AlgoTune benchmark integration
│   │   │       ├── run_algotune.py # AlgoTune task evolution
│   │   │       └── config.py       # AlgoTune configuration
│   │   ├── kiss.py                 # Utility agents (prompt refiner, bash agent)
│   │   ├── swe_agent_verified/     # SWE-bench Verified benchmark integration
│   │   │   ├── run_swebench.py     # Main runner with CLI support
│   │   │   ├── config.py           # Configuration for SWE-bench runs
│   │   │   └── README.md           # SWE-bench documentation
│   │   ├── self_evolving_multi_agent/  # Self-evolving multi-agent with planning
│   │   │   ├── multi_agent.py          # Main multi-agent implementation
│   │   │   ├── agent_evolver.py        # Agent evolution using KISSEvolve
│   │   │   └── config.py
│   │   └── arvo_agent/             # ARVO vulnerability detection agent
│   │       ├── arvo_agent.py       # Arvo-based vulnerability detector
│   │       └── arvo_tags.json      # Docker image tags for Arvo
│   ├── core/            # Core framework components
│   │   ├── kiss_agent.py      # KISS agent with native function calling
│   │   ├── formatter.py       # Output formatting base class
│   │   ├── simple_formatter.py # Rich-formatted output
│   │   ├── config.py          # Configuration
│   │   ├── config_builder.py  # Dynamic config builder with CLI support
│   │   ├── kiss_error.py      # Custom error class
│   │   ├── utils.py           # Utility functions
│   │   └── models/            # Model implementations
│   │       ├── model.py           # Model interface
│   │       ├── gemini3_model.py   # Gemini model implementation
│   │       ├── openai_model.py    # OpenAI model implementation
│   │       ├── openai_compatible_model.py # OpenAI-compatible API model
│   │       ├── anthropic_model.py # Anthropic model implementation
│   │       ├── together_model.py  # Together AI model implementation
│   │       ├── openrouter_model.py # OpenRouter model implementation
│   │       └── model_info.py      # Model info: context lengths, pricing, and capabilities
│   ├── docker/          # Docker integration
│   │   └── docker_manager.py
│   ├── multiprocessing/ # Multiprocessing utilities
│   │   └── multiprocess.py
│   ├── rag/             # RAG (Retrieval-Augmented Generation)
│   │   └── simple_rag.py # Simple RAG system with in-memory vector store
│   ├── scripts/         # Utility scripts
│   │   ├── check.py                    # Code quality check script
│   │   └── kissevolve_bubblesort.py    # KISSEvolve example: evolving bubble sort
│   ├── tests/           # Test suite
│   │   ├── test_kiss_agent_agentic.py
│   │   ├── test_kiss_agent_non_agentic.py
│   │   ├── test_kissevolve_bubblesort.py
│   │   ├── test_gepa_squad.py
│   │   ├── test_docker_manager.py
│   │   ├── test_models.py         # Tests for all models based on ModelInfo capabilities
│   │   ├── test_multiprocess.py
│   │   └── test_internal.py
│   └── viz_trajectory/  # Trajectory visualization
│       ├── server.py                    # Flask server for trajectory visualization
│       ├── README.md                    # Trajectory visualizer documentation
│       └── templates/                   # HTML templates for the visualizer
│           └── index.html
├── pyproject.toml       # Project configuration
└── README.md
```

## Versioning

The project uses semantic versioning (MAJOR.MINOR.PATCH). The version is defined in a single source of truth:

- **Version file**: `src/kiss/_version.py` - Edit this file to update the version
- **Package access**: `kiss.__version__` - Access the version programmatically
- **Build system**: `pyproject.toml` automatically reads the version from `_version.py` using dynamic versioning

Example:
```python
from kiss import __version__
print(f"KISS version: {__version__}")
```

To update the version, simply edit `src/kiss/_version.py`:
```python
__version__ = "0.2.0"  # Update to new version
```

## Configuration

Configuration is managed through environment variables and the `DEFAULT_CONFIG` object:

- **API Keys**: Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, and/or `OPENROUTER_API_KEY` environment variables
- **Agent Settings**: Modify `DEFAULT_CONFIG` in `src/kiss/core/config.py`:
  - `max_steps`: Maximum iterations in the ReAct loop (default: 100)
  - `verbose`: Enable verbose output (default: True)
  - `debug`: Enable debug mode (default: False)
  - `max_agent_budget`: Maximum budget per agent run in USD (default: 1.0)
  - `global_max_budget`: Maximum total budget across all agents in USD (default: 10.0)
  - `use_google_search`: Automatically add Google search tool if enabled (default: True)
- **GEPA Settings**: Modify `DEFAULT_CONFIG.gepa` in `src/kiss/agents/gepa/config.py`:
  - `reflection_model`: Model to use for reflection (default: "gemini-3-flash-preview")
  - `max_generations`: Maximum number of evolutionary generations (default: 10)
  - `population_size`: Number of candidates to maintain in population (default: 8)
  - `pareto_size`: Maximum size of Pareto frontier (default: 4)
  - `mutation_rate`: Probability of mutating a prompt template (default: 0.5)
  - `crossover_probability`: Probability of combining with lessons from Pareto frontier (default: 0.3)
  - `rollouts_per_generation`: Number of rollouts per generation (default: 1)
- **AlgoTune Settings**: Modify `DEFAULT_CONFIG.algotune` in `src/kiss/agents/kiss_evolve/algotune/config.py`:
  - `task`: Specific task name to solve (default: "svm")
  - `all_tasks`: Solve all tasks in AlgoTuneTasks directory (default: False)

## Available Commands

### Development

- `uv sync` - Install dependencies
- `uv sync --group dev` - Install dependencies including dev tools (mypy, ruff, pytest)
- `uv build` - Build the project package

### Testing

- `uv run pytest` - Run all tests (uses testpaths from pyproject.toml)
- `uv run pytest src/kiss/tests/ -v` - Run all tests with verbose output
- `uv run pytest src/kiss/tests/test_kiss_agent_agentic.py -v` - Run agentic agent tests
- `uv run pytest src/kiss/tests/test_kiss_agent_non_agentic.py -v` - Run non-agentic agent tests
- `uv run pytest src/kiss/tests/test_models.py -v` - Run model tests
- `uv run pytest src/kiss/tests/test_multiprocess.py -v` - Run multiprocessing tests
- `uv run python -m unittest src.kiss.tests.test_gepa_squad -v` - Run GEPA Squad tests (unittest)
- `uv run python -m unittest src.kiss.tests.test_docker_manager -v` - Run docker manager tests (unittest)
- `uv run python -m unittest discover -s src/kiss/tests -v` - Run all tests using unittest

### Code Quality

- `uv run check` - Run all code quality checks (fresh dependency install, build, lint, and type check)
- `uv run ruff format src/` - Format code with ruff (line-length: 100, target: py313)
- `uv run ruff check src/` - Lint code with ruff (selects: E, F, W, I, N, UP)
- `uv run mypy src/` - Type check with mypy (python_version: 3.13)

### Utilities

- `uv run python -m kiss.agents.arvo_agent.arvo_agent` - Run the ARVO Vulnerability Detector agent
- `uv run src/kiss/agents/swe_agent_verified/run_swebench.py` - Run the SWE-bench Verified agent
- `uv run python -m kiss.scripts.kissevolve_bubblesort` - Run the KISSEvolve bubble sort evolution example
- `uv run python -m kiss.viz_trajectory.server artifacts` - Start the trajectory visualizer server

### Cleanup

```bash
rm -rf build/ dist/ .pytest_cache .mypy_cache .ruff_cache && \
find . -type d -name __pycache__ -exec rm -r {} + && \
find . -type f -name "*.pyc" -delete
```

## Trajectory Saving and Visualization

Agent trajectories are automatically saved to the artifacts directory (default: `artifacts/`). Each trajectory includes:
- Complete message history with token usage and budget information appended to each message
- Tool calls and results
- Configuration used
- Timestamps
- Budget and token usage statistics

### Visualizing Trajectories

The framework includes a web-based trajectory visualizer for viewing agent execution histories:

```bash
# Run the visualizer server
uv run python -m kiss.viz_trajectory.server artifacts

# Or with custom host/port
uv run python -m kiss.viz_trajectory.server artifacts --host 127.0.0.1 --port 5050
```

Then open your browser to `http://127.0.0.1:5050` to view the trajectories.

The visualizer provides:
- **Modern UI**: Dark theme with smooth animations
- **Sidebar Navigation**: List of all trajectories sorted by start time
- **Markdown Rendering**: Full markdown support for message content
- **Code Highlighting**: Syntax highlighting for fenced code blocks
- **Message Display**: Clean, organized view of agent conversations
- **Metadata Display**: Shows agent ID, model, steps, tokens, and budget information

> 📖 **For detailed trajectory visualizer documentation, see [Trajectory Visualizer README](src/kiss/viz_trajectory/README.md)**


**Supported Models**: The framework includes context length, pricing, and capability flags for:

**Generation Models** (text generation with function calling support):
- **OpenAI**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-5.1, gpt-5.2
- **Anthropic**: claude-opus-4-5, claude-opus-4-1, claude-sonnet-4-5, claude-haiku-4-5
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3-pro-preview, gemini-3-flash-preview
- **Together AI (Llama)**: Llama-4-Scout/Maverick (with function calling), Llama-3.x series (generation only)
- **Together AI (Qwen)**: Qwen2.5-72B-Instruct-Turbo, Qwen3 series (with function calling)
- **Together AI (DeepSeek)**: DeepSeek-R1, DeepSeek-V3.1 (with function calling)
- **Together AI (Other)**: Kimi-K2-Instruct, GLM-4.5/4.6, Nemotron-Nano-9B
- **OpenRouter**: Access to 400+ models from multiple providers via unified API:
  - OpenAI (gpt-4.1, gpt-4o, gpt-5, gpt-5.1, gpt-5.2, o1, o3, o3-pro, o4-mini, codex-mini)
  - Anthropic (claude-3-haiku, claude-3.5-haiku, claude-3.5-sonnet, claude-3.7-sonnet, claude-sonnet-4/4.5, claude-haiku-4.5, claude-opus-4/4.1/4.5)
  - Google (gemini-2.0-flash, gemini-2.5-flash/pro, gemini-3-flash/pro-preview, gemma-3-27b)
  - Meta Llama (llama-3.3-70b, llama-4-maverick/scout)
  - DeepSeek (deepseek-chat, deepseek-r1, deepseek-v3.1/v3.2)
  - Qwen (qwen-2.5-72b, qwen-turbo/plus/max, qwen3-8b/14b/32b/235b, qwen3-coder, qwq-32b)
  - Amazon Nova (nova-micro/lite/pro, nova-2-lite, nova-premier)
  - Cohere (command-r, command-r-plus, command-a)
  - X.AI Grok (grok-3/3-mini, grok-4/4-fast, grok-4.1-fast)
  - MiniMax (minimax-m1, minimax-m2/m2.1)
  - ByteDance Seed (seed-1.6, seed-1.6-flash)
  - MoonshotAI (kimi-k2, kimi-k2-thinking)
  - Mistral (codestral, devstral, mistral-large/medium/small, mixtral)
  - And many more (nvidia, z-ai/glm, inception, arcee-ai, etc.)

**Embedding Models** (for RAG and semantic search):
- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Google**: text-embedding-004
- **Together AI**: BAAI/bge-large-en-v1.5, BAAI/bge-base-en-v1.5, m2-bert-80M-32k-retrieval, multilingual-e5-large-instruct, gte-modernbert-base

Each model in `MODEL_INFO` includes capability flags:
- `is_function_calling_supported`: Whether the model reliably supports tool/function calling
- `is_generation_supported`: Whether the model supports text generation
- `is_embedding_supported`: Whether the model is an embedding model

> **Note**: Additional models can be used, but context length, pricing, and capability information must be added to `src/kiss/core/models/model_info.py` for accurate token tracking, budget monitoring, and test filtering.

Token counts are extracted directly from API responses, ensuring accuracy and supporting multiple agents sharing the same model instance.

### Embedding Support

The framework provides embedding generation capabilities through the `get_embedding()` method on model instances:

- **OpenAI Models**: Full embedding support via OpenAI's embeddings API
  - Default model: `text-embedding-3-small` (can be customized)
  - Usage: `model.get_embedding(text, embedding_model="text-embedding-3-small")`
- **Together AI Models**: Full embedding support via Together AI's embeddings API
  - Default model: `togethercomputer/m2-bert-80M-8k-retrieval` (can be customized)
  - Usage: `model.get_embedding(text, embedding_model="togethercomputer/m2-bert-80M-8k-retrieval")`
- **Anthropic Models**: Embeddings not supported (raises `NotImplementedError`)
- **Gemini Models**: Embeddings not supported (raises `NotImplementedError`)

Embeddings are primarily used by the `SimpleRAG` system for document retrieval. When using `SimpleRAG`, ensure you use an OpenAI or Together AI model that supports embeddings.

## Contributing

Contributions are welcome! Please ensure your code:
- Follows the KISS principle
- Passes all tests (`uv run pytest`)
- Passes linting (`uv run ruff check src/`)
- Passes type checking (`uv run mypy src/`)

## License

Apache-2.0

## Authors

- Koushik Sen (ksen@berkeley.edu)
