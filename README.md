![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/818u234myu55pxt0wi7j.jpeg)

**Version:** 0.1.6

# When Simplicity Becomes Your Superpower: Meet KISS Agent Framework

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

______________________________________________________________________

## 🎯 The Problem with AI Agent Frameworks Today

Let's be honest. The AI agent ecosystem has become a jungle.

Every week brings a new framework promising to revolutionize how we build AI agents. They come loaded with abstractions on top of abstractions, configuration files that rival tax forms, and dependency trees that make `node_modules` look tidy. By the time you've figured out how to make your first tool call, you've already burned through half your patience and all your enthusiasm.

**What if there was another way?**

What if building AI agents could be as straightforward as the name suggests?

Enter **KISS** — the *Keep It Simple, Stupid* Agent Framework.

## 🚀 Your First Agent in 30 Seconds

Let me show you something beautiful:

```python
from kiss.core.kiss_agent import KISSAgent

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = KISSAgent(name="Math Buddy")
result = agent.run(
    model_name="gemini-3-flash-preview",
    prompt_template="Calculate: {question}",
    arguments={"question": "What is 15% of 847?"},
    tools=[calculate]
)
print(result)  # 127.05
```

That's a fully functional AI agent that uses tools. No boilerplate. No ceremony. Just intent, directly expressed.

KISS uses **native function calling** from the LLM providers. Your Python functions become tools automatically. Type hints become schemas. Docstrings become descriptions. Everything just works.

## Overview

KISS is a lightweight agent framework that implements a ReAct (Reasoning and Acting) loop for LLM agents. The framework provides:

- **Simple Architecture**: Clean, minimal core that's easy to understand and extend
- **GEPA Implementation From Scratch**: Genetic-Pareto prompt optimization for compound AI systems
- **KISSEvolve Implementation From Scratch**: Evolutionary algorithm discovery framework with LLM-guided mutation and crossover
- **Model Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Gemini, Together AI, OpenRouter)
- **Native Function Calling**: Seamless tool integration using native function calling APIs (OpenAI, Anthropic, Gemini, Together AI, and OpenRouter)
- **Docker Integration**: Built-in Docker manager for running agents in isolated environments
- **Trajectory Tracking**: Automatic saving of agent execution trajectories
- **Token Usage Tracking**: Built-in token usage tracking with automatic context length detection and step counting
- **Budget Tracking**: Automatic cost tracking and budget monitoring across all agent runs
- **Self-Evolution**: Framework for agents to evolve and refine other multi agents.
- **SWE-bench Dataset Support**: Built-in support for downloading and working with SWE-bench Verified dataset
- **RAG Support**: Simple retrieval-augmented generation system with in-memory vector store
- **Useful Agents**: Pre-built utility agents including prompt refinement and general bash execution agents
- **Multiprocessing Support**: Utilities for parallel execution of functions using multiprocessing
- **Trajectory Visualization**: Web-based visualizer for viewing agent execution trajectories with modern UI

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

## Output Formatting

Unlike other agentic systems, you do not need to specify the output schema for the agent. Just create
a suitable "finish" function with parameters. The parameters could be treated as the top level keys
in a json format.

### KISSAgent API Reference

> 📖 **For detailed KISSAgent API documentation, see [API.md](API.md)**

## 🤝 Multi-Agent Orchestration

Here's where KISS really shines — composing multiple agents into systems greater than the sum of their parts.

KISS includes utility agents that work beautifully together. Let's build a **self-improving coding agent** that writes code, tests it, and refines its own prompts based on failures:

```python
import json
from kiss.core.kiss_agent import KISSAgent
from kiss.agents.kiss import refine_prompt_template, get_run_simple_coding_agent

# Step 1: Define a test function for our coding task
def test_fibonacci(code: str) -> bool:
    """Test if the generated fibonacci code is correct."""
    try:
        namespace = {}
        exec(code, namespace)
        fib = namespace.get('fibonacci')
        if not fib:
            return False
        # Test cases
        return (fib(0) == 0 and fib(1) == 1 and 
                fib(10) == 55 and fib(20) == 6765)
    except Exception:
        return False

# Step 2: Create the coding agent
coding_agent_fn = get_run_simple_coding_agent(test_fibonacci)

# Step 3: Define our initial prompt
prompt_template = """
Write a Python function called 'fibonacci' that returns the nth Fibonacci number.
Requirements: {requirements}
"""

# Step 4: The self-improving loop
original_prompt = prompt_template
current_prompt = prompt_template
max_iterations = 3

for iteration in range(max_iterations):
    print(f"\n{'='*50}")
    print(f"Iteration {iteration + 1}")
    print(f"{'='*50}")
    
    # Run the coding agent
    coding_agent = KISSAgent(name=f"Coder-{iteration}")
    try:
        result = coding_agent_fn(
            prompt_template=current_prompt,
            arguments={"requirements": "Use recursion with memoization for efficiency"},
            model_name="gpt-4o"
        )
        print(f"✅ Code generated successfully!")
        print(f"Result: {result[:100]}...")
        break  # Success! Exit the loop
        
    except Exception as e:
        print(f"❌ Attempt failed: {e}")
        
        # Get the trajectory to understand what went wrong
        trajectory = coding_agent.get_trajectory()
        
        # Use the Prompt Refiner agent to improve our prompt
        print("🔄 Refining prompt based on failure...")
        current_prompt = refine_prompt_template(
            original_prompt_template=original_prompt,
            previous_prompt_template=current_prompt,
            agent_trajectory=trajectory,
            model_name="gpt-4o"
        )
        print(f"📝 New prompt:\n{current_prompt[:200]}...")
```

**What's happening here?**

1. **Coding Agent** ['get_run_simple_coding_agent'](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/agents/kiss.py): Generates code and validates it against test cases
1. **Prompt Refiner Agent** ['refine_prompt_template'](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/agents/kiss.py): Analyzes failures and evolves the prompt
1. **Orchestration**: A simple Python loop (not to be confused with the ReAct loop) coordinates the agents

No special orchestration framework needed. No message buses. No complex state machines. Just Python functions calling Python functions.

### Why This Matters

Most multi-agent frameworks require you to learn a new paradigm: graphs, workflows, channels, and supervisors. KISS takes a different approach: **agents are just functions**.

```python
# Agent 1: Research
research_result = research_agent.run(model_name="gpt-4o", more_args)

# Agent 2: Write (uses research)
draft = writer_agent.run(
    model_name="claude-sonnet-4-5",
    arguments={"research": research_result},
    # ...
)

# Agent 3: Edit (uses draft)
final = editor_agent.run(
    model_name="gemini-3-pro-preview", 
    arguments={"draft": draft},
    # ...
)
```

Each agent can use a different model. Each agent has its own budget. Each agent saves its own trajectory. And you compose them with the most powerful orchestration tool ever invented: **regular Python code**.

### Using Agent Creator and Optimizer

> 📖 **For detailed Agent Creator documentation, see [Agent Creator README](src/kiss/agents/agent_creator/README.md)**

The Agent Creator module provides tools to automatically evolve and optimize AI agents for **token efficiency** and **execution speed** using evolutionary algorithms with Pareto frontier maintenance.

**Key Components:**

- **ImproverAgent**: Optimizes existing agent code through iterative improvement
- **AgentEvolver**: Maintains a population of agent variants and evolves them using mutation and crossover operations

```python
import anyio
from kiss.agents.agent_creator import ImproverAgent, AgentEvolver

# Option 1: Improve an existing agent
async def improve_existing_agent():
    improver = ImproverAgent(
        model_name="claude-sonnet-4-5",
        max_steps=150,
        max_budget=15.0,
    )

    success, report = await improver.improve(
        source_folder="/path/to/agent",
        target_folder="/path/to/improved_agent",
    )

    if success and report:
        print(f"Improvement completed in {report.improved_time:.2f}s")
        print(f"Tokens used: {report.improved_tokens}")

# Option 2: Evolve a new agent from a task description
async def evolve_new_agent():
    evolver = AgentEvolver(
        task_description="Build a code analysis assistant that can parse and analyze large codebases",
        max_generations=10,
        max_frontier_size=6,
        mutation_probability=0.8,
    )

    best_variant = await evolver.evolve()

    print(f"Best agent: {best_variant.folder_path}")
    print(f"Tokens used: {best_variant.tokens_used}")
    print(f"Execution time: {best_variant.execution_time:.2f}s")

anyio.run(improve_existing_agent)
```

**Features:**

- **Multi-Objective Optimization**: Optimizes for both token usage and execution time
- **Pareto Frontier Maintenance**: Keeps track of all non-dominated solutions
- **Evolutionary Operations**: Supports mutation (improving one variant) and crossover (combining ideas from two variants)
- **Automatic Pruning**: Removes dominated variants to manage memory and storage

For usage examples, API reference, and configuration options, please see the [Agent Creator README](src/kiss/agents/agent_creator/README.md).

### Using GEPA for Prompt Optimization

> 📖 **For detailed GEPA documentation, see [GEPA README](src/kiss/agents/gepa/README.md)**

KISS has a fresh implementation of GEPA with some improvements. GEPA (Genetic-Pareto) is a prompt optimization framework that uses natural language reflection to evolve prompts. It maintains an instance-level Pareto frontier of top-performing prompts and combines complementary lessons through structural merge. GEPA is based on the paper ["GEPA: REFLECTIVE PROMPT EVOLUTION CAN OUTPERFORM REINFORCEMENT LEARNING"](https://arxiv.org/pdf/2507.19457).

For usage examples, API reference, and configuration options, please see the [GEPA README](src/kiss/agents/gepa/README.md).

### Using KISSEvolve for Algorithm Discovery

> 📖 **For detailed KISSEvolve documentation, see [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md)**

KISSEvolve is an evolutionary algorithm discovery framework that uses LLM-guided mutation and crossover to evolve code variants. It supports advanced features including island-based evolution, novelty rejection sampling, and multiple parent sampling methods.

For usage examples, API reference, and configuration options, please see the [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md).

### Using Self-Evolving Multi-Agent

> 📖 **For detailed Self-Evolving Multi-Agent documentation, see [Self-Evolving Multi-Agent README](src/kiss/agents/self_evolving_multi_agent/README.md)**

The goal of this project is to create an optimal multi-agent using KISSAgent and ClaudeCodingAgent given a set of tasks.

```python
from kiss.agents.self_evolving_multi_agent import SelfEvolvingMultiAgent, run_task

# Option 1: Using the class directly
agent = SelfEvolvingMultiAgent()
result = agent.run("""
    Create a Python script that:
    1. Generates the first 20 Fibonacci numbers
    2. Saves them to 'fibonacci.txt'
    3. Reads the file and prints the sum
""")
print(result)

# Access execution statistics
stats = agent.get_stats()
print(f"Completed todos: {stats['completed']}/{stats['total_todos']}")
print(f"Dynamic tools created: {stats['dynamic_tools']}")

# Option 2: Using run_task (for evolver integration)
result = run_task("Create a calculator module with tests")
print(f"Result: {result['result']}")
print(f"Metrics: {result['metrics']}")
print(f"Stats: {result['stats']}")
```

For usage examples, API reference, and configuration options, please see the [Self-Evolving Multi-Agent README](src/kiss/agents/self_evolving_multi_agent/README.md).

### Using Claude Coding Agent

The Claude Coding Agent uses the Claude Agent SDK to generate tested Python programs with file system access controls:

```python
from kiss.core.claude_coding_agent import ClaudeCodingAgent
import anyio

# Create agent with a name
agent = ClaudeCodingAgent(name="My Coding Agent")

async def main():
    # Run a coding task with path restrictions
    result = await agent.run(
        model_name="claude-sonnet-4-5",
        prompt_template="""
            Write, test, and optimize a fibonacci function in Python
            that is efficient and correct.
        """,
        readable_paths=["src/"],  # Allowed read paths
        writable_paths=["output/"],  # Allowed write paths
        base_dir="workdir"  # Base working directory
    )
    if result:
        print(f"Result: {result}")

anyio.run(main)
```

**Built-in Tools Available:**

- `Read`, `Write`, `Edit`, `MultiEdit`: File operations
- `Glob`, `Grep`: File search and content search
- `Bash`: Shell command execution
- `WebSearch`, `WebFetch`: Web access

### Using Gemini CLI Agent

The Gemini CLI Agent uses the Google ADK (Agent Development Kit) to generate tested Python programs:

```python
from kiss.core.gemini_cli_agent import GeminiCliAgent
import anyio

# Create agent with a name (must be a valid identifier - use underscores, not hyphens)
agent = GeminiCliAgent(name="my_coding_agent")

async def main():
    result = await agent.run(
        model_name="gemini-2.5-flash",
        prompt_template="Write a fibonacci function with tests",
        readable_paths=["src/"],
        writable_paths=["output/"],
        base_dir="workdir"
    )
    if result:
        print(f"Result: {result}")

anyio.run(main)
```

### Using OpenAI Codex Agent

The OpenAI Codex Agent uses the OpenAI Agents SDK to generate tested Python programs:

```python
from kiss.core.openai_codex_agent import OpenAICodexAgent
import anyio

agent = OpenAICodexAgent(name="My Coding Agent")

async def main():
    result = await agent.run(
        model_name="gpt-5.2-codex",
        prompt_template="Write a fibonacci function with tests",
        readable_paths=["src/"],
        writable_paths=["output/"],
        base_dir="workdir"
    )
    if result:
        print(f"Result: {result}")

anyio.run(main)
```

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

### Using SimpleRAG for Retrieval-Augmented Generation

SimpleRAG provides a lightweight RAG system with in-memory vector storage and similarity search:

> **Note**: SimpleRAG requires a model with embedding support. Currently, OpenAI, Together AI, and Gemini models support embeddings. Anthropic models do not provide embedding APIs.

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
from kiss.agents.kiss import refine_prompt_template

refined_prompt = refine_prompt_template(
    original_prompt_template="Original prompt...",
    previous_prompt_template="Previous version...",
    agent_trajectory=agent.get_trajectory(),  # Returns JSON string
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
│   │   ├── agent_creator/          # Agent evolution and improvement
│   │   │   ├── agent_evolver.py    # Evolutionary agent optimization
│   │   │   ├── improver_agent.py   # Agent improvement through generations
│   │   │   ├── config.py           # Agent creator configuration
│   │   │   └── README.md           # Agent creator documentation
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
│   │   ├── base_agent.py      # Base agent class with common functionality
│   │   ├── kiss_agent.py      # KISS agent with native function calling
│   │   ├── claude_coding_agent.py # Claude Coding Agent using Claude Agent SDK
│   │   ├── gemini_cli_agent.py    # Gemini CLI Agent using Google ADK
│   │   ├── openai_codex_agent.py  # OpenAI Codex Agent using OpenAI Agents SDK
│   │   ├── formatter.py       # Output formatting base class
│   │   ├── simple_formatter.py # Rich-formatted output
│   │   ├── config.py          # Configuration
│   │   ├── config_builder.py  # Dynamic config builder with CLI support
│   │   ├── kiss_error.py      # Custom error class
│   │   ├── utils.py           # Utility functions
│   │   └── models/            # Model implementations
│   │       ├── model.py           # Model interface
│   │       ├── gemini_model.py    # Gemini model implementation
│   │       ├── openai_compatible_model.py # OpenAI-compatible API model (OpenAI, Together AI, OpenRouter)
│   │       ├── anthropic_model.py # Anthropic model implementation
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
│   │   ├── test_models_quick.py   # Quick tests for models based on ModelInfo capabilities
│   │   ├── run_all_models_test.py # Comprehensive tests for all models
│   │   ├── test_multiprocess.py
│   │   ├── test_internal.py
│   │   ├── test_claude_coding_agent.py  # Tests for Claude Coding Agent
│   │   ├── test_gemini_cli_agent.py     # Tests for Gemini CLI Agent
│   │   ├── test_openai_codex_agent.py   # Tests for OpenAI Codex Agent
│   │   └── test_agent_creator.py        # Tests for Agent Creator
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
  - `max_agent_budget`: Maximum budget per agent run in USD (default: 10.0)
  - `global_max_budget`: Maximum total budget across all agents in USD (default: 200.0)
  - `use_web_search`: Automatically add web search tool if enabled (default: True)
- **GEPA Settings**: Modify `DEFAULT_CONFIG.gepa` in `src/kiss/agents/gepa/config.py`:
  - `reflection_model`: Model to use for reflection (default: "gemini-3-flash-preview")
  - `max_generations`: Maximum number of evolutionary generations (default: 10)
  - `population_size`: Number of candidates to maintain in population (default: 8)
  - `pareto_size`: Maximum size of Pareto frontier (default: 4)
  - `mutation_rate`: Probability of mutating a prompt template (default: 0.5)
- **AlgoTune Settings**: Modify `DEFAULT_CONFIG.algotune` in `src/kiss/agents/kiss_evolve/algotune/config.py`:
  - `task`: Specific task name to solve (default: "matrix_multiplication")
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
- `uv run pytest src/kiss/tests/test_models_quick.py -v` - Run quick model tests
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

- **OpenAI**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-5, gpt-5.1, gpt-5.2
- **Anthropic**: claude-opus-4-5, claude-opus-4-1, claude-sonnet-4-5, claude-sonnet-4, claude-haiku-4-5
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
- **Gemini (preview, unreliable function calling)**: gemini-3-pro-preview, gemini-3-flash-preview, gemini-2.5-flash-lite
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
  - Default model: `togethercomputer/m2-bert-80M-32k-retrieval` (can be customized)
  - Usage: `model.get_embedding(text, embedding_model="togethercomputer/m2-bert-80M-32k-retrieval")`
- **Gemini Models**: Full embedding support via Google's embedding API
  - Default model: `text-embedding-004` (can be customized)
  - Usage: `model.get_embedding(text, embedding_model="text-embedding-004")`
- **Anthropic Models**: Embeddings not supported (raises `NotImplementedError`)

Embeddings are primarily used by the `SimpleRAG` system for document retrieval. When using `SimpleRAG`, ensure you use an OpenAI, Together AI, or Gemini model that supports embeddings.

## Contributing

Contributions are welcome! Please ensure your code:

- Follows the KISS principle
- Passes all tests (`uv run pytest`)
- Passes linting (`uv run ruff check src/`)
- Passes type checking (`uv run mypy src/`)

## License

Apache-2.0

## Authors

- Koushik Sen (ksen@berkeley.edu) | [LinkedIn](https://www.linkedin.com/in/koushik-sen-80b99a/) | [X @koushik77](https://x.com/koushik77)
