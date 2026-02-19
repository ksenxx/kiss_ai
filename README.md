![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/818u234myu55pxt0wi7j.jpeg)

**Version:** 0.1.17

# When Simplicity Becomes Your Superpower: Meet KISS Multi Agent Multi Optimization Framework

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

______________________________________________________________________

KISS stands for ["Keep it Simple, Stupid"](https://en.wikipedia.org/wiki/KISS_principle) which is a well known software engineering principle.

## Installation

Install from PyPI with pip:

```bash
pip install kiss-agent-framework
```

## 🎯 The Problem with AI Agent Frameworks Today

Let's be honest. The AI agent ecosystem has become a jungle. Every week brings a new framework promising to revolutionize how we build AI agents. They come loaded with abstractions on top of abstractions, bloated with techniques that are unnecessary. By the time you've figured out how to make your first tool call, you've already burned through half your patience and all your enthusiasm.

**What if there was another way?**

What if building AI agents could be as straightforward as the name suggests?

Enter **KISS** — the *Keep It Simple, Stupid* Agent Framework.

## 🚀 Your First Agent in 30 Seconds.

Try the interactive Jupyter notebook by running `uv run notebook --lab`.

Let me show you something beautiful:

```python
from kiss.core.kiss_agent import KISSAgent

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = KISSAgent(name="Math Buddy")
result = agent.run(
    model_name="gemini-2.5-flash",
    prompt_template="Calculate: {question}",
    arguments={"question": "What is 15% of 847?"},
    tools=[calculate]
)
print(result)  # 127.05
```

That's a fully functional AI agent that uses tools. No annotations. No boilerplate. No ceremony. Just intent, directly expressed.
Well you might ask "**Why not use LangChain, DSpy, OpenHands, MiniSweAgent, CrewAI, Google ADK, Claude Agent SDK, or some well established agent frameworks?**" Here is my response:

- **KISS comes with [Repo Optimizer](src/kiss/agents/coding_agents/repo_optimizer.py) that will not only enable you write or create agents, but also automatically optimize the agents for efficiency and cost.**
- **It has the GEPA prompt optimizer builtin with a simple API.**
- **It has a [RelentlessCodingAgent](src/kiss/agents/coding_agents/relentless_coding_agent.py), which is pretty straightforward in terms of implementation, but it can work for very very long tasks. It was self evolved over time and is still evolving.**
- **No bloat and simple codebase.**
- **Optimization strategies can be written in plain English.**
- **New techniques will be incorporated to the framework as I research them.**
- **The project effectively applies various programming language and software engineering principles and concepts that I learned since 1995.**

## 🤝 Multi-Agent Orchestration is Function Composition

Here's where KISS really shines — composing multiple agents into systems greater than the sum of their parts.

Since agents are just functions, you orchestrate them with plain Python. Here's a complete **research-to-article pipeline** with three agents:

```python
from kiss.core.kiss_agent import KISSAgent

# Agent 1: Research a topic
researcher = KISSAgent(name="Researcher")
research = researcher.run(
    model_name="gpt-4o",
    prompt_template="List 3 key facts about {topic}. Be concise.",
    arguments={"topic": "Python asyncio"},
    is_agentic=False  # Simple generation, no tools
)

# Agent 2: Write a draft using the research
writer = KISSAgent(name="Writer")
draft = writer.run(
    model_name="claude-sonnet-4-5",
    prompt_template="Write a 2-paragraph intro based on:\n{research}",
    arguments={"research": research},
    is_agentic=False
)

# Agent 3: Polish the draft
editor = KISSAgent(name="Editor")
final = editor.run(
    model_name="gemini-2.5-flash",
    prompt_template="Improve clarity and fix any errors:\n{draft}",
    arguments={"draft": draft},
    is_agentic=False
)

print(final)
```

**That's it.** Each agent can use a different model. Each agent saves its own trajectory. And you compose them with the most powerful orchestration tool ever invented: **regular Python code**.

No special orchestration framework needed. No message buses. No complex state machines. Just Python functions calling Python functions.

## 💪 Using Relentless Coding Agent

The **flagship** coding agent of KISS is the [relentless coding agent](src/kiss/agents/coding_agents/relentless_coding_agent.py). For very long running coding tasks, use the `RelentlessCodingAgent`. The agent will work relentlessly to complete your task using a single-agent architecture with smart continuation. It can run for hours to days to complete a task:

```python
from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

agent = RelentlessCodingAgent(name="Simple Coding Agent")

result = agent.run(
    prompt_template="""
        Create a Python script that reads a CSV file,
        filters rows where age > 18, and writes to a new file.
    """,
    model_name="claude-sonnet-4-5",
    work_dir="./workspace",
    max_steps=200,
    max_sub_sessions=200
)
print(f"Result: {result}")
```

**Running with Docker:**

You can optionally run bash commands inside a Docker container for isolation:

```python
from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

agent = RelentlessCodingAgent(name="Dockered Relentless Coding Agent")

result = agent.run(
    prompt_template="""
        Install numpy and create a script that generates 
        a random matrix and computes its determinant.
    """,
    docker_image="python:3.11-slim",  # Bash commands run in Docker
    max_steps=200,
    max_sub_sessions=2000
)
print(f"Result: {result}")
```

**Key Features:**

- **Single-Agent with Auto-Continuation**: A single agent executes the task across multiple sub-sessions, automatically continuing where it left off via structured JSON progress tracking
- **Structured Progress Tracking**: Each sub-session reports completed and remaining tasks in JSON format (done/next items), which is deduplicated and passed to subsequent sub-sessions
- **Compressed Prompts**: Minimal, high-signal task prompts with critical rules (use Write() for new files, bounded poll loops, immediate finish on success)
- **Multi-Tool Execution**: The agent can execute multiple tool calls in a single step for faster task completion
- **Efficiency Rules**: Built-in prompt instructions enforce immediate completion when tests pass, timeout guidance for bash, and bounded loops for background jobs
- **Retry with Context**: Failed sub-sessions automatically pass structured progress summaries to the next sub-session
- **Configurable Sub-Sessions**: Set high sub-session counts (e.g., 200+) for truly relentless execution
- **Docker Support**: Optional isolated execution via Docker containers
- **Path Access Control**: Enforces read/write permissions on file system paths
- **Built-in Tools**: Bash, Read, Edit, Write, search_web, and fetch_url tools for file and web operations
- **Safer Bash Execution**: `Bash()` validates file paths inside inline interpreter flags (e.g., `python -c`, `node -e`) against path access controls, and disallows shell control commands like `env`, `eval`, and `exec`
- **Budget & Token Tracking**: Automatic cost and token usage monitoring across all sub-sessions

## 💬 Browser-Based Assistant

KISS includes a browser-based assistant UI for interacting with agents. It provides a rich web interface with real-time streaming output, task history with autocomplete, and a file browser for the working directory.

```bash
# Launch the assistant (opens browser automatically)
uv run assistant

# Or with a custom working directory
uv run assistant --work-dir ./my-project
```

The assistant features:

- **Real-time streaming**: See agent thinking, tool calls, and results as they happen
- **Structured result display**: Results with success/failure status are rendered with markdown formatting
- **Task history**: Previously submitted tasks are saved and available via autocomplete
- **File browser**: Browse files in the working directory from the sidebar
- **Modern UI**: Dark theme with collapsible sections for tool calls and thinking

## 🔧 Using Repo Optimizer

**This is the most important and useful feature of KISS.** The `RepoOptimizer` (`repo_optimizer.py`) uses the `RelentlessCodingAgent` to optimize code within your own project repository. It runs the target program, monitors output in real time, fixes errors, and iteratively optimizes for speed and cost — all without changing the agent's interface. The code can be found [here.](src/kiss/agents/coding_agents/repo_optimizer.py). It has less than 100 lines of code consisting of a detailed short prompt. Modify the prompt in `repo_optimizer` to provide the agent path that you want to optimize. You can run it using the following command.

```bash
# Default: optimize the RelentlessCodingAgent
uv run python -m kiss.agents.coding_agents.repo_optimizer

# Optimize a specific agent file with a different model
uv run python -m kiss.agents.coding_agents.repo_optimizer \
    --agent-code src/kiss/agents/imo_agent/imo_agent.py \
    --model gemini-2.5-pro \
    --project-root /path/to/project
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--project-root` | Auto-detected project root | Project root directory |
| `--agent-code` | `src/kiss/agents/coding_agents/relentless_coding_agent.py` | Path to agent code to optimize |
| `--model` | `claude-opus-4-6` | Model name to use |

**How It Works:**

1. Runs the target program (e.g., `relentless_coding_agent.py`) and monitors output in real time
1. If repeated errors are observed, fixes them and reruns
1. Once the program succeeds, analyzes output and optimizes the source for speed and cost
1. Repeats until running time and cost are reduced significantly

**Optimization Strategies:**

- Shorter system prompts preserving meaning
- Removing redundant instructions and minimizing conversation turns
- Batching operations and using early termination
- Applying latest agentic patterns for long-horizon tasks
- Inventing and implementing new agent architectures for efficiency and reliability

📖 **For the full story of how the repo optimizer self-optimized the RelentlessCodingAgent, see [BLOG.md](src/kiss/agents/coding_agents/BLOG.md)**

## 🎨 Output Formatting

Unlike other agentic systems, you do not need to specify the output schema for the agent. Just create
a suitable "finish" function with parameters. The parameters could be treated as the top level keys
in a json format.

**Example: Custom Structured Output**

```python
from kiss.core.kiss_agent import KISSAgent

# Define a custom finish function with your desired output structure
def finish(
    sentiment: str,
    confidence: float,
    key_phrases: str,
    summary: str
) -> str:
    """
    Complete the analysis with structured results.
    
    Args:
        sentiment: The overall sentiment ('positive', 'negative', or 'neutral')
        confidence: Confidence score between 0.0 and 1.0
        key_phrases: Comma-separated list of key phrases found in the text
        summary: A brief summary of the analysis
    
    Returns:
        The formatted analysis result
    """
    ...

```

The agent will automatically use your custom `finish` function instead of the default one which returns its argument. The function's parameters define what information the agent must provide, and the docstring helps the LLM understand how to format each field.

## 📊 Trajectory Saving and Visualization

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

![Trajectory Visualizer](assets/image-0478c494-2550-4bbe-8559-f205a4544bec.png)

📖 **For detailed trajectory visualizer documentation, see [Trajectory Visualizer README](src/kiss/viz_trajectory/README.md)**

## 📖 Features of The KISS Framework

KISS is a lightweight, yet powerful, multi agent framework that implements a ReAct (Reasoning and Acting) loop for LLM agents. The framework provides:

- **Simple Architecture**: Clean, minimal core that's easy to understand and extend
- **Multi-Tool Execution**: Agents can execute multiple tool calls in a single step for faster task completion
- **Relentless Coding Agent**: Single-agent coding system with smart auto-continuation for long-running tasks
- **Browser-Based Assistant**: Interactive web UI for agents with real-time streaming and task history
- **Repo Optimizer**: Uses RelentlessCodingAgent to iteratively optimize code in your project for speed and cost (💡 new idea)
- **IMO Agent**: Verification-and-refinement pipeline for solving competition math problems (based on [arXiv:2507.15855](https://arxiv.org/abs/2507.15855))
- **GEPA Implementation From Scratch**: Genetic-Pareto prompt optimization for compound AI systems
- **KISSEvolve Implementation From Scratch**: Evolutionary algorithm discovery framework with LLM-guided mutation and crossover
- **Model Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Gemini, Together AI, OpenRouter)
- **Native Function Calling**: Seamless tool integration using native function calling APIs (OpenAI, Anthropic, Gemini, Together AI, and OpenRouter)
- **Docker Integration**: Built-in Docker manager for running agents in isolated environments
- **Trajectory Tracking**: Automatic saving of agent execution trajectories with unified state management
- **Structured Result Display**: Console and browser printers parse YAML result content to show success/failure status with markdown rendering
- **Token Streaming**: Real-time token streaming via async callback for all providers (OpenAI, Anthropic, Gemini, Together AI, OpenRouter), including thinking/reasoning tokens and tool execution output
- **Token Usage Tracking**: Built-in token usage tracking with automatic context length detection and step counting
- **Budget Tracking**: Automatic cost tracking and budget monitoring across all agent runs
- **Self-Evolution**: Framework for agents to evolve and refine other multi agents
- **SWE-bench Dataset Support**: Built-in support for downloading and working with SWE-bench Verified dataset
- **RAG Support**: Simple retrieval-augmented generation system with in-memory vector store
- **Useful Agents**: Pre-built utility agents including prompt refinement and general bash execution agents
- **Multiprocessing Support**: Utilities for parallel execution of functions using multiprocessing
- **Trajectory Visualization**: Web-based visualizer for viewing agent execution trajectories with modern UI

## 📦 Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone/download KISS and navigate to the directory
cd kiss

# Create virtual environment
uv venv --python 3.13

# Install all dependencies (full installation)
uv sync

# (Optional) activate the venv for convenience (uv run works without activation)
source .venv/bin/activate

# Set up API keys (optional, for LLM providers)
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export TOGETHER_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"
```

### Selective Installation (Dependency Groups)

KISS supports selective installation via dependency groups for minimal footprints:

```bash
# Minimal core only (no model SDKs) - for custom integrations
uv sync --group core

# Core + specific provider support
uv sync --group claude    # Core + Anthropic Claude
uv sync --group openai    # Core + OpenAI Compatible Models
uv sync --group gemini    # Core + Google Gemini

# Docker support (for running agents in isolated containers)
uv sync --group docker

# Evals dependencies (for running benchmarks)
uv sync --group evals

# Development tools (mypy, ruff, pytest, jupyter, etc.)
uv sync --group dev

# Combine multiple groups as needed
uv sync --group claude --group dev
```

**Dependency Group Contents:**

| Group | Description | Key Packages |
|-------|-------------|--------------|
| `core` | Minimal core module | pydantic, rich, requests, beautifulsoup4, playwright, flask |
| `claude` | Core + Anthropic | core + anthropic |
| `openai` | Core + OpenAI | core + openai |
| `gemini` | Core + Google | core + google-genai |
| `docker` | Docker integration | docker, types-docker |
| `evals` | Benchmark running | datasets, swebench, orjson, scipy, scikit-learn |
| `dev` | Development tools | mypy, ruff, pyright, pytest, jupyter, notebook |

> **Optional Dependencies:** All LLM provider SDKs (`openai`, `anthropic`, `google-genai`) are optional. You can import `kiss.core` and `kiss.agents` without installing all of them. When you try to use a model whose SDK is not installed, KISS raises a clear `KISSError` telling you which package to install.

## 📚 KISSAgent API Reference

📖 **For detailed KISSAgent API documentation, see [API.md](API.md)**

## 🎯 Using GEPA for Prompt Optimization

KISS has a fresh implementation of GEPA with some key improvements. GEPA (Genetic-Pareto) is a prompt optimization framework that uses natural language reflection to evolve prompts. It maintains an instance-level Pareto frontier of top-performing prompts and combines complementary lessons through structural merge. It also supports optional batched evaluation via `batched_agent_wrapper`, so you can plug in prompt-merging inference pipelines to process more datapoints per API call. GEPA is based on the paper ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457).

📖 **For detailed GEPA documentation, see [GEPA README](src/kiss/agents/gepa/README.md)**

## 🧪 Using KISSEvolve for Algorithm Discovery

This is where I started building an optimizer for agents. Then I switched to [`agent evolver`](src/kiss/agents/create_and_optimize_agent/agent_evolver.py) because `KISSEvolver` was expensive to run. Finally I switched to \[`repo_optimizer`\] for efficiency and simplicity. I am still keeping KISSEvolve around. KISSEvolve is an evolutionary algorithm discovery framework that uses LLM-guided mutation and crossover to evolve code variants. It supports advanced features including island-based evolution, novelty rejection sampling, and multiple parent sampling methods.

For usage examples, API reference, and configuration options, please see the [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md).

📖 **For detailed KISSEvolve documentation, see [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md)**

## ⚡ Multiprocessing

KISS provides utilities for parallel execution of Python functions using multiprocessing. This is useful for running multiple independent tasks concurrently to maximize CPU utilization.

### Basic Usage

```python
from kiss.multiprocessing import run_functions_in_parallel

def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

# Define tasks as (function, arguments) tuples
tasks = [(add, [1, 2]), (multiply, [3, 4])]
results = run_functions_in_parallel(tasks)
print(results)  # [3, 12]
```

### With Keyword Arguments

```python
from kiss.multiprocessing import run_functions_in_parallel_with_kwargs

def greet(name, title="Mr."):
    return f"Hello, {title} {name}!"

functions = [greet, greet]
args_list = [["Alice"], ["Bob"]]
kwargs_list = [{"title": "Dr."}, {}]

results = run_functions_in_parallel_with_kwargs(functions, args_list, kwargs_list)
print(results)  # ["Hello, Dr. Alice!", "Hello, Mr. Bob!"]
```

### 💻 Checking Available Cores

```python
from kiss.multiprocessing import get_available_cores

num_cores = get_available_cores()
print(f"Available CPU cores: {num_cores}")
```

The multiprocessing utilities automatically scale to the number of available CPU cores, using at most as many workers as there are tasks to avoid unnecessary overhead.

## 🐳 Docker Manager

KISS provides a `DockerManager` class for managing Docker containers and executing commands inside them. This is useful for running code in isolated environments, testing with specific dependencies, or working with SWE-bench tasks.

### Basic Usage

```python
from kiss.docker import DockerManager

# Create a Docker manager for an Ubuntu container
with DockerManager(image_name="ubuntu", tag="22.04", workdir="/app") as docker:
    # Run commands inside the container
    output = docker.run_bash_command("echo 'Hello from Docker!'", "Print greeting")
    print(output)
    
    output = docker.run_bash_command("python3 --version", "Check Python version")
    print(output)
```

### Manual Lifecycle Management

```python
from kiss.docker import DockerManager

docker = DockerManager(image_name="python", tag="3.11", workdir="/workspace")
docker.open()  # Pull image and start container

try:
    output = docker.run_bash_command("pip install numpy", "Install numpy")
    output = docker.run_bash_command("python -c 'import numpy; print(numpy.__version__)'", "Check numpy")
    print(output)
finally:
    docker.close()  # Stop and remove container
```

### Port Mapping

```python
from kiss.docker import DockerManager

# Map container port 8080 to host port 8080
with DockerManager(image_name="nginx", ports={80: 8080}) as docker:
    # Start a web server
    docker.run_bash_command("nginx", "Start nginx")
    
    # Get the actual host port (useful when Docker assigns a random port)
    host_port = docker.get_host_port(80)
    print(f"Server available at http://localhost:{host_port}")
```

### Configuration Options

- `image_name`: Docker image name (e.g., 'ubuntu', 'python:3.11')
- `tag`: Image tag/version (default: 'latest')
- `workdir`: Working directory inside the container (default: '/')
- `mount_shared_volume`: Whether to mount a shared volume for file transfer (default: True)
- `ports`: Port mapping from container to host (e.g., `{8080: 8080}`)

The Docker manager automatically handles image pulling, container lifecycle, and cleanup of temporary directories.

## 📁 Project Structure

```
kiss/
├── src/kiss/
│   ├── agents/          # Agent implementations
│   │   ├── assistant/              # Assistant agent with coding + browser tools
│   │   │   ├── assistant_agent.py      # AssistantAgent with coding and browser automation
│   │   │   ├── assistant.py            # Browser-based assistant UI
│   │   │   ├── relentless_agent.py     # RelentlessAgent base class
│   │   │   └── config.py               # Assistant agent configuration
│   │   ├── create_and_optimize_agent/  # Agent evolution and improvement
│   │   │   ├── agent_evolver.py        # Evolutionary agent optimization
│   │   │   ├── improver_agent.py       # Agent improvement through generations
│   │   │   ├── config.py               # Agent creator configuration
│   │   │   ├── BLOG.md                 # Blog post about agent evolution
│   │   │   └── README.md               # Agent creator documentation
│   │   ├── gepa/                   # GEPA (Genetic-Pareto) prompt optimizer
│   │   │   ├── gepa.py
│   │   │   ├── config.py           # GEPA configuration
│   │   │   └── README.md           # GEPA documentation
│   │   ├── imo_agent/              # IMO mathematical problem-solving agent
│   │   │   ├── __init__.py
│   │   │   ├── imo_agent.py            # Verification-and-refinement pipeline (arXiv:2507.15855)
│   │   │   ├── imo_problems.py         # IMO 2025 problem statements, validation criteria, and difficulty
│   │   │   ├── imo_agent_creator.py    # Repo agent that created the IMO agent
│   │   │   └── config.py               # IMO agent configuration
│   │   ├── kiss_evolve/            # KISSEvolve evolutionary algorithm discovery
│   │   │   ├── kiss_evolve.py
│   │   │   ├── novelty_prompts.py  # Prompts for novelty-based evolution
│   │   │   ├── config.py           # KISSEvolve configuration
│   │   │   └── README.md           # KISSEvolve documentation
│   │   ├── coding_agents/          # Coding agents for software development tasks
│   │   │   ├── relentless_coding_agent.py # Single-agent system with smart auto-continuation
│   │   │   ├── claude_coding_agent.py     # Claude-based coding agent
│   │   │   ├── repo_optimizer.py          # Iterative code optimizer using RelentlessCodingAgent
│   │   │   ├── repo_agent.py              # Repo-level task agent using RelentlessCodingAgent
│   │   │   ├── config.py                  # Coding agent configuration (RelentlessCodingAgent)
│   │   │   └── BLOG.md                    # Blog post about self-optimization
│   │   ├── self_evolving_multi_agent/  # Self-evolving multi-agent system
│   │   │   ├── agent_evolver.py       # Agent evolution logic
│   │   │   ├── multi_agent.py         # Multi-agent orchestration
│   │   │   ├── config.py              # Configuration
│   │   │   └── README.md              # Documentation
│   │   └── kiss.py                 # Utility agents (prompt refiner, bash agent)
│   ├── core/            # Core framework components
│   │   ├── base.py            # Base class with common functionality for all KISS agents
│   │   ├── kiss_agent.py      # KISS agent with native function calling (supports multi-tool execution)
│   │   ├── printer.py         # Abstract Printer base class and MultiPrinter
│   │   ├── print_to_console.py # ConsolePrinter: Rich-formatted terminal output
│   │   ├── print_to_browser.py # BrowserPrinter: SSE streaming to browser UI
│   │   ├── browser_ui.py      # Browser UI base components and utilities
│   │   ├── config.py          # Configuration
│   │   ├── config_builder.py  # Dynamic config builder with CLI support
│   │   ├── kiss_error.py      # Custom error class
│   │   ├── utils.py           # Utility functions (finish, resolve_path, is_subpath, etc.)
│   │   ├── useful_tools.py    # UsefulTools class with path-restricted Read, Write, Bash, Edit, search_web, fetch_url
│   │   ├── web_use_tool.py    # WebUseTool class with Playwright-based browser automation
│   │   └── models/            # Model implementations
│   │       ├── model.py           # Model interface with TokenCallback streaming support
│   │       ├── gemini_model.py    # Gemini model implementation
│   │       ├── openai_compatible_model.py # OpenAI-compatible API model (OpenAI, Together AI, OpenRouter)
│   │       ├── anthropic_model.py # Anthropic model implementation
│   │       └── model_info.py      # Model info: context lengths, pricing, and capabilities
│   ├── docker/          # Docker integration
│   │   └── docker_manager.py
│   ├── evals/            # Benchmark and evaluation integrations
│   │   ├── algotune/               # AlgoTune benchmark integration
│   │   │   ├── run_algotune.py     # AlgoTune task evolution
│   │   │   └── config.py           # AlgoTune configuration
│   │   ├── arvo_agent/             # ARVO vulnerability detection agent
│   │   │   ├── arvo_agent.py       # Arvo-based vulnerability detector
│   │   │   └── arvo_tags.json      # Docker image tags for Arvo
│   │   ├── hotpotqa/               # HotPotQA benchmark integration
│   │   │   ├── hotpotqa_benchmark.py # HotPotQA benchmark runner
│   │   │   └── README.md           # HotPotQA documentation
│   │   └── swe_agent_verified/     # SWE-bench Verified benchmark integration
│   │       ├── run_swebench.py     # Main runner with CLI support
│   │       ├── config.py           # Configuration for SWE-bench runs
│   │       └── README.md           # SWE-bench documentation
│   ├── multiprocessing/ # Multiprocessing utilities
│   │   └── multiprocess.py
│   ├── rag/             # RAG (Retrieval-Augmented Generation)
│   │   └── simple_rag.py # Simple RAG system with in-memory vector store
│   ├── demo/            # Demo scripts
│   │   └── kiss_demo.py               # Interactive demo with streaming output to terminal and browser
│   ├── scripts/         # Utility scripts
│   │   ├── check.py                    # Code quality check script
│   │   ├── notebook.py                 # Jupyter notebook launcher and utilities
│   │   └── kissevolve_bubblesort.py    # KISSEvolve example: evolving bubble sort
│   ├── tests/           # Test suite
│   │   ├── conftest.py              # Pytest configuration and fixtures
│   │   ├── test_kiss_agent_agentic.py
│   │   ├── test_kiss_agent_non_agentic.py
│   │   ├── test_kiss_agent_coverage.py    # Coverage tests for KISSAgent
│   │   ├── test_kissevolve_bubblesort.py
│   │   ├── test_gepa_hotpotqa.py
│   │   ├── test_gepa_batched.py           # Tests for GEPA batched wrapper behavior and performance
│   │   ├── test_gepa_progress_callback.py # Tests for GEPA progress callbacks
│   │   ├── test_docker_manager.py
│   │   ├── test_model_implementations.py  # Integration tests for model implementations
│   │   ├── run_all_models_test.py # Comprehensive tests for all models
│   │   ├── test_multiprocess.py
│   │   ├── test_internal.py
│   │   ├── test_core_branch_coverage.py   # Branch coverage tests for core components
│   │   ├── test_gemini_model_internals.py # Tests for Gemini model internals
│   │   ├── test_cli_options.py            # Tests for CLI option parsing
│   │   ├── test_claude_coding_agent.py    # Tests for coding agents
│   │   ├── test_evolver_progress_callback.py # Tests for AgentEvolver progress callbacks
│   │   ├── test_token_callback.py         # Tests for async token streaming callback
│   │   ├── test_coding_agent_token_callback.py # Tests for token callback in coding agents
│   │   ├── test_a_model.py                    # Tests for model implementations
│   │   ├── test_print_to_console.py         # Tests for ConsolePrinter output
│   │   ├── test_print_to_browser.py         # Tests for BrowserPrinter browser output
│   │   ├── test_search_web.py
│   │   ├── test_useful_tools.py
│   │   ├── test_web_use_tool.py             # Tests for WebUseTool browser automation
│   │   ├── test_chatbot_tasks.py            # Tests for assistant task handling
│   │   ├── integration_test_assistant_agent.py  # Integration tests for AssistantAgent
│   │   ├── integration_test_google_search.py    # Integration tests for Google search
│   │   └── integration_test_web_use_tool.py     # Integration tests for WebUseTool
│   ├── py.typed          # PEP 561 marker for type checking
│   └── viz_trajectory/  # Trajectory visualization
│       ├── server.py                    # Flask server for trajectory visualization
│       ├── README.md                    # Trajectory visualizer documentation
│       └── templates/                   # HTML templates for the visualizer
│           └── index.html
├── scripts/             # Repository-level scripts
│   └── release.sh       # Release script
├── API.md               # KISSAgent API reference
├── BLOG.md              # Blog post about the KISS framework
├── CLAUDE.md            # Code style guidelines for LLM assistants
├── kiss.ipynb           # Interactive tutorial Jupyter notebook
├── LICENSE              # Apache-2.0 license
├── pyproject.toml       # Project configuration
└── README.md
```

## 🏷️ Versioning

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

## ⚙️ Configuration

Configuration is managed through environment variables and the `DEFAULT_CONFIG` object:

- **API Keys**: Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, and/or `MINIMAX_API_KEY` environment variables
- **Agent Settings**: Modify `DEFAULT_CONFIG.agent` in `src/kiss/core/config.py`:
  - `max_steps`: Maximum iterations in the ReAct loop (default: 100)
  - `verbose`: Enable verbose output (default: True)
  - `debug`: Enable debug mode (default: False)
  - `max_agent_budget`: Maximum budget per agent run in USD (default: 10.0)
  - `global_max_budget`: Maximum total budget across all agents in USD (default: 200.0)
  - `use_web`: Automatically add web browsing and search tool if enabled (default: True)
  - `print_to_console`: Enable ConsolePrinter for Rich terminal output (default: True). Can be overridden per-call via the `print_to_console` parameter on `run()`.
  - `print_to_browser`: Enable BrowserPrinter for live browser UI output (default: False). Can be overridden per-call via the `print_to_browser` parameter on `run()`.
  - `artifact_dir`: Directory for agent artifacts (default: auto-generated with timestamp)
- **Relentless Coding Agent Settings**: Modify `DEFAULT_CONFIG.coding_agent.relentless_coding_agent` in `src/kiss/agents/coding_agents/config.py`:
  - `model_name`: Model for task execution (default: "claude-opus-4-6")
  - `max_sub_sessions`: Maximum number of sub-sessions for auto-continuation (default: 200)
  - `max_steps`: Maximum steps per sub-session (default: 25)
  - `max_budget`: Maximum budget in USD (default: 200.0)
- **IMO Agent Settings**: Modify `DEFAULT_CONFIG.imo_agent` in `src/kiss/agents/imo_agent/config.py`:
  - `solver_model`: Model for solving IMO problems (default: "gemini-3-pro-preview")
  - `verifier_model`: Model for verifying solutions (default: "gemini-2.5-pro")
  - `validator_model`: Model for independent validation against known answers (default: "gemini-2.5-pro")
  - `max_refinement_rounds`: Max verification-refinement iterations per attempt (default: 2)
  - `num_verify_passes`: Number of verification passes required to accept a solution (default: 1)
  - `max_attempts`: Max independent attempts per problem (default: 1)
  - `max_budget`: Maximum budget in USD per problem (default: 50.0)
- **GEPA Settings**: Modify `DEFAULT_CONFIG.gepa` in `src/kiss/agents/gepa/config.py`:
  - `reflection_model`: Model to use for reflection (default: "gemini-3-flash-preview")
  - `max_generations`: Maximum number of evolutionary generations (default: 10)
  - `population_size`: Number of candidates to maintain in population (default: 8)
  - `pareto_size`: Maximum size of Pareto frontier (default: 4)
  - `mutation_rate`: Probability of mutating a prompt template (default: 0.5)
- **KISSEvolve Settings**: Modify `DEFAULT_CONFIG.kiss_evolve` in `src/kiss/agents/kiss_evolve/config.py`:
  - `max_generations`: Maximum number of evolutionary generations (default: 10)
  - `population_size`: Number of variants to maintain in population (default: 8)
  - `mutation_rate`: Probability of mutating a variant (default: 0.7)
  - `elite_size`: Number of best variants to preserve each generation (default: 2)
  - `num_islands`: Number of islands for island-based evolution, 1 = disabled (default: 2)
  - `migration_frequency`: Number of generations between migrations (default: 5)
  - `migration_size`: Number of individuals to migrate between islands (default: 1)
  - `migration_topology`: Migration topology: 'ring', 'fully_connected', or 'random' (default: "ring")
  - `enable_novelty_rejection`: Enable code novelty rejection sampling (default: False)
  - `novelty_threshold`: Cosine similarity threshold for rejecting code (default: 0.95)
  - `max_rejection_attempts`: Maximum rejection attempts before accepting (default: 5)
  - `parent_sampling_method`: Parent sampling: 'tournament', 'power_law', or 'performance_novelty' (default: "power_law")
  - `power_law_alpha`: Power-law sampling parameter for rank-based selection (default: 1.0)
  - `performance_novelty_lambda`: Selection pressure parameter for sigmoid (default: 1.0)
- **Self-Evolving Multi-Agent Settings**: Modify `DEFAULT_CONFIG.self_evolving_multi_agent` in `src/kiss/agents/self_evolving_multi_agent/config.py`:
  - `model`: LLM model to use for the main agent (default: "gemini-3-flash-preview")
  - `sub_agent_model`: Model for sub-agents (default: "gemini-3-flash-preview")
  - `evolver_model`: Model for evolution (default: "gemini-3-flash-preview")
  - `max_steps`: Maximum orchestrator steps (default: 100)
  - `max_budget`: Maximum budget in USD (default: 10.0)
  - `max_retries`: Maximum retries on error (default: 3)
  - `sub_agent_max_steps`: Maximum steps for sub-agents (default: 50)
  - `sub_agent_max_budget`: Maximum budget for sub-agents in USD (default: 2.0)
  - `docker_image`: Docker image for execution (default: "python:3.12-slim")
  - `workdir`: Working directory in container (default: "/workspace")

## 🛠️ Available Commands

### Development

- `uv sync` - Install all dependencies (full installation)
- `uv sync --group dev` - Install dev tools (mypy, ruff, pytest, jupyter, etc.)
- `uv sync --group <name>` - Install specific dependency group (see [Selective Installation](#selective-installation-dependency-groups))
- `uv build` - Build the project package

### Testing

- `uv run pytest` - Run all tests (uses testpaths from pyproject.toml)
- `uv run pytest src/kiss/tests/ -v` - Run all tests with verbose output
- `uv run pytest src/kiss/tests/test_kiss_agent_agentic.py -v` - Run agentic agent tests
- `uv run pytest src/kiss/tests/test_kiss_agent_non_agentic.py -v` - Run non-agentic agent tests
- `uv run pytest src/kiss/tests/test_multiprocess.py -v` - Run multiprocessing tests
- `uv run python -m unittest src.kiss.tests.test_docker_manager -v` - Run docker manager tests (unittest)
- `uv run python -m unittest discover -s src/kiss/tests -v` - Run all tests using unittest

### Code Quality

- `uv run check` - Run all code quality checks (fresh dependency install, build, lint, and type check)
- `uv run check --clean` - Run all code quality checks (fresh dependency install, build, lint, and type check after removing previous build options)
- `uv run ruff format src/` - Format code with ruff (line-length: 100, target: py313)
- `uv run ruff check src/` - Lint code with ruff (selects: E, F, W, I, N, UP)
- `uv run mypy src/` - Type check with mypy (python_version: 3.13)
- `uv run pyright src/` - Type check with pyright (alternative to mypy, stricter checking)

### Notebook

- `uv run notebook --test` - Test all imports and basic functionality
- `uv run notebook --lab` - Open the tutorial notebook in JupyterLab (recommended)
- `uv run notebook --run` - Open the tutorial notebook in Jupyter Notebook
- `uv run notebook --execute` - Execute notebook cells and update outputs in place
- `uv run notebook --convert` - Convert notebook to Python script

### Assistant

- `uv run assistant` - Launch the browser-based assistant UI (coding + browser automation)
- `uv run assistant --work-dir ./my-project` - Launch with custom working directory

### AlgoTune

- `uv run algotune` - Run the AlgoTune benchmark

### Cleanup

```bash
rm -rf build/ dist/ .pytest_cache .mypy_cache .ruff_cache && \
find . -type d -name __pycache__ -exec rm -r {} + && \
find . -type f -name "*.pyc" -delete
```

## 🤖 Models Supported

**Supported Models**: The framework includes context length, pricing, and capability flags for:

**Generation Models** (text generation with function calling support):

- **OpenAI**: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-4.5-preview, gpt-4-turbo, gpt-4, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5.1, gpt-5.2, gpt-5.2-pro
- **OpenAI (Codex)**: gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini, gpt-5.2-codex, codex-mini-latest
- **OpenAI (Reasoning)**: o1, o1-mini, o1-pro, o3, o3-mini, o3-mini-high, o3-pro, o3-deep-research, o4-mini, o4-mini-high, o4-mini-deep-research
- **OpenAI (Open Source)**: openai/gpt-oss-20b, openai/gpt-oss-120b
- **Anthropic**: claude-opus-4-6, claude-opus-4-5, claude-opus-4-1, claude-sonnet-4-5, claude-sonnet-4, claude-haiku-4-5
- **Anthropic (Legacy)**: claude-3-5-sonnet-20241022, claude-3-5-haiku, claude-3-5-haiku-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro (deprecated), gemini-1.5-flash (deprecated)
- **Gemini (preview, unreliable function calling)**: gemini-3-pro-preview, gemini-3-flash-preview, gemini-2.5-flash-lite
- **Together AI (Llama)**: Llama-4-Scout/Maverick (with function calling), Llama-3.x series (generation only)
- **Together AI (Qwen)**: Qwen2.5-72B/7B-Instruct-Turbo, Qwen2.5-Coder-32B, Qwen2.5-VL-72B, Qwen3-235B series, Qwen3-Coder-480B, Qwen3-Coder-Next, Qwen3-Next-80B, Qwen3-VL-32B/8B, QwQ-32B (with function calling)
- **Together AI (DeepSeek)**: DeepSeek-R1, DeepSeek-V3-0324, DeepSeek-V3.1 (with function calling)
- **Together AI (Kimi/Moonshot)**: Kimi-K2-Instruct, Kimi-K2-Instruct-0905, Kimi-K2-Thinking, Kimi-K2.5
- **Together AI (Mistral)**: Ministral-3-14B, Mistral-7B-v0.2/v0.3, Mistral-Small-24B
- **Together AI (Z.AI)**: GLM-5.0, GLM-4.5-Air, GLM-4.7
- **Together AI (Other)**: Nemotron-Nano-9B, Arcee (Coder-Large, Maestro-Reasoning, Virtuoso-Large, trinity-mini), DeepCogito (cogito-v2 series), google/gemma-2b/3n, Refuel-LLM-2/2-Small, essentialai/rnj-1, marin-community/marin-8b
- **OpenRouter**: Access to 400+ models from 60+ providers via unified API:
  - OpenAI (gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4.1, gpt-4o variants, gpt-5/5.1/5.2 and codex variants, o1, o3, o3-pro, o4-mini, codex-mini, gpt-oss, gpt-audio)
  - Anthropic (claude-3-haiku, claude-3.5-haiku/sonnet, claude-3.7-sonnet, claude-sonnet-4/4.5, claude-haiku-4.5, claude-opus-4/4.1/4.5/4.6 with 1M context)
  - Google (gemini-2.0-flash, gemini-2.5-flash/pro, gemini-3-flash/pro-preview, gemma-2-9b/27b, gemma-3-4b/12b/27b, gemma-3n-e4b)
  - Meta Llama (llama-3-8b/70b, llama-3.1-8b/70b/405b, llama-3.2-1b/3b/11b-vision, llama-3.3-70b, llama-4-maverick/scout, llama-guard-2/3/4)
  - DeepSeek (deepseek-chat/v3/v3.1/v3.2/v3.2-speciale, deepseek-r1/r1-0528/r1-turbo, deepseek-r1-distill variants, deepseek-coder-v2, deepseek-prover-v2)
  - Qwen (qwen-2.5-7b/72b, qwen-turbo/plus/max, qwen3-8b/14b/30b/32b/235b, qwen3-coder/coder-plus/coder-next/coder-flash/coder-30b, qwen3-vl variants, qwq-32b, qwen3-next-80b, qwen3-max/max-thinking)
  - Amazon Nova (nova-micro/lite/pro, nova-2-lite, nova-premier)
  - Cohere (command-r, command-r-plus, command-a, command-r7b)
  - X.AI Grok (grok-3/3-mini/3-beta/3-mini-beta, grok-4/4-fast, grok-4.1-fast, grok-code-fast-1)
  - MiniMax (minimax-01, minimax-m1, minimax-m2/m2.1/m2.5/m2-her)
  - ByteDance Seed (seed-1.6, seed-1.6-flash, seed-2.0, seed-2.0-thinking)
  - MoonshotAI (kimi-k2, kimi-k2-thinking, kimi-k2.5, kimi-dev-72b)
  - Mistral (codestral, devstral/devstral-medium/devstral-small, mistral-large/medium/small, mixtral-8x7b/8x22b, ministral-3b/8b/14b, pixtral, voxtral)
  - NVIDIA (llama-3.1-nemotron-70b/ultra-253b, llama-3.3-nemotron-super-49b, nemotron-nano-9b-v2/12b-v2-vl, nemotron-3-nano-30b)
  - Z.AI/GLM (glm-5, glm-4-32b, glm-4.5/4.5-air/4.5v, glm-4.6/4.6v, glm-4.7/4.7-flash)
  - AllenAI (olmo-2/3-7b/32b-instruct/think, olmo-3.1-32b-instruct/think, molmo-2-8b)
  - Perplexity (sonar, sonar-pro, sonar-pro-search, sonar-deep-research, sonar-reasoning-pro)
  - NousResearch (hermes-2-pro/3/4-llama series, hermes-4-70b/405b, deephermes-3)
  - Baidu ERNIE (ernie-4.5 series including VL and thinking variants)
  - Aurora (openrouter/aurora-alpha — free cloaked reasoning model)
  - And 30+ more providers (ai21, aion-labs, alfredpros, alpindale, anthracite-org, arcee-ai, bytedance, deepcogito, essentialai, ibm-granite, inception, inflection, kwaipilot, liquid, meituan, morph, nex-agi, opengvlab, prime-intellect, relace, sao10k, stepfun-ai, tencent, thedrummer, tngtech, upstage, writer, xiaomi, etc.)

**Embedding Models** (for RAG and semantic search):

- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Google**: text-embedding-004, gemini-embedding-001
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
  - Default model: `text-embedding-004` (can be customized; `gemini-embedding-001` also available)
  - Usage: `model.get_embedding(text, embedding_model="text-embedding-004")`
- **Anthropic Models**: Embeddings not supported (raises `NotImplementedError`)

Embeddings are primarily used by the `SimpleRAG` system for document retrieval. When using `SimpleRAG`, ensure you use an OpenAI, Together AI, or Gemini model that supports embeddings.

## 🤗 Contributing

Contributions are welcome! Please ensure your code:

- Follows the KISS principle
- Passes all tests (`uv run pytest`)
- Passes linting (`uv run ruff check src/`)
- Passes type checking (`uv run mypy src/`)
- Passes type checking (`uv run pyright src/`)

## 📄 License

Apache-2.0

## ✍️ Authors

- Koushik Sen (ksen@berkeley.edu) | [LinkedIn](https://www.linkedin.com/in/koushik-sen-80b99a/) | [X @koushik77](https://x.com/koushik77)
