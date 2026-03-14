<div align="center">

![KISS Framework](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/818u234myu55pxt0wi7j.jpeg)

# When Simplicity Becomes Your Superpower: Meet KISS Multi-Agent Multi-Optimization Framework

[![Version](https://img.shields.io/badge/version-0.2.37-blue?style=flat-square)](https://pypi.org/project/kiss-agent-framework/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue?style=flat-square)](https://www.python.org/)

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

</div>

______________________________________________________________________

KISS stands for ["Keep it Simple, Stupid"](https://en.wikipedia.org/wiki/KISS_principle) which is a well-known software engineering principle.

<details>
<summary><strong>Table of Contents</strong></summary>

- [Installation](#installation)
- [The Problem with AI Agent Frameworks Today](#-the-problem-with-ai-agent-frameworks-today)
- [Your First Agent in 30 Seconds](#-your-first-agent-in-30-seconds)
- [Multi-Agent Orchestration](#-multi-agent-orchestration-is-function-composition)
- [KISS Sorcar](#-kiss-sorcar)
- [Key Features and Ideologies](#-key-features-and-ideologies-behind-kiss-and-sorcar)
- [Repo Optimizer](#-using-repo-optimizer)
- [Output Formatting](#-output-formatting)
- [Trajectory Saving and Visualization](#-trajectory-saving-and-visualization)
- [Features of The KISS Framework](#-features-of-the-kiss-framework)
- [Full Installation & Dependency Groups](#-installation-1)
- [KISSAgent API Reference](#-kissagent-api-reference)
- [GEPA Prompt Optimization](#-using-gepa-for-prompt-optimization)
- [KISSEvolve Algorithm Discovery](#-using-kissevolve-for-algorithm-discovery)
- [Docker Manager](#-docker-manager)
- [Project Structure](#-project-structure)
- [Versioning](#%EF%B8%8F-versioning)
- [Configuration](#%EF%B8%8F-configuration)
- [Available Commands](#%EF%B8%8F-available-commands)
- [Models Supported](#-models-supported)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#%EF%B8%8F-authors)

</details>

## Installation and Launching KISS Sorcar

You must provide ANTHROPIC_API_KEY and GEMINI_API_KEY in the environment.  Then download 
[KISS Sorcar](https://github.com/ksenxx/kiss_ai/releases/download/v0.2.35/kiss-offline-installer.pkg) and
install.


# Introduction to KISS Sorcar

**KISS Sorcar** (named after the [Famous Bengali Magician P.C. Sorcar](https://en.wikipedia.org/wiki/P._C._Sorcar)) is a free alternative to Cursor or Antigravity and **a general purpose agent with web browsing capabilities**. It runs **locally** as a VSCode IDE. It **codes really well** and **works pretty fast**. The agent can **run relentlessly for hours to days**. It is **embedded in a browser** and uses **full-fledged vscode**. It has **full browser** support and limited **multimodal** support. The good part is that KISS Sorcar is **completely free** and **open-source** with **no monthly subscription fees**. KISS Sorcar has been built on top of the KISS Multi Agentic Framework, which I describe in the next section.

#whatispossible #KISSSorcar

![KISS Sorcar Screenshot](assets/KISSSorcar.png)

A video demo of KISS Sorcar can be found [here](https://youtube.com/watch?v=XWLpNJlUgZo) and an older video showing how I used KISS Sorcar to develop KISS Sorcar can be found [here](https://youtu.be/2cNQJFu0Mnk).

```bash
# To install for development
curl -LsSf https://raw.githubusercontent.com/ksenxx/kiss_ai/refs/heads/main/install.sh | sh
cd kiss_ai
# To launch sorcar
./sorcar
# To use sorcar CLI
./sorcar --task "Multiply 45278*2983"
# or
./sorcar --f example_prompt.md


# To install as a library
curl -LsSf https://raw.githubusercontent.com/ksenxx/kiss_ai/refs/heads/main/installlib.sh | sh
# To launch sorcar
cd myproject
uv run sorcar
```

# Introduction to KISS

## 🎯 The Problem with AI Agent Frameworks Today

The AI agent ecosystem has grown increasingly complex. Many frameworks introduce excessive layers of abstraction and unnecessary techniques, resulting in a steep learning curve that can significantly hinder developer productivity from the outset.

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
    model_name="gemini-2.5-flash",
    prompt_template="Calculate: {question}",
    arguments={"question": "What is 15% of 847?"},
    tools=[calculate]
)
print(result)  # 127.05
```

That's a fully functional AI agent that uses tools. No annotations. No boilerplate. No ceremony. Just intent, directly expressed.
Well, you might ask "**Why not use LangChain, DSpy, OpenHands, MiniSweAgent, CrewAI, Google ADK, Claude Agent SDK, or some well-established agent frameworks?**" Here is my response:

- KISS comes with KISS Sorcar, a powerful local code IDE that is free and open-source.
- It has the GEPA prompt optimizer built-in with a simple API.
- It has a [`RelentlessAgent`](src/kiss/core/relentless_agent.py), which is pretty straightforward in terms of implementation, but it can work for very long tasks. It was self-evolved over time to reduce cost and running time.
- No bloat and simple codebase.
- New techniques will be incorporated to the framework as we research them.

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

## 💡 Key Features and Ideologies behind KISS and Sorcar

### Key Features

- **KISSAgent with ReAct Loop**: The core agent runs a generate-execute-observe loop with native function calling, automatic tool schema generation from Python function signatures and docstrings, trajectory saving, and per-step budget tracking.
- **RelentlessAgent for Long-Running Tasks**: Extends `Base` and uses `KISSAgent` for each sub-session, with auto-continuation across multiple sub-sessions (up to 10,000 by default). When a session runs out of steps, it **summarizes progress as a chronologically-ordered list of things the agent did with the reason for doing that along with relevant code snippets**, and continues in a new sub-session with the logged context, enabling agents to run for hours to days.
- **SorcarAgent with Coding and Browser Tools**: Provides `Read`, `Write`, `Edit`, and `Bash` (with streaming output and security-hardened command parsing) for coding tasks, plus full browser automation via Playwright with accessibility-tree-based element selection.
- **Browser-Based IDE**: Embeds `code-server` (VS Code in the browser) with a chatbot interface using Server-Sent Events for real-time streaming, task history and replay, AI-powered input autocomplete, a model selector with pricing info, merge views for reviewing agent changes, and theme syncing with VS Code.
- **Provider-Agnostic Multi-Model Support**: A clean `Model` abstraction supports Anthropic, OpenAI, Gemini, Together AI, OpenRouter (300+ models), and MiniMax — each with native function calling, token streaming, budget/cost calculation, and embedding generation.
- **GEPA Prompt Optimizer**: A Genetic-Pareto prompt optimization framework that evolves prompts through natural language reflection, instance-level Pareto frontiers, and structural merge — based on the paper "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning."
- **KISSEvolve for Algorithm Discovery**: An evolutionary framework using LLM-guided mutation, crossover, island-based evolution, novelty rejection sampling, and power-law parent sampling to discover novel algorithms.
- **RepoOptimizer for Iterative Code Improvement**: Wraps `SorcarAgent` to run a command, monitor output in real time, fix errors, and iteratively optimize code for specified metrics — tracking tried ideas to avoid repeating failed approaches.
- **Accessibility-Tree Browser Automation**: The `WebUseTool` uses Playwright's `aria_snapshot()` to extract the accessibility tree, numbers interactive elements for LLM reference, and avoids JavaScript injection to reduce bot detection.
- **Dynamic Configuration System**: Uses Pydantic models with a `add_config()` pattern that lets each subsystem register its own config at import time, with automatic CLI argument generation from the schema.
- **Trajectory Visualization**: A web-based UI for viewing complete agent execution histories including message flows, tool calls, token usage, and budget stats — with markdown rendering and syntax-highlighted code blocks.
- **Docker Integration**: A `DockerManager` for running agent tasks in isolated containers with automatic image pulling, lifecycle management, port mapping, and cleanup.

### Core Ideologies

- **Radical Simplicity over Abstraction**: KISS rejects the layered abstraction and ceremony found in frameworks like LangChain, CrewAI, or DSPy. Agents are just functions, orchestration is plain Python function composition, and tools are ordinary callables — no decorators, annotations, or boilerplate required.
- **No Bloat, No Overengineering**: Every function should do one thing well. After implementation, code is aggressively simplified — unnecessary attributes, variables, config options, conditional checks, and comments are removed. The codebase stays lean by design.
- **No Mocks, No Patches, No Test Doubles**: Tests must use real inputs and verify real outputs. Integration tests are heavily favored over unit tests. This philosophy ensures tests validate actual behavior rather than implementation assumptions.
- **100% Branch Coverage with Redundancy Detection**: The project targets full branch coverage and uses a custom [`redundancy_analyzer.py`](src/kiss/scripts/redundancy_analyzer.py) (powered by coverage.py's dynamic context feature) to identify and remove tests whose coverage is a strict subset of other tests — keeping the test suite minimal and meaningful.
- **Self-Improvement Loop**: The Sorcar agent maintains a `LESSONS.md` file where it records rules and patterns learned during tasks, reviewing them at the start of each new task to avoid repeating mistakes.
- **Free and Open-Source**: KISS Sorcar is a fully functional alternative to proprietary coding assistants like Cursor, with zero monthly subscription fees and complete source transparency under Apache 2.0.

## 🔧 Using Repo Optimizer

The `RepoOptimizer` ([`repo_optimizer.py`](src/kiss/agents/coding_agents/repo_optimizer.py)) uses the `SorcarAgent` to optimize code within your own project repository. It runs a specified command, monitors output in real time, fixes errors, and iteratively optimizes for specified metrics — all without changing the agent's interface. The code can be found [here](src/kiss/agents/coding_agents/repo_optimizer.py).

```bash
# Optimize a program for speed and cost
uv run python -m kiss.agents.coding_agents.repo_optimizer \
    --command "uv run python my_program.py" \
    --metrics "running time and cost" \
    --work-dir .
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--command` | (required) | Command to run and monitor |
| `--metrics` | (required) | Metrics to minimize (e.g., "running time and cost") |
| `--work-dir` | `.` | Working directory for the agent |

**How It Works:**

1. Runs the specified command and monitors output in real time
1. If repeated errors are observed, fixes the code and reruns
1. Once the command succeeds, analyzes output and optimizes the source to minimize the specified metrics
1. Repeats until the metrics are reduced significantly

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

The agent will automatically use the custom `finish` function instead of the default one which returns its argument. The function's parameters define what information the agent must provide, and the docstring helps the LLM understand how to format each field.

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

The visualizer provides:

- **Modern UI**: Dark theme with smooth animations
- **Sidebar Navigation**: List of all trajectories sorted by start time
- **Markdown Rendering**: Full markdown support for message content
- **Code Highlighting**: Syntax highlighting for fenced code blocks
- **Message Display**: Clean, organized view of agent conversations
- **Metadata Display**: Shows agent ID, model, steps, tokens, and budget information

![Trajectory Visualizer](assets/KISSTrajectory.png)

📖 **For detailed trajectory visualizer documentation, see [Trajectory Visualizer README](src/kiss/viz_trajectory/README.md)**

## 📦 Developer Installation

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

# Sorcar agent tools (web tools, browser UI)
uv sync --group sorcar

# Docker support (for running agents in isolated containers)
uv sync --group docker

# Development tools (mypy, ruff, pytest, etc.)
uv sync --group dev

# Combine multiple groups as needed
uv sync --group claude --group dev
```

**Dependency Group Contents:**

| Group | Description | Key Packages |
|-------|-------------|--------------|
| `core` | Minimal core module | pydantic, pydantic-settings, pyyaml, rich |
| `claude` | Core + Anthropic | core + anthropic |
| `openai` | Core + OpenAI | core + openai |
| `gemini` | Core + Google | core + google-genai |
| `sorcar` | Agent tools & browser UI | playwright, uvicorn, starlette |
| `docker` | Docker integration | docker |
| `dev` | Development tools | mypy, ruff, pyright, pytest, pytest-cov, mdformat |

> **Optional Dependencies:** All LLM provider SDKs (`openai`, `anthropic`, `google-genai`) are optional. You can import `kiss.core` and `kiss.agents` without installing all of them. When you try to use a model whose SDK is not installed, KISS raises a clear `KISSError` telling you which package to install.

## 📚 KISSAgent API Reference

📖 **For detailed KISSAgent API documentation, see [API.md](API.md)**

## 🎯 Using GEPA for Prompt Optimization

KISS has a fresh implementation of GEPA with some key improvements. GEPA (Genetic-Pareto) is a prompt optimization framework that uses natural language reflection to evolve prompts. It maintains an instance-level Pareto frontier of top-performing prompts and combines complementary lessons through structural merge. It also supports optional batched evaluation via `batched_agent_wrapper`, so you can plug in prompt-merging inference pipelines to process more datapoints per API call. GEPA is based on the paper ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457).

📖 **For detailed GEPA documentation, see [GEPA README](src/kiss/agents/gepa/README.md)**

## 🧪 Using KISSEvolve for Algorithm Discovery

This is where I started building an optimizer for agents. Then I switched to [`agent evolver`](src/kiss/agents/create_and_optimize_agent/agent_evolver.py) because `KISSEvolver` was expensive to run. Finally I switched to [`repo_optimizer`](src/kiss/agents/coding_agents/repo_optimizer.py) for efficiency and simplicity. I am still keeping KISSEvolve around. KISSEvolve is an evolutionary algorithm discovery framework that uses LLM-guided mutation and crossover to evolve code variants. It supports advanced features including island-based evolution, novelty rejection sampling, and multiple parent sampling methods.

For usage examples, API reference, and configuration options, please see the [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md).

📖 **For detailed KISSEvolve documentation, see [KISSEvolve README](src/kiss/agents/kiss_evolve/README.md)**

## 🐳 Docker Manager

KISS provides a `DockerManager` class for managing Docker containers and executing commands inside them. This is useful for running code in isolated environments, testing with specific dependencies, or working with SWE-bench tasks.

### Basic Usage

```python
from kiss.docker import DockerManager

# Create a Docker manager for an Ubuntu container
with DockerManager(image_name="ubuntu", tag="22.04", workdir="/app") as docker:
    # Run commands inside the container
    output = docker.Bash("echo 'Hello from Docker!'", "Print greeting")
    print(output)

    output = docker.Bash("python3 --version", "Check Python version")
    print(output)
```

### Manual Lifecycle Management

```python
from kiss.docker import DockerManager

docker = DockerManager(image_name="python", tag="3.11", workdir="/workspace")
docker.open()  # Pull image and start container

try:
    output = docker.Bash("pip install numpy", "Install numpy")
    output = docker.Bash("python -c 'import numpy; print(numpy.__version__)'", "Check numpy")
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
    docker.Bash("nginx", "Start nginx")
    
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
│   │   ├── sorcar/                 # Sorcar agent with coding + browser tools
│   │   │   ├── sorcar_agent.py         # SorcarAgent with coding and browser automation
│   │   │   ├── sorcar.py              # Browser-based Sorcar UI
│   │   │   ├── browser_ui.py           # Browser UI base components and BaseBrowserPrinter
│   │   │   ├── chatbot_ui.py           # Chatbot UI templates: CSS, JavaScript, HTML
│   │   │   ├── code_server.py          # Code-server setup and git diff/merge utilities
│   │   │   ├── prompt_detector.py      # Prompt detection utilities
│   │   │   ├── task_history.py         # Task history, proposals, and file usage persistence
│   │   │   ├── useful_tools.py         # UsefulTools class with Read, Write, Bash, Edit
│   │   │   ├── web_use_tool.py         # WebUseTool with Playwright-based browser automation
│   │   │   └── config.py               # Sorcar agent configuration
│   │   ├── coding_agents/          # Coding agents for software development tasks
│   │   │   ├── repo_optimizer.py          # Iterative code optimizer using SorcarAgent
│   │   │   ├── repo_agent.py              # Repo-level task agent
│   │   │   ├── agent_optimizer.py         # Meta-optimizer for agent source code
│   │   │   └── config.py                  # Coding agent configuration
│   │   ├── gepa/                   # GEPA (Genetic-Pareto) prompt optimizer
│   │   │   ├── gepa.py
│   │   │   ├── config.py
│   │   │   └── README.md
│   │   ├── kiss_evolve/            # KISSEvolve evolutionary algorithm discovery
│   │   │   ├── kiss_evolve.py
│   │   │   ├── novelty_prompts.py  # Prompts for novelty-based evolution
│   │   │   ├── simple_rag.py       # Simple RAG with in-memory vector store
│   │   │   ├── config.py
│   │   │   └── README.md
│   │   ├── create_and_optimize_agent/  # Agent evolution (deprecated)
│   │   │   ├── agent_evolver.py
│   │   │   ├── improver_agent.py
│   │   │   ├── config.py
│   │   │   └── README.md
│   │   ├── self_evolving_multi_agent/  # Self-evolving multi-agent (deprecated)
│   │   │   ├── agent_evolver.py
│   │   │   ├── multi_agent.py
│   │   │   ├── config.py
│   │   │   └── README.md
│   │   ├── autoresearch/           # Autoresearch agent for autonomous ML experiments
│   │   │   ├── autoresearch_agent.py
│   │   │   ├── config.py
│   │   │   └── README.md
│   │   └── kiss.py                 # Utility agents (prompt refiner, bash agent)
│   ├── core/            # Core framework components
│   │   ├── base.py            # Base class with common functionality
│   │   ├── kiss_agent.py      # KISS agent with native function calling
│   │   ├── printer.py         # Abstract Printer base class and MultiPrinter
│   │   ├── print_to_console.py # ConsolePrinter: Rich-formatted terminal output
│   │   ├── config.py          # Configuration
│   │   ├── config_builder.py  # Dynamic config builder with CLI support
│   │   ├── kiss_error.py      # Custom error class
│   │   ├── utils.py           # Utility functions
│   │   ├── relentless_agent.py # RelentlessAgent base class
│   │   └── models/            # Model implementations
│   │       ├── model.py           # Model interface with Attachment support
│   │       ├── gemini_model.py    # Gemini model implementation
│   │       ├── openai_compatible_model.py # OpenAI-compatible API model
│   │       ├── anthropic_model.py # Anthropic model implementation
│   │       └── model_info.py      # Model info: pricing, context, capabilities
│   ├── channels/        # Communication channel integrations
│   │   ├── gmail_agent.py          # Gmail agent with OAuth2 authentication
│   │   └── slack_agent.py          # Slack bot agent
│   ├── docker/          # Docker integration
│   │   └── docker_manager.py
│   ├── scripts/         # Utility scripts
│   │   ├── check.py                    # Code quality check script
│   │   ├── fix.py                      # Code fix/formatting script
│   │   ├── generate_api_docs.py        # API documentation generator
│   │   ├── redundancy_analyzer.py      # Code redundancy analyzer
│   │   └── update_models.py            # Model info updater script
│   ├── tests/           # Test suite
│   │   ├── _sorcar_test_server.py
│   │   ├── _sorcar_test_server_with_cov.py
│   │   ├── conftest.py
│   │   ├── run_all_models_test.py          # Run tests across all models
│   │   ├── test_a_model.py
│   │   ├── test_ask_user_browser_action.py
│   │   ├── test_ask_user_browser_multistep.py
│   │   ├── test_ask_user_question.py
│   │   ├── test_assistant_multi_session.py
│   │   ├── test_assistant_redundancies.py
│   │   ├── test_autoresearch.py
│   │   ├── test_browser_close_shutdown.py
│   │   ├── test_chat_history_events.py
│   │   ├── test_chatbot_ui_spinner.py
│   │   ├── test_chatbot_ui.py
│   │   ├── test_clear_btn_welcome.py
│   │   ├── test_cli_options.py
│   │   ├── test_code_review_fixes.py
│   │   ├── test_code_server_keybinding.py
│   │   ├── test_code_server_watchdog.py
│   │   ├── test_commit_push.py
│   │   ├── test_core_branch_coverage.py
│   │   ├── test_cs_data_dir_isolation.py
│   │   ├── test_current_editor_file.py
│   │   ├── test_disable_copilot_scm.py
│   │   ├── test_docker_manager.py
│   │   ├── test_error_handling.py
│   │   ├── test_evolver_progress_callback.py
│   │   ├── test_file_usage.py
│   │   ├── test_gemini_model_internals.py
│   │   ├── test_generate_api_docs.py
│   │   ├── test_gepa_batched.py
│   │   ├── test_gepa_progress_callback.py
│   │   ├── test_gepa_sanitize.py
│   │   ├── test_gmail_agent.py
│   │   ├── test_integration_branch_coverage.py
│   │   ├── test_internal.py
│   │   ├── test_kiss_agent_agentic.py
│   │   ├── test_kiss_agent_coverage.py
│   │   ├── test_kiss_agent_non_agentic.py
│   │   ├── test_merge_restore_on_close.py
│   │   ├── test_merge_view_second_change.py
│   │   ├── test_model_base_class.py
│   │   ├── test_model_implementations.py
│   │   ├── test_multimodal.py
│   │   ├── test_print_to_browser.py
│   │   ├── test_print_to_console.py
│   │   ├── test_printer_parity.py
│   │   ├── test_race_conditions.py
│   │   ├── test_redundancy_analyzer.py
│   │   ├── test_run_prompt_button.py
│   │   ├── test_scm_commit_message.py
│   │   ├── test_slack_agent.py
│   │   ├── test_sorcar_bash_streaming.py
│   │   ├── test_sorcar_branch_coverage.py
│   │   ├── test_sorcar_cli_callbacks.py
│   │   ├── test_sorcar_coverage.py
│   │   ├── test_sorcar_cs_integ.py
│   │   ├── test_sorcar_instance_isolation.py
│   │   ├── test_sorcar_integ_branch.py
│   │   ├── test_sorcar_integration.py
│   │   ├── test_sorcar_race_conditions.py
│   │   ├── test_sorcar_race_fixes.py
│   │   ├── test_sorcar_run_selection.py
│   │   ├── test_sorcar_server_endpoints.py
│   │   ├── test_sorcar_text_wrap.py
│   │   ├── test_sse_reconnection.py
│   │   ├── test_stop_agent_thread.py
│   │   ├── test_stream_event_parser.py
│   │   ├── test_task_history_jsonl.py
│   │   ├── test_task_history_scale.py
│   │   ├── test_token_callback.py
│   │   ├── test_tool_exception_handling.py
│   │   ├── test_uncovered_branches.py
│   │   ├── test_update_models_preview.py
│   │   ├── test_useful_tools.py
│   │   ├── test_vscode_panel.py
│   │   ├── test_web_use_tool.py
│   │   ├── integration_test_assistant_agent.py
│   │   ├── integration_test_gmail_login.py
│   │   ├── integration_test_google_search.py
│   │   └── integration_test_web_use_tool.py
│   ├── _version.py       # Single source of truth for package version
│   ├── py.typed          # PEP 561 marker for type checking
│   └── viz_trajectory/  # Trajectory visualization
│       ├── server.py                    # Flask server for trajectory visualization
│       ├── README.md
│       └── templates/
│           └── index.html
├── scripts/             # Repository-level scripts
│   ├── release.sh                  # Release script
│   ├── test_fresh_lib_install.sh   # Fresh library install test
│   └── test_fresh_repo_install.sh  # Fresh repo install test
├── API.md               # KISSAgent API reference
├── example_prompt.md    # Example prompt for the assistant
├── install.sh           # Installation script (development)
├── installlib.sh        # Installation script (library)
├── LICENSE              # Apache-2.0 license
├── pyproject.toml       # Project configuration
└── README.md
```

## 🏷️ Versioning

The project uses semantic versioning (MAJOR.MINOR.PATCH). The version is defined in a single source of truth:

- **Version file**: [`src/kiss/_version.py`](src/kiss/_version.py) - Edit this file to update the version
- **Package access**: `kiss.__version__` - Access the version programmatically
- **Build system**: [`pyproject.toml`](pyproject.toml) automatically reads the version from [`_version.py`](src/kiss/_version.py) using dynamic versioning

Example:

```python
from kiss import __version__
print(f"KISS version: {__version__}")
```

To update the version, simply edit [`src/kiss/_version.py`](src/kiss/_version.py):

```python
__version__ = "0.2.0"  # Update to new version
```

## ⚙️ Configuration

Configuration is managed through environment variables and the `DEFAULT_CONFIG` object:

- **API Keys**: Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, and/or `MINIMAX_API_KEY` environment variables
- **Agent Settings**: Modify `DEFAULT_CONFIG.agent` in [`src/kiss/core/config.py`](src/kiss/core/config.py):
  - `max_steps`: Maximum iterations in the ReAct loop (default: 100)
  - `verbose`: Enable verbose output (default: True)
  - `debug`: Enable debug mode (default: False)
  - `max_agent_budget`: Maximum budget per agent run in USD (default: 10.0)
  - `global_max_budget`: Maximum total budget across all agents in USD (default: 200.0)
  - `artifact_dir`: Directory for agent artifacts (default: auto-generated with timestamp)
- **Relentless Coding Agent Settings**: Modify `DEFAULT_CONFIG.coding_agent.relentless_coding_agent` in [`src/kiss/agents/coding_agents/config.py`](src/kiss/agents/coding_agents/config.py):
  - `model_name`: Model for task execution (default: "claude-opus-4-6")
  - `max_steps`: Maximum steps per sub-session (default: 100)
  - `max_budget`: Maximum budget in USD (default: 200.0)
  - `max_sub_sessions`: Maximum number of sub-sessions for auto-continuation (default: 10000)
- **GEPA Settings**: Modify `DEFAULT_CONFIG.gepa` in [`src/kiss/agents/gepa/config.py`](src/kiss/agents/gepa/config.py):
  - `reflection_model`: Model to use for reflection (default: "gemini-3-flash-preview")
  - `max_generations`: Maximum number of evolutionary generations (default: 10)
  - `population_size`: Number of candidates to maintain in population (default: 8)
  - `pareto_size`: Maximum size of Pareto frontier (default: 4)
  - `mutation_rate`: Probability of mutating a prompt template (default: 0.5)
- **KISSEvolve Settings**: Modify `DEFAULT_CONFIG.kiss_evolve` in [`src/kiss/agents/kiss_evolve/config.py`](src/kiss/agents/kiss_evolve/config.py):
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
- **Self-Evolving Multi-Agent Settings**: Modify `DEFAULT_CONFIG.self_evolving_multi_agent` in [`src/kiss/agents/self_evolving_multi_agent/config.py`](src/kiss/agents/self_evolving_multi_agent/config.py):
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
- `uv sync --group dev` - Install dev tools (mypy, ruff, pyright, pytest, etc.)
- `uv sync --group <name>` - Install specific dependency group (see [Selective Installation](#selective-installation-dependency-groups))
- `uv build` - Build the project package

### Testing

- `uv run pytest` - Run all tests (uses testpaths from pyproject.toml)
- `uv run pytest src/kiss/tests/ -v` - Run all tests with verbose output
- `uv run pytest src/kiss/tests/test_kiss_agent_agentic.py -v` - Run agentic agent tests
- `uv run pytest src/kiss/tests/test_kiss_agent_non_agentic.py -v` - Run non-agentic agent tests
- `uv run python -m unittest src.kiss.tests.test_docker_manager -v` - Run docker manager tests (unittest)
- `uv run python -m unittest discover -s src/kiss/tests -v` - Run all tests using unittest

### Code Quality

- `uv run check` - Run all code quality checks (fresh dependency install, build, lint, and type check)
- `uv run check --clean-only` - Only clean build artifacts, do not run checks
- `uv run ruff format src/` - Format code with ruff (line-length: 100, target: py313)
- `uv run ruff check src/` - Lint code with ruff (selects: E, F, W, I, N, UP)
- `uv run mypy src/` - Type check with mypy (python_version: 3.13)
- `uv run pyright src/` - Type check with pyright (alternative to mypy, stricter checking)

### Sorcar

- `uv run sorcar` - Launch the browser-based Sorcar UI (coding + browser automation)
- `uv run sorcar ./my-project` - Launch with custom working directory
- `uv run sorcar --model_name "gemini-2.5-pro"` - Launch with a specific default model

### Documentation

- `uv run generate-api-docs` - Generate API documentation

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
- **OpenRouter**: Access to 300+ models from 60+ providers via unified API:
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

> **Note**: Additional models can be used, but context length, pricing, and capability information must be added to [`src/kiss/core/models/model_info.py`](src/kiss/core/models/model_info.py) for accurate token tracking, budget monitoring, and test filtering.

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

Embeddings are primarily used by the `SimpleRAG` system ([`src/kiss/agents/kiss_evolve/simple_rag.py`](src/kiss/agents/kiss_evolve/simple_rag.py)) for document retrieval. When using `SimpleRAG`, ensure you use an OpenAI, Together AI, or Gemini model that supports embeddings.

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
