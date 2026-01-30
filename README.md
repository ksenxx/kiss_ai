# KISS Framework

**Keep It Simple, Stupid** - A modern AI agent framework built on radical simplicity.

## Overview

KISS is a Python framework for building AI agents that emphasizes clarity, composability, and evolutionary optimization. It provides a clean API for creating ReAct agents (Reasoning + Acting) with native function calling support across multiple LLM providers.

### Key Philosophy

- **Simple Core**: Core agent implementation in ~300 lines of code
- **No Complexity**: No state machines, workflow graphs, or unnecessary abstractions
- **Native Function Calling**: Leverage built-in function calling from LLM providers
- **Function Composition**: Multi-agent systems through simple Python functions
- **Evolutionary**: Agents, prompts, and code that self-improve through optimization

## Features

### Core Capabilities

- **🤖 ReAct Agents** - Simple reasoning + acting loop with automatic tool schema generation
- **🔧 Native Function Calling** - Direct support for OpenAI, Anthropic, Google, Together AI, OpenRouter
- **💰 Budget Tracking** - Automatic token counting, cost calculation, and budget limits
- **🔄 Multi-Agent Orchestration** - Compose agents through simple function calls
- **🐳 Docker Integration** - Sandboxed code execution in containers
- **🔍 RAG System** - Simple in-memory vector store for document retrieval
- **📊 Trajectory Visualization** - Web-based UI for viewing agent conversations

### Advanced Features

- **🧬 Evolutionary Optimization**
  - **GEPA**: Genetic-Pareto prompt evolution
  - **KISSEvolve**: Algorithm discovery through code evolution
  - **AgentEvolver**: Optimize entire agent implementations
- **🎯 Specialized Coding Agents**
  - **KISSCodingAgent**: Multi-agent coding with planning and orchestration
  - **ClaudeCodingAgent**: Uses Claude Agent SDK
  - **GeminiCliAgent**: Uses Google ADK
  - **OpenAICodexAgent**: Uses OpenAI Agents SDK
- **🔒 Path Access Control** - Restrict agent file system access
- **⚡ Parallel Execution** - Built-in multiprocessing utilities

## Quick Start

### Installation

```bash
pip install kiss-framework
```

### Basic Agent Example

```python
from kiss.core import KISSAgent

# Define a simple tool
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b

# Create and run agent
agent = KISSAgent("calculator")
result = agent.run(
    model_name="gpt-4o",
    prompt_template="What is 15 + 27?",
    tools=[add],
    max_steps=10,
    max_budget=1.0
)
print(result)  # "42"
```

### Multi-Agent Coding Example

```python
from kiss.core import KISSCodingAgent

agent = KISSCodingAgent("coder")
result = agent.run(
    model_name="claude-sonnet-4-5",
    prompt_template="""
        Write a fibonacci function in Python with tests.
        Save it to fibonacci.py and run the tests.
    """,
    readable_paths=["src/"],
    writable_paths=["output/"],
    trials=3
)
print(result)
```

### Evolutionary Optimization Example

```python
import anyio
from kiss.agents.agent_creator import AgentEvolver

async def main():
    evolver = AgentEvolver(
        task_description="Build a code analysis assistant",
        max_generations=5,
        max_frontier_size=4
    )

    best_variant = await evolver.evolve()
    print(f"Best variant: {best_variant.folder_path}")
    print(f"Tokens used: {best_variant.tokens_used}")
    print(f"Execution time: {best_variant.execution_time:.2f}s")

anyio.run(main)
```

## Architecture

### Core Components

```
kiss/
├── core/                      # Core framework (~500 lines)
│   ├── kiss_agent.py         # ReAct agent implementation
│   ├── kiss_coding_agent.py  # Multi-agent coding orchestrator
│   ├── claude_coding_agent.py # Claude SDK integration
│   ├── gemini_cli_agent.py   # Google ADK integration
│   ├── openai_codex_agent.py # OpenAI SDK integration
│   ├── base.py               # Common agent functionality
│   ├── config.py             # Configuration system
│   └── models/               # LLM provider implementations
│
├── agents/                    # Pre-built agents
│   ├── kiss.py               # Utility agent functions
│   ├── gepa/                 # Prompt evolution
│   ├── kiss_evolve/          # Algorithm discovery
│   └── agent_creator/        # Agent program evolution
│
├── docker/                    # Container management
├── rag/                       # RAG system
├── multiprocessing/          # Parallel execution
└── viz_trajectory/           # Web visualization
```

### Agent Types

| Agent | Purpose | Key Feature |
|-------|---------|-------------|
| **KISSAgent** | General ReAct agent | Native function calling |
| **KISSCodingAgent** | Complex coding tasks | Multi-agent orchestration with planner |
| **ClaudeCodingAgent** | Claude SDK coding | Built-in file operations (Read, Write, Edit) |
| **GeminiCliAgent** | Google ADK coding | Google-optimized tooling |
| **OpenAICodexAgent** | OpenAI SDK coding | OpenAI-specific features |

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4.1, GPT-4o, GPT-5, GPT-5.2 series |
| **Anthropic** | Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 |
| **Google** | Gemini 2.5/3 Pro, Gemini Flash |
| **Together AI** | Llama 4, Qwen 3, DeepSeek R1/V3 |
| **OpenRouter** | 400+ models from all providers |

Switch models with a single parameter: `model_name="gpt-4o"` or `model_name="claude-sonnet-4-5"`

## Configuration

KISS uses a Pydantic-based configuration system. Set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export SERPER_API_KEY="your-key"  # Optional: for web search
```

Or configure programmatically:

```python
from kiss.core.config import DEFAULT_CONFIG

DEFAULT_CONFIG.api_keys.openai_api_key = "your-key"
DEFAULT_CONFIG.agent.max_steps = 100
DEFAULT_CONFIG.agent.max_agent_budget = 10.0
DEFAULT_CONFIG.agent.verbose = True
```

## Advanced Usage

### Custom Tools

Tools are regular Python functions with type hints and docstrings:

```python
def search_database(query: str, limit: int = 10) -> str:
    """Search the database for relevant entries.

    Args:
        query: The search query string
        limit: Maximum number of results to return

    Returns:
        JSON string containing search results
    """
    # Implementation
    results = db.search(query, limit=limit)
    return json.dumps(results)

agent = KISSAgent("searcher")
result = agent.run(
    model_name="gpt-4o",
    prompt_template="Find information about {topic}",
    arguments={"topic": "machine learning"},
    tools=[search_database]
)
```

### Path Access Control

Restrict file system access for coding agents:

```python
agent = KISSCodingAgent("secure_coder")
result = agent.run(
    model_name="gpt-4o",
    prompt_template="Analyze and improve code in src/",
    readable_paths=["src/", "tests/"],
    writable_paths=["output/"],
    base_dir="workdir"
)
```

The agent will automatically validate file access before operations.

### Docker Sandboxing

Run code in isolated containers:

```python
from kiss.docker import DockerManager

with DockerManager("ubuntu:latest", ports={80: 8080}) as env:
    output = env.run_bash_command(
        "apt-get update && apt-get install -y python3",
        "Install Python"
    )
    print(output)

    # Get mapped host port
    host_port = env.get_host_port(80)
    print(f"Container port 80 mapped to host port {host_port}")
```

### RAG System

Simple document retrieval:

```python
from kiss.rag import SimpleRAG

rag = SimpleRAG(model_name="gpt-4o")

# Add documents
documents = [
    {
        "id": "doc1",
        "text": "Python is a programming language",
        "metadata": {"source": "manual"}
    },
    {
        "id": "doc2",
        "text": "Machine learning uses algorithms",
        "metadata": {"source": "textbook"}
    }
]
rag.add_documents(documents)

# Query
results = rag.query("What is Python?", top_k=2)
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']}")
```

### Prompt Evolution with GEPA

Automatically optimize prompts:

```python
from kiss.agents.gepa import GEPA

def agent_wrapper(prompt_template, arguments):
    agent = KISSAgent("test_agent")
    result = agent.run(
        model_name="gpt-4o",
        prompt_template=prompt_template,
        arguments=arguments
    )
    return result, agent.get_trajectory()

def evaluation_fn(result):
    # Return dict of metrics (higher is better)
    return {"accuracy": compute_accuracy(result)}

gepa = GEPA(
    agent_wrapper=agent_wrapper,
    initial_prompt_template="Solve: {problem}",
    evaluation_fn=evaluation_fn,
    max_generations=10
)

# Train with examples
train_examples = [
    {"problem": "What is 2+2?"},
    {"problem": "What is 10*5?"},
]

best_candidate = gepa.optimize(train_examples)
print(f"Optimized prompt: {best_candidate.prompt_template}")
```

### Algorithm Discovery with KISSEvolve

Evolve code through generations:

```python
from kiss.agents.kiss_evolve import KISSEvolve

def code_agent_wrapper(model_name, prompt_template, arguments):
    agent = KISSAgent("coder")
    return agent.run(
        model_name=model_name,
        prompt_template=prompt_template,
        arguments=arguments,
        is_agentic=False
    )

def evaluation_fn(code):
    # Test the code and return metrics
    try:
        exec(code)
        return {
            "fitness": compute_performance(code),
            "metrics": {"speed": measure_speed(code)}
        }
    except Exception as e:
        return {"fitness": 0.0, "error": str(e)}

evolver = KISSEvolve(
    code_agent_wrapper=code_agent_wrapper,
    initial_code="def sort(arr): return sorted(arr)",
    evaluation_fn=evaluation_fn,
    model_names=[("gpt-4o", 0.7), ("claude-sonnet-4-5", 0.3)],
    population_size=10,
    max_generations=20
)

best_variant = evolver.evolve()
print(f"Best code: {best_variant.code}")
print(f"Fitness: {best_variant.fitness}")
```

### Parallel Execution

Run multiple agents concurrently:

```python
from kiss.multiprocessing import run_functions_in_parallel

def task1():
    agent = KISSAgent("agent1")
    return agent.run(model_name="gpt-4o", prompt_template="Task 1")

def task2():
    agent = KISSAgent("agent2")
    return agent.run(model_name="gpt-4o", prompt_template="Task 2")

tasks = [(task1, []), (task2, [])]
results = run_functions_in_parallel(tasks)
print(results)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest src/kiss/tests/

# Run specific test file
pytest src/kiss/tests/test_kiss_agent_agentic.py -v

# Run with coverage
pytest src/kiss/tests/ --cov=kiss
```

## Visualization

View agent trajectories in a web UI:

```python
from kiss.viz_trajectory import server

# Start visualization server
server.run(port=8080)
```

Navigate to `http://localhost:8080` to view saved agent trajectories.

## Best Practices

### 1. Budget Management

Always set reasonable budget limits:

```python
agent.run(
    model_name="gpt-4o",
    prompt_template="...",
    max_budget=5.0,  # Per-agent limit
    max_steps=50     # Prevent infinite loops
)
```

Global budget in config:

```python
DEFAULT_CONFIG.agent.global_budget = 100.0
```

### 2. Tool Design

Make tools focused and well-documented:

```python
def good_tool(specific_param: str) -> str:
    """Clear single-purpose description.

    Args:
        specific_param: Exact description of input

    Returns:
        Exact description of output
    """
    return result

# Avoid: tools that do too many things
# Avoid: vague parameter descriptions
```

### 3. Error Handling

Let tools return error strings rather than raising exceptions:

```python
def safe_tool(param: str) -> str:
    """Tool with error handling."""
    try:
        result = risky_operation(param)
        return f"Success: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. Prompt Templates

Use clear, structured prompts:

```python
prompt_template = """
You are a {role} agent. Your task is to {task}.

Input: {input}

Instructions:
1. First, analyze the input
2. Then, use available tools
3. Finally, call finish() with your result

Remember to {constraint}.
"""
```

### 5. Multi-Agent Composition

Keep orchestration simple:

```python
# Good: Simple sequential composition
planner = KISSAgent("planner")
plan = planner.run(
    model_name="gpt-4o",
    prompt_template="Create plan for: {task}",
    arguments={"task": user_task}
)

executor = KISSAgent("executor")
result = executor.run(
    model_name="gpt-4o",
    prompt_template="Execute plan: {plan}",
    arguments={"plan": plan},
    tools=[relevant_tools]
)

# Avoid: Complex orchestration frameworks
```

## Project Structure

```
kiss/
├── src/kiss/
│   ├── core/                 # Core framework
│   ├── agents/              # Pre-built agents
│   ├── docker/              # Container management
│   ├── rag/                 # RAG system
│   ├── multiprocessing/    # Parallel execution
│   ├── viz_trajectory/     # Visualization
│   └── tests/              # Test suite
├── README.md               # This file
├── API.md                  # Detailed API reference
├── pyproject.toml          # Package configuration
└── setup.py                # Package setup
```

## Contributing

Contributions are welcome! Please:

1. Follow the KISS philosophy - simplicity over complexity
1. Add tests for new features
1. Update documentation
1. Ensure backwards compatibility

## License

[Add your license information here]

## Documentation

- [API Reference](API.md) - Detailed API documentation
- [Examples](examples/) - Example scripts and notebooks
- [Tests](src/kiss/tests/) - Test suite with usage examples

## Citation

If you use KISS in your research, please cite:

```bibtex
@software{kiss_framework,
  title = {KISS: Keep It Simple, Stupid - AI Agent Framework},
  author = {[Author Names]},
  year = {2025},
  url = {https://github.com/[username]/kiss}
}
```

## Acknowledgments

KISS builds on ideas from:

- ReAct: Reasoning and Acting in Language Models
- GEPA: Reflective Prompt Evolution
- Evolutionary Algorithms and Genetic Programming

Special thanks to the LLM provider teams (OpenAI, Anthropic, Google) for their excellent APIs.
