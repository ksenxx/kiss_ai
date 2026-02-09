![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/818u234myu55pxt0wi7j.jpeg)

# When Simplicity Becomes Your Superpower: Meet KISS Agent Framework

*"Everything should be made as simple as possible, but not simpler." — Albert Einstein*

______________________________________________________________________

## 🎯 The Problem with AI Agent Frameworks Today

Let's be honest. The AI agent ecosystem has become a jungle.

Every week brings a new framework promising to revolutionize how we build AI agents. They come loaded with abstractions on top of abstractions, configuration files that rival tax forms, and dependency trees that make `node_modules` look tidy. By the time you've figured out how to make your first tool call, you've already burned through half your patience and all your enthusiasm.

**What if there was another way?**

What if building AI agents could be as straightforward as the name suggests?

Enter **KISS** — the *Keep It Simple, Stupid* Agent Framework.

______________________________________________________________________

## 💡 The Philosophy: Radical Simplicity

KISS isn't just a clever acronym. It's a design philosophy that permeates every line of code in this framework.

Born of the frustration of wrestling with overly complex agent architectures, KISS strips away the unnecessary and focuses on what actually matters: **getting intelligent agents to solve real problems**. The KISS Agent API is simple and straightforward. It is so simple that a coding agent can write complex AI pipelines, called **agent programs**, from natural language descriptions. You can also optimize an agent program using a KISS evolver. The result is a self-evolving agent program.

Every KISS agent is a ReAct agent by default:

```
1. You give the agent a prompt
2. The agent thinks and calls tools
3. Repeat until done
4. That's it. That's the framework.
```

No workflow graphs. No state machines. No PhD required.

______________________________________________________________________

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

## 🤝 Multi-Agent Orchestration: Agents That Improve Each Other

Here's where KISS really shines — composing multiple agents into systems greater than the sum of their parts.

KISS includes utility agents that work beautifully together. Let's build a **self-improving coding agent** that writes code, tests it, and refines its own prompts based on failures:

```python
import json
from kiss.core.kiss_agent import KISSAgent
from kiss.agents.kiss import prompt_refiner_agent, get_run_simple_coding_agent

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
        current_prompt = prompt_refiner_agent(
            original_prompt_template=original_prompt,
            previous_prompt_template=current_prompt,
            agent_trajectory_summary=trajectory,
            model_name="gpt-4o"
        )
        print(f"📝 New prompt:\n{current_prompt[:200]}...")
```

**What's happening here?**

1. **Coding Agent** ['get_run_simple_coding_agent'](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/agents/kiss.py): Generates code and validates it against test cases
1. **Prompt Refiner Agent** [`prompt_refiner_agent`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/agents/kiss.py): Analyzes failures and refines the prompt
1. **Orchestration**: A simple Python loop coordinates the agents

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

______________________________________________________________________

______________________________________________________________________

## 🧬 GEPA: Teaching Your Agents to Evolve

But KISS isn't just about simplicity — it's about *intelligent* simplicity.

Meet **GEPA** (Genetic-Pareto Prompt Evolution), a prompt optimization system that sounds like science fiction but works like magic.

Traditional prompt engineering is like trying to tune a guitar by ear in a noisy room. You make changes, hope for the best, and repeat until exhaustion or deadline, whichever comes first.

GEPA takes a different approach:

```
1. Run your agent
2. Reflect on what went wrong (using AI!)
3. Evolve the prompt based on insights
4. Maintain a Pareto frontier of best performers
5. Combine winning strategies through crossover
6. Repeat until your prompts are superhuman
```

This isn't just iteration — it's **evolution**. GEPA maintains multiple prompt candidates, each optimized for different objectives. Want an agent that's both accurate AND concise? GEPA finds the sweet spot on the Pareto frontier.

```python
from kiss.agents.gepa import GEPA

gepa = GEPA(
    agent_wrapper=my_agent_function,
    initial_prompt_template="You are a helpful assistant...",
    evaluation_fn=score_the_result,
    max_generations=10,
    population_size=8
)

best = gepa.optimize(
    train_examples=[
        {"task": "Write a poem"},
        {"task": "Explain physics"},
        {"task": "Create a recipe"},
        {"task": "Describe ML"},
    ]
)
# Your prompt just evolved beyond human-crafted limits
```

The research paper backing this? ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457). Yes, prompt evolution can beat RL. Let that sink in.

______________________________________________________________________

## 🔬 KISSEvolve: When Algorithms Write Themselves

Here's where things get really interesting.

What if you could start with a bubble sort and end up with quicksort — without writing a single line of sorting code yourself?

**KISSEvolve** is an evolutionary algorithm discovery framework. You provide:

- Starting code (even a naive implementation)
- A fitness function
- An LLM to guide mutations
- It includes features of OpenEvolve and some new ideas.

KISSEvolve does the rest:

```python
from kiss.agents.kiss_evolve.kiss_evolve import KISSEvolve
from kiss.core.kiss_agent import KISSAgent

# Start with O(n²) bubble sort
initial_code = """
def sort_array(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

def code_agent_wrapper(model_name: str, prompt_template: str, arguments: dict) -> str:
    agent = KISSAgent(name="Code Optimizer")
    return agent.run(model_name=model_name, prompt_template=prompt_template, arguments=arguments)

optimizer = KISSEvolve(
    code_agent_wrapper=code_agent_wrapper,
    initial_code=initial_code,
    evaluation_fn=measure_performance,
    model_names=[("gemini-3-flash-preview", 0.5), ("gemini-3-pro-preview", 0.5)],
    population_size=8,
    max_generations=10
)

best = optimizer.evolve()
# Discovers O(n log n) algorithms like quicksort or mergesort!
```

The framework includes advanced features that would make any evolutionary computation researcher smile:

- **Island-Based Evolution**: Multiple populations evolving in parallel with periodic migration
- **Novelty Rejection Sampling**: Ensures diversity by filtering redundant solutions
- **Power-Law & Performance-Novelty Sampling**: Sophisticated parent selection strategies
- **Multi-Model Support**: Use different LLMs with configurable probabilities

This isn't theoretical. The included `kissevolve_bubblesort.py` script demonstrates how to find an O(n log n) sorting algorithm.

______________________________________________________________________

## 🏗️ Real-World Ready: SWE-bench Integration

KISS isn't a toy. It's battle-tested against one of the most challenging AI benchmarks: **SWE-bench Verified**.

SWE-bench presents AI systems with real GitHub issues from major Python repositories. The task? Generate patches that actually fix the bugs. It's the software engineering equivalent of a PhD qualifying exam.

KISS includes first-class support:

```bash
uv run src/kiss/evals/swe_agent_verified/run_swebench.py \
    --swebench_verified.model gemini-3-flash-preview \
    --swebench_verified.instance_id "django__django-11099"
```

The agent:

1. Spins up a Docker container with the exact repository state
1. Reads the issue description
1. Explores the codebase using bash commands
1. Generates a patch
1. Gets evaluated by the official SWE-bench harness

All with built-in trajectory saving, budget tracking, and automatic evaluation.

______________________________________________________________________

## 🌐 Model Agnostic: Your LLM, Your Choice

KISS doesn't lock you into any single provider. Out of the box, it supports:

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4.1, GPT-4o, GPT-5 series |
| **Anthropic** | Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 |
| **Google** | Gemini 2.5/3 Pro, Gemini Flash |
| **Together AI** | Llama 4, Qwen 3, DeepSeek R1/V3 |
| **OpenRouter** | 400+ models from all providers |

Each model includes accurate pricing, context length limits, and capability flags. Token usage and costs are tracked automatically across all agent runs.

```python
# Switch models with a single parameter
result = agent.run(model_name="claude-sonnet-4-5", ...)
result = agent.run(model_name="gemini-3-pro-preview", ...)
result = agent.run(model_name="openrouter/x-ai/grok-4", ...)
```

______________________________________________________________________

## 🐳 Docker Integration: Safe Sandboxing

Giving AI agents the ability to execute code is powerful but dangerous. KISS includes a `DockerManager` that makes sandboxing trivial:

```python
from kiss.docker.docker_manager import DockerManager

with DockerManager("ubuntu:latest") as env:
    agent = KISSAgent(name="Safe Agent")
    result = agent.run(
        model_name="gemini-3-flash-preview",
        prompt_template="Install nginx and configure it",
        tools=[env.run_bash_command]
    )
```

The agent can execute any bash command, but it's all contained. When the context manager exits, the container is destroyed. Your host system remains pristine.

______________________________________________________________________

## 📊 Trajectory Visualization: See What Your Agents Think

Debugging AI agents is notoriously difficult. What was the agent thinking? Why did it make that tool call?

KISS automatically saves complete trajectories to YAML files. But reading YAML isn't fun. That's why KISS includes a **web-based trajectory visualizer**:

```bash
uv run python -m kiss.viz_trajectory.server artifacts
```

Open your browser and you get:

- Dark-themed modern UI
- Markdown rendering with syntax highlighting
- Complete message history with timestamps
- Token usage and budget tracking per step
- Tool calls and their results

It transforms agent debugging from archaeology into insight.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/dsvaylst4o0mpmg737kb.png)

______________________________________________________________________

## 🔍 RAG Made Simple

Need retrieval-augmented generation? KISS includes `SimpleRAG`:

```python
from kiss.rag import SimpleRAG

rag = SimpleRAG(model_name="gpt-4o", metric="cosine")

rag.add_documents([
    {"id": "1", "text": "Python is a programming language", "metadata": {"topic": "coding"}},
    {"id": "2", "text": "Machine learning uses algorithms", "metadata": {"topic": "AI"}},
])

results = rag.query("What is Python?", top_k=2)
```

In-memory vector store. Cosine or L2 similarity. Filter functions. Collection statistics. Everything you need, nothing you don't.

______________________________________________________________________

## 💰 Built-in Budget Tracking

AI API calls cost money. KISS tracks every token:

```python
agent.run(
    model_name="gpt-4o",
    max_budget=1.0,  # USD limit for this run
    ...
)

print(f"Budget used: ${agent.budget_used:.4f}")
print(f"Tokens used: {agent.total_tokens_used}")
print(f"Global budget: ${KISSAgent.global_budget_used:.4f}")
```

Set per-agent limits. Set global limits. Get automatic cost calculation based on actual model pricing. Never get surprised by your API bill again.

______________________________________________________________________

## 🎨 The Architecture

```
kiss/
├── core/           # The heart: KISSAgent, models, formatters
│   ├── kiss_agent.py      # ~450 lines. That's the whole agent.
│   └── models/            # OpenAI, Anthropic, Gemini, Together, OpenRouter
├── agents/         # Pre-built agents and optimization frameworks
│   ├── gepa/              # Genetic-Pareto prompt evolution
│   ├── kiss_evolve/       # Evolutionary algorithm discovery
│   └── swe_agent_verified/# SWE-bench benchmark integration
├── docker/         # Container management
├── rag/            # Simple retrieval-augmented generation
└── viz_trajectory/ # Web-based trajectory visualizer
```

The entire core agent implementation is under 500 lines. Not because features are missing, but because every line earns its place.

______________________________________________________________________

## 🛠️ Getting Started

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/your-repo/kiss_ai.git
cd kiss_ai
uv venv --python 3.13
uv sync --group dev

# Set your API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# Run your first agent
uv run python -c "
from kiss.core.kiss_agent import KISSAgent
agent = KISSAgent('Hello World')
print(agent.run('gpt-4o', 'Say hello!', is_agentic=False))
"
```

______________________________________________________________________

## 🌟 Why KISS?

In a world obsessed with complexity, KISS is a rebellion.

It's for developers who believe that:

- **Simplicity is a feature**, not a limitation
- **Code should be readable** by humans, not just machines
- **Agents should be tools**, not black boxes
- **Evolution beats engineering** when the search space is vast

KISS doesn't try to be everything. It tries to be **exactly what you need** — a clean, powerful foundation for building AI agents that actually work.

______________________________________________________________________

## 🔮 What's Next?

KISS is actively evolving (pun intended). The roadmap includes:

- More benchmark integrations
- Enhanced multi-agent orchestration
- Improved evolution strategies
- Community-contributed tools and agents
- Asynchronous tool calling support

But the core philosophy will never change: **Keep It Simple, Stupid**.

______________________________________________________________________

## 📚 Resources

- **GitHub**: [KISS Agent Framework](https://github.com/ksenxx/kiss_ai)
- **GEPA Paper**: [arXiv:2507.19457](https://arxiv.org/pdf/2507.19457)
- **SWE-bench**: [swebench.com](https://www.swebench.com/)

______________________________________________________________________

*Built with ❤️ by Koushik Sen (ksen@berkeley.edu)*

*Because the best code is the code you don't have to write.*

______________________________________________________________________

**License**: Apache-2.0

**Python**: ≥3.13

**Philosophy**: KISS
