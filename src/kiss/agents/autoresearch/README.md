# Autoresearch Agent

A KISS-based implementation of Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) — an autonomous AI
agent that runs ML experiments overnight, iterating on a training script to
minimize validation loss without human intervention.

## What is Autoresearch?

Autoresearch gives an AI agent a small but real LLM training setup and lets it
experiment autonomously. The agent reads a `program.md` file for instructions,
then enters an infinite loop:

1. Modify `train.py` with an experimental idea
1. Git commit the change
1. Run training for a fixed 5-minute time budget
1. Check if validation loss (`val_bpb`) improved
1. **Keep** the change if it improved, **discard** (git reset) if not
1. Log results to `results.tsv`
1. Repeat indefinitely

The human wakes up to a log of ~100 experiments and (hopefully) a better model.

## How It Works

### Architecture

```
AutoresearchAgent (autoresearch_agent.py)
    └── RelentlessAgent (core/relentless_agent.py)
            └── KISSAgent (core/kiss_agent.py)
                    └── LLM API (Claude, GPT-4, Gemini, etc.)
```

The `AutoresearchAgent` extends `RelentlessAgent`, which provides automatic
sub-session continuation for long-running tasks. When the agent approaches its
step limit, it summarizes progress and continues in a new sub-session — enabling
indefinite autonomous operation.

### Key Files

| File | Purpose |
|---|---|
| `autoresearch_agent.py` | Agent class, CLI entry point, program file loading |
| `config.py` | Pydantic configuration (model, steps, budget, etc.) |
| `__init__.py` | Module init, registers config |

### Tools Available

The agent has four tools for code manipulation and experiment execution:

- **`Bash`** — Run shell commands (git, `uv run train.py`, grep results, etc.)
- **`Read`** — Read file contents
- **`Edit`** — Make precise string replacements in files
- **`Write`** — Create or overwrite files

No browser tools are included — autoresearch is entirely CLI-driven.

### Program File

The agent's behavior is driven by `program.md`, a Markdown file containing:

- Setup instructions (create git branch, read repo files, verify data)
- Experimentation rules (what can/cannot be modified, the metric to optimize)
- Output format and logging conventions
- The experiment loop (modify → commit → train → evaluate → keep/discard)

The default `program.md` from the original repo instructs the agent to:

- Only modify `train.py` (architecture, optimizer, hyperparameters — everything
  is fair game)
- Never modify `prepare.py` (data loading, evaluation, constants)
- Run training for exactly 5 minutes wall clock
- Log results to `results.tsv` with commit hash, val_bpb, memory, status, and
  description
- Never stop or ask for permission — run autonomously until interrupted

## Usage

### As a Python Library

```python
from kiss.agents.autoresearch.autoresearch_agent import AutoresearchAgent

agent = AutoresearchAgent("my-research")
result = agent.run(
    model_name="claude-opus-4-6",
    work_dir="/path/to/autoresearch/repo",
    max_steps=100,        # steps per sub-session
    max_budget=200.0,     # USD budget cap
    max_sub_sessions=10000,  # effectively unlimited continuation
)
```

#### With a Custom Program File

```python
result = agent.run(
    work_dir="/path/to/repo",
    program_file="/path/to/custom_program.md",
)
```

#### With a Direct Task Prompt

```python
result = agent.run(
    work_dir="/path/to/repo",
    prompt_template="Read program.md and start experimenting.",
)
```

### From the Command Line

```bash
# Run with default program.md in working directory
python -m kiss.agents.autoresearch.autoresearch_agent \
    --work_dir /path/to/autoresearch/repo

# Run with a specific task
python -m kiss.agents.autoresearch.autoresearch_agent \
    --task "Read program.md and kick off experiments" \
    --work_dir /path/to/repo

# Run with a custom program file
python -m kiss.agents.autoresearch.autoresearch_agent \
    --program /path/to/custom_program.md \
    --work_dir /path/to/repo

# Full options
python -m kiss.agents.autoresearch.autoresearch_agent \
    --model_name claude-opus-4-6 \
    --max_steps 100 \
    --max_budget 200.0 \
    --work_dir /path/to/repo \
    --verbose true
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `claude-opus-4-6` | LLM model to use |
| `--max_steps` | `100` | Maximum steps per sub-session |
| `--max_budget` | `200.0` | Maximum budget in USD |
| `--work_dir` | current directory | Working directory containing the repo |
| `--program` | `None` | Path to program.md (defaults to `work_dir/program.md`) |
| `--verbose` | `true` | Print output to console |
| `--task` | `None` | Direct task prompt (overrides program file) |

### Configuration

Defaults are configured in `config.py` and can be overridden via KISS config:

```yaml
autoresearch:
  autoresearch_agent:
    model_name: claude-opus-4-6
    max_steps: 100
    max_budget: 200.0
    max_sub_sessions: 10000
    verbose: false
```

## KISS Autoresearch vs. Original Autoresearch

The [original autoresearch](https://github.com/karpathy/autoresearch) is not a
standalone agent — it's a repo with a `program.md` file that you point any
coding agent at (Claude Code, Codex, etc.). The KISS implementation wraps
this pattern into a proper, self-contained agent with several advantages:

### 1. Automatic Sub-Session Continuation

The original approach relies on whatever agent you use to handle context window
limits. If the agent runs out of context or hits its step limit, the experiment
loop stops.

KISS's `RelentlessAgent` base class handles this automatically. When the agent
approaches its step limit, it:

- Summarizes all progress so far
- Starts a fresh sub-session with the summary as context
- Continues exactly where it left off

With `max_sub_sessions=10000`, the agent can run for days without human
intervention — each sub-session picks up from the last, maintaining the
experiment loop across context window boundaries.

### 2. Budget and Cost Control

The original has no built-in cost control — you're at the mercy of whatever API
limits your agent provider has.

KISS tracks token usage and cost in real-time via `budget_used` and
`total_tokens_used`. The `max_budget` parameter sets a hard USD cap, preventing
runaway API costs during overnight runs.

### 3. Multi-Model Support

The original assumes you'll use whatever model your agent tool provides
(typically Claude via Claude Code).

KISS's model layer supports Claude, GPT-4, Gemini, and other providers through
a unified interface. Switch models with a single `--model_name` flag — useful
for comparing how different models approach research problems.

### 4. Structured Output and Logging

The original outputs results to the terminal of whatever agent tool you use.

KISS produces structured YAML output with `success` and `summary` keys, saves
full agent trajectories to disk (in `artifacts/trajectories/`), and provides
configurable output via the `Printer` interface. This makes it easy to
programmatically analyze experiment results and agent behavior.

### 5. Docker Isolation

KISS supports running tools inside a Docker container via the `docker_image`
parameter. This is valuable for autoresearch since the agent runs arbitrary
training code — Docker isolation prevents accidental damage to the host system.

### 6. Programmatic Integration

The original is designed for interactive use — you open a terminal, start your
agent, and paste in a prompt.

KISS's `AutoresearchAgent` is a Python class you can instantiate and call from
any code. This enables:

- Orchestrating multiple autoresearch agents on different GPUs
- Integrating with CI/CD pipelines
- Building higher-level research automation on top

### Summary Table

| Feature | Original | KISS |
|---|---|---|
| Auto-continuation across context limits | ✗ | ✓ (RelentlessAgent) |
| Budget/cost control | ✗ | ✓ (max_budget in USD) |
| Multi-model support | ✗ | ✓ (Claude, GPT-4, Gemini, etc.) |
| Structured output & trajectory logging | ✗ | ✓ (YAML + saved trajectories) |
| Docker isolation | ✗ | ✓ (docker_image parameter) |
| Programmatic API | ✗ | ✓ (Python class) |
| CLI entry point | ✗ | ✓ (--task, --program, etc.) |
| Configuration management | ✗ | ✓ (Pydantic config) |
| Agent is self-contained | ✗ (needs external agent) | ✓ |
