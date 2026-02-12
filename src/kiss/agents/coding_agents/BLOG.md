![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/fua0wd1uhu227x18ohpz.jpeg)

*No manual tuning. No architecture redesign. Just a plain-English instruction and a feedback loop.*

______________________________________________________________________

## The Setup

I maintain [KISS](https://github.com/ksenxx/kiss_ai), a minimalist multi-agent framework built on one principle: keep it simple, stupid. The framework's flagship coding agent, `RelentlessCodingAgent`, is a single-agent system with smart auto-continuation — it runs sub-sessions of an LLM-powered coding loop, tracks progress across sessions, and keeps hammering at a task until it succeeds or exhausts its budget. The agent was self-evolved to run relentlessly.

It works. But it was expensive. A single run with Claude Sonnet 4.5 cost $3–5 and took 600–800 seconds. For an agent framework that preaches simplicity and efficiency, that felt like hypocrisy.

So I built a 69-line Python script and told it, in plain English, to fix the problem.

## The Tool: `repo_optimizer.py`

The entire optimizer is a `RelentlessCodingAgent` pointed at its own source code. Here is the core of it:

```python
from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

TASK = """
Can you run 'uv run src/kiss/agents/coding_agents/relentless_coding_agent.py'
in the background so that I can see its output and you monitor the output in real time?
If you observe any repeated errors in the output, please fix them and run the command again.
Once the command succeeds, analyze the output and optimize
src/kiss/agents/coding_agents/relentless_coding_agent.py
so that it runs reliably, faster, and with less cost.
Keep repeating the process until the running time and the cost are reduced significantly,
such 99%.
...
"""

agent = RelentlessCodingAgent("RepoAgent")
result = agent.run(prompt_template=TASK, model_name="claude-opus-4-6", work_dir=PROJECT_ROOT)
```

That's it. The agent runs itself, watches the output, diagnoses problems, edits its own code, and runs itself again — in a loop — until the numbers drop.

No gradient descent. No hyperparameter grid search. No reward model. Just an LLM reading logs and rewriting source files.

## What the Optimizer Actually Does

The feedback loop works like this:

1. **Run** the target agent on a benchmark task and capture the output.
1. **Monitor** the logs in real time. If the agent crashes or hits repeated errors, fix the code and rerun.
1. **Analyze** a successful run: wall-clock time, token count, dollar cost.
1. **Optimize** the source code using strategies specified in plain English — compress prompts, switch models, eliminate wasted steps.
1. **Repeat** until the metrics plateau or the target reduction is hit.

The strategies themselves are just bullet points in the task prompt:

- Shorter system prompts that preserve meaning
- Remove redundant instructions
- Minimize conversation turns
- Batch operations, use early termination
- Search the web for agentic patterns that improve efficiency and reliability

The optimizer isn't hard-coded to apply any particular technique. It reads, reasons, experiments, and iterates. Which techniques it picks depend on what the logs reveal.

## The Results

After running overnight, the optimizer produced this report:

| Metric | Before (Claude Sonnet 4.5) | After (Gemini 2.5 Flash) | Reduction |
|---|---|---|---|
| **Time** | ~600–800s | **169.5s** | ~75% |
| **Cost** | ~$3–5 | **$0.12** | ~96–98% |
| **Tokens** | millions | **300,729** | massive |

All three benchmark tests passed after optimization: diamond dependency resolution, circular detection, and failure propagation.

## What the Optimizer Changed

The optimizer made concrete modifications, all discovered autonomously:

1. **Model switch**: Claude Sonnet 4.5 ($3/$15 per million tokens) to Gemini 2.5 Flash ($0.30/$2.50 per million tokens) — 10x cheaper input, 6x cheaper output.
1. **Compressed prompts**: Stripped verbose `CODING_INSTRUCTIONS` boilerplate, shortened `TASK_PROMPT` and `CONTINUATION_PROMPT` without losing meaning. The task prompt was reduced to a handful of critical rules.
1. **Added `Write()` tool**: The original agent only had `Edit()`, which fails on uniqueness conflicts. Each failure wasted 2–3 steps. Adding `Write()` eliminated that. The prompt now instructs: "Use Write() for new/full files. Edit() only for tiny fixes."
1. **Stronger finish instruction**: "IMMEDIATELY call finish once tests pass. NO extra verification." — stopped the agent from burning tokens on redundant confirmation runs.
1. **Bash timeout guidance**: "set `timeout_seconds=120` for test runs" — prevented hangs on parallel bash execution.
1. **Bounded poll loops**: "use bounded poll loops, never unbounded waits" — eliminated infinite-loop risks on background processes.
1. **Reduced `max_steps`**: 25 down to 15. Forced the agent to be efficient. Still enough to complete the task.
1. **Simplified step threshold**: Always `max_steps - 2` instead of a complex adaptive calculation.
1. **Removed `CODING_INSTRUCTIONS` import**: Eliminated unnecessary token overhead loaded into every prompt.
1. **Fixed Anthropic `max_tokens` default**: The Anthropic model's default output token limit was 4096, causing large `Write()` tool calls to be truncated — the model would generate `file_path` but the `content` argument would be cut off. Increased to 16384 and added safety handling to discard truncated tool_use blocks when `stop_reason` is `max_tokens`.

None of these changes are exotic. Each one is obvious in hindsight. But together they compound into a 98% cost reduction. The point is that no human sat down and applied them — the optimizer discovered and validated each one through experimentation.

## Why This Works

The `RelentlessCodingAgent` is a general-purpose coding loop: it gets a task in natural language, has access to Bash, Read, Edit, and Write tools, and runs sub-sessions until it succeeds. The `repo_optimizer.py` simply reuses this same loop, pointed inward.

This is possible because of three properties of the KISS framework:

- **Agents are just Python functions.** There's no config ceremony or deployment pipeline. An agent is a class you instantiate and call `.run()` on. So an agent can instantiate and run another agent — or itself.
- **Tools are just Python functions.** `Bash()`, `Read()`, `Edit()`, `Write()` — plain functions with type hints. The agent calls them natively. No wrappers, no adapters.
- **Tasks are just strings.** The optimization strategy, the constraints, the success criteria — all expressed in the task prompt. Changing what the optimizer does means editing a paragraph, not rewriting a pipeline.

The result is a self-improving system built from the same primitives as every other KISS agent.

## The Bigger Picture: `repo_agent.py`

The optimizer is actually a specialization of an even simpler tool: `repo_agent.py`. This is a 28-line script that takes any task as a command-line argument and executes it against your project root:

```bash
uv run python -m kiss.agents.coding_agents.repo_agent "Add retry logic to the API client."
```

The repo agent and the repo optimizer share the same engine (`RelentlessCodingAgent`) and the same interface (a string). The only difference is the task. The optimizer's task happens to be "optimize this agent for speed and cost." It could just as easily be "add comprehensive test coverage" or "migrate from REST to GraphQL."

The agents in KISS don't care what you ask them to do. They care about doing it relentlessly until it's done.

## Try It Yourself

```bash
# [Install KISS](https://github.com/ksenxx/kiss_ai/README.md)
# Run the repo optimizer on your own codebase
uv run python -m kiss.agents.coding_agents.repo_optimizer

# Or give the repo agent any task in plain English
uv run python -m kiss.agents.coding_agents.repo_agent "Refactor the database layer for connection pooling."
```

The framework, the agents, and the optimizer are all open source: [github.com/ksenxx/kiss_ai](https://github.com/ksenxx/kiss_ai).

______________________________________________________________________

*KISS is built by [Koushik Sen](mailto:ksen@berkeley.edu). Contributions welcome.*
