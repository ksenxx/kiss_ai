#whatispossible

# RelentlessAgent: The Elegant Engine Behind Multi-Hour Agentic Tasks in KISS Sorcar IDE

*How ~300 lines of Python solve the hardest problem in agentic coding — and why it deliberately ignores the patterns everyone else uses.*

______________________________________________________________________

## Overview

The `RelentlessAgent` (defined in [`relentless_agent.py`](relentless_agent.py)) is the KISS framework's answer to a deceptively hard problem: **how do you make an AI agent work on a task for hours or even days without losing its mind?**

Cursor, Claude Code, and every other agentic coding tool eventually hit the same wall — the context window fills up, the model starts forgetting earlier decisions, and the work degrades or halts. The industry's response has been a growing stack of complexity: vector databases for long-term memory, RAG pipelines for retrieval, compaction APIs, embedding-based search, layered memory hierarchies (short-term, mid-term, long-term), and elaborate context management subsystems.

The RelentlessAgent takes the opposite path. It uses a single, radical mechanism: **session boundaries with chronological progress summaries with explanations and relevant code snippets**. No memory databases. No embeddings. No compaction. Just plain Python, a for loop, and the insight that an LLM can summarize its own work better than any retrieval system can reconstruct it.

______________________________________________________________________

## How It Works: The Technical Details

### The Core Loop

The entire architecture revolves around the `perform_task` method and a simple `for` loop over sub-sessions:

```python
for session in range(self.max_sub_sessions):
    executor = KISSAgent(f"{self.name} Session-{session}")
    result = executor.run(...)

    if not is_continue or success:
        return result

    summary = payload.get("summary", "")
    progress_section = CONTINUATION_PROMPT.format(progress_text=summary)
```

Each sub-session is a fresh `KISSAgent` instance, which is a simple agent running a ReAct loop, with a clean context window. It receives the original task description plus a progress summary from all previous sessions. When the task completes, the agent calls `finish(success=True, is_continue=False, summary="...")` and the loop exits. When it cannot finish within its step or context limit, it calls `finish(success=False, is_continue=True, summary="...")` with a detailed chronological account of what it did and why. The next sub-session picks up where the last one left off.

That is the entire continuation mechanism. There is no other trick.

### The Self-Aware Step Threshold

The prompt injected into each sub-session contains a critical instruction:

```
At step {step_threshold}: you MUST call finish(success=False, is_continue=True,
summary="precise chronologically-ordered list of things the agent did
with the reason for doing that along with relevant code snippets")
if the task is not complete and you are at risk of running out of steps or context length.
```

The `step_threshold` is set to `max_steps - 2`, giving the agent exactly two steps of margin before the hard limit. This is not an afterthought — it is the mechanism that makes the entire system work. The agent knows it is going to be interrupted. It knows when. And it knows exactly what format to use for the handoff. The `is_continue=True` signal explicitly tells the orchestrator that work remains, distinguishing an incomplete task from an outright failure. This transforms what would be a failure (context window or step exhaustion) into a structured checkpoint.

### The Continuation Prompt

When the agent reports failure with a summary, the next sub-session receives:

```
# Task Progress
{progress_text}

# Continue
- Complete the rest of the task.
- **DON'T** redo completed work.
```

The simplicity is intentional. The progress text is the agent's own words — a chronologically-ordered list of what was done, why it was done, and relevant code snippets. The continuation prompt does not attempt to reconstruct state from tool outputs, parse file diffs, or query a memory store. It trusts the LLM's ability to compress its own trajectory into a coherent narrative.

### Error Recovery with Trajectory Summarization

When a sub-session crashes (API failure, tool exception, unexpected error), the RelentlessAgent does not simply fail. It spawns a separate summarizer agent that reads the crashed session's trajectory file and produces a summary:

```python
summarizer_agent = KISSAgent(f"{self.name} Summarizer")
summarizer_result = summarizer_agent.run(
    model_name=self.model_name,
    prompt_template=SUMMARIZER_PROMPT,
    tools=[shell_tools.Read, shell_tools.Bash],
    arguments={"trajectory_file": str(trajectory_path)},
)
```

This agentic call uses `Read` and `Bash` tools to read the (potentially large) trajectory file and distill whatever work was done before the crash. The next sub-session then receives this summary as progress context, allowing it to continue from the last known good state rather than restarting from scratch. The error recovery is itself just another application of the same summarize-and-continue pattern.

### The `finish` Function as Structured Output

The `finish` function doubles as both a tool and a schema definition:

```python
def finish(success: bool, is_continue: bool, summary: str) -> str:
    """Finish execution with status and summary.

    Args:
        success: True if the agent has successfully completed the task, False otherwise
        is_continue: True if the task is incomplete and should continue, False otherwise
        summary: precise chronologically-ordered list of things the
            agent did with the reason for doing that along with
            relevant code snippets
    """
```

The three-parameter design gives the orchestrator an unambiguous signal: `success=True` means the task is done, `is_continue=True` means more work is needed, and `success=False, is_continue=False` means the agent has given up. The LLM reads the parameter names, types, and documentation through native function calling, then produces YAML output. No Pydantic response models, no JSON schema definitions, no output parsers. The docstring is the schema.

### Budget and Token Tracking Across Sessions

Each sub-session's cost accumulates into the parent:

```python
self.budget_used += executor.budget_used
self.total_tokens_used += executor.total_tokens_used
```

This gives the RelentlessAgent awareness of total spend across all sub-sessions for tracking and reporting. Hard budget caps that prevent runaway costs operate at two levels: `max_budget` caps each individual sub-session (since each `KISSAgent` starts with `budget_used=0`), while the global budget (`Base.global_budget_used` checked against `config.agent.global_max_budget`) caps cumulative spending across all sub-sessions and agents in the process. Both checks happen at every step boundary inside `KISSAgent._check_limits()`.

### Docker Isolation

The agent supports running tools inside Docker containers via a context manager:

```python
if self.docker_image:
    with DockerManager(self.docker_image) as docker_mgr:
        self.docker_manager = docker_mgr
        return self.perform_task(tools or [], attachments=attachments)
```

This is relevant for long-running tasks because it provides repeatable environments — the agent can install dependencies, modify system configurations, and run arbitrary commands without affecting the host machine, even across dozens of sub-sessions.

______________________________________________________________________

## How Cursor and Claude Code Handle the Same Problem

### Cursor's Approach: In-Place Context Compression

Cursor operates within a single continuous conversation. When the context window fills up, it automatically summarizes older messages and replaces them with compressed versions. It preserves key decisions, file paths, and errors while discarding exact tool outputs and intermediate reasoning. Cursor also silently injects substantial context (open files, git status, terminal output, linter errors) into every message.

The problem: **context compression is lossy and cumulative**. Each compression pass discards information that might be needed later. After several rounds of compression, the agent effectively works from a degraded version of its own history. Research from 2025-2026 shows that 65% of enterprise AI agent failures were attributed to context drift — the gradual degradation of understanding during multi-step reasoning — not raw context exhaustion. Cursor users consistently report effective context limits of 70K-120K tokens despite models formally supporting 200K, because internal truncation and compression reduce practical capacity.

Cursor also developed Composer, a specialized MoE model trained via reinforcement learning for software engineering, and supports sub-agents for parallel work. But the core context management remains in-place summarization within a single session.

### Claude Code's Approach: Compaction API + Flat History

Claude Code uses a `while(tool_call)` loop with flat message history — no threading, no complex state machines. When approaching approximately 92% of the context window, it triggers a compactor that summarizes the conversation and moves compressed information to Markdown documents serving as project memory. Anthropic released the `compact-2026-01-12` server-side compaction API to automate this.

Claude Code's design is deliberately simple: a single agent loop with well-designed tools, regex over embeddings for search, and Markdown files instead of databases. It allows at most one sub-agent branch to maintain single-threaded reliability.

The limitation: **compaction happens within the conversation**, meaning the model must simultaneously hold its compressed history, the current task context, and the new work it is generating. The compressed context competes for tokens with the active work. After multiple compaction cycles, the effective "working memory" for the current task shrinks. Claude Code's practical performance degrades around 120K tokens, and complex multi-hour tasks often require manual intervention (`/compact` commands, `CLAUDE.md` files) to stay on track.

______________________________________________________________________

## What Makes the RelentlessAgent Different

### 1. Clean Session Boundaries Eliminate Context Drift

The RelentlessAgent never compresses context in-place. Instead, each sub-session starts with a **fresh context window** containing only the original task description and a summary of previous progress. There is no accumulated noise, no partially-compressed history, no context from five sessions ago competing for tokens with the current work. Context drift — the primary killer of long-running agents — is architecturally impossible because each session's context is constructed from scratch.

This is the fundamental insight: **a fresh start with a good summary beats a degraded continuation every time.** The 65% failure rate attributed to context drift in other systems simply does not apply here.

### 2. The Summary Is First-Person, Not Reconstructed

When Cursor or Claude Code compress context, a separate mechanism (compactor model, summarization pass) reads the conversation and decides what to keep. This is inherently a lossy third-party process — the compressor does not know what will matter for future work.

The RelentlessAgent's summary is written by the working agent itself, in the moment, as part of its final action before the session ends. The agent knows what it was trying to do, what worked, what failed, and what remains. The summary is a first-person account with the agent's own reasoning, not a post-hoc reconstruction. The docstring explicitly requests "a precise chronologically-ordered list of things the agent did with the reason for doing that along with relevant code snippets" — this format preserves causal chains that generic compression destroys.

### 3. 10,000 Sub-Sessions, Not One Long Conversation

The default `max_sub_sessions` is 10,000. With 100 steps per session, that is 1,000,000 potential steps (each step can include multiple tool calls). No context window management is needed to reach this scale because each session is independent. Cursor and Claude Code can theoretically run indefinitely with compaction, but their effective capacity degrades with each cycle. The RelentlessAgent's capacity is flat — session 9,999 has exactly the same working memory as session 1.

### 4. Total Cost: ~300 Lines

The entire [`relentless_agent.py`](relentless_agent.py) is approximately 300 lines of Python, including imports, docstrings, and the prompt templates. For comparison:

- Cursor's context management involves dynamic context discovery, summarization systems, sub-agent orchestration, and a custom-trained MoE model.
- Claude Code's context management involves the compaction API, session memory systems, CLAUDE.md file management, and manual intervention workflows.

The RelentlessAgent achieves comparable (and for very long tasks, superior) results with a `for` loop, three prompt templates, and a `finish` function. This is not minimalism for its own sake — it means there is almost nothing that can break, almost nothing that needs debugging, and almost nothing that resists understanding on first read.

### 5. No Subscription, No Proprietary Infrastructure

Cursor requires a monthly subscription ($20/month Pro, $200/month Ultra) and runs on proprietary infrastructure. Claude Code requires an Anthropic API key with per-token billing through their infrastructure, or a $100-200/month Max plan. The RelentlessAgent runs on any LLM provider (Anthropic, OpenAI, Gemini, Together AI, OpenRouter), on your own machine, with no monthly fees beyond API usage. You can switch models between sessions if needed.

______________________________________________________________________

## Why Explicit Agent Memory Is Not Used

Standard agentic AI patterns in 2025-2026 recommend layered memory architectures:

- **Short-term memory**: The active context window (2K-8K tokens of working state)
- **Mid-term memory**: Episodic memory of recent sessions (50-200 turns, compressed via summarization)
- **Long-term memory**: Semantic patterns and facts stored in vector databases, retrieved via embedding similarity

The RelentlessAgent deliberately uses none of these. Here is why:

### Vector Memory Adds Complexity Without Proportional Value

RAG-based memory requires embedding generation, vector storage, similarity search, and retrieval chunking. Research consistently shows that RAG matches or outperforms long-context loading for factual Q&A, but its advantage does not extend to the kind of holistic reasoning that coding tasks demand. A coding agent needs to understand the causal chain of its own work — "I changed file X because of error Y, which I discovered while testing Z." Embedding-based retrieval fragments this chain into independently-retrieved chunks, destroying exactly the causal structure that matters.

### The Summary Already Is the Memory

The chronological progress summary produced at each session boundary is a natural language compression of the agent's trajectory. It captures what happened, why, and what code was affected — precisely the information needed to continue. This is equivalent to mid-term episodic memory, but it requires no infrastructure. No database to maintain, no embedding model to run, no retrieval pipeline to tune.

### Statelessness Is a Feature

Each sub-session is stateless except for the summary it receives. This means:

- Sessions can run on different machines or in different containers
- A crashed session loses nothing — the summarizer recovers what was done
- There is no state corruption from concurrent access, stale embeddings, or database failures
- Debugging is trivial: read the summary, understand the state

Explicit memory systems introduce failure modes (stale entries, embedding drift, retrieval misses) that scale with the size of the memory. The RelentlessAgent's memory is a string. It cannot be stale, it cannot drift, and it cannot miss — because it is read in full every time.

### The KISS Principle

The KISS framework's core philosophy is that unnecessary abstraction is the enemy of reliability. Explicit memory systems solve a real problem — but they solve it for agents that maintain a single continuous conversation. The sub-session architecture makes that problem disappear. Adding memory infrastructure on top would be solving a problem the design has already eliminated.

______________________________________________________________________

## The Elegance in the Design

There is a pattern in engineering where the most sophisticated solution is also the simplest. The RelentlessAgent embodies this:

1. **The for loop is the orchestrator.** No state machines, no message buses, no DAG executors. Just iteration with a break condition.

1. **The prompt is the protocol.** The agent knows its step limit, knows it must summarize, and knows the format. The continuation prompt tells the next session what happened. All coordination is encoded in natural language.

1. **The finish function is the schema.** Python function signatures and docstrings define the structured output. The LLM reads them through native function calling. No separate output parsers needed.

1. **Error recovery reuses the same pattern.** A crashed session is summarized by an agentic LLM call (with `Read` and `Bash` tools to handle large trajectory files), and the summary feeds into the next session identically to a normal handoff. There is no separate error recovery codepath.

1. **The agent evolves itself.** The README notes that the RelentlessAgent "was self-evolved over time to reduce cost and running time." The KISS Sorcar agent — itself powered by RelentlessAgent — was used to iteratively simplify and improve the RelentlessAgent's own code, prompts, and parameters.

The result is a system that can run relentlessly for hours to days, across thousands of sub-sessions, with flat performance characteristics, full budget awareness, optional Docker isolation, and complete trajectory logging — all in a file small enough to read in five minutes and understand completely.

That is the RelentlessAgent's argument: **you do not need a complex system to solve a complex problem. You need the right decomposition.**

______________________________________________________________________

## How SorcarAgent Uses RelentlessAgent

The `SorcarAgent` ([`src/kiss/agents/sorcar/sorcar_agent.py`](../agents/sorcar/sorcar_agent.py)) is a coding-plus-general-purpose agent built as a thin subclass of `RelentlessAgent`. It demonstrates how concrete agents are assembled on top of the continuation engine without altering any of its mechanics.

### What SorcarAgent Adds

SorcarAgent's job is to supply **tools**, **system instructions**, and **prompt enrichment** — the three things the RelentlessAgent is deliberately agnostic about.

**Tools.** The `_get_tools()` method assembles the tool list that each sub-session receives: `Bash`, `Read`, `Edit`, `Write`, and `ask_user_question` for coding work and human-in-the-loop interaction, plus a full set of browser automation tools (`go_to_url`, `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`) from `WebUseTool`. If a Docker image is configured, the Bash tool is swapped for a Docker-isolated variant. This tool set is passed to `super().run(tools=self._get_tools())`, and from that point the RelentlessAgent's sub-session loop takes over — each fresh `KISSAgent` session receives these tools unchanged.

```python
def _get_tools(self) -> list:
    stop_event = getattr(self, "_stop_event", None)
    useful_tools = UsefulTools(stream_callback=_stream, stop_event=stop_event)
    bash_tool = self._docker_bash if self.docker_manager else useful_tools.Bash
    tools = [bash_tool, useful_tools.Read, useful_tools.Edit, useful_tools.Write]
    if self.web_use_tool:
        tools.extend(self.web_use_tool.get_tools())
    tools.append(ask_user_question)
    return tools
```

**System instructions.** SorcarAgent prepends the framework's `SYSTEM_PROMPT` (which contains the code style guidelines, testing instructions, and web tool usage rules that the agent sees in every session) and appends the path to a task history file so the agent has cross-task context.

**Prompt enrichment.** Before calling `super().run()`, the `run()` method appends contextual hints to the user's task prompt: the path of the currently active editor file, and a note about any attached images or PDFs instructing the agent to examine them directly rather than through browser tools.

### What SorcarAgent Does Not Do

SorcarAgent does not touch the continuation loop, the session boundary mechanism, the summarization logic, or the `finish` function protocol. It does not manage context windows or track progress across sub-sessions. All of that is inherited from `RelentlessAgent`. The subclass is roughly 220 lines of logic (excluding the CLI `main()`), and its entire `run()` method ends with:

```python
return super().run(
    model_name=model_name,
    system_instructions=system_instructions,
    prompt_template=prompt,
    tools=self._get_tools(),
    ...
)
```

This is the intended extension pattern: concrete agents configure *what* the agent can do (tools and context), while `RelentlessAgent` handles *how long* and *how reliably* it can do it.
