# LinkedIn posts — KISS Sorcar §2.3 (Sorcar Agent)

Two variants. Both are honest summaries of what §2.3 of the paper actually
claims; nothing is exaggerated beyond the text.

---

## Option A — feature spotlight (recommended)

**One task, many models: how KISS Sorcar makes LLM choice a *tool call*, not a setting.**

Most coding agents lock you into a single model for the duration of a task.
The Sorcar Agent in KISS Sorcar takes a different approach.

In Section 2.3 of our paper, we describe `set_model` — a tool that every
Sorcar Agent has built in. When the agent calls it, the framework:

• constructs the new backend (Anthropic / OpenAI / Google / OpenRouter)
  through the same factory as the original
• copies the conversation history and cumulative usage counters into it
• rebuilds the tools schema in the new provider's dialect
• persists the choice so the next task in the same chat session reuses it

The live conversation continues on a different LLM, mid-task, with budget
accounting and continuation summaries staying coherent across the switch.

This unlocks three patterns we use every day:

1. **Scout-then-edit** — a cheap, fast model (e.g. gemini-2.5-flash) reads
   the repo and gathers context, then hands off to a stronger reasoner for
   the actual edit.
2. **Generate-then-review** — a primary model produces a change; a
   second-opinion model re-reads the diff, runs the tests, and verifies the
   change against the original request.
3. **Cost-aware long-loop** — an open-ended optimization task spends a
   cheap model on bookkeeping and journal-keeping, and pays for a frontier
   model only at decision points.

What makes this practical is that none of it requires special APIs or
modes. Each pattern is a single natural-language instruction you paste into
the IDE or CLI. For example, straight from our INJECTIONS.md:

  "Use claude-opus-4-7 for all coding, bug fixing, and test creation. Use
   gpt-5.5 (not codex) for thorough review and debugging. Check if the
   other model has missed code or introduced bugs."

The rest of the Sorcar Agent tool layer is deliberately small: a shell
executor with streaming output and cancellation, file Read/Edit/Write,
Playwright-driven browser automation, parallel sub-agents in a thread
pool, an ask-user tool for clarifications, and optional Docker-isolated
variants for untrusted work.

That's the whole layer. Everything else — planning, testing discipline,
verification, budget control — lives in prompts on top of it.

📄 Paper: github.com/ksenxx/kiss_ai (Section 2.3)

#AI #LLM #AgenticAI #SoftwareEngineering #DeveloperTools

---

## Option B — narrative / problem-first

**"Which model is best?" is the wrong question for agentic coding.**

The honest answer is *it depends on the step*. Reading 40 files to find a
call site is not the same job as writing a careful patch, and neither is
the same job as independently reviewing that patch for missing code or
subtle bugs.

In KISS Sorcar (paper §2.3), we treat this directly. Every Sorcar Agent
exposes a `set_model` tool. The agent can switch the underlying LLM during
a single task — same conversation, same transcript, same usage counters —
and the framework rebuilds the tools schema in the new provider's dialect
so nothing breaks.

In practice this lets one task look like:

• gemini-2.5-flash scouts the repository (cheap, fast)
• claude-opus-4-7 writes the change and the tests (strong reasoner)
• gpt-5.5 reviews the diff, runs the tests, and flags anything the first
  model missed (independent second opinion)

All in one conversation. All under one budget. Triggered by a single
sentence in the prompt.

The rest of the Sorcar Agent layer — Bash / Read / Edit / Write, a
Playwright browser, parallel sub-agents, ask-user-question, optional
Docker isolation — is intentionally minimal. The interesting capability
isn't more tools. It's letting the agent pick the right model for the
right step, and have the framework keep the books straight when it does.

📄 Section 2.3 of the paper: github.com/ksenxx/kiss_ai

#AI #LLM #AgenticCoding #DeveloperProductivity
