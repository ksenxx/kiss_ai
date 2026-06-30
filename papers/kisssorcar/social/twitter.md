# Twitter / X posts — KISS Sorcar §2.3 (Sorcar Agent)

Section 2.3 is about the Sorcar Agent's tool layer — and especially `set_model`,
which lets one running task hand off the live conversation to a different LLM
without restarting it. The posts below are written to be attention-grabbing
but factually grounded in the paper.

---

## Option A — lead with set_model (recommended)

Most "agent frameworks" force you to pick one model per task.

KISS Sorcar's Sorcar Agent ships a `set_model` tool that swaps the LLM
mid-task — same conversation, same history, same budget counters.

Three patterns we actually use:

🛰️  scout-then-edit: cheap model greps the repo, strong model writes the patch
🔍  generate-then-review: model A codes, model B re-reads the diff + runs tests
💸  cost-aware long-loop: cheap model does bookkeeping, frontier model only at decisions

It's not a special mode. It's a tool the agent calls.

Paper §2.3 → github.com/ksenxx/kiss_ai

---

## Option B — punchier, thread-style

1/ "Which model should I use?" is the wrong question.

KISS Sorcar lets the agent change its mind mid-task. One tool call,
`set_model("claude-opus-4-7")`, and the live conversation continues on a
different LLM — history, tool schemas, and usage counters all carried over.

2/ This turns model choice from a session setting into a *programmable
task-level resource*. You can literally paste this as a prompt:

  "Use claude-opus-4-7 for coding. Use gpt-5.5 to review the work and find
   bugs the first model missed."

No API glue. No orchestrator. Just a sentence.

3/ The Sorcar Agent layer is small on purpose: Bash / Read / Edit / Write,
a Playwright browser tool, parallel sub-agents, ask-user-question, optional
Docker isolation, and `set_model`. That's it. Everything else is prompts.

Paper §2.3: github.com/ksenxx/kiss_ai

---

## Option C — short and quotable

The Sorcar Agent in KISS Sorcar gives the LLM a `set_model` tool.

The agent can switch model — Anthropic → OpenAI → Gemini → OpenRouter —
in the middle of a task, keeping the full transcript and budget intact.

Cheap model scouts. Strong model edits. Independent model reviews.
All in one conversation.

(Paper §2.3, github.com/ksenxx/kiss_ai)
