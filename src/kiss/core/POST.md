#whatispossible
**How do you make an AI agent work on a task for hours — without losing its mind in [KISS Sorcar](https://github.com/ksenxx/kiss_ai)?**
Cursor compresses context in-place. Claude Code triggers compaction APIs. Both degrade over time as context drift accumulates.
We took the opposite path: **session boundaries with first-person progress summaries.** No vector databases. No RAG. No embeddings. No compaction. Just a `for` loop and the insight that an LLM can summarize its own work better than any retrieval system can reconstruct it.
The result is [`RelentlessAgent`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/core/relentless_agent.py) — 282 lines of Python that can run for thousands of sub-sessions with **flat performance characteristics**. Session 9,999 has exactly the same working memory as session 1.

How it works:
→ Each sub-session gets a **fresh context window** — original task + chronological progress summary with explanation and relevant code snippets
→ The agent knows its step limit and writes its own handoff summary before time runs out
→ Crashed sessions are recovered by summarizing whatever work was done — same pattern, no special codepath
→ The `finish(success, is_continue, summary)` function is both the tool and the schema — the docstring *is* the protocol
What it deliberately ignores:
✗ Vector memory / RAG pipelines
✗ Layered memory hierarchies (short/mid/long-term)
✗ In-place context compression
✗ Proprietary infrastructure or subscriptions
What it achieves:
✓ Up to 10,000 sub-sessions (1,000,000 steps)
✓ Zero context drift — architecturally impossible
✓ Error recovery reuses the same summarize-and-continue pattern
✓ Works with any LLM provider (Anthropic, OpenAI, Gemini, etc.)
✓ Full budget/token tracking, optional Docker isolation
✓ Small enough to read in 5 minutes and understand completely
**You don't need a complex system to solve a complex problem. You need the right decomposition.**
Part of the open-source KISS framework / Sorcar IDE.
📄 Deep dive: https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/core/RELENTLESS_AGENT.md
💻 Source: https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/core/relentless_agent.py
🔗 Repository: https://github.com/ksenxx/kiss_ai
#AI #LLM #AgenticAI #OpenSource #SoftwareEngineering #AIAgents #ContextWindow #Python #cursor #claudecode #claude
