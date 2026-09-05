# KISS Sorcar — Tell it what to do, in English. It picks the best LLM and ships the work.

> Open-source AI agent. Tell it what to do in English — it picks the best LLM, runs on your laptop, and ships the work. Apache 2.0, 634 models. Works as a VS Code extension, a web/mobile app, and a Python API.

This is the plain-Markdown twin of <https://kisssorcar.github.io/>. Machine-readable entry points: [/llms.txt](https://kisssorcar.github.io/llms.txt) and [/llms-full.txt](https://kisssorcar.github.io/llms-full.txt).

## What Is KISS Sorcar

**KISS Sorcar is a free, simple, local-first, bring-your-own-key AI agent framework** for long-horizon tasks and AI discovery. It runs as a VS Code extension and a browser/mobile web app, both served by a local daemon (`kiss-web`), and offers a Python client API for scripting tasks. Your prompts and code are sent directly to the model provider or local endpoint you configure — not through our servers. It supports multi-model workflows just via prompts. All agents run as daemons. Complex AI systems/techniques can be replaced with a paragraph of prompt.

- **License:** Apache-2.0 · **Source:** <https://github.com/ksenxx/kiss_ai> · **PyPI:** `kiss-agent-framework`
- **634 models** across 9 provider categories (OpenAI, Anthropic, Gemini/Google, Together AI, Z.AI, Moonshot AI, OpenRouter, Claude Code CLI, Codex CLI)
- **32 channel agents** (Slack, Gmail, Email, WhatsApp, SMS, iMessage, Telegram, Discord, Signal, Phone Control, Home Assistant, …)
- Unique features: AI discovery via prompt, GEPA prompt optimization via prompt, multi-vendor multi-model tasks, dynamic `set_model` switching, steering-on-the-fly, git-worktree task isolation, wake-word voice chat

## Install

```bash
# Full install (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash

# Or Python package only (Python 3.13+)
pipx install kiss-agent-framework

# Start the daemon that serves the VS Code extension and the web app
kiss-web
```

Then open the KISS Sorcar sidebar in VS Code, or the remote web app URL from the Settings panel. From Python: `kiss.server.sorcar.run("What is 2435*234?")`.

## Documentation

- [Documentation index](docs/index.md)
- [Overview & comparison vs Claude Code / Cursor](docs/overview.md)
- [Installation](docs/installation.md)
- [Client interfaces](docs/cli.md)
- [Python API reference](docs/api.md)
- [Supported models](docs/models.md)
- [Messaging & third-party agents](docs/messaging-agents.md)
- [Sample tasks](docs/sample-tasks.md)
- [Prompt tricks](docs/prompt-tricks.md)
- [Tips](docs/tips.md)

## Papers

- **KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant** — [PDF](assets/kiss_sorcar.pdf) · [arXiv:2604.23822](https://arxiv.org/abs/2604.23822). Introduces KISS Sorcar and the underlying KISS Agent Framework: a deliberately minimal, open-source agent runtime designed for long-horizon tasks, rigorous review, multi-model orchestration, and AI discovery.
- **Software Engineering KISS Sorcar with KISS Sorcar** — [PDF](assets/se_kiss_sorcar.pdf). A case study of building KISS Sorcar with KISS Sorcar itself: over 44 days the developer issued 3,099 tasks through the system's own interface; nine recurring human–AI collaboration patterns are surfaced from the SQLite usage log.
- **Writing a Research Paper with an AI Agent** — [PDF](assets/writing_paper.pdf). A nine-day, hundred-task chronicle of KISS Sorcar drafting, citing, compiling, and debugging its own research paper.
- **HydraKV: Adversarial AI Discovery of a Larger-than-Memory Key-Value Store that Outperforms FASTER on Skewed YCSB-A** — [PDF](assets/hydra_kv.pdf). A larger-than-memory key-value store in dependency-free C++17, designed, implemented, and tested almost entirely by KISS Sorcar, verified by independent audits.
- **SWEDefend: A Confidence-Gated Intent-Alignment Judge with Capability-Diff Reasoning for Automated-Program-Repair Backdoor Defense** — [PDF](assets/swedefend.pdf). Defends LLM program-repair agents against backdoored patches; evaluated under adaptive attack with an honestly reported partial-defense result.
- **Cleverest+: A Fixed-Budget Portfolio and Signature-Grounded Oracle for LLM-Based Commit-Directed Test Generation** — [PDF](assets/cleverest_plus.pdf). Improves commit-directed test generation with a fixed-budget three-model portfolio and sanitizer-signature-grounded oracles.

## Blog

- **[Making LZ4's Multithreaded File Compression Scale](blog/lz4-optimization-blog.html)** (10 Aug 2026). Rewires lz4 v1.10.0's multithreaded file-mode pipeline to 1.88–2.57× stock throughput at level -1, byte-identical output, head-to-head vs pigz/pzstd/zstd -T.
- **[Optimizing DuckDB Against Its Official and Academic Benchmarks](blog/duckdb-optimization-blog.html)** (10 Aug 2026). Verified 1.152–1.237× geometric-mean speedups per suite on TPC-H, TPC-DS, IMDB/JOB, h2oai, and ClickBench.
- **[Reaching 99+ on Biomni × TusoAI-Style Biology Benchmarks with AI Discovery](blog/tuso-evolved-blog.html)** (9 Aug 2026). An AI-discovery loop evolves one method scoring ≥99/100 on perturbation-response and enhancer–gene-linking benchmarks.
- **[Optimizing SQLite Against Its Official and Academic Benchmarks](blog/sqlite-optimization-blog.html)** (8 Aug 2026). Verified 1.59× geometric-mean speedup on speedtest1, kvtest, TATP, and the Star Schema Benchmark.
- **[Verification of the sqlite-optimized Repository](blog/sqlite-optimized-verification-report.html)** (8 Aug 2026). Independent rebuild-and-reproduce audit of the SQLite optimization work.

## Citation

```bibtex
@misc{sen2026kisssorcar,
  title         = {KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant},
  author        = {Sen, Koushik},
  year          = {2026},
  eprint        = {2604.23822},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2604.23822}
}
```

______________________________________________________________________

KISS Sorcar · Apache 2.0 · [GitHub](https://github.com/ksenxx/kiss_ai) · Named after [P. C. Sorcar](https://en.wikipedia.org/wiki/P._C._Sorcar), the legendary Bengali magician.
