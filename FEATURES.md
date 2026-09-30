# KISS Sorcar Feature Inventory

Version 2026.9.27 · HEAD `5eb6496fd` (4,926 commits) · compiled 2026-09-30 · supersedes the 2026-09-21 inventory

Every count and feature in this document was checked against the source tree at `5eb6496fd`, the packaged assets under `src/kiss/`, and the task database `~/.kiss/sorcar.db` on 2026-09-30. Items marked **NEW** did not exist at the previous inventory's commit `85d9f3cda` (2026-09-22); the 360 commits between the two are summarised in section 2. Where the code and a document disagree, the code wins and the disagreement is listed in section 23.

## Contents

1. [At a glance](#1-at-a-glance)
2. [What changed since 2026-09-21](#2-what-changed-since-2026-09-21)
3. [Research drivers, papers and website](#3-research-drivers-papers-and-website)
4. [Architecture](#4-architecture)
5. [Interfaces and prompt surfaces](#5-interfaces-and-prompt-surfaces)
6. [Agent runtime](#6-agent-runtime)
7. [Built-in tools and tool profiles](#7-built-in-tools-and-tool-profiles)
8. [Models, routing and cost accounting](#8-models-routing-and-cost-accounting)
9. [Sorcar Extension Agents and slash commands](#9-sorcar-extension-agents-and-slash-commands)
10. [Sub-agents and parallelism](#10-sub-agents-and-parallelism)
11. [Task classification, tags and chat summaries](#11-task-classification-tags-and-chat-summaries)
12. [Git worktrees, merging and auto-commit](#12-git-worktrees-merging-and-auto-commit)
13. [Persistent memory and repository knowledge](#13-persistent-memory-and-repository-knowledge)
14. [The kiss-web daemon and remote web app](#14-the-kiss-web-daemon-and-remote-web-app)
15. [Chat client features](#15-chat-client-features)
16. [Browser, web research and sign-in hand-off](#16-browser-web-research-and-sign-in-hand-off)
17. [Voice](#17-voice)
18. [Messaging and third-party agents](#18-messaging-and-third-party-agents)
19. [Scheduled automations](#19-scheduled-automations)
20. [Installation, deployment, Docker and release](#20-installation-deployment-docker-and-release)
21. [Developer tooling and tests](#21-developer-tooling-and-tests)
22. [What the trajectories show](#22-what-the-trajectories-show)
23. [Caveats, corrections and stale documentation](#23-caveats-corrections-and-stale-documentation)

## 1. At a glance

- **One agent, many surfaces**: the same `SorcarAgent` runs in the VS Code extension, the remote web app served by the `kiss-web` daemon, the `sorcar` CLI, the Python client API, 44 messaging and service channels, and cron jobs. Clients connected to the daemon share its tab registry (`~/.kiss/tabs.json`) for chat tabs, and sub-agent and browser tabs are synchronised by events, so an open task, sub-agent or browser tab appears in every client; only the standalone `sorcar` CLI runs its agent in the terminal instead. **NEW**
- **706 model entries** in `src/kiss/core/models/MODEL_INFO.json` (687 generation, 525 with function calling, 7 embedding): 165 direct API models, 412 through OpenRouter, 16 `cc/` (Claude Code) and 10 `codex/` run-to-completion adapters, the rest through Together and other hosts. Two router entries, `autorouter` and `bestrouter`, appear in the model picker and are implemented as SEAs. **NEW**
- **17 bundled Sorcar Extension Agents** (slash commands) in `src/kiss/agents/seas/`, 13 of them new since 2026-09-22, plus 44 channel SEAs in `src/kiss/agents/third_party_agents/`; both packages use a folder-per-SEA layout with a mandatory `description()`.
- **33 built-in tool functions** in a full-profile run (shell/file 6, browser 9, memory 7, MCP sign-in 2, dispatch 3, interaction 6 including `finish` and, when Jev is enabled, `decide`), before skill, MCP-server and caller-supplied tools; five tool profiles (`full`, `review`, `shell`, `assistant`, `bash`).
- **50 console scripts** in `pyproject.toml` (`sorcar`, `kiss-web`, `kiss-cron`, `check`, `generate-api-docs`, `swedefend-eval` and 44 `kiss-<channel>` CLIs, including the new `kiss-overleaf`).
- **Prompt assets**: `SYSTEM.md` 3,971 words, `SYSTEM_LITE.md` 753 words, 23 tips, 6 bundled promptlets, 12 sample tasks.
- **Codebase**: 160,637 lines of non-test Python under `src/kiss` (server 37,727; `agents/sorcar` 32,532), 1,360 `test_*.py` files with 10,682 test functions (382,938 lines across the 1,405 Python files under `src/kiss/tests`), 11,326 lines of TypeScript plus the 23,860-line shared `main.js` in the VS Code extension (43,649 lines with the other media scripts and vendored JavaScript) and 369 JS test files.
- **Task database**: 27,673 task rows (6,457 top-level, 21,216 sub-agent) in 3,487 chats, 10.6 million persisted events, 572,526 tool calls, 125 distinct working directories on top-level tasks (430 across all rows), 52 named models. Recorded spend on top-level tasks is **$40.3K** across 28.5 billion tokens and 315K steps (section 22 explains why this is lower than the figure in earlier inventories).

## 2. What changed since 2026-09-21

360 commits between `85d9f3cda` and `5eb6496fd`, dated 2026-09-22 through 2026-09-30 (UTC) with a peak of 69 commits on 09-26; version bumps 2026.9.21 through 2026.9.27. Grouped by theme:

| Theme | What landed | Commits |
| --- | --- | --- |
| Tab synchronisation | Canonical `TabRegistry` persisted in `~/.kiss/tabs.json`; sub-agent tabs open on every surface while the child runs, close everywhere when it finishes or when any client closes them; side-channel tabs (`/ask`, task-update) never reopen on replay | `bf21e4172` `2864c0c61` `da0b73eed` `6ac2e7476` `d02d80f0b` |
| Live browser tab | The daemon machine's browser is streamed as a "KISS Browser" tab to every client (JPEG frames, 15 fps cap, clicks and keys sent back); agent `go_to_url`/`show_browser` and third-party sign-in pages open there | `27f3051b3` `70e1d2e60` `b751e06a0` |
| Sidebar subpanels | Schedule (cron jobs), Apps (channel auth state, click to connect) and Spend (per-day, per-model cost graph) added to the Task Info panel; all five subpanels collapsible, resizable and equal-height by default | `7c337d8d1` `350dc1e2d` `613dcd72f` `bbb2b1aa1` |
| Bundled SEAs | Folder-per-SEA layout and `description()` contract; new `/autorouter`, `/bestrouter`, `/write`, `/write_paper`, `/review_paper`, `/revise_and_review_paper`, `/remember`, `/forget`, `/rsi7d`, `/skillopt`, `/task_update`, `/git_extract_knowledge`, `/coding`; `/ask` moved into the package | `86c09b15d` `6271d7973` `66b29f0db` `a6bb1f61d` `904f76514` `c95f4ca33` `8868e2bad` `77d58a865` `d54868487` `397cb77a7` `e8c046264` `af8a7a002` |
| Model routing | `autorouter` (tiered cheapest-model-first with escalation, ledger in `~/.kiss/MODEL_DECISIONS.md`, evidence in `~/.kiss/AUTOROUTER.md`) and `bestrouter` (claude-fable-5-1 writes, gpt-6-astra reviews) as picker entries; picking autorouter schedules a weekly `/rsi7d` job | `bba4ac2b0` `29f82ed0b` `6271d7973` `aa215fa17` `3d0245792` `2b5ba2192` |
| Memory | Memory moved to `~/.kiss/memories` with per-repository domain memories (`kiss/`); latency work (racy-clean stat checks, batched embeddings, numpy scan); deletion tombstones; memory sync between machines on `rsorcar` deploy; `/git_extract_knowledge` builds an FTS5 block store per repository | `c2964bf1b` `be1320246` `9834afaf7` `52f88b55e` `900802a14` `12ec468f2` |
| Task database | `tags` and `sea` columns on `task_history`, a `chat_summaries` table (6 to 8 word chat titles), per-call `llm_call` events with cost; History panel filters by tag and by SEA | `a1ebcbc07` `e3da3cc35` `958b68916` `a9eaa206a` |
| Cost accounting | OpenRouter provider-reported cost, no double billing of BYOK upstream cost, billed-but-rejected responses charged, stopped or timed-out sub-tasks charged to the caller, side-channel spend charged to the owning task, audio token pricing, per-model split in the Spend panel | `5c7998c75` `a157e83d9` `327a682c7` `d765ec581` |
| Auth and channels | Overleaf service agent; generic MCP OAuth client with `connect_mcp_server`; Composio project-key guidance for the six Google agents; Gmail scope narrowed to `gmail.modify` and the permanent-delete tool removed; Muse-auth daemon spawn lock | `092fe942a` `8b7d20108` `d53bfdcaa` `ed5a9f4d6` `628e056ae` `fc9417631` |
| Daemon | TLS trust installer (`kiss-web --trust-ca`, `GET /ca.crt`); reconnect in place without page reload; home-wide file index for `@` mentions; bare-path prompts open the file instead of starting a task; replay serialisation fixes; stall watchdog stop | `25d60c66f` `284200b0c` `d22391679` `ab964ea14` `e2087df58` `58897520c` `86d737949` `02455a12a` |
| Client UI | Design tokens and `--fs-*` font scale; light theme default on the remote app; unified welcome page with logo; themed tips popup reopening after each update; promptlet edit/copy/delete; working directory per chat tab; Working/Memory directory fields removed from Settings; shared pdf.js viewer; Monaco menu bar and themes; branding overlay for white-label builds; auto-reload after update | `425dc0c45` `7e755d73f` `ab1f44abb` `2ca71f214` `7b2a7534a` `ddef00939` `8a7c0aa8d` `0e9ae17eb` `4b7f8a961` `72618394b` `c7f53054f` `49c5df249` `d7b8af126` |
| Runtime | Static system prompt and tools under one Anthropic cache breakpoint; `assistant` tool profile; Docker work dir bind-mounted at its host path; Jev decisions model gated behind an off-by-default setting; review fan-out guardrails removed; `WEB_TOOLS_OFF_NOTE` | `38a7bca32` `8493b350b` `5ceb28e9a` `ab1f44abb` `ee6ba3d38` |
| Worktrees and git | Spare worktree prewarmed at daemon start and refilled when idle; merge blocked only by modified tracked files in the main tree; 200 KB cap on diffs fed to commit-message generation; orphaned merge claims self-heal; HTML task results converted to Markdown in commit messages | `b3be38c0d` `38a7bca32` `f7d726827` `9a38e4302` `ba6e53974` `6e1a9ed8f` |
| Cron | Schedules evaluated in `America/Los_Angeles`; `ensure` action for idempotent jobs | `59561f6ed` `3d0245792` |
| Install and release | Node v22.23.3 and uv 0.12.19 pins; web app opened after install; PyPI size check and sdist `only-include`; pytest subprocess reaper; Windows test run fixes | `442c54960` `ab1f44abb` `253372168` `e4d8192f1` `537de3e65` `52a718e42` |

## 3. Research drivers, papers and website

**Prompt-level drivers** in SYSTEM.md: the 7-step AI discovery loop (baseline → SOTA search → `tmp/ideas.md` → pairwise judging → implement and evaluate → `tmp/explored-ideas.md` → repeat until the metric is met with a held-out check), adversarial testing (one sub-task breaks, another fixes), adversarial training (iterated adversarial datasets), Deep Work (read the target state fully, concrete values, planned changes, a verification per change), and the 10-site web research rule.

**Papers** (`papers/`, 10 directories, 6,050 files): every paper is LaTeX with its PDF checked in, written and revised by the agent through the `/write_paper`, `/review_paper` and `/revise_and_review_paper` SEAs (section 9; 41 of the 64 slash-command tasks in the database are paper tasks, section 22). 69 commits touched `papers/` between 09-22 and 09-30. Dates are the first and last commit to each directory; page counts are from the checked-in PDFs.

| Directory | Title | Format | Pages | Active | What it claims |
| --- | --- | --- | --- | --- | --- |
| `kisssorcar/kiss_sorcar.tex` | *KISS Sorcar: A Stupidly-Simple General-Purpose and Software Engineering AI Assistant* | NeurIPS 2026 preprint (arXiv 2604.23822), 43 sections, 6 appendices, checklist | 53 | 04-24 → 09-30 | Five agent classes of 4,531 significant lines under a 3,851-word prompt; ablation on 10 tasks × 6 arms (120 cells) finds the rules change how the agent works but not the hidden-test pass rate, and the second-vendor reviewer found real defects in 15 of 20 reviewed runs at 2.3× cost / 3.1× wall time; HydraKV 5.50 Mops/s on YCSB-A, TPC-H engine 34.5× faster; 30 Terminal-Bench 2.0 tasks × 7 models pooled 75.6% (74.1% under the 100-turn cap) vs Pi 70.0%; 59 held-out tasks 79.7%; paired Pi 0.87.1 on 57 held-out tasks (`qemu-alpine-ssh`, `qemu-startup` excluded for a verifier apt confound) **82.5% vs 71.9%** **NEW** |
| `kisssorcar/ks_assistant.tex` | *KS Gov: Long-Running AI Discovery and Adversarial Testing from a System Prompt* | ICLR 2027 double-blind (anonymised as "KS Gov") | 15 | 09-25 → 09-26 | Two long runs (HydraKV over 17.2 agent hours, TPC-H 34.5×) as case studies of the discovery loop and adversarial testing; TB2 comparison against the HarnessTax study; revised through nine review rounds; ships a renamed supplement (`make_supplement.py`, `supplement_README.md`: `ks` package, `ksgov.db`) **NEW** |
| `fable_sol/` | *A Trace-Based Study of Heterogeneous LLM Review and Repair in a Ten-Week Deployment* | ICLR 2027 double-blind, one directory per section, `math_commands.tex` | 29 | 09-24 → 09-25 | 486 tasks and 1.1M events from a deployment where an Anthropic model wrote and an OpenAI model reviewed read-only: 3 of 2,164 attributable reviewer calls performed a write, none to tracked source; review took 9.3% of task cost on the 42 separable tasks; 1,943 self-reported findings and 1,779 fixes, a second coder counts a fifth fewer; 60-task re-coded sample; no comparison arm; scripts in `scripts/` (`corpus_selection.py`, `review_passes.py`, `later_round_recoding.py`, `reviewer_child_costs.py`); the reviewer's own round in `fable_sol.txt` (rating 6) **NEW** |
| `sesorcar/` | *Rules in the Prompt, Guards in the Tools: Software Engineering Discipline in a Self-Hosted Coding Agent* | ACM `acmart` (FSE 2027, anonymous review) | 22 | 09-28 → 09-29 | Six months of the agent's own task database and git history (20,074 full-prompt tasks): 5.2% of `Edit` calls went to a file not opened with `Read`, 0.17% to a file never named in any call; after the read guard, repeat violations fell from 49.5% to 11.3% of later files at one wasted step per 35 guarded tasks; strict success rose 79.5% → 84.3% over six months but not within a single model; full-vs-lite controlled study; 464-file `evidence/` (`events_export.jsonl.gz`, `edit_pairs.py`, `classify_quality.py`, `ablation_summary.py`), macros recomputed by `se_numbers.py` and `quality_macros.py` into `macros.tex`, figures by `figures.py` **NEW** |
| `sorcarccl/` and `Collective_Algebra_…/` | *Collective Algebra Above a Closed Library: A Research-Loop Agent Finds the Rewrites That Strategy Enumeration Misses* | ICLR 2027 double-blind | 15 / 15 | 09-15; 09-25 | The research-loop agent replaces OverlayCCL's strategy enumerator above the closed Trainium collective library; on a 143-problem pool the agent's rewrite beats the enumerator by >5% in the simulator on 55 (09-15 build) / 40 (09-25 ICLR build) anchors, 45 / 40 confirmed on 224 Trainium cores at 1.06–3.38× / 1.12–4.37× warm-cache speedup, the enumerator wins 0; step time 2.46× (9.75B Llama-style) and 2.15× (9.70B GPT-3-class) faster. The 09-25 directory adds the 858-file `sorcarccl-production-set-main/` (40 production all-reduce / all-gather / reduce-scatter problems, harness, results), `compute_numbers.py`, `make_figures.py`, `make_tables.py`, `notes/citation-verification.md` and `notes/review-dispositions.md` **NEW** |
| `sekisssorcar/` | *Software Engineering KISS Sorcar with KISS Sorcar* | NeurIPS 2026 style, with `se_kiss_sorcar_slides.pptx` | 19 | 05-15 → 06-30 | 3,099 tasks over 44 days (April 22 to June 5) mined from the SQLite log into nine recurring patterns (test-first bug fixing, paper–code co-evolution, defensive revert, cross-model iterative review, ...); six SE principles encoded in the prompt and a five-layer hierarchy of roughly 2,400 lines |
| `kvstorepaper/` | *HydraKV: Adversarial AI Discovery of a Larger-than-Memory Key-Value Store* | NeurIPS 2026 style, `social/hackernews.md` | 19 | 07-18 → 09-15 | YCSB-A (Zipfian θ = 0.95) under a hard memory budget on 64 vCPUs and eight NVMe SSDs: 5.50 Mops/s vs FASTER's 0.93 (5.9×), about 89% of the miss-bandwidth bound, 3.87–5.67 Mops/s on four adversarial workload variants; ~4,000 lines of dependency-free C++17 (O_DIRECT log, fingerprint hash index, admission-controlled write-back cache, `io_uring` read-miss path, scan-based crash recovery, hole-punch compaction) |
| `swedefend/` | *SWEDefend: A Confidence-Gated Intent-Alignment Judge with Capability-Diff Reasoning for Automated-Program-Repair Backdoor Defense* | NeurIPS 2026 style | 14 | 07-12 → 07-20 | Defends APR agents against SWExploit-style magic-string-gated CWE payloads; a naive fail-closed intent judge reaches 100% catch at ~20% false positives, three design elements lower that operating point; the evaluator ships as the `swedefend-eval` console script and `projects/swedefend` is in the sdist (sections 1 and 20) |
| `cleverest_plus_paper/` | *Cleverest+: A Fixed-Budget Portfolio and Signature-Grounded Oracle for LLM-Based Commit-Directed Test Generation* | NeurIPS 2026 style, `REVIEW_NOTES.md` | 11 | 07-10 → 07-11 | Four mechanisms on top of Cleverest (JSON/base64 candidates with an allowlisted argument vector, shell-free invocation, sanitizer signatures, a (4,3,3) portfolio of DeepSeek-R1 and GPT-4o over exactly 10 trials fixed before any outcome); a three-subject mini-benchmark because the 72-commit benchmark's eight builds are not co-located |
| `writingpaper/` | *Writing a Research Paper with an AI Agent: A Chronicle of KISS Sorcar Writing Its Own Paper* | NeurIPS 2026 style | 9 | 04-30 → 06-06 | Nine days and over one hundred user tasks, each in the task database with its git diff, organised into seven phases; the human steered at the level of intent while the agent edited LaTeX, searched for citations, ran `pdflatex` and fixed its own bugs when the diff/merge interface failed |

**Numbers are regenerated, not typed.** Every headline figure above is a `\newcommand` macro (326 in `kiss_sorcar.tex`) recomputed by a script in the paper directory: `kisssorcar/ks_numbers.py` and `ks_tb2.py` (the latter also writes `tables/ks_tb2_table.tex` and `figures/ks_tb2_frontier.pdf`), `paste_macros.py`, `ks_figures.py`; `sesorcar/se_numbers.py`; `sorcarccl/compute_numbers.py`. Inputs live beside the papers: `kisssorcar/evidence/` (`mine_sorcar_db.py`, `mine_case_studies.py` → `case_studies.json`, `classify_reviews.py` → `review_verdicts.jsonl`, `db_metrics.json`, `tb2_trials.json` with the `main`, `heldout` and `pi` runs, `tb2_prompt.txt`, `tb2_blog_tables.json`) and the 4,575-file `kisssorcar/ablation/` (`arms.py` freezes the six prompt arms at the commit they ran on, `run_all.py`/`run_one.py`/`run_staged.py`, `paired_stats.py`, `rate_findings.py`, `PREREG_staged.md`, 122 result files plus `results_hard`, `results_heldout`, `results_heldout_calib`, `results_review` and their ratings). The `/write_paper` SEA's `check_paper` gate (AI-slop and consistency checks) and `build_paper` (pdflatex + bibtex) run on every revision.

**Around the papers**: the 44-page `KISS_Sorcar_Lecture.pdf` / `.pptx` and a one-slide deck in `kisssorcar/`; six social drafts in `kisssorcar/social/` (X and LinkedIn posts for the paper and for the HarnessTax TB2 result, rewritten four times on 09-29 and 09-30 to lead with the headline and match the blog's voice) and `kvstorepaper/social/hackernews.md`; the "Harness Tax, Audited" blog (`website/.../blog/harness-tax-terminal-bench-blog.html`, updated 30 September) **NEW**, which joins five optimisation blogs (LZ4, DuckDB, SQLite ×2, Tuso).

**Known gaps in `papers/`**: the Collective Algebra paper exists twice, and the two `sorcarccl.tex` files disagree on `\Anchors` (55 vs 40), `\RTconf` (45 vs 40) and the speedup range, so a reader must know that the long-named directory is the ICLR build; `\PromptWordsNow` is 3,851 in both `kisssorcar` papers against 3,971 words in SYSTEM.md today, and `\LocFive` is 4,531 in `kiss_sorcar.tex` but 4,449 in `ks_assistant.tex`; `KISS_Sorcar_Lecture.pdf` (built 06-09) predates the folder-per-SEA layout, the routers and the memory move; `website/.../llms.txt` lists six paper PDFs and omits `fable_sol`, `ks_assistant`, `sesorcar` and Collective Algebra.

**Website** (`website/kisssorcar.github.io/`): 11 docs pages, 6 blog pages, `privacy.html` **NEW**, `llms.txt` / `llms-full.txt`, sitemap.

## 4. Architecture

```
 Clients (one shared webview, media/chat.html + main.js)
 ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌────────────────────────┐
 │ VS Code extension│ │ Remote web app   │ │ sorcar CLI   │ │ 44 channel CLIs, cron, │
 │ (editor tabs or  │ │ (WSS, PWA, light │ │ Python API   │ │ A2A and OpenAI-compat  │
 │ secondary bar)   │ │ theme default)   │ │ (CLI: local) │ │ servers                │
 └────────┬─────────┘ └────────┬─────────┘ └──────┬───────┘ └───────────┬────────────┘
          │ UDS ~/.kiss/sorcar.sock or WSS :8787 (same JSON commands, sorcar.py API catalog)
 ┌────────▼──────────────────────▼────────────────▼──────────────────────▼────────────┐
 │ kiss-web daemon  src/kiss/server (31 modules, 37,727 lines)                        │
 │  VSCodeServer: TabRegistry (~/.kiss/tabs.json), replay, sub-agent tab lifecycle    │
 │  task_runner · json_printer · merge_flow · browser_tab (KISS Browser) · file_index │
 │  sidebar_panels (Schedule/Apps/Spend) · task_update · tls_certs/tls_trust · voice  │
 │  cron scheduler thread · SEA registry watcher (2 s) · worktree pool prewarm        │
 └────────┬───────────────────────────────────────────────────────────────────────────┘
          │ one agent per task or sub-agent
 ┌────────▼───────────────────────────────────────────────────────────────────────────┐
 │ WorktreeSorcarAgent ⊂ ChatSorcarAgent ⊂ SorcarAgent ⊂ RelentlessAgent ⊃ KISSAgent  │
 │  tools: Bash/bash_job/Read/Edit/Write/run_commands_parallel · 9 browser · 7 memory │
 │         run_agent/run_parallel/number_of_cores · MCP + connect_mcp_server · skill  │
 │         ask_user_question/talk/set_model/decide/summary/finish                     │
 │  prompt: SYSTEM.md (SYSTEM_LITE if simple) + ~/.kiss/SORCAR.md + memory protocol   │
 │  cost ledger · context compaction · prompt-cache keep-alive · fallback model       │
 └────────┬───────────────────────────────────────────────────────────────────────────┘
          │
 ┌────────▼───────────────┐ ┌──────────────────────┐ ┌───────────────────────────────┐
 │ Models (706 entries)   │ │ State in ~/.kiss     │ │ Extension points              │
 │ Anthropic/OpenAI/Gemini│ │ sorcar.db · memories/│ │ SEAs (seas/, SEAS.md folders) │
 │ OpenRouter/Together/   │ │ cron/jobs.json ·     │ │ skills (SKILL.md) · MCP       │
 │ Z.ai/Moonshot · cc/    │ │ tabs.json · SORCAR.md│ │ servers · tools= files ·      │
 │ codex/ CLIs · autorouter│ │ AUTOROUTER.md · TLS  │ │ INJECTIONS.md · MY_INJECTION  │
 └────────────────────────┘ └──────────────────────┘ └───────────────────────────────┘
```

Figure 1. The data path from clients to models. Every client speaks the same command catalogue (`src/kiss/server/sorcar.py`, `API`), whether over the Unix socket or a WebSocket, and receives the same events; the daemon runs the agents for every client except the standalone `sorcar` CLI, which runs `SorcarAgent` in its own process.

## 5. Interfaces and prompt surfaces

| Surface | How it is reached | Notes |
| --- | --- | --- |
| VS Code extension `ksenxx.kiss-sorcar` | Activity-bar History view; chat as editor tabs (default, `kissSorcar.editorTabsMode`) or in the secondary sidebar; Task Info in the secondary sidebar | 13 commands, 4 keybindings (`Ctrl/Cmd+T` new conversation, `Ctrl/Cmd+D` focus chat, `Ctrl/Cmd+E` run selection, `Ctrl/Cmd+L` insert selection), 4 VS Code settings (`defaultModel`, `kissProjectPath`, `editorTabsMode`, `checkForUpdates`); SCM sparkle generates commit messages |
| Remote web app | `https://<host>:8787` served by `kiss-web`, password from Settings; Cloudflare tunnel URL on remote deploys; installable PWA with an offline shell (`sw.js`) | Same `chat.html`/`main.js` as the extension; light theme by default **NEW**; reconnects in place with backoff and a 45 s half-open detector **NEW** |
| `sorcar` CLI | `sorcar -t "task"` or `-f file`, with `-m model`, `-b budget`, `--work-dir` | Runs `SorcarAgent` in the terminal process, outside the daemon; the 44 `kiss-<channel>` CLIs run through the daemon instead |
| Python client API | `from kiss.server import sorcar; sorcar.run(task, ...)` returning `TaskResult(text, success, cost, tokens, steps, chat_id, task_id)` | 27 keyword options including `tools=` (path of a file whose `get_tools()` returns callables), `extension_agent_path`, `tool_profile`, `docker_image`, `timeout=3600`, `stop_on_timeout` |
| Slash commands | `/<name> <text>` at position 0 of a prompt | Registered from three sources (section 9); `/<name> help` prints the SEA's `description()` |
| Channels | `kiss-<channel>` CLIs, `run_agent(agent="slack", ...)`, always-on gateways scheduled by cron | 44 agents (section 18) |
| Voice | In-page wake word "Hey Sorcar" **NEW wording**, host-side Vosk listener, `talk()` playback on every open tab | Section 17 |

Prompt assets shipped in `src/kiss/`:

- `SYSTEM.md` (3,971 words): identity, rule precedence, visibility constraint, tool usage rules, web research protocol (10 sites, 1 for real-time data), code style, pre-flight checks, the 7-step AI discovery loop, adversarial testing and training, deep work, complex task planning, file browsing, testing, pre-finish verification, Sorcar-specific rules. The static part is sent under one Anthropic `cache_control` breakpoint; the per-task "Task Settings" block follows uncached. **NEW**
- `SYSTEM_LITE.md` (753 words): the reduced prompt for tasks the classifier marks as simple; `/ask` ships its own `_ask_system_lite.md` (608 words) plus an answering playbook.
- `~/.kiss/SORCAR.md`: standing user instructions appended to every system prompt, managed by `/remember` and `/forget`. **NEW management**
- `INJECTIONS.md` (6 promptlets) plus `~/.kiss/MY_INJECTION.md` (user promptlets, editable in place from the Inject panel **NEW**); `TIPS.md` (23 tips shown once per version **NEW cadence**); `SAMPLE_TASKS.md` (12 sample tasks).
- `~/.kiss/SEAS.md`: extra SEA folders, one per line, bottom line wins.

## 6. Agent runtime

### KISSAgent (`src/kiss/core/kiss_agent.py`)

- Single agentic loop: `run(model_name, prompt_template, arguments, system_prompt, tools, is_agentic, max_steps, max_budget, model_config, printer, attachments, llm_call_hook, tool_call_hook)`. Stops after 3 consecutive errors or 2 consecutive responses without a tool call; CLI (run-to-completion) models default to a 3,600 s output-silence timeout, overridable through `model_config["timeout"]`.
- Context limit fraction 0.7 (`KISS_CONTEXT_LIMIT_FRACTION`); a session that hits it hands off to the next RelentlessAgent session.
- **Tool-output compaction**: once the conversation passes 100K tokens, and every further 100K, tool results longer than 500 characters and older than the six most recent (never `Edit`, `Write`, `finish`) are replaced by a stub holding a 200-character preview, unless that would drop less than 25% or the session is within 80% of its hand-off point.
- **Prompt-cache keep-alive**: fan-out tools, `bash_job(action="wait")` and any tool call with a timeout of 300 s or more run under a thread that pings the model's prompt cache every 240 s (at most 10 pings); each ping is billed and recorded as an `llm_call` event.
- **Fallback model** on non-retryable errors (authentication, permission, missing API key): `model_info.get_fallback_model()`.
- **Per-call cost events** **NEW**: every billed model call emits `llm_call` with `model`, `step`, `duration_ms`, input/output/cache-read/cache-write tokens and cost; audio tokens are priced separately; billed-but-rejected responses (truncations, refusals) are charged.

### RelentlessAgent (`src/kiss/agents/sorcar/relentless_agent.py`)

- Runs KISSAgent sessions until the task finishes; a session ending on the context limit is summarised (60K-character cap) into a hand-off; the run stops after 2 zero-progress sessions. Defaults: model `claude-opus-4-6`, budget $200.
- Appends `~/.kiss/SORCAR.md` to every system prompt, reading it with writer-tolerant IO so a rename in progress cannot abort the task.
- Usage ledger with epochs and seen-maps so sub-agent, classifier, TTS and side-channel spend are counted exactly once, including across stop interrupts; printer offsets are re-based before the failed-session summariser so the UI cost never drops. **NEW**
- Docker mode: `docker_image=` starts a container with the task work dir bind-mounted at its host path and set as `WorkingDir`; `container:<id>` attaches to a running container without mounts; CLI models cannot be combined with Docker. **NEW mount**

### SorcarAgent (`sorcar_agent.py`, 3,511 lines)

- Builds the tool set (section 7), the system prompt (SYSTEM.md + restricted-profile note + `WEB_TOOLS_OFF_NOTE` when browsing is off **NEW** + memory protocol + SEA additions), the memory root and domain memories, and the model (a picker SEA such as `autorouter` resolves to its `model()`).
- `set_model` rebuilds the tool schema for the new model and shows it in the picker; `summary` is a no-op tool whose every-10-steps cadence is enforced by the prompt only.
- Review fan-out guardrails (`ReviewQuota`, `MAX_REVIEW_ROUNDS`, `MIN_SUBAGENT_BUDGET`, `KISS_REVIEW_BUDGET_FRACTION`) were removed on 2026-09-25; a sub-agent's budget share is now `remaining / (num_tasks + 1)`. **CHANGED**

### ChatSorcarAgent and WorktreeSorcarAgent

- `ChatSorcarAgent` persists chats and task chains in `sorcar.db`; prompts carry at most 10 prior tasks, normally the newest two in full and the older ones as 600/300-character task/result digests, shortened further to fit a 6,000-character prefix cap (`chat_history_digest`, default on). It records the SEA name of the run (`task_history.sea`) **NEW** and bare-path prompts are turned into open-file directives **NEW**.
- `WorktreeSorcarAgent` gives a development-classified task its own `git worktree` under `.kiss-worktrees/` on branch `kiss/wt-*` (worktrees off, non-development verdicts, non-git directories and detached HEADs run directly); a spare worktree is prewarmed at daemon start and refilled after each worktree task ends **NEW**; outcomes: committed and removed, preserved (no auto-commit, commit failed, sub-agent active, rescue failed). Section 12 has the merge rules.

### Configuration knobs (`src/kiss/core/config.py`)

`KISS_READ_DEDUPE` (on), `KISS_READ_OUTLINE_LINES` (2000), `KISS_TOOL_OUTPUT_COMPACTION` (on), `KISS_COMPACTION_START_TOKENS` / `KISS_COMPACTION_STEP_TOKENS` (100,000), `KISS_TOOL_OUTPUT_MAX_CHARS` (50,000), `KISS_CONTEXT_LIMIT_FRACTION` (0.7), `KISS_TOOL_PROFILES` (on), `KISS_CHAT_HISTORY_DIGEST` (on), `KISS_DISPATCH_PATH_REWRITE` (on), `KISS_USE_MEMORY` (one-process override), `KISS_HOME` (default `~/.kiss`), `KISS_DISABLE_WORKTREE_POOL`, `KISS_DISABLE_TASK_CLASSIFIER`, `KISS_MUSE_AUTH=0`; seven API-key variables plus a workspace id (`ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`). Per-run settings in `~/.kiss/config.json`: `max_budget`, `auto_commit_mode`, `is_worktree`, `use_web_browser`, `classify_tasks` (on), `classify_with_decisions` (off), `use_memory` (on), `memory_dir` (empty = `~/.kiss/memories`), voice options, remote password, custom models.

## 7. Built-in tools and tool profiles

| Group | Tools | Notes |
| --- | --- | --- |
| Files and shell | `Bash`, `bash_job`, `run_commands_parallel`, `Read`, `Edit`, `Write` | `Bash(background=true)` detaches with `nohup` and returns a job id; `bash_job` tails, waits or kills; `run_commands_parallel` runs shell commands in threads with per-command exit codes; a whole-file `Read` of a file over 2,000 lines returns an outline, and an unchanged range already shown returns a one-line stub (`force=True` re-reads); `Edit` rejects a file not read in the session and `Write` rejects overwriting an unread existing file (new files and scratch files under `tmp/` are exempt). In Docker the file tools execute inside the container without this guard and `bash_job` is absent. |
| Browser | `go_to_url`, `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`, `show_browser`, `close_browser` | Patchright/Chromium; `show_browser` streams the page into the KISS Browser tab on every surface **NEW**; sub-agents get an ephemeral profile; only in the `full` profile with "Use web tools" on |
| Memory | `memory_search`, `memory_pull`, `memory_read`, `memory_write`, `memory_list`, `memory_delete`, `memory_refresh` | Section 13; `memory=` narrows to the general or a domain memory **NEW** |
| Dispatch | `run_agent`, `run_parallel`, `number_of_cores` | Section 10; `run_parallel` and `number_of_cores` only in parallel mode |
| MCP | tools of configured MCP servers, `connect_mcp_server`, `finish_mcp_server_connect` **NEW** | OAuth sign-in for remote MCP servers (Notion, Linear, Asana, Zoom or any URL); tokens in `~/.kiss/mcp_auth/<server>.json` |
| Skills | `skill` | Present when the work dir or `~/.kiss/skills`, `.kiss/skills`, `.agents/skills` or Claude skill directories hold a `SKILL.md` |
| Interaction | `ask_user_question`, `talk(language, text, emotion)`, `set_model`, `decide`, `summary`, `finish` | `decide` only when the Jev decisions model is enabled (section 11); `finish` carries `summary_in_html`, `is_continue`, `suggested_next_task` |

Tool profiles (`TOOL_PROFILES`, `sorcar_agent.py:71-92`):

| Profile | Tools kept | Used by |
| --- | --- | --- |
| `full` | everything above | default |
| `review` | `Bash`, `bash_job`, `Read`, `run_commands_parallel`, `memory_search`, `memory_pull`, `memory_read`, `memory_list`, `decide`, `summary` | reviewer children of `run_parallel`, `/ask` |
| `shell` | `Bash`, `bash_job`, `Read`, `run_commands_parallel` | `/skillopt` |
| `assistant` **NEW** | shell profile + `ask_user_question`, `talk`, `decide`, `summary`, `set_model` | conversational runs without file edits |
| `bash` | `Bash` | `/sh`, `/remember`, `/forget`, `/task_update` |

Every restricted profile also receives `ask_user_question`, `talk`, `set_model`, `summary` (and `decide` when available) before filtering, and `finish` is always added. A restricted run gets `RESTRICTED_PROFILE_NOTE` in its system prompt listing the tools it has.

## 8. Models, routing and cost accounting

- **Catalogue**: 706 entries; direct families are `gpt-*` (100 entries with `-low`/`-high`/`-xhigh` effort aliases), `gemini-*` (20), `claude-*` (16), `glm-*` (8), `kimi-*` (7), `o1`/`o3`/`o4` (8); OpenRouter (412), Together-hosted open weights (Qwen 24, Meta Llama 16, DeepSeek 13, Moonshot 10, Z.ai 10, DeepCogito 6, Mistral 6), `cc/*` (16) and `codex/*` (10) drive the Claude Code and Codex CLIs to completion. Price fields include cache read/write, long-context tiers and audio input. Custom OpenAI-compatible endpoints can be added from Settings with name, endpoint, key and headers.
- **Picker routers** **NEW**: `autorouter` splits a task into units and runs each on the cheapest tier (small, medium, frontier) that passes its acceptance check, escalating on failure and logging every decision to `~/.kiss/MODEL_DECISIONS.md`; its per-model evidence lives in `~/.kiss/AUTOROUTER.md` (capped at 2,500 characters, refreshed weekly by `/rsi7d`). `bestrouter` runs everything on `claude-fable-5-1` and has `gpt-6-astra` review read-only through `run_parallel` with at most 75% of the budget. Picking a router SEA fires its `on_picked_as_model` hook (15 s cap); autorouter's hook ensures the weekly cron job.
- **Cost pipeline** **NEW**: OpenRouter is billed from the provider-reported `usage.cost` (BYOK upstream cost added only when `is_byok`); classifier, merge-agent, TTS, side-channel and late-arriving sub-agent spend fold into the owning task via `_net_totals` and `_add_late_task_usage`; a sub-task stopped on timeout still charges its caller (`StoppedOnTimeoutError`); the Spend subpanel splits each task's cost across the models its sub-agent tree used.
- **Live accounting**: cost, tokens, steps and remaining budget are printed in every tool result and shown in the status bar and Task Info; the daily cost-calculation audit is a cron job on the development machine.
- **Model updates**: the "Update Models" button and the daily `update_models.py` cron job refresh prices; the `-low`/`-high` aliases mirror their base model's prices.

## 9. Sorcar Extension Agents and slash commands

An SEA is a Python file exposing optional zero-argument getters (`system_prompt`, `model`, `tools`, `tool_profile`, `max_budget`, `use_worktree`, `auto_commit`, `use_memory`, `use_web_tools`, `is_parallel`, `classify_tasks`, `append_to_system_prompt`, `add_to_system_prompt`, `append_to_prompt`, `dispatch_timeout`, `register_as_model`, `llm_call_hook`, `tool_call_hook`), an optional picker hook `on_picked_as_model(work_dir)` and, since `86c09b15d`, a mandatory `description()` returning one sentence that `/<name> help` prints. **NEW contract**

Registry (`src/kiss/agents/sorcar/sea_commands.py`): a folder `<name>/` becomes `/<name>` only if it contains `<name>_sea.py` (loose `*_sea.py` files are not commands); sources in precedence order are `third_party_agents/` (highest), folders listed in `~/.kiss/SEAS.md` (bottom line beats top line), then bundled `seas/`; a watcher thread rescans every 2 s and open chats receive the new command list; a miss triggers an immediate rescan. A matched command is rewritten into a `run_agent` directive with the SEA path and the verbatim text, adding `timeout=` when the SEA defines `dispatch_timeout()`.

The 17 bundled SEAs (`src/kiss/agents/seas/`):

| Command | Purpose | Profile / budget / timeout | Since |
| --- | --- | --- | --- |
| `/ask` | Answer a question about the running task from its digested trajectory (side channel in the task's tab) | `review`, no web, no memory | moved 09-27 |
| `/autorouter` | Tiered cheapest-model routing with escalation and a decision ledger; picker entry | orchestrator model = first runnable frontier model | **NEW** |
| `/bestrouter` | claude-fable-5-1 writes, gpt-6-astra reviews read-only (≤75% budget); picker entry | default | **NEW** |
| `/coding` | Unattended Sorcar inside a Docker container for benchmark trials (`ContainerHarness`, JSONL trajectories); generates per-trial SEAs | default | relocated |
| `/dummy` | Plain Sorcar sub-agent; what `run_agent(agent="")` runs | default | |
| `/forget` | Remove a standing instruction from `~/.kiss/SORCAR.md` | `bash`, $1 | **NEW** |
| `/git_extract_knowledge` | Index every tracked file and commit of a repository into its domain memory and an FTS5 block store; `update`, `ask` modes; daily 04:00 PT refresh job | `full`, parallel, no memory tools | **NEW** |
| `/merge` | Resolve git merge conflicts and stage the result; also run automatically by auto-commit merges | $5, no worktree | |
| `/remember` | Append a standing instruction to `~/.kiss/SORCAR.md` | `bash`, $1 | **NEW** |
| `/review_paper` | Review a paper (PDF/.tex/.md/.txt) for a venue with web search, seven 1-10 scores, word-limit and AI-slop gates; second opinion from `gpt-5.6-sol` | web on, parallel, 2 h | **NEW** |
| `/revise_and_review_paper` | Loop `/write_paper` and `/review_paper` (default 6 rounds, 1,000-word reviews) until strong accept or no further improvement | `full`, 24 h | **NEW** |
| `/rsi7d` | Mine the last 7 days of SEA runs in `sorcar.db` for mistakes and cost sinks, patch SEA prompts (replaying file-modifying tasks in clones), refresh autorouter evidence, and with permission patch `SYSTEM.md`, `SORCAR.md` and Sorcar's code | $2,000, memory on | **NEW** |
| `/sh` | Run one shell command in the tab's work dir and return its verbatim output | `bash`, no worktree | |
| `/skillopt` | SkillOpt outer loop over a SKILL.md, a SEA prompt or a module constant against a JSON eval set; writes `<target>.proposed` | `shell` | **NEW** |
| `/task_update` | Report what a task has done from its persisted transcript; run by the Task Info panel on first show, every 10 min and on refresh | `bash`, $1 | **NEW** |
| `/write` | Concise professional prose with machine-text vocabulary banned | default + writing protocol | **NEW** |
| `/write_paper` | Write or revise a LaTeX paper with `check_paper` (AI-slop and consistency gates) and `build_paper` (pdflatex + bibtex) and a read-only reviewer model | web on, parallel, 6 h | **NEW** |

Files these SEAs keep under `~/.kiss`: `SORCAR.md`, `SEAS.md`, `AUTOROUTER.md`, `MODEL_DECISIONS.md`, `memories/<repo>/knowledge.sqlite3`, `cron/jobs.json` (weekly rsi7d and daily knowledge jobs), and read-only access to `sorcar.db`.

## 10. Sub-agents and parallelism

- `run_agent(task, agent="", workspace, model_name, max_budget, timeout, chat_id, system_prompt, tools, model_config, use_worktree, auto_commit, use_web_tools, classify_tasks, use_memory, is_parallel, append_basic_tools, append_to_system_prompt, append_to_prompt, tool_profile)`: `agent` is empty (plain sub-agent), a channel name, `"cron"`, or a `.py` path. Default wait 300 s; on timeout the child is stopped and its spend still charged **NEW**. Path-named agents honour the persisted worktree and auto-commit settings **NEW**. Task text naming the parent repository path is rewritten to the active worktree (`dispatch_path_rewrite`).
- `run_parallel(tasks, max_workers, model_name, tool_profile)`: independent LLM sub-agents, each in its own tab on every surface; fan-out may recurse (the prompt asks for at most two levels, the code sets no limit); children classified as reviewers get the `review` profile when tool profiles are on and the task needs no implementation, unless `tool_profile=` says otherwise.
- `run_commands_parallel(commands, max_workers, timeout_seconds, max_output_chars)`: shell commands in threads, no LLM.
- Sub-agent tabs: opened on all clients while the child runs, closed everywhere by `subagentDone` when it finishes or by `closeSubagentTab` when any client closes them; the user's close is remembered so a replay does not reopen it. **NEW**
- 21,216 of the 27,673 recorded task rows are sub-agents; 4,461 top-level tasks ran in parallel mode.

## 11. Task classification, tags and chat summaries

- **Pre-run classifier** (`task_classifier.py`, "Classify tasks before running", default on): decides `is_simple` and `is_development` so trivial prompts skip the worktree and the full prompt; verdicts are cached; classifier spend folds into the task. Launch phases "Classifying task…" and "Preparing worktree…" are shown in the tab. **NEW**
- **Jev decisions model** (`openrouter/~typesafe/jev-latest`): one gate, `decisions_tool_available()`, controls both the classifier's Jev route and the `decide` tool; it needs the "Use Jev (decisions model)" setting (default **off** since 09-27), an OpenRouter key and the model in the catalogue.
- **Tags** **NEW**: when a task finishes, a regex classifier (no model call) writes up to 6 comma-separated tags to `task_history.tags`: `work`/`personal` first, then `secret`, `chore`, `question`, `coding`, `testing`, `debugging`, `review`, `research`, `paper`, `writing`, `docs`, `data`, `devops`, `messaging`, `browsing`, `shopping`, `finance`, `scheduling`, `failed`, `subagent`. The History panel's tag filter is applied in SQL by the daemon.
- **SEA column** **NEW**: `task_history.sea` holds the file stem of the agent script that ran the task (`bestrouter_sea`, `review_paper_sea`, `cron_agent`, `slack_sea`, ...), back-filled for sub-agents from their parent's `run_agent` call; the History panel's SEA filter offers `All`, `None` and one entry per script seen.
- **Chat summaries** **NEW**: `chat_summaries(chat_id, summary, last_launched)` stores a title of up to 8 words per chat (target 6-8, derived from the chat's first five top-level tasks, sometimes shorter or empty); History shows it as the group header when present and the first task's text otherwise (2,743 rows on the development machine). `src/kiss/scripts/backfill_task_metadata.py` fills tags and chat summaries in batches of 500 and SEA values per parent task.

## 12. Git worktrees, merging and auto-commit

- With "Use worktree" on (default), a task the classifier marks as development runs in its own worktree under `<repo>/.kiss-worktrees/` on a `kiss/wt-*` branch; the main checkout's `node_modules` directories are symlinked in; `core.untrackedCache` is enabled when unset (an existing setting is preserved). **NEW cache**
- **Worktree pool** **NEW**: one spare per repository, registered with `git worktree add --no-checkout` under the repo lock and populated outside it; prewarmed at daemon start (`prewarm_worktree_pool`), consumed by the next task, refilled after a worktree task ends; disabled by `KISS_DISABLE_WORKTREE_POOL` (the test suite sets it).
- **Merge flow** (`src/kiss/server/merge_flow.py`, 2,212 lines): with auto-commit on, the branch is squash-merged into the main tree when the task succeeds (failed or stopped tasks stay pending for an explicit Merge); conflicts are handed to the `/merge` SEA running in-process as a nested sub-agent of the task, a rejected or unresolved resolution reports `CONFLICT` again, and a failed commit resets the tree with `MERGE_FAILED`. Without auto-commit the user picks Merge, Leave as is, or Discard; automatic cleanup rescues ignored task output, an explicit Discard does not. A non-worktree task on another tab blocks a merge only when `git status --porcelain -uno` on the main tree is non-empty, failing closed on git errors **NEW**; deferred merges are retried after task completion, Git Commit or a main-tree Discard and proceed only when the tracked main tree is clean; orphaned `is_merging` claims whose thread died are released automatically **NEW**.
- **Commit messages**: generated from the staged diff, capped at 200,000 bytes streamed through `_git_stdout_head` with a `--stat` fallback so a huge diff cannot hang the merge **NEW**; HTML task results are converted to Markdown before they are stamped into the message **NEW**; `has_staged_changes` (`git diff --cached --quiet`) replaces whole-patch emptiness checks.
- **Orphan recovery**: `sweep_orphaned_state` and `reclaim_orphaned_worktrees` clean up after a crashed daemon; the `owner` column in `task_history` records the process token; a stale stop signal is cleared before deferred disposal so a later "cron list" is not refused with "A worktree merge is in progress" **NEW**.
- 3,197 of 27,673 task rows carry `is_worktree = 1`; 3,966 of 6,457 top-level tasks ran with auto-commit on.

## 13. Persistent memory and repository knowledge

`kiss.core.memoryfield` (`pages.py`, `index.py`, `tools.py`, `evaluate.py`; 2,431 lines):

- **Pages**: a flat directory of Markdown files with YAML frontmatter (`title`, `uuid`, `summary`, `created`, `updated`), names `^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$`, an 8 KB guideline (larger pages are saved with a warning and only the first 8,192 bytes are embedded), atomic publish; deletions leave `.tombstones/<name>` so a sync cannot resurrect a page **NEW**.
- **Index**: SQLite per embedder (`text-embedding-3-small.sqlite3` when `OPENAI_API_KEY` is set, otherwise the offline 1,024-dimension `hashed-bow-v1.sqlite3`), schema v3 with a read-only fast path, git-style racy-clean stat checks (3 s window) so unchanged pages with a mature stat key are not re-read (racy files and `memory_refresh` verification still hash), batched embeddings (256 inputs / 250 KB per batch via the new `Model.get_embeddings`), compare-and-swap row updates, a cached numpy float32 matrix scan with exact `math.sumprod` rescoring, a 64-entry query cache, near-duplicate detection at cosine 0.9. **NEW performance work**
- **Tools**: `memory_search(query, k, memory)`, `memory_pull` (full pages up to 24,000 characters), `memory_read`, `memory_write` (rejects bad names, warns on oversize pages and on ephemeral names such as `round-3`), `memory_list`, `memory_delete`, `memory_refresh(stale_days=30, duplicate_threshold=0.9)`. The `review` profile keeps the four read-only ones.
- **Location and domains** **NEW**: the general memory is `~/.kiss/memories/`; a repository gets a domain memory `~/.kiss/memories/<slug of the main checkout's directory name>/` (`general` maps to `general-repo`) shared by its worktrees (`git rev-parse --git-common-dir`), addressed as `<memory>/<page>`; searches span all memories unless `memory=` narrows them. "Use persistent memory" (default on) can be overridden per run, by `KISS_USE_MEMORY`, or by a SEA's `use_memory()`; four hard gates disable it (no basic tools, Docker, `cc/`/`codex/` models, caller-supplied `system_instruction`). On the development machine: 759 general pages, `kiss/` 374 pages.
- **Repository knowledge** (`/git_extract_knowledge`) **NEW**: curated pages (`overview`, `architecture`, `conventions`, `history`, `module-*`, `faq`, `recent-changes`, `knowledge-lookup`) plus an FTS5 block store `knowledge.sqlite3` with kinds `repo, dir, file, chunk, symbol, commit, change, tag, branch, author, note`; incremental (blob hashes, `rev-list --all` set difference, children-first writes), 80-line chunks, per-language symbol patterns, BM25 with title 4x / key 2x and history rank factors 0.6/0.8; clone URLs are checked out under `~/.kiss/knowledge/checkouts/`; a daily 04:00 PT cron job refreshes it ($25 + $2 relay, 4 h).
- **Memory sync on deploy** **NEW**: `rsorcar` step 4c streams pages and tombstones both ways as tar over ssh and merges with `merge_memory_pages.py` (newest `updated` wins, mtime tie-break for hand edits, same-second conflicts named on stdout, tombstones propagate, indexes never copied); clock skew between machines is measured first and widens the tolerance.
- **Recall evaluation**: `python -m kiss.core.memoryfield.evaluate` builds a memory from past tasks and reports Recall@1/3/5 and MRR for vector, hashed, BM25 and hybrid retrievers; `src/kiss/scripts/memory_search_eval.py` runs a YAML question set (76 questions for the `kiss` memory) with a `--min-recall5` gate.

## 14. The kiss-web daemon and remote web app

`src/kiss/server/` (31 modules, 37,727 lines; `web_server.py` 10,881). One daemon per machine owns the agents, the tab registry, the cron scheduler, the SEA watcher, the worktree pool, the browser tab service, the file index and the voice pipeline.

- **Transports**: Unix socket `~/.kiss/sorcar.sock` (newline-delimited JSON) and WSS on port 8787 with the same command catalogue (`sorcar.py` `API`, `validate_command`, `ServerApi.dispatch`; 76 whitelisted webview commands). A `heartbeat` frame drives the client's half-open detector.
- **Tab registry** **NEW**: `TabRegistry` persisted atomically to `~/.kiss/tabs.json` (cap 512) is mirrored by every client; `ready` replays the registry, the running tasks' recordings and the browser-tab snapshot to the connecting client only (`singleTabId` for an editor-tab panel); replay snapshots are taken under the delivery lock with a reserved send slot so no event is duplicated or lost **NEW**.
- **KISS Browser tab** **NEW**: `BrowserTabService` launches the daemon machine's browser with a persistent profile (`~/.kiss/browser-tab-profile`), mirrors each page as a `browser__*` tab (1280×800, JPEG quality 60, at most 15 frames/s, 90 s open timeout) and relays clicks, keys and viewport changes; commands `browserOpen`, `browserClose`, `browserNavigate`, `browserInput`, `browserViewport`; agents receive it as `live_browser` so `go_to_url` and `show_browser` stream into the same tabs; third-party sign-in pages open there through `browser_handoff.set_browser_tab_opener`.
- **Sidebar data** **NEW**: `getCronJobs` (Schedule), `getAppsStatus` (Apps; every channel module probed in a subprocess, cached 30 s), `getSpendReport` (Spend; total, per day, per model from persisted `cost`/`tokens`, sub-agent rows excluded), `getTaskUpdate` (Task update; runs the `/task_update` SEA on the first poll and again once the previous report is 600 s old, or at once on Refresh, as a side channel whose spend is charged to the task).
- **TLS** **NEW**: a local CA and leaf certificate (`tls_certs.py`, renewed automatically); `kiss-web --trust-ca` installs the CA into NSS databases, the macOS login keychain or the Windows user store; other devices fetch it from `GET /ca.crt`.
- **File index** **NEW**: a persistent home-wide index (`~/.kiss/file-index`, depth 12, 1,000,000-entry cap, dot-dirs, junk dirs and literal positive `.gitignore` entries skipped, bulk data dirs collapsed) warmed at startup serves the `@` picker with `./` matches from the work dir followed by `~/` matches; refreshed by directory mtime, 60 s staleness.
- **Prompt shortcuts** **NEW**: a prompt that is only a filesystem path opens the file (delivered as base64; PDFs in the shared pdf.js viewer, images in the image preview) instead of starting a task.
- **Branding** **NEW**: `kiss/core/brand.py` reads `media/brand.json` (`PRODUCT_NAME`, `SHORT_NAME`, tagline, identity) and the page template injects `brand.css`, `PRODUCT_NAME` and `BRAND_JSON`; `install.sh` overlays a git-ignored `.brand/` directory onto the extension media and `package.json` during the VSIX build for white-label builds.
- **Resilience**: stall watchdog (`faulthandler` dump after 60 s of a blocked GIL, disarmed at shutdown **NEW**), `_MAX_INPUT_HISTORY` 500, service worker cache `kiss-shell-<version>` (`/media/*` cache-first, `/` network-first with offline shell, `/ws`, `/api/*`, `/trajectories` and the voice model never intercepted), forced restart from the extension when the daemon is wedged **NEW**, "Reset Server" with an in-panel confirmation when a task is running.
- **Trajectory sharing**: `shareChat` writes `reports/chat-<id>.html` with `share.js` (collapsible think panels, nested `run_parallel` groups, a sub-agent tab bar, light/dark theme).

## 15. Chat client features

One `chat.html` (639 lines, 34 template placeholders) and `main.js` (23,860 lines) render the VS Code sidebar chat, each editor tab, the History view, Task Info and the remote app.

- **Composer**: `@` file mentions from the home-wide index, `#` promptlet injection, drag-and-drop and Attach files, ghost autocompletion from input history, Listening overlay for voice, Suggested-next bar that survives a report tab taking focus **NEW**, `/` slash commands from the live SEA registry.
- **Footer menu**: history toggle, new chat, Inject promptlet (edit, copy and delete user promptlets in place **NEW**), Working directory (scoped to the active idle chat tab; a running task keeps its folder **NEW**), Voice trigger, Share chat, Attach files, Git Commit, open the daemon machine's browser as a tab **NEW**, Switch theme, Settings.
- **Model picker** with search, custom models, and the `autorouter`/`bestrouter` router entries **NEW**; `set_model` calls by the agent update it live.
- **History panel**: search, flat or grouped view, filters for Running / Errored / Succeeded, workspace-only, favourites, date range, **tag** and **SEA** **NEW**; chat headers show the stored chat summary when present **NEW**; task rows end with age, tags and SEA **NEW**; favourite, copy and resume actions; adjacent-task navigation.
- **Task Info panel**: five collapsible, resizable, equal-height-by-default sections **NEW**: Task Info (tokens, cost, steps, time, machine, work dir, budget, date, model, worktree, parallel, chat/task/parent ids), Task update (SEA report, refresh button), Schedule (cron jobs with copy buttons), Apps (channel auth state; click an unconnected app to start a sign-in task), Spend (per-day, per-model graph).
- **Explorer and Source control** views in the left sidebar: file tree with context menus, git status, log and graph, worktree and main-tree actions, commit-message generation.
- **Settings panel** (14 groups): Remote password, Max budget per task, Auto commit, Use worktree, Use web tools, Classify tasks, Use Jev (off), Use persistent memory, Chat in the editor (VS Code only), Auto submit spoken task, Wake word sensitivity (0-100, default 80), buttons Tips / Update / Reset Server / Update Models, collapsible API Keys (8) and Custom Models. The Working directory and Memory directory fields were removed on 09-27. **CHANGED**
- **Editors and viewers**: Monaco editor tabs with a menu bar and themes derived from the VS Code palette **NEW**; shared pdf.js viewer (pdfjs-dist 6.3.289) with Download link, editable page indicator and keyboard shortcuts (Ctrl/Cmd+0 fit width, Ctrl/Cmd with `+`/`-` zoom, Home/End, PageUp/PageDown, Left/Right when the pages fit horizontally) **NEW**; image preview; report tabs for generated HTML.
- **Look and feel** **NEW**: 86 root design tokens in `main.css` (`--fs-xs` … `--fs-2xl` font scale, spacing, radii, shadows, 17-layer z-index scale, motion) with 67 remote overrides; unified welcome page with the product logo (suggestion chips removed); themed tips popup that reopens once per device after every update; light theme default on the remote app.
- **Extension host**: `DependencyInstaller.ts` installs uv 0.12.19 and Git (MinGit 2.55.0.5 on Windows) while `install.sh` installs Node; PyPI update check at most every 6 h with Update / Update when idle / Snooze; automatic window reload after `install.sh` finishes **NEW**; VSIX excludes `**/*.kiss-rescued-*` siblings **NEW**.

## 16. Browser, web research and sign-in hand-off

- **Agent browser**: Patchright (Chromium) driven by `WebUseTool`; accessibility-tree page content with `[N]` ids, screenshots, typing, scrolling; persistent cookies and logins across tasks; `show_browser()` puts the live page in the KISS Browser tab on every surface so the user can solve CAPTCHAs or log in while the agent keeps driving it **NEW**.
- **Web research protocol** (SYSTEM.md): at least 10 distinct sites per research session via `go_to_url` (curl/wget do not count), a running `./tmp/information-<id>.md` with a visited counter, exceptions for real-time lookups (1 authoritative site); when Google blocks, open the search in the visible browser and ask the user for the bot check.
- **Sign-in hand-off** **NEW**: `browser_handoff.open_for_user(url)` tries the streamed tab first, then the machine's default browser, else reports the URL; `portal_handoff` is what channel auth tools call, so OAuth consent, device-code, QR and portal pages appear in the tab without asking the user to open a link.
- 9,961 `go_to_url`, 655 `screenshot`, 525 `click`, 190 `get_page_content` and 48 `show_browser` calls are recorded in the task database.

## 17. Voice

- In-page wake-word listener (`voice.js` + `vosk.js`) and a host-side listener (`voiceWake.ts`, `voice_wake.py`); the wake phrase is **"Hey Sorcar"** (changed from "Sorcar" on 09-26); sensitivity 0-100 in Settings; spoken text is either submitted at once or inserted at the cursor ("Auto submit spoken task"); an acknowledgement clip (`working-on-it.mp3`) plays when a spoken task starts.
- Spoken input arrives as `Speaker #n:` text; the agent answers with `talk(language, text, emotion)` (emotions such as calm, cheerful, curious, serious, warm), played on every device with the task's tab open (`talk_player.py`); TTS spend folds into the task.
- 116 top-level voice tasks and 265 `talk` calls recorded.

## 18. Messaging and third-party agents

`src/kiss/agents/third_party_agents/` holds **44 agents**, one folder each (`<name>/<name>_sea.py`), plus `muse_auth/`, `auth_status.py`, `govee.py` (Govee lights CLI, not one of the 44) and shared helpers (`_channel_agent_utils.py` 2,040 lines, `_channel_cli.py`, `_composio_google.py`, `_device_auth.py`, `_oauth_apps.py`, `_overleaf_realtime.py`).

- **32 messaging and device channels**: bluebubbles, dingtalk, discord, email, feishu, gmail, googlechat, homeassistant, imessage, irc, line, matrix, mattermost, msteams, nextcloud, nostr, ntfy, phone, qq, signal, simplex, slack, sms, synology, telegram, tlon, twitch, webhook, wecom, weixin, whatsapp, zalo.
- **10 service APIs**: brave, firecrawl, gcal, gdocs, gdrive, github, gsheets, notion, **overleaf** **NEW** (48 tools over the user's own browser session cookie, CSRF token, socket.io file tree and Git bridge), postgres.
- **2 infrastructure surfaces**: `kiss-a2a` (Agent-to-Agent protocol server) and `kiss-oai` (OpenAI-compatible HTTP server); both excluded from channel dispatch.
- **Dispatch**: `run_agent(agent="<channel>", task, workspace)` from any task; channel and cron sub-tasks never use a worktree or auto-commit; a channel session gets its own tab on every surface.
- **Gateways and scheduled delivery**: a channel that defines `_make_backend()` (25 messaging channels; not gmail, homeassistant, imessage, nostr, tlon, twitch, wecom) supports one-poll ticks `kiss-<channel> --channel=<chat> [--allow-users] [--pairing] [--quiet]`, pairing admin `--approve CODE` / `--list-pending`, thread continuity, a delivery ledger, per-channel model and budget overrides, and cron delivery targets such as `telegram:123`, `slack:eng`, `ntfy`. An always-on gateway is that tick scheduled as a cron command job (a Slack gateway ticks every minute on the development machine). `[SILENT]` / `NO_REPLY` suppress delivery.
- **Sign-in** **NEW**: the Apps subpanel lists every channel's auth state (`auth_status.py`, 20 s probes, 8 workers) and a click starts a "Connect my <app>" task; public OAuth apps use device code (GitHub, Teams), PKCE on `http://localhost:53682/callback` (Slack, Discord), Nextcloud Login Flow v2, Matrix device grant, `signal-cli link` QR; the six Google agents go through Composio Connect Links and `tools.proxy` so no Google token reaches the process (project keys `ak_…`; consumer `ck_…` keys are rejected); Gmail authenticates through Composio (scopes come from its auth configuration) and exposes 13 tools with no permanent delete; remote MCP servers sign in through `connect_mcp_server` (CIMD or dynamic client registration, loopback port 53683, 600 s).
- **Muse auth**: a credential-isolation daemon holding real tokens in a vault and handing surrogate tokens to 18 connectors (17 SEAs plus Govee) under a host allow-list and read/write policy; CLI verbs `status, import, grant, revoke, audit, clear, daemon, stop, export`; daemon spawn serialised behind `spawn.lock` **NEW**; Microsoft Teams requires it; `KISS_MUSE_AUTH=0` opts out.
- Recorded use: `slack_sea` 33 runs, `gmail_sea` 16, `github_sea` 5, `whatsapp_sea` 3, `gcal_sea` 2 in the `sea` column; 109 `list_messages`, 37 `check_slack_auth`, 36 `post_message`, 22 `check_gmail_auth` tool calls.

## 19. Scheduled automations

`cron_agent.py` (1,813 lines) is an agent script dispatched with `run_agent(agent="cron", task=...)`; only that session holds the `cron_job` tool, so the chat agent never edits schedules directly.

- **Store**: `~/.kiss/cron/jobs.json` with atomic writes and an `flock`; scratch dirs `~/.kiss/cron/runs/<id>-<rand>`; outputs in `~/.kiss/cron/output`.
- **Scheduler**: a thread inside the blocking `kiss-web` lifecycle ticks about every 60 s, reschedules due jobs before running them and runs them concurrently; `kiss-cron --tick` / `--daemon` run it standalone. Schedules: `every 30m`, 5-field cron, one-shot durations, ISO timestamps; cron expressions and offset-less ISO timestamps are evaluated in **America/Los_Angeles** (an explicit offset is kept) **NEW**.
- **Job kinds**: *prompt* jobs write a per-run SEA file (prompt, model, budget, work dir, worktree and auto-commit pinned, classifier off) and launch it through `run_agent`; *command* jobs run a shell command with no LLM and deliver stdout verbatim; channel gateways are command jobs.
- **Tool actions**: `create` (refuses exact duplicates), `ensure` (idempotent by name; resumes a paused job) **NEW**, `list`, `remove`, `pause`, `resume`, `run_now`; `deliver` accepts a comma list of channels, `local` or `none`.
- **Auto-scheduled jobs** **NEW**: picking `autorouter` ensures "Weekly rsi7d: autorouter evidence and prompt (Sat 1am PT)" (`0 1 * * 6`, claude-fable-5-1, $25 + $5 relay, 2 h nested run inside a 2 h 10 min job); `/git_extract_knowledge` ensures "git-knowledge daily update: <repo>" at 04:00 PT.
- The Schedule subpanel lists jobs with next-run times and copy buttons. On the development machine 12 jobs are configured (6 prompt, 6 command; 10 enabled): daily test fix, README update, cost audit, model update, knowledge refresh, two weekly rsi7d jobs, a Slack gateway every minute, a 4-hourly sync of `sorcar.db` and memories to a shared machine, a nightly work index, and two paused reinstall jobs. 180 top-level tasks mention cron; 109 `cron_job` calls are recorded.

## 20. Installation, deployment, Docker and release

- **`install.sh`** (73,902 bytes): re-executes under `setsid`, takes a cross-process `flock` (dead holders no longer block retries **NEW**), five steps (git, Node **v22.23.3** **NEW pin**, VS Code CLI, extension build, extension install), installs the `rsorcar` and `sorcar-docker` launchers, applies the `.brand/` overlay **NEW**, and finally waits up to 900 s for the daemon and opens the web app (`kiss-web --trust-ca`, tunnel URL on a remote) **NEW**; `KISS_SKIP_LAUNCH`, `KISS_NONINTERACTIVE`, `KISS_CODE_CLI` control it. The Python package is `kiss-agent-framework` on PyPI; the extension is `ksenxx.kiss-sorcar` on the Marketplace.
- **`rsorcar user@host`** (57,275 bytes), ten steps: ssh check and prerequisites, refuse to restart a remote with a running task unless `SORCAR_FORCE_RESTART=1`, disk headroom check, copy `~/.ssh` (never `authorized_keys`), two-way repository sync through `origin` (never force-push, never a `kiss/wt-*` branch), API keys only to a remote that has none with a probe that fails closed **NEW**, two-way `sorcar.db` merge, **memory sync with tombstones and clock-skew tolerance** **NEW**, sync of `AUTOROUTER.md` and `MODEL_DECISIONS.md` **NEW**, remote `install.sh` (code-server when no `code` CLI), `kiss-web` as a lingering systemd user service, Cloudflare tunnel with ntfy notification, GitHub credentials over stdin, public-URL verification, local `install.sh`. Thirteen `SORCAR_*` environment overrides.
- **Docker**: `sorcar-docker [PORT] [--rebuild]` builds `codercom/code-server` with uv **0.12.19** **NEW pin**, clones the repository, runs `install.sh` and serves code-server on 8080 with the extension preinstalled, forwarding `GH_TOKEN` and the API keys. Inside tasks, `docker_image=` runs the shell and file tools in a container with the work dir mounted at its host path **NEW**.
- **Release** (`scripts/release.sh`, 13 steps): purge private paths from public history, bump the version in `_version.py`, README, SYSTEM.md and `package.json`, build the VSIX into the release commit, push filtered history and tag to `ksenxx/kiss_ai`, GitHub release, PyPI publish (100 MiB per-file check before upload; sdist restricted to `src/kiss` and `projects/swedefend` **NEW**), Marketplace publish, local reinstall.
- **Windows**: MinGit 2.55.0.5 and uv installed by the extension; the 2026-09-24 Windows test run fixed cross-platform bugs in memoryfield, `sorcar_md`, and channel tests.

## 21. Developer tooling and tests

- `uv run check --full`: `uv sync`, `generate-api-docs`, `compileall`, `ruff`, `mypy`, then `pyright` and, when the extension's `node_modules` and npm are present, `npm run typecheck` and `npm run lint` (eslint, stylelint, htmlhint); failed stages are listed at the end with their output. `--no-clean` and `--clean-only` flags; the extension version is synced first.
- **Tests**: 1,360 `test_*.py` files with 10,682 test functions (382,938 lines across the 1,405 Python files under `src/kiss/tests/`), 369 JavaScript test files under `src/kiss/agents/vscode/test/` (`node test/run-all.js`); the suite disables the worktree pool and the classifier, raises the open-file soft limit to 4,096, and pins `pytest>=9.0.3,<9.1`.
- **Subprocess reaper** **NEW** (`kiss.tests.subprocess_reaper`): wraps `Popen` so every child is attributed to the running test, fixture or session and swept with SIGTERM, 5 s, SIGKILL, emitting `LeakedSubprocessWarning` with the command; written after a run left 529 detached `muse_auth.daemon` processes.
- **Testing policy** (SYSTEM.md): end-to-end tests only, no mocks or structural tests, 100% branch coverage of new code where reachable, test splits run in parallel with `run_commands_parallel` on `cores - 2` workers, a load-dependent flake is named rather than chased.
- **Repository hygiene**: worktree tasks get `node_modules` symlinks and must compile the extension once; the Bash tool refuses commands naming the main checkout path from inside a worktree; `tmp/` for scratch files; `reports/` for deliverables.

## 22. What the trajectories show

Figures from `~/.kiss/sorcar.db` on the development machine, 2026-09-30 (first row 2026-04, plus two rows with a 2009 clock artefact).

| Measure | Value |
| --- | --- |
| Task rows | 27,673 (6,457 top-level, 21,216 sub-agents, 1,646 side channels) |
| Chats | 3,487 (2,743 with a stored summary) |
| Persisted events | 10,625,419 (14,433 `llm_call` events since 09-27) |
| Tool calls | 572,526 |
| Spend on top-level tasks | $40,287 · 28.5 billion tokens · 314,908 steps |
| Sum over every row (double counts folded sub-agent spend) | $71,085 · 49.4 billion tokens · 438,413 steps |
| Tasks over one hour / over 1,000 steps / over $100 | 2,300 / 32 / 55 |
| Longest task | 8,056 steps, $762 ("Find all race conditions, deadlocks, obvious bugs, missing wiring …") |
| Distinct top-level working directories / named models | 125 (430 across all rows) / 52 (64 rows record no model) |
| Worktree rows / auto-commit top-level tasks / parallel top-level tasks | 3,197 / 3,966 / 4,461 |
| Voice tasks / slash-command tasks | 116 / 64 (`/review_paper` 19, `/write_paper` 18, `/sh` 6, `/remember` 6, `/revise_and_review_paper` 4) |
| Since 2026-09-21 | 3,455 rows, 735 top-level, 2,720 sub-agents |

Figure 2. Tool calls by name (571,303 of the 572,526 recorded calls; the remaining 1,223 belong to smaller tools such as `type_text` 141, `press_key` 133, `scroll` 119 and `list_messages` 109).

| Tool | Calls | Group |
| --- | --- | --- |
| `Bash` | 273,786 | shell |
| `Read` | 157,211 | files |
| `Edit` | 46,493 | files |
| `summary` | 27,046 | protocol |
| `finish` | 20,797 | protocol |
| `go_to_url` | 9,961 | browser |
| `Write` | 9,694 | files |
| `run_parallel` | 4,498 | dispatch |
| `set_model` | 4,212 | protocol |
| `memory_search` | 2,869 | memory |
| `task_transcript` | 2,032 | SEA (`/ask`, `/task_update`) |
| `run_commands_parallel` | 1,836 | shell |
| `memory_pull` | 1,651 | memory |
| `code_graph` | 1,524 | retired tool |
| `memory_write` | 1,441 | memory |
| `memory_read` | 839 | memory |
| `number_of_cores` | 750 | dispatch |
| `bash_job` | 676 | shell |
| `screenshot` | 655 | browser |
| `click` | 525 | browser |
| `decide` | 505 | protocol |
| `run_agent` | 463 | dispatch |
| `ask_user_question` | 446 | interaction |
| `talk` | 265 | interaction |
| `close_browser` | 248 | browser |
| `get_page_content` | 190 | browser |
| `build_paper` / `read_paper` / `check_review` / `check_paper` | 173 / 162 / 127 / 71 | paper SEAs |
| `cron_job` | 109 | cron |
| `show_browser` | 48 | browser |

Figure 3. Top-level tasks and spend per month.

| Month | Tasks | Spend |
| --- | --- | --- |
| 2026-04 | 513 | $414 |
| 2026-05 | 1,807 | $1,506 |
| 2026-06 | 888 | $2,451 |
| 2026-07 | 883 | $6,885 |
| 2026-08 | 843 | $10,382 |
| 2026-09 (to the 30th) | 1,521 | $18,650 |

Models on top-level tasks: `claude-fable-5` 1,972, `claude-opus-4-7` 1,829, `claude-opus-4-6` 1,044, `claude-fable-5-1` 779, `claude-opus-4-8` 231, `gpt-5.6-sol` 125, `gpt-5.5` 105, `claude-opus-5-5` 90, `claude-opus-5` 63, `cc/opus` 37. Across all rows `gpt-5.6-sol` leads with 8,079 (plus 2,972 `-xhigh`); 10,902 of those 11,051 rows are sub-agents and 8,878 of them carry the `review` tag, the read-only reviewers dispatched by `run_parallel`; then `claude-fable-5` 7,353 and `claude-fable-5-1` 3,210; `gpt-6-astra` has 333 rows and `gpt-6-astra-high` 35.

Kinds of work by keyword in top-level task text: test 1,408, fix 1,340, bug 1,244, review 1,177, git 598, report 495, paper 471, implement 227, cron 180, slack 120, benchmark 107, optimiz* 80, audit 61, gmail 60, research 60, blog 51, rsorcar 44, refactor 43, docker 42, slides 24, deploy 14, whatsapp 11, latex 10, telegram 6.

Tags written since 09-27 (27,664 of the 27,673 rows carry tags): the most frequent combinations are `work` alone (1,865), `work,coding,testing,debugging,subagent` (1,860), `work,chore,question,subagent` (1,505) and `work,testing,debugging,subagent` (1,494); 886 rows carry `work,review,subagent,failed`.

## 23. Caveats, corrections and stale documentation

- **Spend figure corrected.** Earlier inventories reported "$53.9K across 36.3 billion tokens" by summing every `task_history` row. Because sub-agent spend is folded into the parent's row by the usage ledger, that sum double counts; the top-level-only total is $40.3K (28.5 billion tokens) and the all-row sum today is $71.1K. Both are given in section 22.
- **Worktree rows.** 3,197 rows carry `is_worktree = 1` (the 2026-09-21 inventory said 2,767; the earlier "22,023 of 24,063" claim was wrong).
- **`knowledge/` residue.** Commit `12ec468f2` seeded a tracked `knowledge/` directory and `c2964bf1b` moved it to `~/.kiss/memories/kiss/` the same day, but 10 files (8 research notes and two `.sqlite3` indexes) committed by parallel tasks remain tracked; no code reads them.
- **Docs behind the code.** `website/.../docs/sea-commands.md` names 9 of the 17 bundled SEAs.
- **Gmail scope.** The `gmail.modify` narrowing (`628e056ae`) preceded the move to the Composio proxy the same day, so no scope constant remains in the tree; what persists is the 13-tool surface without permanent delete.
- **Per-call cost is not a table.** `llm_call` events live in `events`, not in a dedicated table; the autorouter's cost tools and the task panel read them from there.
- **`summary` cadence is prompt-enforced only**; the tool itself is a no-op.
- **Docker runs** have no `bash_job` and no memory tools; browser tools stay available when web tools are on.
- **Counts are from one machine.** Section 22 describes the development machine's database; the two rows dated 2009 come from a clock artefact and are excluded from the per-month table.
