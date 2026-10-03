# KISS Sorcar Feature Inventory

Version 2026.9.27 · HEAD `5eb6496fd` (4,926 commits) · compiled 2026-09-30 · supersedes the 2026-09-21 inventory

Every count and feature in this document was checked against the source tree at `5eb6496fd`, the packaged assets under `src/kiss/`, and the task database `~/.kiss/history.db` on 2026-09-30. Items marked **NEW** did not exist at the previous inventory's commit `85d9f3cda` (2026-09-22); the 360 commits between the two are summarised in section 2. Where the code and a document disagree, the code wins and the disagreement is listed in section 23.

## Contents

1. [At a glance](#1-at-a-glance)
2. [What changed since 2026-09-21](#2-what-changed-since-2026-09-21)
3. [AI discovery and optimisation on real software](#3-ai-discovery-and-optimisation-on-real-software)
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

## 3. AI discovery and optimisation on real software

**The drivers.** Three blocks of SYSTEM.md turn a one-paragraph task into a long unattended run: the 7-step AI discovery loop (baseline → SOTA search → `tmp/ideas.md` → pairwise judging → implement and measure → `tmp/explored-ideas.md`, where a losing idea is marked failed and never retried → repeat until the metric is met, then a held-out check), adversarial testing (one sub-task writes breaking inputs or workloads, another repairs) and adversarial training. The shipped prompts ("Sorcar for AI Discovery", "Sorcar for Optimization", the GEPA optimiser) add a numeric stopping condition ("do not STOP until ...") and a report. None of this is code: the model runs the loop with `Bash`, the browser, `run_parallel` and `Write`. What follows is what happened when those blocks were pointed at real software, taken from the LaTeX in `papers/`, the project logs in `projects/` and the six posts on kisssorcar.github.io (live pages checked 09-30; the checked-in copies under `website/` carry the same text). Every run below is a single run with no control arm, and the task prompts were written by the framework's author, so these are records, not experiments.

| Target | Baseline → result | Held-out and correctness gates | Agent effort |
| --- | --- | --- | --- |
| **HydraKV**, a larger-than-memory key-value store written from scratch (~4,470 lines of dependency-free C++17) for YCSB-A: 250 M keys, 100-byte values, Zipf 0.95, 8 GiB memory cap, 64 vCPUs, eight NVMe | Starting engine 1.85 Mops/s → **5.50 Mops/s**, 5.9× Microsoft FASTER's 0.93 on the same harness; ~89% of the 6.2 Mops/s miss-bandwidth bound at the observed 77% cache hit rate | 250,000,000/250,000,000 keys validated; adversarial workload variants 5.67 (sparse keys), 5.56 (clustered), 3.97 (drifting hot set); held-out 60/25/15 mix generated after the loop closed **3.87** (29% below the scored trace); 28 fault-injecting e2e tests, ASan/UBSan/TSan clean, 93% line coverage; three independent audits, each finding bugs the previous round introduced or missed | 8 tasks, 22.6 h, $516, 1,959 steps, 7 continuations, 11 review sub-agents; seven task prompts and a few steering messages from the human |
| **Bespoke OLAP TPC-H engine** (VLDB'26 artifact, single-threaded, 22 templates), 32-core Linux, SF10 | 3,132.2 ms → **90.8 ms (34.5×)** summed over 22 queries; 10 queries under 0.1 ms; SF1 276.8 → 14.2 ms (19.5×); ingest 52.5 → 63.0 s, repaid after 4 suite runs; 3.1× ahead of the 284 ms the authors' own multithreaded follow-up reports (a README figure without a run log) | Seed 999 excluded from development: **85.1 ms**, tracking seed 42 query by query; DuckDB as differential oracle on six (SF, seed) configurations; 265-case guard/fallback audit against the pristine artifact; read-only reviewer found an OpenMP race in Q20 and an `int64_t` accumulator in Q1 (baseline uses `__int128`, overflow near SF 8,300), neither visible to the oracle | 3 tasks, 2.4 h, $90, 475 steps; development 1.3 h, 162 steps, $43, zero steering |
| **LZ4** v1.10.0 CLI, multithreaded file mode, 16 cores / 32 threads, 1.7 GB corpus in tmpfs | Level -1 at 32 threads 1,519 → **3,810 MB/s (2.51×)**, 2.57× at 64, 1.88× at 8; levels -3/-6/-9 1.09–1.27×; scaling 3.6× → 9.1×; faster than pzstd (2,788–2,888) and pigz (1,314) at 32 threads | Compressed output byte-identical to stock at every level; 712-case oracle matrix; stock binary decodes patched output; breaker campaign found 2 bugs (a >300 s hang on a 1-byte file, a SIGBUS on mid-run truncation); reviewer found 4 more | ~7.5 h, under $600 |
| **DuckDB** development tree (32-core VM), five suites | Geometric means: TPC-H 1.152×, TPC-DS 1.184×, IMDB/JOB 1.215×, h2oai 1.234×, ClickBench 1.237×; largest single query ClickBench q28 ~1.73× | IMDB/JOB, h2oai and ClickBench held out of all PGO training; interleaved A/B because run-to-run drift reached 30%; 6,377-test suite with zero attributable failures; ~15,500 differential hardening statements plus 49,000 fuzz statements byte-identical to stock; reviewer found 5 issues including a NULL-handling regression | not stated |
| **SQLite** 3.54.0 trunk (32-core GCP VM), four benchmarks | speedtest1 2.06×, TATP 1.90×, SSB 1.30×, kvtest 1.25×; **geomean 1.59×** (1.54× at `synchronous=FULL`) | `testrunner.tcl full`: 0 errors in 1,032,940 tests; checksum-verified benchmark harness; 37-script differential corpus under ASan/UBSan; breaker found 2 bugs, a second hardening pass 1; an independent rebuild reproduced **1.50–1.55×** and documented the 2.5% shortfall | under 8 h, under $150 |
| **Biomni × TusoAI biology benchmarks** (reconstructed: three perturbation-response tasks, one enhancer–gene task) | Naive baseline 78.71–92.20 → a single 157-line method scoring **99.66–99.71** on every sealed test set; no reference method (kNN, random forest, gradient boosting on raw features) clears 99 on all four | Sealed train/val/test splits, a regenerated dataset (99.70–99.72), shuffled-label collapse (R² → 0, AUC → 50), process isolation with labels only in the parent; 27 adversarial and security tests; two gate bugs found (a `SystemExit(0)` that passed scoring, `PYTHONHASHSEED`-dependent data) | one autonomous session |
| **Collective communication above AWS Trainium's closed library** (SorcarCCL vs OverlayCCL's strategy enumeration), 224 NeuronCores | 40 production all-reduce / all-gather / reduce-scatter problems in 8 families: **40/40 confirmed on hardware at 1.12–4.37×** (median 1.46×, geomean 1.86×), enumerator wins 0; e.g. an MoE routing renormaliser collapsed from 8 all-reduces to 2 (4.06×), a per-block normalisation loop vectorised to `view(B,S)` + `mean(dim=1)` (4.31×); the earlier 143-problem build: 55 anchors, 88 ties, 45 confirmed at 1.06–3.38×, step time **2.46×** (9.75B Llama-style) and **2.15×** (9.70B GPT-3-class) faster | Rank-asymmetric correctness oracle at world sizes 4 and 8 plus bf16; nine simulator seeds with Mann-Whitney and bootstrap bounds, then one warm-cache hardware pair per problem; no regression on OverlayCCL's own 8 problems | 30 steps and $5 of Claude Sonnet 4.5 per problem; agent $5.37 vs enumerator $7.77 over 24 problems |

**How the runs went.** The loop's value shows in the ideas it measured and threw away as much as in the ones it kept. HydraKV's first engine (a flat 4-byte-per-key array over 128-byte slots) worked only because the published trace's keys fit in 32 bits; the first adversarial variant (sparse SplitMix64 keys) OOM-killed it in minutes and forced a fingerprint hash index that scored the same 1.87 Mops/s. Profiling showed a 72% cache hit rate and one contended I/O queue; two rebuilds regressed to 1.10 and 0.47 before an `fio` measurement of the device (718k random 4 KiB reads/s) led to eight raw `io_uring` rings and an asynchronous read-miss state machine (5.27) and SLRU segmentation (5.50). Six cache designs were measured and reverted: strict probation 0.78, two-way probation 0.95, page-neighbour co-admission 1.98, admitting load-phase-clean keys 4.17, cleaners flushing any dirty entry 0.47, heat-clustered relocation 4.03–4.34. The drifting-hot-set variant stayed at 3.97 because "no engine change we tried recovered the difference without hurting the stationary workloads". The 10 Mops/s goal set mid-run and the 7.0 stretch goal were never met: at 50% reads the device bounds this design class near 6.2 Mops/s unless the hit rate reaches 86%, and every admission policy tried saturated at 76.8–77.1%. Off the scored mix the engine is far slower (YCSB-B 2.93, YCSB-C 2.78, write-only 5.13, read-modify-write 0.30, delete-heavy time series 0.08).

On the TPC-H engine the human supplied a researched plan (5,979 characters: remove the `AffinityGuard`, OpenMP first, precomputed join artifacts second), so the agent ran the implement-and-measure step without the search step; its own contributions were the per-query choice of artifact (date-indexed prefix cubes for Q1, Q4–Q8, Q12, Q14, Q15; CSR join indexes for Q9, Q17, Q20; per-order aggregates for Q10, Q18; pre-filtered rows for Q13, Q16, Q19, Q21, Q22), a guarded fallback to the baseline algorithm on every fast path, the profile-driven removal of a serial 60-million-row scan in Q19, and the 265-case audit. Q2 was left alone because its parameters admit no small prefix structure and is the one query slower than baseline (by 0.3 ms); Q10 is 43% of the final total. In the three blog projects the rejected ideas are named with their numbers: LZ4's `--no-frame-crc` reached 6.4–8.1 GB/s and was rejected as cheating because it changes the output bytes (sequential XXH32 caps the honest path near 3.7 GB/s); DuckDB's row-based aggregate `Combine` rewrite won individual queries (TPC-DS q32 1.088×, q18 1.142×) but moved suite geomeans only 1.006–1.011×, under the pre-registered 1.05× keep bar, and was kept as a patch file; gcc `-march=native` alone regressed TPC-DS and IMDB and lost to clang-18 on every suite. Two of the biggest LZ4 gains were build fixes rather than algorithm changes: PGO had silently never applied (the object cache was keyed on flags, leaving an LTO-only "PGO" binary) and training on level 1 alone cost the HC levels 5–6% until retraining on levels 1, 3 and 9 turned that into +7%.

**What the developer model's own tests missed.** In every project the defects that survived a green test suite were found by a different mechanism: a read-only reviewer from a second vendor (`gpt-5.6-sol`, capped at 20% of the budget), a breaker sub-agent writing adversarial inputs, or an external audit with its own oracle. HydraKV's second audit reproduced four bugs on hardware: an upsert/delete deadlock, an OOM on a full disk, a recovery that dropped 480K of 976K keys and reported success, and deletes 160× slower than upserts (244 µs vs 1.5 µs). The third audit showed that the slow-delete fix had made deletes non-durable: 5,000 of 5,000 deleted keys came back after a kill in the minimal repro. The repair was a delete-intent log written before acknowledgement (final `Delete` 1.0 µs); the reviewer's second round on that repair found a linearizability race between the registry mark and the cache sweep that no fault had triggered. The multi-workload task found that values over 101 bytes had no disk path at all (100% stale reads at 1 KB values) and that clean shutdown lost data on every workload, a bug present in the engine long before the campaign. On the TPC-H engine the DuckDB oracle passed while Q20 had a data race (both racing stores wrote 1) and Q1 accumulated in a narrowed type. LZ4's reviewer found silent corruption when stdin was at a non-zero offset; DuckDB's found a NULL-handling change from a hoisted rewrite validation; SQLite's found two opcodes on the fallback dispatch and a harness that printed checksums without enforcing them. Each of the three blog projects' breaker campaigns found one or two real bugs (LZ4 2, SQLite 2, Tuso 2, DuckDB none). The KISS Sorcar paper's summary of the list ("a delete-durability regression, an aliasing fingerprint, an OpenMP race, a narrowed accumulator") is also its argument for the two-vendor review rule in the system prompt.

**Where the agent went wrong.** Two failure modes are recorded and both changed the prompt. In HydraKV's bug-hunting task the fixes were layered onto a rejected slower branch and the engine sat at 3.6 Mops/s until the human asked "Did you start with the most performant variant of the engine?"; bisection and a rebase restored 5.51. In the same project the agent once retried an idea its own log had marked failed, which is why the discovery block now says "mark it as failed so it is never retried". The agent's phrase "production-hardened" is contradicted by its own record (three audits, each finding new bugs), and the literal demands of 100% coverage and "all bugs" were not met (coverage stopped at 92.84–93.3%). The Q1 accumulator was a violation of a stated hard constraint. On the biology benchmark the feature engineering was informed by the documented data generator, which the post says outright. The sqlite-optimized verification found that "full test suite green" held only for the source changes under default build options: the changed runtime defaults fail 26 upstream tests that assert the old defaults, and WAL-by-default aborts `attach2` and `autovacuum`.

**The agent as harness.** The same coding rules, cut to 727 words, were run on the 30 Terminal-Bench 2.0 tasks and seven models of the HarnessTax study: 630 attempts, **75.6%** of attempts solved against Pi 70.0%, Codex CLI 65.7% and Claude Code 65.1%, the best point estimate on all seven models (74.1% if solves past 100 turns are counted as failures). A paired rerun of Pi 0.87.1 on Claude Fable 5 gave +11.1 points (95% interval +3.3 to +20.0) at five cents more per attempt; the same frozen prompt scored 79.1% on the 59 tasks the study did not sample, and a second paired Pi rerun on 57 of them gave +10.5 points (+2.9 to +18.7) at 63 cents more per attempt. Discovery, adversarial testing, memory and the reviewer were all switched off for these runs, so they measure the loop, six tools and the rules, not the procedures above.

**Research artifacts built the same way.** Two July papers were built by Sorcar tasks (SWEDefend's under a prompt ending "Do AI discovery to get better results"): SWEDefend, a defence against SWExploit-style backdoored program-repair patches, whose combined pipeline at confidence threshold 0.9 catches 97.96% of 49 malicious cases with 0 of 100 benign vetoes where a naive fail-closed judge sat at ~20% false positives, and which reports its own defeat by an adaptive attacker (3 of 3 seeds by the third iteration); and Cleverest+, a fixed-budget (4,3,3) three-model portfolio with sanitizer-signature oracles that solved 6 of 6 issues on a three-subject mini-benchmark in 10 trials each, with no run yet on the 72-commit benchmark it targets.

**The loop turned on the agent itself.** A speed audit of one week of the task database (2,786 tasks) measured where 110 hours of user waiting went: 38% in the agent's own LLM round trips (a trivial step takes 4.4 s at under 25k tokens of context and 9.8 s above 300k), 33% waiting on `run_parallel`, 25% in reviewer loops that never converged (0 of 135 rounds in tasks with three or more rounds came back clean; the same "run all tests" prompt took 0.2 h with 27 parallel splits and 7.9 h with a 12-round review). The cost-lever work that followed removed the mandatory `AGENTS.md` read (306 → 0 per day), shell-wrapper sub-agents (258 → 0) and reviewer overspend (7 trees → 0), reached a 0.970 prompt-cache hit ratio, and then found that its own context compaction was cache-hostile (57 of 57 compactions missed the cache, net +$130), which produced the cache-aware gate and the keep-alive ping (replay: input bill 0.66× of production). Six months of the agent developing its own repository (section 22) gave the controlled result behind the rules: the full prompt versus a lite one changed how the agent works (tests written in 68 of 71 cells vs 12) but not the hidden-test pass rate (+0.9 points, interval −2.8 to +4.1), while a second-vendor reviewer was the one intervention with a measured gain (+4.7 points, +1.2 to +9.4, at 2.5× the cost); in the ten-week two-vendor deployment the reviewer made 3 writes in 2,164 calls, all to `tmp/`, took 9.3% of task cost and reported roughly 1,500–2,000 findings, 53% of the coded sample functional bugs.

**Provenance.** Every headline figure above is a `\newcommand` macro recomputed by a script beside the paper (`kisssorcar/ks_numbers.py` and `ks_tb2.py`, `sesorcar/se_numbers.py`, the Collective Algebra `compute_numbers.py`, 326 macros in the main paper alone) from checked-in evidence (`kisssorcar/evidence/`, the 4,575-file `kisssorcar/ablation/`, `projects/kv_adversarial/DISCOVERY_LOG.md`, `projects/bespoke_tpch_x4/results/`, the 858-file `sorcarccl-production-set-main/`). Where sources disagree the numbers above follow the latest paper: the HydraKV process record is 8 tasks / 22.6 h / $516 in `kiss_sorcar.tex` but 6 / 17.2 h / $352 in `ks_assistant.tex` (which also attributes the starting engine to Claude Code rather than a Sorcar task); the held-out drop is "29%", "more than a quarter" and "a third" in three places; the Hacker News draft still says six prompts, ~3,970 lines and 5.51 Mops/s; the Collective Algebra paper exists twice with 55 vs 40 anchors and the end-to-end step-time result only in the older build; the Cleverest+ text still calls its campaign "not yet wired" in three paragraphs that its results section contradicts. The `/write_paper` SEA's `check_paper` gate (AI-slop and consistency checks) and `build_paper` (pdflatex + bibtex) run on every revision (section 9; 41 of the 64 slash-command tasks in the database are paper tasks).

**Website** (`website/kisssorcar.github.io/`): 11 docs pages, the 6 blog pages above, the paper PDFs, `privacy.html` **NEW**, `llms.txt` / `llms-full.txt` (still listing six paper PDFs and omitting `fable_sol`, `ks_assistant`, `sesorcar` and Collective Algebra), sitemap. The home page carries the Terminal-Bench 2.0 table and the paired-Pi figures.

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
 │  prompt: SYSTEM.md (SYSTEM_LITE if simple) + ~/.kiss/AGENTS.md + memory protocol   │
 │  cost ledger · context compaction · prompt-cache keep-alive · fallback model       │
 └────────┬───────────────────────────────────────────────────────────────────────────┘
          │
 ┌────────▼───────────────┐ ┌──────────────────────┐ ┌───────────────────────────────┐
 │ Models (706 entries)   │ │ State in ~/.kiss     │ │ Extension points              │
 │ Anthropic/OpenAI/Gemini│ │ history.db · memories/│ │ SEAs (seas/, SEAS.md folders) │
 │ OpenRouter/Together/   │ │ cron/jobs.json ·     │ │ skills (SKILL.md) · MCP       │
 │ Z.ai/Moonshot · cc/    │ │ tabs.json · AGENTS.md│ │ servers · SEA add_to_tools() ·│
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
| Python client API | `from kiss.server import sorcar; sorcar.run(task, ...)` returning `TaskResult(text, success, cost, tokens, steps, chat_id, task_id)` | 25 keyword options including `extension_agent_path` (a SEA whose `add_to_tools()` / `tools()` return the extra tool callables), `tool_profile`, `docker_image`, `timeout=3600`, `stop_on_timeout` |
| Slash commands | `/<name> <text>` at position 0 of a prompt | Registered from three sources (section 9); `/<name> help` prints the SEA's `description()` |
| Channels | `kiss-<channel>` CLIs, `run_agent(agent="slack", ...)`, always-on gateways scheduled by cron | 44 agents (section 18) |
| Voice | In-page wake word "Hey Sorcar" **NEW wording**, host-side Vosk listener, `talk()` playback on every open tab | Section 17 |

Prompt assets shipped in `src/kiss/`:

- `SYSTEM.md` (3,971 words): identity, rule precedence, visibility constraint, tool usage rules, web research protocol (10 sites, 1 for real-time data), code style, pre-flight checks, the 7-step AI discovery loop, adversarial testing and training, deep work, complex task planning, file browsing, testing, pre-finish verification, Sorcar-specific rules. The static part is sent under one Anthropic `cache_control` breakpoint; the per-task "Task Settings" block follows uncached. **NEW**
- `SYSTEM_LITE.md` (753 words): the reduced prompt for tasks the classifier marks as simple; `/ask` ships its own `_ask_system_lite.md` (608 words) plus an answering playbook.
- `~/.kiss/AGENTS.md`: standing user instructions appended to every system prompt, managed by `/remember` and `/forget`. **NEW management**
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
- Appends `~/.kiss/AGENTS.md` to every system prompt, reading it with writer-tolerant IO so a rename in progress cannot abort the task.
- Usage ledger with epochs and seen-maps so sub-agent, classifier, TTS and side-channel spend are counted exactly once, including across stop interrupts; printer offsets are re-based before the failed-session summariser so the UI cost never drops. **NEW**
- Docker mode: `docker_image=` starts a container with the task work dir bind-mounted at its host path and set as `WorkingDir`; `container:<id>` attaches to a running container without mounts; CLI models cannot be combined with Docker. **NEW mount**

### SorcarAgent (`sorcar_agent.py`, 3,511 lines)

- Builds the tool set (section 7), the system prompt (SYSTEM.md + restricted-profile note + `WEB_TOOLS_OFF_NOTE` when browsing is off **NEW** + memory protocol + SEA additions), the memory root and domain memories, and the model (a picker SEA such as `autorouter` resolves to its `model()`).
- `set_model` rebuilds the tool schema for the new model and shows it in the picker; `summary` is a no-op tool whose every-10-steps cadence is enforced by the prompt only.
- Review fan-out guardrails (`ReviewQuota`, `MAX_REVIEW_ROUNDS`, `MIN_SUBAGENT_BUDGET`, `KISS_REVIEW_BUDGET_FRACTION`) were removed on 2026-09-25; a sub-agent's budget share is now `remaining / (num_tasks + 1)`. **CHANGED**

### ChatSorcarAgent and WorktreeSorcarAgent

- `ChatSorcarAgent` persists chats and task chains in `history.db`; prompts carry at most 10 prior tasks, normally the newest two in full and the older ones as 600/300-character task/result digests, shortened further to fit a 6,000-character prefix cap (`chat_history_digest`, default on). It records the SEA name of the run (`task_history.sea`) **NEW** and bare-path prompts are turned into open-file directives **NEW**.
- `WorktreeSorcarAgent` gives a development-classified task its own `git worktree` under `.kiss-worktrees/` on branch `kiss/wt-*` (worktrees off, non-development verdicts, non-git directories and detached HEADs run directly); a spare worktree is prewarmed at daemon start and refilled after each worktree task ends **NEW**; outcomes: committed and removed, preserved (no auto-commit, commit failed, sub-agent active, rescue failed). Section 12 has the merge rules.

### Configuration knobs (`src/kiss/core/config.py`)

`KISS_READ_DEDUPE` (on), `KISS_READ_OUTLINE_LINES` (2000), `KISS_TOOL_OUTPUT_COMPACTION` (on), `KISS_COMPACTION_START_TOKENS` / `KISS_COMPACTION_STEP_TOKENS` (100,000), `KISS_TOOL_OUTPUT_MAX_CHARS` (50,000), `KISS_CONTEXT_LIMIT_FRACTION` (0.7), `KISS_TOOL_PROFILES` (on), `KISS_CHAT_HISTORY_DIGEST` (on), `KISS_DISPATCH_PATH_REWRITE` (on), `KISS_USE_MEMORY` (one-process override), `KISS_HOME` (default `~/.kiss`), `KISS_DISABLE_WORKTREE_POOL`, `KISS_DISABLE_TASK_CLASSIFIER`, `KISS_MUSE_AUTH=0`; seven API-key variables plus a workspace id (`ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`). Per-run settings in `~/.kiss/config.json`: `max_budget`, `auto_commit_mode`, `is_worktree`, `use_web_browser`, `classify_tasks` (on), `classify_with_decisions` (on), `use_memory` (on), `memory_dir` (empty = `~/.kiss/memories`), voice options, remote password, custom models.

## 7. Built-in tools and tool profiles

| Group | Tools | Notes |
| --- | --- | --- |
| Files and shell | `Bash`, `bash_job`, `run_commands_parallel`, `Read`, `Edit`, `Write` | `Bash(background=true)` detaches with `nohup` and returns a job id; `bash_job` tails, waits or kills; `run_commands_parallel` runs shell commands in threads with per-command exit codes; a whole-file `Read` of a file over 2,000 lines returns an outline, and an unchanged range already shown returns a one-line stub (`force=True` re-reads); `Edit` rejects a file not read in the session and `Write` rejects overwriting an unread existing file (new files and scratch files under `tmp/` are exempt). In Docker the file tools execute inside the container without this guard and `bash_job` is absent. |
| Browser | `go_to_url`, `click`, `type_text`, `press_key`, `scroll`, `screenshot`, `get_page_content`, `show_browser`, `close_browser` | Patchright/Chromium; `show_browser` streams the page into the KISS Browser tab on every surface **NEW**; sub-agents get an ephemeral profile; only in the `full` and `review` profiles with "Use web tools" on |
| Memory | `memory_search`, `memory_pull`, `memory_read`, `memory_write`, `memory_list`, `memory_delete`, `memory_refresh` | Section 13; `memory=` narrows to the general or a domain memory **NEW** |
| Dispatch | `run_agent`, `run_parallel`, `number_of_cores` | Section 10; `run_parallel` and `number_of_cores` only in parallel mode |
| MCP | tools of configured MCP servers, `connect_mcp_server`, `finish_mcp_server_connect` **NEW** | OAuth sign-in for remote MCP servers (Notion, Linear, Asana, Zoom or any URL); tokens in `~/.kiss/mcp_auth/<server>.json` |
| Skills | `skill` | Present when the work dir or `~/.kiss/skills`, `.kiss/skills`, `.agents/skills` or Claude skill directories hold a `SKILL.md` |
| Interaction | `ask_user_question`, `talk(language, text, emotion)`, `set_model`, `decide`, `summary`, `finish` | `decide` only when the Jev decisions model is enabled (section 11); `finish` carries `summary_in_html`, `is_continue`, `suggested_next_task` |

Tool profiles (`TOOL_PROFILES`, `sorcar_agent.py:71-92`):

| Profile | Tools kept | Used by |
| --- | --- | --- |
| `full` | everything above | default |
| `review` | `Bash`, `bash_job`, `Read`, `run_commands_parallel`, `memory_search`, `memory_pull`, `memory_read`, `memory_list`, `decide`, `summary`, `talk`, the browser tools | reviewer children of `run_parallel`, `/ask` |
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
| `/forget` | Remove a standing instruction from `~/.kiss/AGENTS.md` | `bash`, $1 | **NEW** |
| `/git_extract_knowledge` | Index every tracked file and commit of a repository into its domain memory and an FTS5 block store; `update`, `ask` modes; daily 04:00 PT refresh job | `full`, parallel, no memory tools | **NEW** |
| `/merge` | Resolve git merge conflicts and stage the result; also run automatically by auto-commit merges | $5, no worktree | |
| `/remember` | Append a standing instruction to `~/.kiss/AGENTS.md` | `bash`, $1 | **NEW** |
| `/review_paper` | Review a paper (PDF/.tex/.md/.txt) for a venue with web search, seven 1-10 scores, word-limit and AI-slop gates; second opinion from `gpt-5.6-sol` | web on, parallel, 2 h | **NEW** |
| `/revise_and_review_paper` | Loop `/write_paper` and `/review_paper` (default 6 rounds, 1,000-word reviews) until strong accept or no further improvement | `full`, 24 h | **NEW** |
| `/rsi7d` | Mine the last 7 days of SEA runs in `history.db` for mistakes and cost sinks, patch SEA prompts (replaying file-modifying tasks in clones), refresh autorouter evidence, and with permission patch `SYSTEM.md`, `AGENTS.md` and Sorcar's code | $2,000, memory on | **NEW** |
| `/sh` | Run one shell command in the tab's work dir and return its verbatim output | `bash`, no worktree | |
| `/skillopt` | SkillOpt outer loop over a SKILL.md, a SEA prompt or a module constant against a JSON eval set; writes `<target>.proposed` | `shell` | **NEW** |
| `/task_update` | Report what a task has done from its persisted transcript; run by the Task Info panel on first show, every 10 min and on refresh | `bash`, $1 | **NEW** |
| `/write` | Concise professional prose with machine-text vocabulary banned | default + writing protocol | **NEW** |
| `/write_paper` | Write or revise a LaTeX paper with `check_paper` (AI-slop and consistency gates) and `build_paper` (pdflatex + bibtex) and a read-only reviewer model | web on, parallel, 6 h | **NEW** |

Files these SEAs keep under `~/.kiss`: `AGENTS.md`, `SEAS.md`, `AUTOROUTER.md`, `MODEL_DECISIONS.md`, `memories/<repo>/knowledge.sqlite3`, `cron/jobs.json` (weekly rsi7d and daily knowledge jobs), and read-only access to `history.db`.

## 10. Sub-agents and parallelism

- `run_agent(task, agent="", workspace, model_name, max_budget, timeout, chat_id, system_prompt, model_config, use_worktree, auto_commit, use_web_tools, classify_tasks, use_memory, is_parallel, append_to_system_prompt, append_to_prompt, tool_profile)`: `agent` is empty (plain sub-agent), a channel name, `"cron"`, or a `.py` path. Default wait 300 s; on timeout the child is stopped and its spend still charged **NEW**. Path-named agents honour the persisted worktree and auto-commit settings **NEW**. Task text naming the parent repository path is rewritten to the active worktree (`dispatch_path_rewrite`).
- `run_parallel(tasks, max_workers, model_name, tool_profile)`: independent LLM sub-agents, each in its own tab on every surface; fan-out may recurse (the prompt asks for at most two levels, the code sets no limit); children classified as reviewers get the `review` profile when tool profiles are on and the task needs no implementation, unless `tool_profile=` says otherwise.
- `run_commands_parallel(commands, max_workers, timeout_seconds, max_output_chars)`: shell commands in threads, no LLM.
- Sub-agent tabs: opened on all clients while the child runs, closed everywhere by `subagentDone` when it finishes or by `closeSubagentTab` when any client closes them; the user's close is remembered so a replay does not reopen it. **NEW**
- 21,216 of the 27,673 recorded task rows are sub-agents; 4,461 top-level tasks ran in parallel mode.

## 11. Task classification, tags and chat summaries

- **Pre-run classifier** (`task_classifier.py`, "Classify tasks before running", default on): decides `is_simple` and `is_development` so trivial prompts skip the worktree and the full prompt; verdicts are cached; classifier spend folds into the task. Launch phases "Classifying task…" and "Preparing worktree…" are shown in the tab. **NEW**
- **Jev decisions model** (`openrouter/~typesafe/jev-latest`): one gate, `decisions_tool_available()`, controls both the classifier's Jev route and the `decide` tool; it needs the "Use Jev (decisions model)" setting (default **on** since 10-03; off from 09-27 to 10-03), an OpenRouter key and the model in the catalogue.
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
- **Settings panel** (14 groups): Remote password, Max budget per task, Auto commit, Use worktree, Use web tools, Classify tasks, Use Jev (on), Use persistent memory, Chat in the editor (VS Code only), Auto submit spoken task, Wake word sensitivity (0-100, default 80), buttons Tips / Update / Reset Server / Update Models, collapsible API Keys (8) and Custom Models. The Working directory and Memory directory fields were removed on 09-27. **CHANGED**
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
- The Schedule subpanel lists jobs with next-run times and copy buttons. On the development machine 12 jobs are configured (6 prompt, 6 command; 10 enabled): daily test fix, README update, cost audit, model update, knowledge refresh, two weekly rsi7d jobs, a Slack gateway every minute, a 4-hourly sync of `history.db` and memories to a shared machine, a nightly work index, and two paused reinstall jobs. 180 top-level tasks mention cron; 109 `cron_job` calls are recorded.

## 20. Installation, deployment, Docker and release

- **`install.sh`** (73,902 bytes): re-executes under `setsid`, takes a cross-process `flock` (dead holders no longer block retries **NEW**), five steps (git, Node **v22.23.3** **NEW pin**, VS Code CLI, extension build, extension install), installs the `rsorcar` and `sorcar-docker` launchers, applies the `.brand/` overlay **NEW**, and finally waits up to 900 s for the daemon and opens the web app (`kiss-web --trust-ca`, tunnel URL on a remote) **NEW**; `KISS_SKIP_LAUNCH`, `KISS_NONINTERACTIVE`, `KISS_CODE_CLI` control it. The Python package is `kiss-agent-framework` on PyPI; the extension is `ksenxx.kiss-sorcar` on the Marketplace.
- **`rsorcar user@host`** (57,275 bytes), ten steps: ssh check and prerequisites, refuse to restart a remote with a running task unless `SORCAR_FORCE_RESTART=1`, disk headroom check, copy `~/.ssh` (never `authorized_keys`), two-way repository sync through `origin` (never force-push, never a `kiss/wt-*` branch), API keys only to a remote that has none with a probe that fails closed **NEW**, two-way `history.db` merge, **memory sync with tombstones and clock-skew tolerance** **NEW**, sync of `AUTOROUTER.md` and `MODEL_DECISIONS.md` **NEW**, remote `install.sh` (code-server when no `code` CLI), `kiss-web` as a lingering systemd user service, Cloudflare tunnel with ntfy notification, GitHub credentials over stdin, public-URL verification, local `install.sh`. Thirteen `SORCAR_*` environment overrides.
- **Docker**: `sorcar-docker [PORT] [--rebuild]` builds `codercom/code-server` with uv **0.12.19** **NEW pin**, clones the repository, runs `install.sh` and serves code-server on 8080 with the extension preinstalled, forwarding `GH_TOKEN` and the API keys. Inside tasks, `docker_image=` runs the shell and file tools in a container with the work dir mounted at its host path **NEW**.
- **Release** (`scripts/release.sh`, 13 steps): purge private paths from public history, bump the version in `_version.py`, README, SYSTEM.md and `package.json`, build the VSIX into the release commit, push filtered history and tag to `ksenxx/kiss_ai`, GitHub release, PyPI publish (100 MiB per-file check before upload; sdist restricted to `src/kiss` and `projects/swedefend` **NEW**), Marketplace publish, local reinstall.
- **Windows**: MinGit 2.55.0.5 and uv installed by the extension; the 2026-09-24 Windows test run fixed cross-platform bugs in memoryfield, `agents_md`, and channel tests.

## 21. Developer tooling and tests

- `uv run check --full`: `uv sync`, `generate-api-docs`, `compileall`, `ruff`, `mypy`, then `pyright` and, when the extension's `node_modules` and npm are present, `npm run typecheck` and `npm run lint` (eslint, stylelint, htmlhint); failed stages are listed at the end with their output. `--no-clean` and `--clean-only` flags; the extension version is synced first.
- **Tests**: 1,360 `test_*.py` files with 10,682 test functions (382,938 lines across the 1,405 Python files under `src/kiss/tests/`), 369 JavaScript test files under `src/kiss/agents/vscode/test/` (`node test/run-all.js`); the suite disables the worktree pool and the classifier, raises the open-file soft limit to 4,096, and pins `pytest>=9.0.3,<9.1`.
- **Subprocess reaper** **NEW** (`kiss.tests.subprocess_reaper`): wraps `Popen` so every child is attributed to the running test, fixture or session and swept with SIGTERM, 5 s, SIGKILL, emitting `LeakedSubprocessWarning` with the command; written after a run left 529 detached `muse_auth.daemon` processes.
- **Testing policy** (SYSTEM.md): end-to-end tests only, no mocks or structural tests, 100% branch coverage of new code where reachable, test splits run in parallel with `run_commands_parallel` on `cores - 2` workers, a load-dependent flake is named rather than chased.
- **Repository hygiene**: worktree tasks get `node_modules` symlinks and must compile the extension once; the Bash tool refuses commands naming the main checkout path from inside a worktree; `tmp/` for scratch files; `reports/` for deliverables.

## 22. What the trajectories show

Figures from `~/.kiss/history.db` on the development machine, 2026-09-30 (first row 2026-04, plus two rows with a 2009 clock artefact).

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
