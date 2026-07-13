# `code_graph` A/B Benchmark

## Verdict

**This evaluation does not verify an overall improvement.** On the tested KISS
corpus, both versions answered every task correctly, while the reviewed
`code_graph` treatment used **12.1% more tokens**, cost **1.7% more**, and took
**10.8% longer** on average. Task-cluster bootstrap intervals include zero for
all efficiency deltas, so neither a win nor a stable regression is established
from this suite.

The benchmark did expose two real implementation defects. Fixing them reduced
the treatment's one-trial token overhead from **+128% to +15%** and its cost
overhead from **+44% to approximately parity**. The fixes are included with
this benchmark, but even the corrected feature did not beat the baseline
across all three requested dimensions.

## Final reviewed experiment

### Design

- **84 independent agent runs:** 14 tasks × 2 arms × 3 trials.
- **Tasks:** 11 structure/navigation questions and 3 general code questions.
  Each answer is graded by deterministic, case-insensitive gold-fact coverage;
  there is no LLM judge.
- **Corpus:** a 222 MB copy of KISS containing 885 Python files, plus its other
  source/docs. Git history, `.kiss`, `tmp`, `node_modules`, `.venv`, and caches
  were excluded. Neither arm could recover answers from Git history.
- **Graph:** 18,008 nodes and 47,480 edges from 1,036 source files. A clean full
  build took **2.17 seconds and $0**; incremental updates were not included in
  per-task latency.
- **Model:** `claude-fable-5`; `max_steps=15`, `max_budget=$1` per run.
- **Controlled feature ablation:** both arms use the same current `KISSAgent`,
  prompt, corpus, Bash, and Read implementations. Baseline has no graph and no
  `code_graph` tool; treatment has a prebuilt graph, the tool schema, and real
  production query-before-grep interception. This isolates the feature, but it
  is **not** a full checkout-level execution of commit `a0694449`.
- **Isolation:** separate corpus copies; baseline graph absence asserted before
  every run. All 42 baseline transcripts contain zero graph hints.
- **Measurements:** final-answer accuracy, total model tokens, model-reported
  USD cost, wall time, steps, graph hints, explicit graph calls, and errors.
- **Final matrix spend:** **$5.44**. All raw results and trajectories are
  committed alongside this report.

### Aggregate results

Means are per task-trial (42 observations per arm).

| Metric | Baseline | `code_graph` | Relative delta |
|---|---:|---:|---:|
| Gold-fact accuracy | **100.0%** | **100.0%** | tie / ceiling |
| Tokens | **6,413** | 7,188 | **+12.1%** |
| Cost | **$0.06428** | $0.06536 | **+1.7%** |
| Wall time | **18.81 s** | 20.84 s | **+10.8%** |
| Agent steps | **3.05** | 3.24 | **+6.3%** |
| Errors | 0 | 0 | tie |
| Grep interceptions | 0 | 25 | — |
| Explicit `code_graph` calls | 0 | 0 | — |

Task-cluster bootstrap 95% intervals (50,000 deterministic resamples; task is
the resampling unit):

| Delta | Point estimate | 95% interval |
|---|---:|---:|
| Tokens | +12.1% | −4.9% to +33.5% |
| Cost | +1.7% | −8.9% to +13.6% |
| Wall time | +10.8% | −5.9% to +33.6% |
| Steps | +6.3% | −7.5% to +24.5% |

All intervals cross zero. With 14 tasks and three stochastic trials, the
honest conclusion is **no statistically stable efficiency difference**, not
that the small cost delta is meaningful.

### Results by task category

| Category | Arm | Accuracy | Tokens | Cost | Time | Steps |
|---|---|---:|---:|---:|---:|---:|
| Structure (33 runs) | baseline | 100% | **6,389** | **$0.06329** | **18.51 s** | **3.09** |
| Structure (33 runs) | treatment | 100% | 7,350 | $0.06556 | 21.45 s | 3.30 |
| General (9 runs) | baseline | 100% | **6,502** | $0.06792 | 19.90 s | **2.89** |
| General (9 runs) | treatment | 100% | 6,592 | **$0.06463** | **18.59 s** | 3.00 |

The structure subset—where a graph should help most—still used 15.0% more
tokens and took 15.9% longer. The small general-task cost/time win is based on
only three tasks and is not evidence of a general effect.

### Per-task means across three trials

Both arms scored 100% on every row, so the table focuses on efficiency.

| Task | Base tokens | Graph tokens | Δ tokens | Base cost | Graph cost | Base s | Graph s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `callers_kill_process_group` | 12,307 | **11,440** | −7.0% | $0.1041 | **$0.0892** | 30.7 | **27.2** |
| `where_class_useful_tools` | **2,464** | 2,744 | +11.4% | **$0.0280** | $0.0298 | 10.6 | **10.3** |
| `where_class_kissagent` | **4,255** | 5,919 | +39.1% | **$0.0439** | $0.0533 | **16.7** | 19.5 |
| `callers_make_skill_tool` | **3,600** | 6,188 | +71.9% | **$0.0463** | $0.0592 | **11.8** | 21.0 |
| `callers_discover_skills` | **7,783** | 10,060 | +29.3% | **$0.0864** | $0.0899 | **13.8** | 27.3 |
| `where_docker_manager` | **2,450** | 2,727 | +11.3% | **$0.0274** | $0.0294 | **10.3** | 10.7 |
| `callers_format_bash_result` | 13,572 | **8,888** | −34.5% | $0.1120 | **$0.0768** | 34.6 | **24.4** |
| `sorcar_agent_parent` | **5,106** | 5,530 | +8.3% | **$0.0552** | $0.0572 | **18.2** | 22.5 |
| `callers_load_skill_content` | **4,604** | 9,426 | +104.7% | **$0.0602** | $0.0818 | **12.5** | 26.9 |
| `callers_catalog_xml` | **6,980** | 8,685 | +24.4% | **$0.0651** | $0.0742 | 22.2 | **21.8** |
| `guard_wiring` | **7,157** | 9,247 | +29.2% | **$0.0675** | $0.0802 | **22.3** | 24.6 |
| `general_skill_sources` | **5,826** | 6,218 | +6.7% | **$0.0576** | $0.0589 | 20.4 | **16.6** |
| `general_bundled_skills_dir` | 10,426 | **10,025** | −3.9% | $0.0964 | **$0.0831** | **25.1** | 25.2 |
| `general_kissagent_run_params` | **3,252** | 3,535 | +8.7% | **$0.0498** | $0.0519 | 14.2 | **14.0** |

The graph had meaningful wins on two caller questions, but substantial losses
on several others. For searches that never triggered interception, treatment
still paid for the additional tool schema on every model turn.

## Bugs found and fixed by the independent review

The initial implementation was developed with `claude-fable-5`; the required
`gpt-5.6-sol` review read the harness, all final answers, and representative
trajectories, then found these defects:

1. **Relevant graph answers were truncated out.** `query()` performed a
   three-hop BFS but alphabetically sorted every discovered node and rendered
   *all nodes before all edges*. On a real repository, a file hub expanded to
   thousands of nodes, so a 1,500-character grep hint for
   `_format_bash_result` began with unrelated `__init__`/alphabetical nodes and
   omitted the queried function, direct callers, and relationships.
   The fix orders by BFS distance and interleaves each node with the edge back
   toward the seed. A regression test recreates the hub under a tight budget.
   A later review also found exact labels (for example `UsefulTools`) could lose
   their seed slot to alphabetically earlier substring matches
   (`AnotherUsefulToolsTest`); exact-match scoring and seed-rank rendering now
   prevent that failure.
1. **Repeated denial loops.** When a model re-ran grep to verify an incomplete
   graph answer, every equivalent grep was denied again. The fix returns one
   graph hint per distinct answer per `UsefulTools`/Docker tool instance; a
   repeated search falls through to real grep for verification. This preserves
   correctness while preventing agent loops.
1. **Instrumentation conflated two events.** The original counter treated
   `[code_graph]` tool-result labels as grep hints and failed to detect textual
   `code_graph(action=...)` calls. Raw transcripts were reprocessed with
   separate exact-marker and model-call checks. Released-feature diagnostic:
   14 actual hints and 8 explicit calls, not 22 hints and 0 calls.
1. **One incomplete rubric.** The `Skill.source` gold list omitted `user`.
   It was added and all saved answers were regraded (scores were unchanged
   because every answer had included it).
1. **Non-reproducible setup and single-trial weakness.** The runner now creates
   sealed corpora via `--prepare`, validates arm purity, supports resumable
   arm/trial result files, uses monotonic timing, and ran three final trials.

Validation after these fixes: **106 code-graph tests passed** and
`code_graph.py` retained **100% statement and branch coverage**.

### Diagnostic progression

These one-trial figures show why transcript review mattered. They are retained
in `results_pre_review.json` / `transcripts_pre_review/` and
`results_query_fix.json` / `transcripts_query_fix/`.

| Treatment stage | Accuracy | Tokens/task | Cost/task | Time/task |
|---|---:|---:|---:|---:|
| Baseline trial 1 | 100% | 6,791 | $0.0693 | 18.75 s |
| Released feature (before review) | 100% | 15,500 | $0.0999 | 26.82 s |
| Relevance ordering fixed | 100% | 9,349 | $0.0781 | 21.93 s |
| Ordering + one-shot fallback (trial 1) | 100% | 7,801 | $0.0686 | 21.38 s |

The review removed most of the regression, but the final three-trial matrix
still does not establish a net win.

## Interpretation and limitations

- **Accuracy is inconclusive due to a ceiling.** Both arms reached 100%; this
  suite cannot test Graphify's reported accuracy lift. Harder multi-hop or
  held-out code-change tasks are needed.
- **The graph tool itself was never selected in the final 42 treatment runs.**
  The observed effect comes from 25 grep interceptions plus the tool-schema
  token overhead—not from deliberate `query/path/explain` usage.
- **The corpus is large enough to stress graph fan-out, but not ~1M LOC.** It is
  substantially larger than a toy repository, yet smaller than the corpus on
  which Graphify reports its largest gains.
- **Three trials reveal substantial model variance.** For example, baseline
  `callers_kill_process_group` ranged from 5,886 to 18,812 tokens. This is why
  the report uses all trials and uncertainty intervals rather than selecting a
  favorable run.
- **Order was fixed baseline-then-treatment within each task** and provider
  latency was not controlled. Wall-time deltas therefore include API noise.
- **Gold-fact substring grading measures recall, not unsupported claims.** The
  reviewer manually checked every final answer; no false-positive grades were
  found, but a larger benchmark should add structural precision and
  unanswerable cases.
- **Ablation vs checkout:** using the same executable code isolates the feature
  cleanly, but does not measure unrelated behavioral changes between complete
  historical revisions or the full Sorcar tool suite.

## Recommended next changes

1. Do **not** enable query-before-grep globally based on this evidence. Keep it
   opt-in until it wins a harder benchmark.
1. Replace the 1,500-character generic subgraph with a compact operation-shaped
   result (`callers`, `callees`, `definition`) containing `file:line`—the shape
   agents repeatedly sought.
1. Suppress test/obsolete nodes by default for production-code questions and
   down-rank file/class hubs.
1. Avoid paying the `code_graph` tool schema on turns where only transparent
   interception is desired, or teach the prompt/tool schema when to call
   `query`, `path`, and `explain`.
1. Next evaluation: 20–50 harder tasks, including multi-hop questions,
   unanswerable cases, and held-out edit/test tasks on a ≥1M-LOC repository;
   randomize arm order and run at least three trials.

## Research basis

The methodology follows the sources researched before implementation:

- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — trials, deterministic graders, isolated environments, trajectories, and token/latency tracking.
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — realistic tasks, verifiable outcomes, tool-call/runtime/token analysis, and raw transcript review.
- [Databricks: Benchmarking coding agents on a multi-million-line codebase](https://www.databricks.com/blog/benchmarking-coding-agents-databricks-multi-million-line-codebase) — cost per task, deterministic grading, same harness, and sealing Git history.
- [SWE-bench](https://www.swebench.com/) — common harnesses, resolve rate, average cost, and published trajectories.
- [CodexGraph (arXiv:2408.03910)](https://arxiv.org/abs/2408.03910) — graph-assisted repository navigation precedent and structure-aware retrieval.
- [Graphify README](https://github.com/Graphify-Labs/graphify/blob/main/README.md) and [architecture](https://github.com/Graphify-Labs/graphify/blob/main/ARCHITECTURE.md) — token-per-query methodology and corpus/subgraph comparisons.
- [Mem0: State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026) — report correctness, tokens, and latency together on the application's own workload.

## Reproduction

```bash
uv run python benchmarks/code_graph_eval/runner.py --prepare
uv run python benchmarks/code_graph_eval/runner.py --trial 1
uv run python benchmarks/code_graph_eval/runner.py --trial 2
uv run python benchmarks/code_graph_eval/runner.py --trial 3
```

`results.json` contains all 84 final records. `transcripts/` contains the 84
corresponding trajectories. The archived pre-review files preserve the failed
implementation rather than hiding it.
