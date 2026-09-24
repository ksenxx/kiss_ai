---
name: model-cost-router
description: Routes every unit of agent work (sub-agent dispatch, review, exploration, test runs, planning) to the cheapest model tier that will finish it correctly on the first attempt, escalating only on a verified failure. Use when a task spawns sub-agents or parallel workers, when the budget is limited, when the user mentions cost, tokens, cheap models, model selection, or routing, or before any run_parallel / run_agent / set_model call. Classifies with a non-generative decide() call, picks concrete priced models from the local catalog, and logs every decision.
license: MIT
compatibility: Designed for KISS Sorcar (run_parallel, run_agent, set_model, decide). Also applies to Claude Code, Codex and other Agent Skills clients via references/other-clients.md. scripts/route.py needs Python 3.11+, no third-party packages.
metadata:
  author: Koushik Sen
  version: "1.0"
  researched: "2026-09-24"
---

# Model cost router

Frontier models cost 40 to 100 times more per token than small models
(`claude-opus-5-5` $4/$20 vs `gpt-6-luna` $0.10/$0.50 per 1M tokens), and most
agent tokens are spent on exploration, file reads, test output, and mechanical
edits that a small model handles as well. This skill moves those tokens to the
cheapest capable tier while keeping reasoning-heavy and irreversible work on a
frontier model. The objective is **cost per accepted task**, not cost per token:
a cheap model that fails, retries, and then escalates costs more than routing
correctly the first time.

## When to run the protocol

Run it before **every** dispatch of work to another model or model change:
`run_parallel(...)`, `run_agent(...)`, `set_model(...)`, or, in other clients,
an Agent-tool call with a `model` parameter. Also run it once at task start to
decide the phase plan (Step 4). Never route a single tool call or a single
turn: routing granularity is one *unit of work* with a checkable result.

## Step 1: Split the task into units with a mechanical acceptance check

A unit is dispatchable when you can state what proves it done without reading
the transcript: a passing test command, a `git diff --stat` that touches only
the expected files, a grep that finds the new symbol, a file that exists with
the expected sections. Units without such a check stay in your own loop.

## Step 2: Classify each unit with `decide()`

`decide()` is a non-generative call that costs a fraction of a cent; use it
instead of reasoning about the tier yourself. Put the unit description,
the acceptance check, the file count and size, and any prior failure of the
same unit into `state`, and ask:

```json
{
  "tier": {
    "type": "choice",
    "instructions": "Cheapest model tier that completes this unit correctly on the first attempt, judged by the acceptance check.",
    "criteria": {
      "small": "Mechanical or lookup work with an unambiguous spec: read/summarize files or logs, grep and report, run tests and report failures, rename or move symbols, format, write boilerplate or docstrings from a template, fill a table, translate a config between formats.",
      "medium": "Standard engineering with a clear spec: implement a function or endpoint from a description, write tests for existing code, fix a bug whose cause is known, refactor within one module, write documentation for existing behaviour, review a small diff against stated rules.",
      "frontier": "Open-ended reasoning: find the root cause of a failure, design across modules or services, resolve conflicting requirements, security or concurrency review, anything where a wrong answer is expensive to detect, and the final acceptance judgement of the whole task."
    }
  },
  "guarded": {
    "type": "noul",
    "instructions": "Does the unit touch authentication, secrets, payments, data deletion, migrations, or any action the user cannot undo, or does it decide whether the whole task is complete?"
  }
}
```

Apply the result mechanically:

- `guarded` probability >= 0.5: **frontier**, whatever `tier` says.
- `tier` confidence < 0.6: move one tier **up**. Ties go to the expensive tier.
- The same unit already failed once on a tier: start at the next tier up and
  pass `--exclude <failed model>` in Step 3. Never retry the same tier with
  a longer prompt or higher effort; that is the failure mode that makes
  cheap routing cost more than no routing.

## Step 3: Pick a concrete model from the local catalog

```bash
python <skill_dir>/scripts/route.py pick --tier medium --in 300000 --out 30000
python <skill_dir>/scripts/route.py menu                   # all tiers, priced
python <skill_dir>/scripts/route.py estimate --model gpt-6-sol --in 300000 --out 30000
```

`pick` prints the first candidate of the tier whose provider credential is
configured (`"runnable": true`), with its `$ /1M` prices and the estimated
cost for the token counts you pass (`--in` defaults to 200k prompt tokens,
`--out` to 20k; a sub-agent that reads a medium codebase and runs tests uses
that much). Outside Sorcar, where the `kiss` package is not importable,
availability is unknown and `pick` returns the first candidate with
`"runnable": null`; confirm the credential yourself before dispatching.
Candidate order per tier lives in `assets/tiers.json` and is by measured
coding quality per dollar as of 2026-09-24; edit it when prices or models
change. `references/routing-table.md` explains each choice.

Then dispatch with the model on the sub-agent boundary, never mid-context:

```text
run_parallel(tasks=[...], model_name="<picked>")     # independent units
run_agent(task=..., model_name="<picked>")           # one delegated unit
```

Prompt caches are per model. Switching the main loop with `set_model` throws
the cache away and repays the whole context at the new model's input price,
so use `set_model` only at a phase boundary (Step 4), never inside a phase.

## Step 4: Phase plan for the main loop

Plan and final acceptance on the frontier model; execution on medium; volume
work (exploration, test runs, log reading, documentation drafts) on small
sub-agents that return a summary, not raw output. This is the pattern Claude
Code ships as `opusplan`. Concretely, when you start on a frontier model and
the task is a multi-file implementation:

1. Read the code, write the plan and the acceptance checks (frontier).
2. `set_model("<medium pick>")` once, implement the plan, run the tests.
3. Dispatch reviews and test runs to small or medium sub-agents.
4. `set_model("<original frontier model>")` once for root-cause work if a
   check fails twice, and for the final verification of every requirement.

Skip the downgrade when the task is short (under roughly 10 tool calls), when
the user pinned a model, or when the remaining budget is not a constraint and
the user asked for the best result rather than the cheapest.

## Step 5: Budget gate

Sorcar prints `Budget: $spent/$max` after every tool result. Before a
dispatch, compare the `estimated_usd` from `pick` with the remaining budget:

- estimate > 25% of remaining: split the unit or drop a tier if the classifier
  allows it (never below the `guarded` floor); if neither is possible, tell the
  user before dispatching.
- reviewer share named by the user ("at most N% for reviewing"): reviewers
  run on the small tier unless the diff touches guarded code.

## Step 6: Verify, then escalate up only

Accept a sub-agent's result only through its acceptance check, never through
its own summary. Small-tier agents over-report success; run the check
yourself (`uv run pytest <impacted tests>`, `git diff --stat`, `grep -n`).
On failure, escalate **one tier up** with the failure evidence in the new
prompt and the failed model in `--exclude`. Two tiers up when the failure is a
reasoning failure (wrong approach, misunderstood spec) rather than a slip
(typo, missed file). Never escalate more than twice for the same unit; on the
third failure stop and report.

## Step 7: Log every decision

```bash
python <skill_dir>/scripts/route.py log --unit "write tests for parser" --tier medium \
    --model gemini-3.8-flash --reason "clear spec, existing code" --outcome "passed 12/12"
```

Appends a row to `./tmp/MODEL_DECISIONS.md`. Update the outcome column when
the check runs. The ledger is what lets you (or the user) see whether a tier
is failing too often for a kind of unit and move the boundary.

## Hard rules

- Never route to small or medium: security-sensitive code, credentials,
  payments, deletions, migrations, the final acceptance of the whole task,
  root-cause analysis after two failed fixes, or user-facing prose that will
  be sent on the user's behalf.
- Never downgrade a model the user named explicitly.
- Never spawn a sub-agent to run a shell command; run it inline
  (`run_commands_parallel` for many). A sub-agent costs a full model context.
- Never let a small-tier agent expand its own scope; its prompt must name
  the files it may touch and the check that ends it.
- Do not invent prices or model names: use `route.py menu`.

## Reference files

- `references/routing-table.md`: tiers, candidate models with prices, the
  effort axis, and where cheap routing fails.
- `references/evidence.md`: the sourced research and practitioner numbers
  behind each rule (RouteLLM, FrugalGPT, cascade routing, Claude Code docs).
- `references/other-clients.md`: the same protocol for Claude Code
  (`model:` frontmatter, `CLAUDE_CODE_SUBAGENT_MODEL`, `opusplan`, effort),
  Codex and other Agent Skills clients.
- `assets/tiers.json`: the ordered candidate list `scripts/route.py` reads.
