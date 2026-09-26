---
title: SkillOpt SEA - optimizing a skill or SEA prompt against an eval set
uuid: cdb050de-f3aa-4904-a2db-76ebe49ca69a
summary: 'skillopt_sea.py optimizes prompt text of a SKILL.md, SEA or module constant:
  rollouts, analysts, ranker, pre-rollout gate, strict-improvement acceptance, .proposed
  output, resume.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# SkillOpt SEA

`src/kiss/agents/seas/skillopt_sea.py` implements the outer loop of SkillOpt (arXiv 2605.23904,
cited in the module docstring). It never edits the target: accepted text is written to
`<target>.proposed` for the user to adopt.

## Targets (`load_target`)

- `SkillTarget`: a `SKILL.md`; the whole file is trainable and is appended to the default
  Sorcar system prompt during rollouts.
- `SeaTarget`: a `*_sea.py`; the trainable text is the string constant returned by
  `system_prompt()` (directly or through a module constant, found by `_prompt_constant` via
  AST). `validate` rejects a candidate that does not compile, changes any code outside that
  constant (AST fingerprint compare), cannot be loaded, or whose `system_prompt()` does not
  return the proposed text. Candidates are written as `<stem>_candidate.py` (never a `_sea.py`
  name, so they do not become slash commands).
- `ConstantTarget` (`--constant NAME`): any module-level string constant; may be a
  `str.format` template, and candidates must keep exactly the same replacement fields. Used for
  SEAs that build the prompt inside a class or function.

Because candidates run from scratch directories, a SEA target must be self-contained (no sibling
imports, no files next to `__file__`), or it fails the gate with "cannot be loaded as a SEA".

## One round (`Optimizer._round`)

1. Roll the current best out on the training batch (imported trajectories from
   `train_rollouts` stand in for their tasks) and on the selection tasks.
2. Failure and success analysts (`FAILURE_ANALYST_PROMPT`, `SUCCESS_ANALYST_PROMPT`) read
   minibatches (`minibatch=8`) of trajectories and propose patches (`PATCH_FORMAT`).
3. One ranker call (`RANKER_PROMPT`) merges and ranks the patches plus earlier rejected ones down
   to the round's edit budget; `edit_budget` cosine-decays from `edit_budget` (4) to
   `edit_floor` (2) over the run.
4. Apply patches, run the target's pre-rollout gate, roll the candidate out on the selection
   tasks, and accept only if its pass rate is **strictly** higher than the best's.
   Rejected candidates are recorded in `state["rejected"]` and shown to later rankers.

Rounds per epoch = ceil(train tasks / `batch_size` (40)). The loop stops early when
`total_cost >= max_cost` (default $20). Each rollout is capped by `rollout_budget` ($1) and
`max_steps` (30); `max_workers` (4) rollouts run in parallel threads.

Rollouts (`run_rollout`) run in-process with `SorcarAgent` (no daemon) in a per-task scratch
dir, with defaults `web_tools=False, is_parallel=False, use_memory=False`, overridden by the
eval set's `rollout` object and then by the target's getters (`SeaTarget.rollout_kwargs`).
Spend is attributed to the calling agent (`_attribute`).

## Artifacts in out_dir

`state.json` (resumable; resumed only when `target`, `evals`, `task_ids`, `original_text` match,
`_RESUME_KEYS`), `log.txt`, and `round_NN/` with `train/`, `select_best/`,
`select_candidate/` (each with `rollouts.json`), `proposed_patches.json`, `candidate.txt`.
`fresh=True` deletes earlier state, logs, round dirs and the proposal. `status(out_dir)`
reports from `state.json` without spending.

Gotcha: the default `out_dir` in code is `<target dir>/.skillopt/<target stem>`, while the
agent's `SYSTEM_PROMPT` says `<work_dir>/tmp/skillopt/<target stem>`; the agent passes its own
`out_dir` when following its prompt.

## Running

```
/skillopt optimize src/kiss/agents/seas/sh_sea.py with src/kiss/agents/seas/evals/sh_sea_evals.json for one epoch

uv run python -m kiss.agents.seas.skillopt_sea --target src/kiss/agents/seas/sh_sea.py \
    --evals src/kiss/agents/seas/evals/sh_sea_evals.json --out-dir tmp/skillopt/sh_sea --model <model>
```

From another agent use `run_agent` with the file path (a bare `"skillopt"` name is not
resolved; see `seas-writing-and-running`). Eval format: `seas-eval-set-format`.

## Sources
- `src/kiss/agents/seas/skillopt_sea.py` (`SkillTarget`, `SeaTarget`, `ConstantTarget`, `load_target`, `OptimizeConfig`, `run_rollout`, `edit_budget`, `Optimizer`, `optimize`, `status`, `main`, `SYSTEM_PROMPT`)
- Commit `77d58a865` "add SkillOpt SEA for prompt optimization, validate on sh_sea"
