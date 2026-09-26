---
title: SEA eval set format (SkillOpt evals JSON, grading, custom Env, sh_sea evals)
uuid: be94c554-0c3d-4c65-ada6-713f026f3c6a
summary: SkillOpt eval set JSON (tasks with expect, expect_regex, setup, check, split;
  rollout defaults; env class; train_rollouts), how verify grades, bundled sh_sea
  train and held-out sets.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# SEA eval set format

Eval sets are JSON files read by `load_evals` in `skillopt_sea.py`. The bundled ones are in
`src/kiss/agents/seas/evals/`.

## Shape

```json
{"rollout": {"web_tools": false, "is_parallel": false, "use_memory": false, "max_steps": 30},
 "tasks": [
   {"id": "echo", "prompt": "echo hi", "expect": ["hi"]},
   {"id": "two", "prompt": "seq 1 2", "expect_regex": "1\\s*\\n\\s*2",
    "setup": "touch f", "check": "test -f f", "split": "select"}]}
```

- A bare list is accepted as `{"tasks": [...]}`.
- `id` defaults to `task<N>`; ids must be unique. `expect` may be a string or list.
- `split`: `"train"`, `"select"` or absent (both). The optimizer needs at least one train task
  and one selection task.
- `rollout`: `SorcarAgent.run` keyword defaults plus `max_steps`; the target's own getters
  override it.
- `env`: `{"class": "pkg.module:Class", ...kwargs}` naming an `Env` subclass that replaces
  in-process rollouts and grading (for example one Docker container per task with a
  benchmark's official harness). Built by `make_env`; must be an `Env` instance.
- `train_rollouts`: path (relative to the eval file) of a `rollouts.json` as the optimizer
  writes it; those trajectories stand in for their tasks' training rollouts and are never
  re-rolled. They must name `split: "train"` tasks only, because a trajectory of the original
  text cannot serve as a selection rollout of a candidate.

## Grading (`verify`)

The rollout result is converted to plain text (HTML tags stripped, entities unescaped). A task
passes when all hold:
1. every `expect` substring is present;
2. `expect_regex` matches (`re.search`);
3. `check` exits 0. It runs in the rollout's scratch dir with `$RESULT` (the plain text) and
   `$SUCCESS` (`1`/`0`) in its environment, 120 s timeout.
A task with none of the three passes when the rollout finished successfully. `setup` runs in
the scratch dir before the rollout (`check=True`, 120 s). `setup` and `check` use the same shell
as the `Bash` tool (`sh` on POSIX, Git bash on Windows, via `_popen_kwargs`).

## Bundled sets

- `sh_sea_evals.json`: 10 tasks for `sh_sea.py` (echo-basic, arithmetic, multiline-order,
  missing-path-exit-code, no-output, stderr-capture, quoting, files-in-workdir, python-inline,
  nonzero-exit-reported), no `split`, so each is used for both training and selection.
- `sh_sea_heldout_evals.json`: 5 different tasks (word-count, env-variable, sorted-lines,
  failing-grep, file-roundtrip) to check that an accepted prompt generalizes rather than
  overfits the training set.

Both use `rollout` defaults `{"web_tools": false, "is_parallel": false, "use_memory": false}`.

## Writing a good eval set

Follow what the bundled ones do: mechanical checks (substrings, regex, shell `check`) instead of
LLM judgment, one behavior per task, and a separate held-out file. The SkillOpt agent is told not
to write an eval set itself unless asked.

## Sources
- `src/kiss/agents/seas/skillopt_sea.py` (`EvalTask`, `EvalSet`, `load_evals`, `load_rollouts`, `plain_text`, `verify`, `Env`, `InProcessEnv`, `make_env`, `run_rollout`)
- `src/kiss/agents/seas/evals/sh_sea_evals.json`, `src/kiss/agents/seas/evals/sh_sea_heldout_evals.json`
