---
title: Base class state and trajectory YAML persistence
uuid: 8a8c8c2b-ed61-4c29-ae00-6c09d3333308
summary: 'Base class: global agent id counter, printer setup, messages history, and
  the atomic trajectory YAML at {artifact_dir}/trajectories/trajectory_{name}_{id}_{stamp}.yaml.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Base class and trajectory persistence

`Base` (`src/kiss/core/base.py`) is the superclass of `KISSAgent` and of `RelentlessAgent` (which subclasses `Base` directly;
`SorcarAgent` subclasses `RelentlessAgent`). The relentless agents compose `KISSAgent` executors per session.

## State
- `id` comes from a process-wide `Base.agent_counter`, guarded by `_class_lock` so ids stay unique across threads.
- `messages` is the trajectory (not the provider conversation). `_add_message(role, content, timestamp)`
  appends `{unique_id, role, content, timestamp}`. KISSAgent adds `user` (prompt, tool results, nudges) and
  `model` (text + ```python call reprs``` + ```text usage```) entries.
- `model.conversation` (the provider-format messages) is separate. Compaction and `llm_call_hook` change
  only the conversation, never `messages`.

## Saving (`_save`, called from `run()`'s `finally`)
- Path: `{config.artifact_dir}/trajectories/trajectory_{name}_{id}_{_trajectory_stamp}.yaml`. In the name,
  spaces and `/` become `_`.
- `_trajectory_stamp` = `max(run_start_timestamp, previous_stamp + 1)`, so two runs of one instance in the
  same second do not overwrite each other. `run_start_timestamp` stays the real wall clock.
- Contents (`_build_state_dict`): name, id, messages, tool names, start/end timestamps, `config_to_dict()`,
  arguments, prompt_template, is_agentic, model, budget_used/total_budget, tokens_used/max_tokens,
  step_count/max_steps, and the command line.
- The write is atomic (`atomic_write_text`), because the server may be serving the file to the trajectory
  visualizer while it is written.
- `_reset` clears per-run state **before** building the model, so a run that fails on an unknown model name
  does not overwrite the previous run's file.
- `get_trajectory()` returns `messages` as JSON.

## Sources
- `src/kiss/core/base.py` (`Base`, `_save`, `_build_state_dict`, `get_trajectory_path`, `_add_message`)
- `src/kiss/core/kiss_agent.py` (`KISSAgent._reset`)
