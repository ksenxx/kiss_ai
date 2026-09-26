---
title: Pre-run task classification (is_simple, is_development) with Jev decisions
  model and LLM fallback
uuid: 39493393-2386-4bf5-adfc-f21048dcc109
summary: 'Pre-run task classification (is_simple, is_development): Jev decisions classifier,
  LLM fallback, verdict cache, classify_tasks config, KISS_DISABLE_TASK_CLASSIFIER.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Task classifier

Before a run, the classifier answers two questions about the task:
- `is_simple`: involves neither software development nor Internet search. Such a task runs on `SYSTEM_LITE.md`.
- `is_development`: may create or modify project files (code, docs, reports, notebooks, data). Such a task
  must run in a worktree. Git-only requests (commit, merge, rebase, conflict resolution) are not development.
  The verdict can only **demote** a run that asked for a worktree, never promote a pinned
  `use_worktree=False`.

## Two classifiers (`classify_task`)
1. **Decisions classifier**, the default when `decisions_classification_enabled()`: the config key
   `classify_with_decisions` (default true; the settings-panel "Classify with Jev" box), an
   `OPENROUTER_API_KEY`, and `openrouter/~typesafe/jev-latest` in the catalog. It asks one `choice`
   question with 5 kinds (`_DECISIONS_KIND_CRITERIA`), mapped by `_KIND_VERDICTS` to (is_simple, is_development):
   development (F,T), ambiguous (F,T), git_only (T,F), simple (T,F), internet (F,F).
   A chosen kind with probability < `CLASSIFIER_DECISIONS_MIN_PROBABILITY = 0.6` is treated as ambiguous.
   The call times out after 5 s (`CLASSIFIER_DECISIONS_TIMEOUT_SECONDS`), and `KISS_DECISIONS_BASE_URL` overrides the endpoint.
   On the 415-prompt benchmark in `benchmarkings/task_classifier/` it scored 89% agreement vs 84% for the best LLM,
   at about 1/200 of the cost. It is non-generative, so it also works for cc/codex models.
2. **LLM classifier**, used when the decisions classifier is off or failed: one non-agentic `KISSAgent` `generate()` on the
   run's own model, returning `{"is_simple", "is_development"}`. It pins the provider's structured output
   (`_VERDICT_JSON_SCHEMA`) except on Anthropic, where it measured slower and `thinking.type=disabled` gives HTTP 400 on
   adaptive-thinking models. If the structured attempt fails, one plain retry runs. `_parse_verdict` (via `_verdict_bool`) accepts
   real booleans and the strings true/false, yes/no, 1/0 (case-insensitive, trimmed); numbers and null are rejected. Caps: `CLASSIFIER_MAX_TOKENS = 1000`, `CLASSIFIER_TASK_MAX_CHARS = 20_000` (longer tasks are truncated),
   `CLASSIFIER_MAX_BUDGET = 1.0`, and a 60 s stall timeout. It is **skipped for cc/* and codex/* models**: they run
   tasks to completion and could actually *execute* the embedded task.

Any failure returns `classification=None`, and the run behaves exactly as it would without a classifier (full prompt,
requested worktree mode).

## Cache
Verdicts are memoised in `<KISS home>/task_classifier_cache.json`, keyed by the sha256 of (criteria, model,
whole model_config, task). There are at most 2000 entries with a 7-day TTL. Changing the criteria or prompt retires old
memos. A cache hit costs zero, which makes relaunching a repeated prompt near-instant.

## Enabling
`classification_enabled(override)`: `KISS_DISABLE_TASK_CLASSIFIER=1` (the test suite's kill switch) beats
everything, then comes the per-run override (`classify_tasks` / wire `classifyTasks`), then the
`classify_tasks` config key (default true).

## Agent integration
`SorcarAgent._classify_task_once` classifies at most once per run (`_classification_attempted`) and
substitutes the prompt arguments first. `WorktreeSorcarAgent.run` (worktree decision) and `SorcarAgent.run`
(prompt choice) share the verdict. The server runner calls `classify_task_for_run` to *pre-seed* the verdict
for a submission (`_classification_preseeded`), so the worktree bookkeeping and the agent agree across
subtasks. The spend is held in `_ClassifierSpend` and folded into the run's totals after `super().run`
(`_fold_classifier_usage`), because `_reset` would otherwise zero it.

## Sources
- `src/kiss/agents/sorcar/task_classifier.py` (`classify_task`, `_classify_with_decisions`, `_classify_with_llm`, `_KIND_VERDICTS`, `classification_enabled`, `decisions_classification_enabled`, `_cache_key`, `CLASSIFIER_*`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_classify_task_once`, `classify_task_for_run`, `_reset_task_classification`, `_fold_classifier_usage`)
