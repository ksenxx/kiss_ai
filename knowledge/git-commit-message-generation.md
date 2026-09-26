---
title: LLM commit message generation (commit_message.py, staged diff cap, User prompt/Result
  blocks)
uuid: 6cfa7513-c7d9-452f-8add-aef24a8ebf46
summary: 'generate_commit_message_from_diff: get_fast_model one-shot KISSAgent, conventional
  commits, User prompt:/Result: blocks, fallbacks; staged diff capped at 200,000 bytes.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# LLM commit message generation

## `commit_message.generate_commit_message_from_diff(diff_text, user_prompt=None, task_result=None)`
- Model: `kiss.core.models.model_info.get_fast_model()`.
- `_run_oneshot_llm` builds a `KISSAgent("Commit Message Generator")` and calls `run(..., is_agentic=False, verbose=False)`;
  any exception or empty output returns the fallback.
- Prompt asks for a conventional-commit subject (`type: description`) and optional bullet body. With a
  user prompt, the template tells the model to phrase the subject by the user's intent and NOT to repeat the
  prompt (it is appended separately).
- Fallback text: `"kiss: auto-commit agent work"` (empty diff, model selection failure, LLM failure).
- `_append_user_prompt` adds `USER_PROMPT_HEADING` (`"\n\nUser prompt:\n"`) + trimmed prompt;
  `_append_task_result` adds `TASK_RESULT_HEADING` (`"\n\nResult:\n"`) + trimmed result. Both headings are
  defined in `git_worktree.py`.
- `clean_llm_output` strips whitespace, then only PAIRED surrounding quotes, repeatedly. `str.strip('"')` was
  wrong: it corrupted `feat: rename "foo"`.
- Layering: lives in sorcar (not server) because sorcar may depend only on itself and `kiss.core`;
  `kiss.server.helpers` re-exports it.

## Bounded diff input
`GitWorktreeOps.staged_diff(wt_dir, max_bytes=COMMIT_MESSAGE_DIFF_LIMIT_BYTES)` (200,000) reads at most that
many bytes through the streaming `_git_stdout_head`. Larger patches get a truncation note with `--shortstat`
totals and a `--stat` head (quarter of the cap). Added after auto-commit-and-merge hung the daemon on huge
diffs; emptiness checks use `has_staged_changes` (`git diff --cached --quiet`) instead of reading the patch.

## Squash-merge commit message
`GitWorktreeOps._merge_commit_message` reuses the full message of the task branch's HEAD commit (the
auto-commit's LLM message) and appends prompt/result blocks only if not already present with the current
values (`_ensure_task_metadata`); fallback `"kiss: merged from <branch>"`.

## Sources
- `src/kiss/agents/sorcar/commit_message.py` (`generate_commit_message_from_diff`, `_run_oneshot_llm`, `clean_llm_output`, `_append_user_prompt`, `_append_task_result`)
- `src/kiss/agents/sorcar/git_worktree.py` (`COMMIT_MESSAGE_DIFF_LIMIT_BYTES`, `GitWorktreeOps.staged_diff`, `_git_stdout_head`, `_merge_commit_message`, `_ensure_task_metadata`)
