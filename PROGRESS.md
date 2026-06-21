# Progress

## Task: Re-review previous task's worktree-remap fix with gpt-5.5

### Final state — DONE

Commit `339c7e99` (already on the worktree branch, ready for squash-merge to main):

```
fix(sorcar): prevent parent-repo mutations in worktree-isolated tasks

* Add _absolutize() helper to anchor relative paths under work_dir,
  ensuring they resolve to the worktree regardless of host process cwd
  (bug 1: relative paths bypass remap)
* Make Read/Write/Edit path remap unconditional to prevent fallthrough
  when worktree branch deletes files (bugs 2–3)
* Add _bash_parent_repo_guard() to refuse shell commands targeting
  parent-repo absolute paths, blocking sed -i / echo / rm leaks (bug 4)
* Separate Read's stale-worktree fallback into elif for mutual exclusion
  with active remap
* Add 11 regression tests covering all identified bugs and end-to-end
  auto-commit/squash-merge verification
```

### Round 1 — gpt-5.5 review of commit 43a1b70f

Bugs identified:

1. HIGH — Relative paths bypass remap (Path.resolve() uses host cwd).
1. HIGH — Edit silently mutates main when worktree branch deleted file.
1. HIGH — Read returns stale main content when worktree branch deleted file.
1. HIGH — Bash was not touched; sed -i / echo > / rm still leak.
1. LOW — Misleading success messages (deprioritised).
1. MEDIUM — Existing test only checked dirty-state, not main advancement.

### Round 2 — claude-opus-4-7 fix in `src/kiss/agents/sorcar/useful_tools.py`

1. Added `import re`.
1. New helper `_absolutize(file_path, work_dir)`.
1. New helper `_bash_parent_repo_guard(command, work_dir)`.
1. Wired `_absolutize` into `Read`, `Write`, `Edit`.
1. `Read`: unconditional remap; stale-fallback now `elif` (mutex with active remap); removed second `.resolve()` on remap (Bug 5 sub-issue).
1. `Edit`: unconditional remap (dropped `is_file()` gate).
1. `Bash`: wired `_bash_parent_repo_guard` at top of both sync and streaming code paths.

### Round 2 — claude-opus-4-7 tests

`src/kiss/tests/agents/sorcar/test_active_worktree_path_remap_review_bugs.py` (11 tests):

- 3 × Bug 1: relative-path Edit / Write / Read with unrelated host cwd
- 1 × Bug 2: Edit with file deleted from worktree branch must not mutate main
- 1 × Bug 3: Read with file deleted from worktree branch must not return main content
- 3 × Bug 4: Bash `> /main`, `sed -i /main`, `rm /main` all refused
- 2 × Bug 4 positive controls: Bash inside worktree works, Bash outside repo works
- 1 × Bug 6: end-to-end auto-commit + squash-merge actually advances main

### Verification

- 11/11 new tests pass after fix (0.96s).
- 449/449 related tests pass across 8 parallel splits (worktree, useful_tools, pwd_prefix, read_tool, edit_crlf, empty_oldstring, read_binary).
- `uv run check --full` passes (lint, mypy, pyright, mdformat, syntax, VS Code extension check).

### Round 3 — gpt-5.5 re-review of commit `339c7e99`

Examined diff carefully:

- `_absolutize`: sound; only triggers with `work_dir` + relative path.
- Bash guard regex: anchors at path boundary; allows worktree-prefixed paths; refuses sibling worktrees (correctly prevents cross-tab leak); refuses parent-repo paths.
- Edge cases noted but NOT actionable: Windows separator (Unix-first project), symlinked parent paths (pre-existing limitation), heredoc false positives (acceptable trade-off).
- Read/Write/Edit unconditional remap: correct; subsequent `is_file()` / `exists()` produce honest errors against the worktree path rather than leaking main content.
- Stale-fallback `elif`: correct mutual exclusion with active remap.
- Bash guard placement: covers both `_bash_streaming` and `_spawn`.

Verdict: **no new actionable bugs found**. Loop terminates.

### Model usage

- gpt-5.5 (non-codex): Round 1 review, Round 3 re-review.
- claude-opus-4-7: Round 2 tests + fix implementation + parallel test execution.
