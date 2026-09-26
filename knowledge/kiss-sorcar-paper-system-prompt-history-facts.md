---
title: Verified SYSTEM.md history facts used by kiss_sorcar.tex appendices
uuid: bf2902a2-fa9f-4b9e-9eb9-2b73729a76e8
summary: Git-verified dates/commits behind the kiss_sorcar.tex system-prompt commentary
  and the 9 April 2026 worktree session (leniency note, 100-test threshold, run_commands_parallel
  switch, test counts).
created: '2026-09-26T19:52:15Z'
updated: '2026-09-26T19:52:15Z'
---
# Verified SYSTEM.md / development-history facts (kiss_sorcar.tex appendices)

Checked 2026-09-26 with `git log -S` on `src/kiss/SYSTEM.md` and the April commits.

- "Leave pre-existing failures ... list them in the final summary" sentence: added 841887006 (2026-08-15), removed 74999bbb9 (2026-09-18, message "Remove leniency note for pre-existing lint/typecheck failures"). Present in released 2026.9.17 (bump 3e272c0f6), absent in 2026.9.18 (bump dccd5df63). So it was dropped *between* 2026.9.17 and 2026.9.18, not "in 2026.9.17".
- Earliest lint wording: "Run lint and typecheckers; fix all errors including pre-existing ones." (before 3771b4d0c, 2026-08-15).
- Sharding bullet: "If the number of tests is more than 100" threshold added 200f321b7 (2026-05-25), removed 50c851a34 (2026-06-28, message: "Remove conditional threshold from parallel test execution rule—always parallelize tests"); no commit evidence for "because the agent under-estimated it". Formula min(tests, max(1, cores-2)) since 3771b4d0c. run_parallel -> run_commands_parallel in 515b81759 (2026-09-19), i.e. after version 2026.9.18 and before 2026.9.19.
- Ablation `arms.py` reads frozen prompts from `papers/kisssorcar/ablation/prompts/` (SYSTEM.md version 2026.9.18), not the shipped src/kiss/SYSTEM.md; defines 12 arms (6 used in Table 4).
- Worktree session (9 Apr 2026): 825c05158 removed manual_merge (tests: 71 deletions in 3 test files, 70 in the two worktree files; 2+17=19 tests remain; source files touched: worktree_sorcar_agent.py, server.py, main.js, SorcarTab.ts, types.ts + API.md). discard() before that commit did NOT check out the original branch. 15 commits to 40cc83e03 (git diff --cached --quiet + git commit --no-edit in git_worktree.py; test_worktree_extension_workflow.py has 33 tests; total 52).
- CLI multiline session commit 71f74e13f (2026-06-28): test_cli_multiline_input.py has 9 tests; test_at_mention_picker/test_cli_repl/test_cli_panel have 12+15+13=40, so "49/49" = 40 + 9.
- Tool schema from docstrings: `src/kiss/core/models/model.py` (`inspect.getdoc`, `_parse_docstring_params`), called via KissAgent `_build_openai_tools_schema`.
