---
title: KISS Sorcar paper (papers/kisssorcar) verified facts and tool quirks
uuid: 434b9dc7-b5c0-45ef-aa06-01bc7c3a5f67
summary: Verified provenance facts for kiss_sorcar.tex/ks_assistant.tex and the check_paper
  relative-path quirk.
created: '2026-09-26T21:42:15Z'
updated: '2026-09-26T21:42:15Z'
---
- check_paper tool: pass the ABSOLUTE .tex path inside a worktree; a relative path read a different (stale) checkout (2026-09-26).
- HydraKV starting engine (1.85 Mops/s) was written by KISS Sorcar task 3dbc2309065b49859167eda3a4be6f2c (claude-fable-5, $12.26, 45 steps), NOT Claude Code, despite ks_assistant.tex saying so. Verified in ~/.kiss/sorcar.db events (Write of ./tmp/hydra.cc).
- HydraKV case-study set (evidence/mine_case_studies.py) now has 8 tasks incl. 3dbc2309 and ca012ee2 (bug hunting); read-only review sub-agents = 11 (763f68b5 is a developer child, excluded by requiring review AND read-only in prompt).
- ablation/prompts/SYSTEM_LITE.md must stay the frozen 630-word copy (c7f53054f^); a repo-wide {{IDENTITY}} change had altered it.
- Worktrees are used only for tasks the pre-run classifier marks as development (worktree_sorcar_agent.py use_worktree and classification.is_development).
- Edit/Write refuse unread files since 2026.9.19 (commit 2d6a12494); DB and ablation data predate it.
