---
title: Auditing papers/kisssorcar numbers against the repo
uuid: db99e44c-77a4-45f8-b1b6-b33519dc926a
summary: How to recompute and diff the kiss_sorcar.tex / ks_assistant.tex macros (ks_numbers.py
  needs uv run; which evidence file backs each macro family; known drift).
created: '2026-09-26T19:55:13Z'
updated: '2026-09-26T19:55:13Z'
---
# Auditing the KISS Sorcar paper numbers

- `python ks_numbers.py` fails with ModuleNotFoundError (pydantic); run `cd papers/kisssorcar && uv run python ks_numbers.py` (it imports `kiss.core.models.model_info` for the provider count). `uv run python ks_tb2.py` prints the Tb* macros.
- Fast diff: extract `\newcommand{\X}{v}` from the tex (lines 1-330) and from the script output, `join` on the name and print rows whose values differ. Values with spaces (`Pi 0.87.1`, `19.9 billion`) need a split-in-half compare.
- Macro families and their sources: Db* -> evidence/db_metrics.json (mine_sorcar_db.py); Rv* -> evidence/review_verdicts.json; Ab* -> ablation/results/summary.json and ablation/probes/summary.json; AbReviewFound (15) is only derivable from ablation/probes/<task>/NOTES.md (which rep reported a real finding); Bo* -> projects/bespoke_tpch_x4/results/*.json + README.md; Hk*/CsHk* -> projects/kv_adversarial/{refnode_rerun_jul21/scored_fixed_{A,B}.out, refnode_workloads_jul21/README.md, WORKLOAD_HARDENING.md, DISCOVERY_LOG.md, AUDIT2.md, AUDIT3_FIXES.md} and evidence/case_studies.json; Loc* -> ks_numbers.significant_lines over python_files(...).
- `DbReviewsKilledJuly` (1,914) has no key in db_metrics.json; needs the sorcar.db snapshot.
- Provider categories = distinct labels of `get_model_provider()` over MODEL_INFO.json keys (9 at 2026.9.24: OpenRouter, OpenAI, Together, Gemini, Claude Code CLI, Anthropic, Moonshot, Codex CLI, Z.AI).
- Channel taxonomy: 45 `*_sea.py`; `agent_dispatch._NON_CHANNEL_MODULES = {a2a_sea, ask_sea, oai_sea}` -> 42 channel+service agents (32 messaging/device + 10 service per third_party_agents/README.md).
- SAMPLE_TASKS.md ships AI discovery, optimization, GEPA prompts; no adversarial-testing sample prompt (that procedure lives only in SYSTEM.md).
- As of 2026-09-26 HEAD 98a5a859f: ks_assistant.tex macros (ChannelAgents 43, LocCore 25,439, LocPackage 71,152, ModelCatalog 688, "182 files") are stale relative to ks_numbers.py; kiss_sorcar.tex matches the script except line-count drift from later commits (LocPackage 71,556; LocServer 16,910; LocTests 10,036; LocPackageNonBlank 120,168).
