# Cost-lever follow-up (2026-09-20)

Follow-up to `projects/speed-audit-2026-09-19` and the first 24 h KPI check: re-runs the
`cost_report` comparison over 72 h with the ablation orphans and pre-lever trees stripped, and
investigates why tool-output compaction did not lower the share of steps above 100k context.

* `FINDINGS.md` — results and recommendations.
* `compare_kpis.py` — the reusable 72 h comparison (also used by the `cost-levers-72h-recheck` cron job):

  ```bash
  .venv/bin/python projects/cost-levers-followup-2026-09-20/compare_kpis.py \
      --hours 72 --snapshot --out tmp/compare_72h_$(date -u +%F)
  ```

* `compaction_analysis.py`, `compaction_simulation.py` — session reconstruction from
  `events` and replay through `kiss.core.context_compaction.compact_tool_results`:

  ```bash
  sqlite3 ~/.kiss/sorcar.db "VACUUM INTO '/tmp/cost72.db'"
  .venv/bin/python projects/cost-levers-followup-2026-09-20/compaction_analysis.py --db /tmp/cost72.db --since 1789804587
  .venv/bin/python projects/cost-levers-followup-2026-09-20/compaction_simulation.py --db /tmp/cost72.db --since 1789804587
  ```

* `results/` — outputs of the 2026-09-20 06:24 UTC run.

## 2026-09-20 07:30 UTC update

`compaction_simulation.py` now also prices each replay at Fable 5.1 cache rates and includes the
policies implemented in `kiss.core.context_compaction` (H: new defaults, I: new defaults + cache
gate, J: drop gate only).  Results: `results/compaction_sim_gated_2026-09-20.json`,
`results/compare_72h_2026-09-20T0730.*`; discussion in `FINDINGS.md` section 4.
