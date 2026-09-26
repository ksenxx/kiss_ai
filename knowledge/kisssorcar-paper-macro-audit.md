---
title: 'KISS Sorcar paper: ks_assistant.tex vs kiss_sorcar.tex macro audit'
uuid: f6f608dd-7025-48d0-a09f-76c56b1fc463
summary: How to compare the two kisssorcar papers' macro blocks and recompute code-size
  macros without running ks_numbers.py's writers; findings of the 2026-09-26 audit.
created: '2026-09-26T19:51:29Z'
updated: '2026-09-26T19:51:29Z'
---
# ks_assistant.tex vs kiss_sorcar.tex (papers/kisssorcar/)

- ks_assistant.tex (ICLR anonymized "KS Gov" variant, 150 macros) is a subset of kiss_sorcar.tex (249 macros). On 2026-09-26 every ks macro existed in kiss and was used in its body; only four code-size macros differed because ks was computed on older code: LocCore 25,439 vs 25,653; LocPackage 71,152 vs 71,557; ChannelAgents 43 vs 42 (counting rule changed: ks_numbers.py now excludes a2a, ask, oai); ModelCatalog 688 vs 689.
- Tables: tables/ks_tb2_table.tex and tables/kiss_tb2_table.tex differ only in the header label (KS Gov vs KISS Sorcar); figures/ks_tb2_frontier.pdf and kiss_tb2_frontier.pdf are the same plot relabeled. tab:cases and tab:hk-workloads are byte-identical in both papers.
- To recompute the code-size macros without side effects (ks_numbers.py main() writes tables and figures): set sys.modules['ks_tb2'] to a stub module, exec the file text up to `def main(` with `__file__` set, then call significant_lines / python_files on SRC. ks_tb2 imports kiss.core.models.model_info, which fails outside the uv env.
- The HydraKV "reviewer capped at 20% of budget" clause is documented for later tasks too: projects/kv_adversarial/AUDIT3_FIXES.md and WORKLOAD_HARDENING.md.
- Only prose in ks not in kiss: the verbatim "Read before modify rule -- NON-NEGOTIABLE" promptbox (kiss ablation refers to "the read-before-modify sentence" but never quotes it), the related-work clause that Claude Code "refuses to edit a file it has not read", and abstract-level caveats (turn-cap rate, agent hours, cost-indistinguishable) that kiss keeps in the body only (kiss abstract is 249 words).
