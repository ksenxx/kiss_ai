# PROGRESS — HydraKV paper update (papers/kvstorepaper/) for the latest engine

Task: update papers/kvstorepaper/hydra_kv.tex for the post-publication engine
work (AUDIT2 remediation + multi-workload hardening) and rebuild the PDF.

## DONE (single session) — COMPLETE

1. Gathered facts from the repo (no engine changes made):
   - `projects/kv_adversarial/AUDIT2.md` — second independent audit (July 20):
     verified 5.51 Mops/s median (5.9x FASTER) as genuine/unhacked, retracted
     its own overstated findings, reproduced 4 deployment defects on hardware
     (Upsert+Delete deadlock, disk-full OOM 12.1 GiB, silent recovery loss
     with recover_ok=1, Delete ~160x slower than Upsert).
   - `AUDIT2_FIXES.md` — task-5 remediation: non-blocking Delete with
     landing-time tombstone reseal (updel2), fully charged overflow map with
     rejected_mem backpressure (memfull), recover_ok=0 on drops (recoverdrop),
     background delete reaper (delcost), + 4 reviewer fixes (reaper inflight
     race, bounded 64K reap queue, pre-punch compaction-epoch bump, sidecar
     reload cap). Scored 5.50 medians (two suites) with SAME-DAY A/B control
     of the pre-fix engine also 5.50 => zero cost.
   - `WORKLOAD_HARDENING.md` + `refnode_workloads_jul21/README.md` — task-6
     all-workload campaign: 8 bugs fixed (x-records for oversized values,
     ext4 O_DIRECT EOF EINVAL bitmap, clean-shutdown data loss, compactor
     punch of just-reserved extent, recovery-scan region drops, destructor
     std::terminate, stale-position innocent-key tombstone, tombstone
     restaging at compaction). Final engine 5050e98f: scored 5.51/5.52/5.49
     => median 5.51; sweep: 0:100=5.12, 5:95=5.16, B=2.93, C=2.78, RMW=0.30,
     timeseries=0.08 (1.28M deletes, 0 resurrections); TSan compact 15/15;
     matrix 89.30% lines executed.
   - Verified code sizes: hydra.cc = 3,974 lines; test_hydra.cc = 1,980
     lines, 27 tests (counted the registry; README's "28" is off by one).
   - Retrieved verbatim Task-5 and Task-6 prompts from ~/.kiss/sorcar.db
     (task_history ids cdde00e9..., 061e7725...).

2. Updated papers/kvstorepaper/hydra_kv.tex:
   - New constants: \HydraAuditVerified{5.51}, \HydraAuditFixed{5.50},
     \HydraFinalWL{5.51} (+runs), sweep numbers, \EngineLocFinal{~3,970}.
   - Abstract: final engine 5.51 headline, x-records + delete reaper in the
     feature list, new closing narrative (second audit, task 5, task 6).
   - Intro: six tasks; final-engine results sentence; contribution #6
     (deployment-hardening loop closed by a second independent audit);
     expanded artifact footnote (AUDIT2*, WORKLOAD_HARDENING.md, benchmark/,
     refnode_*_jul21/); non-contributions updated.
   - Design: LoC updated; x-records replace the RAM-only oversized path (map
     demoted to charged fail-soft absorber); Delete = background reaper with
     bounded queue + sync fallback, Checkpoint/shutdown drain; destructor now
     lands all sessions' partials + full Checkpoint; sizing paragraph notes
     the x-record path.
   - NEW subsection 4.2 \label{sec:deploy} "Deployment hardening: a second
     audit and an all-workload sweep" (audit findings + retractions, task-5
     fixes + A/B control, task-6 eight bugs + architectural limits).
   - Eval: Table 1 gains rows audit re-run 5.51 / audit-fixed 5.50 / final
     5.51 (bold); NEW Table 2 (tab:wl) workload sweep with honest caption
     (single runs, two engine revisions); main-results text extended;
     testing section: suite growth to 1,980 lines / 27 tests, "Deployment
     tests" paragraph (updel2, delcost, memfull, recoverdrop, bimodal,
     capwarn, oversizebound rewrite, compactcold extension; TSan flake
     story; final-engine coverage 89.3% lines disclosed).
   - Sorcar section: six prompts; Tasks 5 & 6 verbatim promptboxes +
     narratives; Tasks 2--6 role split; reviewer spend note; observation (5)
     "internal rigor did not substitute for external adversaries".
   - Limitations: reaper durability contract + measured SIGKILL loss window
     (~3K/195K); recover_ok now fails on drops; residual risks refreshed
     (index floor ~8 B/key, never-GC'd index words, never-compacted x-record
     extents, compaction-overlap single retry, reaper backpressure); removed
     obsolete "Delete waits politely"; evaluation scope lists all re-scores.
   - Conclusion: six tasks, both audits, updated evidence list.

3. Built the paper: pdflatex (from /Library/TeX/texbin) twice; fixed the one
   overfull hbox by shortening the two new Table-1 row labels. Final build:
   22 pages, 0 errors, 0 overfull, no undefined refs/citations. Visually
   verified pages 1, 9, 11 (both tables), 17 (new promptboxes) via pdftoppm.

4. Cleanup + commit: removed ./tmp/kvpages renders; committed hydra_kv.tex +
   hydra_kv.pdf.
