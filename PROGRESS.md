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

## Session: remove bug-found-and-fixed narratives (reader-facing revision)

User request: "In the paper there is no need to describe what bug you found
and fixed. The readers only want to know how the engine works."

Changes to papers/kvstorepaper/hydra_kv.tex (no number/constant changes):
1. Abstract: hardening/audit/deployment tasks now described by the subsystems
   they added (crash recovery, fail-soft I/O, compaction, x-records,
   non-blocking deletes) and their scores; removed the resurrection-race,
   fingerprint-alias-bug, four-defect, and "eight further bugs" narratives.
2. Intro/contributions: contribution 3 no longer names the delete/read
   resurrection race; contribution 5 drops the coverage-exposed race;
   contribution 6 drops the defect list; non-contributions paragraph
   generalized.
3. Design (Sec 4): fingerprint-alias, overflow-map, pin-ownership,
   Delete/reaper, and clean-shutdown paragraphs rewritten as present-tense
   mechanism descriptions with design rationale (e.g., why landing-time pin
   re-check and tombstone relocation are load-bearing) instead of bug history.
4. Sec 4.1: audit gap list condensed; "O_TRUNC is gone"/"eleven abort()
   sites are gone" replaced with positive statements of current behavior.
5. Sec 4.2 fully rewritten: "Deployment hardening: a second audit and an
   all-workload sweep" -> "Deployment subsystems: oversized values, deletes,
   and shutdown" with four mechanism paragraphs (x-records; non-blocking
   deletes + tombstone_slot guards + compaction tombstone restaging + epoch
   retry; honest accounting/recovery/shutdown; validation across workloads
   with the A/B throughput controls). Audit's genuineness verification kept;
   defect reproductions removed.
6. Sec 5.4 retitled "Correctness and testing"; removed the compactcold
   four-step race story and the stale-position tombstone tale; tests are now
   described by the contracts they verify. Sec 5.1 "after all eight fixes" ->
   "after the campaign".
7. Sec 6: verbatim prompts kept (they are the spec); Task 3/4/5/6 narratives
   condensed to process level (findings counts, review rounds, gates);
   observations (2), (4), (5) generalized away from specific bugs.
8. Limitations: dropped 480K-drop and audit-flagged-false-NotFound historical
   parentheticals; Conclusion: "steered by two independent audits and an
   all-workload sweep", defect framing removed.

Build: pdflatex x2 (at /Library/TeX/texbin, not on default PATH) — 0 errors,
0 overfull, no undefined refs; PDF now 21 pages (was 22). Verified via
pdftotext: 0 hits for "resurrection race"/"eight bugs"; only bug-word hits
remaining are inside the verbatim task prompts.

## Session 3 — consistency read-through after bug-narrative removal — COMPLETE

Task: full read-through of the revised hydra_kv.tex to confirm present-tense
mechanism descriptions still logically justify each design choice without the
removed bug narratives; fix dangling references / broken causal transitions.

Method: read the whole paper end to end; grepped for dangling causal phrases
("described above", "the fix", "discovered", "uncharged", etc.); verified
every suspect claim against ground truth (hydra.cc, kvstore_interface.h,
AUDIT2_FIXES.md, PROD_READINESS.md, WORKLOAD_HARDENING.md).

Findings and fixes (4 edits in hydra_kv.tex):
1. Sec 4 position index: "index repair" was a dangling mechanism name (no
   such path in hydra.cc). The three newest-LSN-wins paths are miss_read,
   the io_uring completion state machine, and recover_log's index rebuild.
   -> "All paths that resolve a key to a slot (the synchronous miss path,
   the io_uring completion path, and recovery's index rebuild)".
2. Sec 4.1 crash recovery: "Oversized values persist through a sidecar file"
   contradicted Sec 4.2 (x-records "need no sidecar"). In the final engine
   the sidecar persists overflow-MAP contents. -> "Overflow-map contents
   (Section 4) persist through a sidecar file ... a key with both sidecar-
   and log-resident versions recovers to the true last write."
3. Limitations: removed "The inline index-capacity overflow path remains
   uncharged by design..." — stale pre-AUDIT2 residual (PROD_READINESS era);
   AUDIT2_FIXES Bug 2 + hydra.cc overflow_put/overflow_absorb show every
   absorb path (disk-full, pinned victim, index/position exhaustion) is now
   charged and budget-capped with honest counted rejection.
4. Sec 4 memory tier: "an honest error status" was inaccurate (harness
   Upsert returns void; rejection is surfaced via rejected_mem/
   rejected_oversize prod stats) -> "surfaced through rejection counters".
   Plus clarity: Sec 4.1 "the overflow keys are dropped" -> "the excess keys
   are dropped ... (and recover_ok reports the failure, Section 4.2)" to
   avoid collision with the overflow map and tie to the recover_ok contract.

All other transitions checked and found self-contained (tombstone-relocation
rationale, pin-ownership landing-time re-check, pre-punch epoch retry,
x-record buffered-descriptor routing, A/B controls, Section 6 process
narratives intentionally retained).

Build: pdflatex x2 (/Library/TeX/texbin) — 0 errors, 0 overfull, 0 undefined
refs, 21 pages. pdftotext confirms all edits present and stale phrases gone.

## Session: AI-slop / AI-tell review of papers/kvstorepaper/hydra_kv.tex (2026-07-21)

Task: review the paper for AI slop or other issues that make it look AI-generated.

Research (10 web sources logged in tmp/information-aislop.md, deleted after
use): Wikipedia:Signs_of_AI_writing (+AI-generated-comments), AI slop article,
Kobak et al. excess-vocabulary (Sci. Adv. 2025), Reinhart et al. rhetorical
styles (PNAS 2025), Simon Willison slop post, Sean Goedecke em-dash analysis,
Google SERP, The Ringer and Night Water em-dash counterpoints.

Diagnosis of the paper:
- LLM marker vocabulary (delve/showcase/underscore/pivotal/tapestry/leverage/
  crucial/meticulous...): ABSENT. No puffery, no "-ing" superficial analyses,
  no boldface/markdown/curly-quote artifacts, no formulaic transitions.
- Real tells found: (a) ~30 prose em dashes, mostly paired parenthetical
  "---X---" constructions (the most recognized tell); (b) high density of
  negative parallelisms ("X, not Y" — WP:AIPARALLEL); (c) rhetorical fragment
  flourishes ("That table's shape is the design speaking", "The measured fact
  is the baseline itself"); (d) repeated signature words: honest x9,
  exactly/precisely x17, load-bearing x2; (e) ~20 single-word rhetorical
  italics (\emph{not}, \emph{before}, \emph{are}...); (f) acknowledgments
  typo "(DESC0021982,)".

Fixes (43 edits, all in hydra_kv.tex; numbers/tables/citations untouched):
- Every paired prose em-dash parenthetical converted to parentheses, commas,
  colons, or split sentences (abstract, intro, contributions item 5, design,
  prod, deploy, testing, observations, limitations, related work,
  conclusion). Remaining "---" are % banners and Table 1 placeholders.
- Both "load-bearing" uses, "the design speaking", "verified the strong way",
  "made loud", and the "not only X but Y" construction reworded.
- exactly/precisely cut where intensifying (9 removals); honest -> counted/
  accounted where it described mechanisms (4), kept the 2 thematic uses.
- Negative parallelisms reduced ("stated precondition, not a discovered bug",
  "relocated, never dropped", "backpressure, not growth", etc.).
- 17 rhetorical single-word italics removed; term-introduction italics kept.
- Acknowledgments comma typo fixed.

Build: pdflatex x2 — 0 errors, 0 overfull boxes, 0 undefined refs, 21 pages.
pdftotext verified: old phrases 0 hits, new phrasings present.

## Session: confine engine-improvement chronology to Section 6 (2026-07)

Task: "Can you not describe all the improvement steps of the engine in paper
except in the section 6? The readers want to know about the engine itself and
not how it got hardened or correct or performant over time."

Changes to papers/kvstorepaper/hydra_kv.tex:
- Abstract: removed the task-by-task narrative (hardening task, production-
  readiness task, second audit re-run, deployment-hardening tasks, per-step
  scores); replaced with one verification-evidence sentence (test matrix,
  cross-model review, fault injection, all-workload sweep, two audits with
  the genuine-and-unhacked verification at 5.51). "designed, implemented,
  debugged, hardened, and finally made production-ready" -> "designed,
  implemented, and tested".
- Intro: six-task enumeration condensed to two sentences (performance problem
  + adversarial loop; four correctness tasks / two audits); results paragraph
  now reports only the final engine (5.51) plus the v4c variant matrix;
  contributions merged 6 -> 4 (hardening/production/deployment protocol items
  folded into one "verification protocol" contribution); non-contributions
  paragraph de-chronologized.
- Section 4: LOC-growth parenthetical removed; 4.1 and 4.2 openers rewritten
  as descriptions of present subsystems with pointers to the released audit
  artifacts and Section 6; "Why the scored number survived" -> "Cost on the
  scored path" (A/B statement instead of revision comparison); 4.2's
  "Validation across workloads" paragraph deleted (sweep facts moved to 5.1,
  audit-verification details moved to Section 6 Task 5, A/B control already
  in Section 6 Task 5).
- Section 5: Table 1 reduced to FASTER + one bold HydraKV row + v4c variant
  block (intermediate-revision rows removed; their numbers remain in Section
  6 prose; caption states the 5.50-5.52 re-score band); 5.1 rewritten around
  the final engine with a stability statement; 5.2's V1 story retold as a
  design-space fact with a Section 6 pointer; 5.3's goal-raising narrative
  and the whole micro-optimization-round paragraph removed (ceiling argument
  kept, now cites 5.51); 5.4 rewritten: final suite only (1,980 lines / 27
  tests / 89.3% lines), tests regrouped by subsystem (recovery+fault,
  compaction, delete/capacity/value-size), tests-first chronology dropped;
  subsection 5.5 "Negative results" deleted from the evaluation.
- Section 6 (only place with chronology, unchanged prompts): Task 4 gained
  the full micro-optimization round and the six-run median/runs; Task 5 intro
  gained the audit verification details (RSS in budget, true O_DIRECT,
  verbatim CRC values, "not reward-hacked, not copied"); new paragraph
  "Accepted and rejected ideas." (label sec:negative preserved) holds the
  moved negative results plus the accepted-side calibration (first engine
  1.85, remaining 3x across redesigns, engineering-log caveat); fixed the
  intra-section self-reference in Task 2.
- Limitations: opening, prod-grade-bar, residual-risk, and evaluation-scope
  paragraphs de-chronologized (evaluation scope now cites the 5.50-5.52 band
  and Section 6 instead of per-task re-scores).
- Conclusion: "and then, steered by..., gave it..." -> "carries ...";
  "7 Mops/s again at production time" -> "a second time".
- Removed now-unused macros \HydraFinal, \EngineLoc, \EngineLocProd,
  \CovBranchesProd.

Build: pdflatex x2 - 0 errors, 0 overfull, 0 undefined refs; 20 pages (was
21). pdftotext: removed narrative phrases 0 hits; the only "hardening task"
mention left is inside Section 6's ideas paragraph. Visual check of page 9
(Table 1 + 4.2 paragraphs) OK.

## Session: de-emphasize methodology in Abstract/Introduction (2026-07-21)

Task: no elaborate discussion of Adversarial AI Discovery / KISS Sorcar
outside Section 6 plus one introduction paragraph; drop the intermediate
"first engine beat FASTER by ~2x then was OOM-killed" story from the
Abstract; readers should meet the engine first.

Changes to papers/kvstorepaper/hydra_kv.tex:
- Abstract rewritten: opens with the engine ("a larger-than-memory key-value
  store for skewed point-operation workloads"), reports 5.51 Mops/s (5.9x)
  plus the 3.87-5.67 adversarial-variant range in one results sentence,
  keeps the architecture list and the test/fault-injection/audit evidence,
  and ends with a single provenance sentence ("designed, implemented, and
  tested almost entirely by an AI agent framework~\cite{kisssorcar} through
  a sequence of natural-language tasks; the development record is reproduced
  in the paper"). Removed: the adversarial-AI-discovery methodology framing,
  the two-agent-activities description, the first-engine OOM story, and
  "keeping optimizing agents honest".
- Introduction: paragraph 2 is now the single designated methodology
  paragraph ("HydraKV was not written by hand. It was produced by KISS
  Sorcar..."), condensing the six tasks, the reward-hacking motivation, the
  adversarial discovery loop, the correctness apparatus, and a pointer that
  Section 6 documents the process and evolution, closing with "The rest of
  the paper describes the engine as it stands." Paragraph 3 now starts
  "HydraKV is" (was "The surviving artifact, HydraKV, is"). Results
  paragraph: agent-refused-page-cache sentence replaced by an engine fact
  (O_DIRECT log). Contributions cut from 4 to 3: the "A methodology" bullet
  deleted (its held-out-variant element folded into the verification bullet,
  which also lost "tests required to fail first" and "cross-model review");
  "the agent's audits" -> "the audits". Scope paragraph's closing claim is
  now engine-focused (conservative composition + released evidence), not
  "a carefully instructed agent... kept honest".
- Section 3 caching paragraph: "every heavier policy the agent tried" ->
  "every heavier policy tried during development"; "the answer the agent
  converged on" -> "the answer".
- Section 5.3: "the agent's documented response" -> "the reason";
  "every admission policy the agent tried" -> "every admission policy tried".
- Conclusion: development-hygiene list condensed to one pointer sentence
  ("The development process, and the experimental hygiene that kept it
  honest, are documented in Section 6...").
- Section 6 untouched; Related Work's AI-driven-code-discovery positioning
  kept (citation-bearing literature comparison, not a development narrative).

Build: pdflatex x2 (/Library/TeX/texbin; note: pdflatex is NOT on the default
PATH in this environment) - 0 errors, 0 undefined refs, 0 overfull; 20 pages.
pdftotext verification: all removed phrases 0 hits; all new phrasings present.
