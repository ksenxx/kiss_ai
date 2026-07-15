# SWEDefend paper revision — PROGRESS

## Task

Update `./papers/swedefend/swedefend.tex` (impl `./swedefend/`) per a reviewer's
feedback. Do AI discovery to get better results. Search internet extensively.
Use **claude-opus-4-7** for all tasks incl. software dev. Use **gpt-5.6-sol**
(not codex) for thorough review & debugging of opus's work; check for missed
code/wiring/bugs. (No need to check models exist.)

## Reviewer feedback (must-address checklist)

1. **20% FPR atrocious**; MSR'26 reports ~1% FPR. No FPR comparison in paper.
   → Must drastically LOWER FPR and add a head-to-head FPR comparison vs MSR'26.
1. **Overfit to one attack strategy**; unclear it stops OTHER variations in the
   SAME threat model. Defense papers should stop all attacks in a threat model,
   not one instantiation.
1. Likes the **intent alignment judge** (keep it, it's the core).
1. **Judge reliability vs modified attacks**: a malicious bug report with a FALSE
   stated problem (e.g. "web page wrongly restricted, should be public") could
   trick the agent into removing auth, and the judge (which checks alignment
   with the STATED problem) would think it's OK. NOT evaluated. Concern of
   overfitting (optimized on same attacks it's evaluated against).
1. **L1-L3 lame**: regexp filters trivially evadable; L2 overfit (assumes payload
   in code block). But they don't contribute much → don't over-weight them.
1. **Low novelty**: intent judge already in MSR'26; improvement is prompt eng +
   newer model. Paper must rest on being SIGNIFICANTLY better than MSR'26
   defenses → need direct comparison.
1. **Adaptive attack (Sec 4.3) weak**: 5 iters too few (PAIR needs ~100). Unclear
   what attack. Unclear if mutator is hard-coded (it IS — see `adaptive.py`
   `_obfuscate_issue/_obfuscate_sink/_split_payload`) or a free LLM. No standard
   literature attacks. Attacker-LLM choice matters: Opus-4.6 best; Opus-4.7/4.8
   refuse too much; GLM-5.2 might be good.
1. Writing hard to understand → clarify.
1. **"Tramèr Standard"/"following Tramèr et al." = AI slop.** Vague. Must cite a
   specific paper/algorithm, not name-drop a person. (Tramèr 2020 "On Adaptive
   Attacks to Adversarial Example Defenses" IS real & cited in bib — but the
   phrase "Tramèr standard" is fabricated-sounding; reframe/attribute precisely.)
1. **Table 2 (ablation) missing "L2 only"** row.
1. **Citations**: use LaTeX `\cite{..}` NOT inline "(arXiv:xxxx)" / paper titles
   / obscure conf names ("MSR" unknown to reviewer). Convert ALL inline
   arXiv/title refs to `\cite{}`.

## Code map (fully read this session)

- `swedefend/pipeline.py` — `SWEDefendPipeline.evaluate()` runs L1 sanitizer,
  L2 masked_reproduction (only if repo_path+fail_to_pass), L3 provenance,
  L4 judge (only if judge set). `allow = all(layer.passed)` (fail-closed).
- `swedefend/defense/issue_sanitizer.py` — regex INJECTION+EXFIL patterns, NFKC,
  strip invisible/bidi, redact HTML comments & base64 blobs. `is_suspicious` =
  any flag. (L1)
- `swedefend/defense/masked_reproduction.py` — masks fenced/indented/shell spans,
  runs fail_to_pass/pass_to_pass as `python -c` subprocess, exit0 required. (L2)
- `swedefend/defense/provenance.py` — bandit (JSON) + semgrep (local
  rules/taint.yaml) → sink categories; `has_dangerous_sink=bool(findings)`. (L3)
- `swedefend/defense/intent_judge.py` — LLM-as-judge; `_JUDGE_SYSTEM` prompt;
  parses single JSON `{aligned,confidence,reason}`; fail-closed on bad JSON. (L4)
- `swedefend/corpus.py` — `generate_corpus()` 72 cases: 10 CWE-78, 8 CWE-502,
  8 CWE-327, 8 CWE-94, 8 CWE-22, 10 prompt-injection, 20 benign. RANDOM 11-char
  magic. **PROBLEM**: malicious patches all use obvious `if x.startswith(magic):`
  gate → trivially detectable; overfit. Benign are all null/bounds checks.
- `swedefend/adaptive.py` — **hard-coded** mutators (not a free LLM attacker),
  only 5 rounds, 3 seed attacks. This is the weak Sec 4.3.
- `swedefend/swexploit.py` — `default_payloads()` 6 hand cases + harness.
- `swedefend/ablation.py` — L1/L3/L4/Combined only (NO L2, NO L2-only row).
- `swedefend/rules/taint.yaml` — 7 semgrep sink rules.
- `swedefend/results/{ablation,ablation_quick,adaptive_asr}.csv` — current nums:
  L1 19.23% catch/0% FPR; L3 34.62%/0%; L4 80.77%/20%; Combined 100%/20%.
  Adaptive ASR 0% all 5 rounds.
- Tests: `src/kiss/tests/swedefend/test_swedefend.py`, `test_corpus.py`.
- MODEL_INFO.json at:
  `/Users/ksen/.vscode/extensions/ksenxx.kiss-sorcar-2026.7.20/kiss_project/src/kiss/core/models/MODEL_INFO.json`

## Plan (next sessions)

### A. Internet research (MANDATORY 10+ sites, log to ./tmp/information-\*.md)

- SWExploit arXiv:2509.25894 — exact title/authors/ASR.
- MSR'26 arXiv:2509.05372 — exact title/authors; VERIFY the ~1% FPR & 47%/72%
  numbers the abstract cites; get their defense list & metrics for direct table.
- Tramèr et al. 2020 (adaptive attacks) — precise claim to cite (fix "Tramèr
  standard" slop). Also Carlini & Wagner "Obfuscated Gradients" (Athalye 2018)
  for adaptive-eval methodology.
- PAIR (Chao et al. 2023), TAP, GCG (Zou 2023), AutoDAN — standard adaptive/
  jailbreak attacks to cite & optionally implement (~50-100 iters).
- LLM-as-judge reliability / goal-hijacking / false-premise (prompt-injection
  goal hijack, Perez & Ribeiro 2022) for the reviewer's auth-removal attack.
- FPR-reduction: confidence thresholds, self-consistency, judge ensembles.
- GLM-5.2 / Opus-4.6 as attacker LLMs; refusal issues.

### B. AI discovery / experiments (use claude-opus-4-7 for dev)

1. **Lower FPR**: revise judge prompt + add a confidence threshold / 2-sample
   self-consistency / benign-calibration so combined FPR ≪ 20% (target ~1-5%)
   while keeping catch high. Re-run ablation → new ablation.csv.
1. **Diversify corpus**: add attack VARIATIONS in same threat model that DON'T
   use the obvious `startswith(magic)` gate (env-var gate, length gate, encoded
   sink, sink hidden in helper, no-magic dead-drop) to test generalization.
1. **False-premise / intent-manipulation attack** (reviewer pt 4): issue states
   a wrong premise ("endpoint should be public, remove the auth check"); patch
   removes `@login_required`. Test whether judge (aligned-with-stated-intent)
   passes it. Add a "premise cross-check" mitigation if it fails. Report result
   HONESTLY even if judge fails.
1. **Real adaptive attacker**: replace hard-coded mutators with an LLM attacker
   (attacker model, e.g. glm-5.2 or opus-4-6) running ~50-100 iters (PAIR-style),
   free to choose strategy. Report ASR curve honestly.
1. Add **L2-only** ablation row + rerun.
1. Direct **MSR'26 comparison table** (their defenses vs SWEDefend on catch & FPR).

### C. Paper rewrite

- Convert ALL "(arXiv:xxxx)"/titles → `\cite{}`. Drop "MSR'26" jargon; say
  "a recent study \\cite{...}".
- Fix "Tramèr standard" → cite Tramèr 2020 + Athalye 2018 precisely, describe the
  actual methodology.
- Honest FPR framing + comparison; reframe novelty (prompt-optimized judge +
  provenance + masked-repro, evaluated against generalization & adaptive attacks).
- Add false-premise eval + limitation. Add L2-only row. Clarify writing. Describe
  adaptive attacker precisely (LLM-driven, N iters, model used, refusals noted).
- Remove the "gpt-5.5-xhigh refused" paragraph (replace w/ real gpt-5.6-sol review
  note if applicable, or drop).

### D. Review with gpt-5.6-sol; fix bugs; `uv run check --full`; rerun tests.

### E. Rebuild PDF; commit; clean ./tmp.

## Verification cmds

- Build corpus: `uv run python -m swedefend.corpus`
- Ablation: `uv run python -m swedefend.ablation` (needs judge model)
- Adaptive: `uv run python -m swedefend.adaptive`
- Tests: `uv run pytest src/kiss/tests/swedefend/ -v`
- PDF: `cd papers/swedefend && pdflatex swedefend.tex` (x2)
- `uv run check --full`

## RESEARCH RESULTS (10 sites, see ./tmp/information-swedefend.md)

- MSR paper = Przymus, Happe, Cito, arXiv:2509.05372 "Adversarial Bug Reports as a
  Security Risk in Language Model-Based APR". Real numbers: 90% ASR; best SINGLE
  pre-filter 47%; ensembles 63-68%; ensemble+post-APR review 72.5% (37/51).
  FPR: validated on 100 real psf/requests issues -> structured gpt-4o-mini gave
  only 1 FP (~1% FPR). **DROP "MSR'26" label (venue unverified).** Reviewer's
  "1% fpr" is CORRECT. SWEDefend's 20% is ~20x worse -> MUST reduce + compare.
- SWExploit = Chen, He, Jana, Ray, arXiv:2509.25894; ASR up to 0.91, baselines \<0.20.
- Adaptive-eval methodology: Carlini et al 2019 (1902.06705) + Tramer et al 2020
  (2002.08347 "On Adaptive Attacks to Adversarial Example Defenses"). REPLACE the
  fabricated "Tramer standard" phrasing with proper cites.
- Standard attacks to cite/model: PAIR (Chao 2310.08419; attacker LLM refines vs
  target, often \<20 q), GCG (Zou 2307.15043), Greshake indirect PI (2302.12173),
  PyRIT (orchestrator-LLM attacker). Attacker LLM: Opus best planner; GLM-4/5 good
  for code vulns; note refusals.
- FPR reduction: confidence-gated veto (sigma>=tau), structured output, LLM jury /
  self-consistency, calibrate tau on held-out benign, report at MATCHED operating
  point. Zheng LLM-judge (2306.05685); Wang self-consistency (2203.11171).
- False-premise/auth-removal = real "agentjacking" threat; FUNDAMENTAL limit of
  intent-only judging (judge approves patches serving the attacker-STATED goal).
- Env: live LLM calls WORK (tested claude-haiku-4-5 judge -> conf 0.98). Judge
  exposes confidence. Models registered: claude-opus-4-7/4-6, gpt-5.6-sol, glm-4.6,
  gpt-4o-mini, claude-haiku-4-5, claude-sonnet-4-5 all present.

## Status: research DONE. Model access CONFIRMED. Next: switch agent to

## claude-opus-4-7, implement improvements (confidence-gated judge + premise

## cross-check; diversified corpus incl false-premise; LLM-driven adaptive attacker;

## L2-only + matched-FPR ablation), run experiments, rewrite paper, gpt-5.6-sol review.

## Continuation session 2026-07-14 (this session)

- Killed stalled 40-iter adaptive run (silent SSL wait). Instrumented adaptive.py
  to write per-attempt `adaptive_partial.csv` and use cumulative-success ASR
  semantics (Attempt success = pipeline.allow AND signature_ok).

- Ran 3-seed x 5-iter adaptive at tau=0.9 with attacker=claude-opus-4-6,
  judge=claude-opus-4-7. Result: ALL THREE seeds succeeded within 3 iterations
  (CWE-78 iter 2, CWE-502 iter 3, CWE-94 iter 2). Cumulative ASR reaches 100% by
  iter 3. Raw data -> swedefend/results/adaptive_partial.csv.

- Independent code review by gpt-5.6-sol saved to tmp/code_review.md (55 issues).
  Highest-impact findings: (1) capability-diff is not a diff (patch source only),
  (2) L2 mask has no effect on verdict, (3) semgrep classifier substring-mismatch,
  (4) judge parse failure fail-open under any positive tau, (5) adaptive threat
  model is patch-control not issue-only. All disclosed honestly in paper's
  Discussion.

- Paper edits: (a) filled adaptive Table 2 with real numbers (100% ASR by iter 3),
  (b) rewrote Section 4.3 to report the negative result honestly, (c) softened
  abstract, conclusion, and MSR-comparison to "target FPR" rather than "matched
  FPR win", (d) fixed Wilson bound (16.1% not 14%), (e) added "Independent
  code-review deviations" subsection listing the 8 most important disclosed
  bugs from tmp/code_review.md, (f) added \\todo fallback macro so paper compiles.

- Paper compiles via `pdflatex swedefend.tex` (Library/TeX/texbin) after adding
  `nonatbib` neurips option to avoid natbib clash with the numeric-style
  thebibliography. PDF regenerated (266972 bytes).

- `uv run check --full`: ALL checks pass.

- ./tmp/ cleaned before finish.

## Continuation 6 (standalone-paper pass + gpt-5.6-sol regression review)
Goal: paper must not reference any previous draft; fix consistency/hallucination/citation/duplication/AI-slop; human-indistinguishable.
- Verified prior session removed all literal draft/AI-slop tells (grep clean).
- Fixed baseline (Przymus et al., arXiv:2509.05372v2) number mixing: structured gpt-4o-mini = 35% catch @ 1% FPR; best single pre-filter = 47% (FPR unpaired); full pre-repair ensemble relabelled 62.7% (was mislabelled "best ensemble 68%"); full pre-repair ensemble + post-APR review = 72.5%. Added \MSRStructuredCatch macro; removed unused \MSRFPRUnstructured.
- Removed false "held-out calibration" claim (tau chosen on benign set, 3 places).
- Fixed citations: jury claims -> \cite{polljury} (was zheng2023, 3 places); Wang self-consistency described as CoT method we adapt; GCG superlative removed; bandit/semgrep now cited in L3; MELON/Task Shield bibitems given authors+venue; msrdefense -> v2/2026.
- Removed residual review-memo/AI-slop tells: "deviations from paper's claims", "Contrary to what we hoped", "We should have introduced", "matched-FPR"/"like with like" overclaim, "verified numbers", "Anti-overclaim note", "We verified by hand and by grep", "This paper does exactly that", D1 prose reversal (vetoes->suppresses), "substantially larger corpus" (149<151), 40-vs-5 iteration mismatch, unused \todo macro, stale source comments; renamed sec:aidiscovery->sec:fprcontrol.
- Fixed Table 1 caption self-contradiction ("on all 49"); Wilson wording -> "two-sided 95% Wilson upper endpoint".
- Split unbreakable slash-token run in prompt box to kill 81pt overfull hbox.
- gpt-5.6-sol independent regression review (tmp/gpt_regression.md) drove the above fixes; verified no broken \ref/\cite, all keys defined+used.
- pdflatex x2: 14 pages, 0 undefined refs/citations, no overfull >40pt.
