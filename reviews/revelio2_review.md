# REVELIO — Improvement-Focused Review for IEEE S&P 2027

*Audience: the authors. Goal: maximize the chance that a stronger version of this paper is accepted at S&P 2027. This is **not** a typical "summary + strengths + weaknesses + score" review. Below are 30+ concrete, actionable suggestions, organized by theme, each with **(W) the underlying problem**, **(H) the proposed change**, and **(A) the reviewer question I expect you to anticipate**. Sources used for cross-referencing claims and baselines are listed at the end.*

---

## TL;DR for the authors

REVELIO is a well-engineered two-stage agent (Haiku hypothesis-funnel → Sonnet sanitizer-grounded PoV synthesizer) with genuinely impressive numbers: 19 zero-day discoveries / 7 CVEs / ~$42 per OSS-Fuzz project, 90% recall / 0% FP on a 10-bug post-cutoff benchmark, 175 / 100 found vulnerabilities at 69% recall / 0% FP / $9.10 per find on a 100-case ARVO/CyberGym subset, all while the four-tier ablation persuasively isolates the contribution of the enforced harness and asymmetric Haiku→Sonnet routing. The submission already has more than enough technical content for S&P; the risk in front of the reviewers is **not novelty**, it is **scientific rigor** of the evaluation. Most of the suggestions below are intended to harden the evaluation so that reviewers can no longer dismiss the headline numbers as a function of (i) a permissive oracle protocol, (ii) selection bias against the baselines, (iii) ill-defined ground truth in "vulnerabilities found", (iv) a single-run measurement, (v) Anthropic-model-specific tuning, or (vi) under-cited / under-baselined contemporary work. Once those are fixed, REVELIO has a strong claim to being the first reproducible LLM-agent system that beats unaided fuzzing on hard real-world C/C++ memory-safety targets at sub-$50 per project. Get there by tightening, not by adding more case studies.

---

## A. Experimental design and comparison fairness

### A1. The chosen baselines are *general* coding agents; the *contemporary* vuln-hunting baselines are missing.
**(W)** The three head-to-head systems in §5.2 (Claude Code, Codex, Sorcar) are *generic* agentic-coding harnesses with bash/edit/finish tools. They are *not* tuned for vuln-hunting; the comparison amounts to "specialized tool vs. unspecialized tools," which reviewers will read as straw-manning. The most cited contemporary direct competitors are absent from the experimental comparison:

- **RepoAudit** (Guo et al., arXiv 2501.18160) — repo-level memory-safety LLM agent, **78% precision**, **$2.54/project**, **185 confirmed new bugs in high-profile OSS**.
- **OSS-Fuzz-Gen** (Google, actively maintained as of Mar 2026, 30+ confirmed CVEs).
- **IRIS** (Li/Dutta/Naik, arXiv 2405.17238) — LLM-assisted CodeQL; +28 / 120 over CodeQL on CWE-Bench-Java.
- **Big Sleep / Naptime** (Project Zero × DeepMind, SQLite seriesBestIndex CVE; variant-analysis seeding).
- **CyberGym** itself (Wang et al., arXiv 2506.02548) ships per-paper baseline agents whose absolute Level-0/Level-1 numbers are leaderboard-comparable; running the same agents under REVELIO's reduced protocol would let reviewers triangulate.

**(H)** Add at minimum RepoAudit and OSS-Fuzz-Gen as head-to-head baselines on the **same 7 OSS-Fuzz projects** in RQ1 and the **same 100 CyberGym subset** in RQ2. For Big Sleep (no API), reproduce its published SQLite bug as a sanity task and add a paragraph in §7 acknowledging variant-analysis seeding as a missing design dimension (not as a baseline). For IRIS, since it is Java-focused, either port a small C/C++ variant or include it only in Related Work — but address the architectural lesson (LLM-infers-specs + symbolic propagation) in the ablation (§ E2 below).

**(A)** *"What is the bug yield per project of RepoAudit on the same seven targets, under the same triage rules?"* If you cannot answer this on submission day, the paper will be rejected from S&P 2027.

### A2. The oracle-file protocol is non-standard and silently *easier* than CyberGym Level-0/1.
**(W)** §5.2 hands every system the *exact vulnerable file* (one file per case) and the harness, but not the textual description. Public CyberGym (v3, March 2026) is a *Level-1* benchmark: agent gets the codebase and a text description; *Level-0* gives no description. Neither corresponds to "exact vulnerable file given." This means REVELIO's 69% recall and the baselines' 31–53% cannot be cross-mapped to the public leaderboard, and reviewers will assume the protocol was chosen post-hoc for favorable numbers.

**(H)** Run REVELIO and *each* baseline at **Level-0**, **Level-1**, and "oracle-file" on the same 100-case subset. Publish all three columns side-by-side. If the relative ordering is stable, you have killed the protocol-fairness objection in advance. If it is not stable, you have found something genuinely interesting to discuss.

**(A)** *"What does REVELIO score on standard CyberGym Level-1 across the full 1,507-vuln benchmark? Why is the random 100-subset representative?"*

### A3. Add a vulnerable-file-localization sub-experiment.
**(W)** REVELIO's Stage 1 is run **per file**; the oracle protocol therefore hides the cost of *finding the right file* in a real repository. In the wild a maintainer does not tell you which file is vulnerable.

**(H)** Add a "full-repo scan" experiment on a random 30-project subset of ARVO: feed every C/C++ file in the project to Stage 1 in parallel, then run Stage 2 on the top-K *across the whole repo*. Report (a) precision of file localization (does the top-K contain the patched file?), (b) total token / dollar cost, (c) recall at the same per-case budget as the oracle setting. This number is what real users care about, and it is the headline you are leaving on the table.

### A4. Give every baseline REVELIO's `validate()` tool.
**(W)** The baselines crash 49% / 60% / 28% of the time with false positives because they are running unaided. REVELIO has a sanitizer-grounded `validate()` tool *and* an independent re-execution loop. This is not a free architectural lunch — it is a feature you can graft onto Claude Code / Codex / Sorcar in a one-page diff.

**(H)** In §5.2 add a variant of each baseline with `validate()` injected. If the baselines now also hit 0% FP, the framing must change to "REVELIO wins on recall, not on precision." That is still a paper. If the baselines stay high-FP, that is *also* a paper — but the reader will now believe you.

**(A)** *"How much of the 0% FP claim is from the validator and how much from the hypothesis ranker?"*

### A5. Report variance — single-shot numbers are not credible at S&P.
**(W)** Every cell in Table 5, Table 7, Table 8, Table 9, Table 10 is a single run. LLM agents are highly non-deterministic; recall variance of ±5–10 points across re-runs is normal. Reviewers cannot tell whether REVELIO 69% vs. Claude Code 53% is a real gap or a 2σ accident.

**(H)** Run each cell **at least 3 times** (5 if budget allows; ~3× the API spend, still <$1,200 total). Report mean ± stdev. Use a paired Wilcoxon test (per-case agreement) to claim significance over Claude Code. Bonferroni-correct across the 4 metrics.

### A6. The pilot sweep used to pick each baseline's best (model, agent) combo overlaps the main set.
**(W)** §5.2's "pilot sweep on 10 random ARVO cases" was used to choose each baseline's strongest model — but those 10 cases are not stated to be *disjoint* from the 100. If they are not disjoint, the baselines are tuned on a subset of the test set, which is acceptable in a typical-agent comparison but unacceptable in a paper whose contribution is "this design beats those."

**(H)** Use a fully disjoint 20-case pilot drawn from a *different* ARVO project pool. Report which (model, agent) combo each baseline ended up using on the test set.

---

## B. Metrics and ground truth

### B1. "Vulnerabilities Found = 175" headline is misleading.
**(W)** Fig. 2's headline 175 vs. 55 vs. 39 vs. 31 implies REVELIO is 3× better than Claude Code. The targeted-recall numbers in Table 8 (69% vs. 53%) are a much smaller, more honest gap. The 175 number includes off-target crashes (any sanitizer-observable bug in the file). For a sample of 100 ARVO cases, 175 > 100 means many cases produced multiple sanitizer-distinct crashes that were *not* the targeted bug, and some may be duplicates collapsed to different stacks.

**(H)** Replace the headline with **Targeted-Recall %** (= number of cases where the *targeted ARVO patch's bug was triggered* divided by total cases). Keep the 175 as a secondary "discovery" bullet with explicit dedup methodology (file × function × bug-class × patch-hunk). Apply identical dedup to baselines.

**(A)** *"How many of the 175 collapse to the same ARVO patch hunk?"*

### B2. Deduplicate via ARVO patch hunks.
**(W)** ARVO ships **vulnerable+patched commit pairs and a triggering input** for >5,000 bugs. The natural unit of "this bug was found" is "the agent's PoV makes the *vulnerable* build crash and the *patched* build not crash on the same sanitizer." This is mechanizable; you should be doing it.

**(H)** Add an ARVO-patch-differential oracle. A find counts only if the PoV crashes on `vulnerable_commit` but not on `patched_commit`. This is the strictest, most-defensible oracle in the literature and would let you state results in a single sentence reviewers cannot argue with.

### B3. "Zero false positives" excludes 16 NULL-deref / OOM / memory-leak crashes.
**(W)** §5.1's footnote acknowledges this. Reviewers reading quickly will see "0% FP" and miss the footnote. Worse, the baselines' 49/60/28% FPs are *not* filtered through the same triage. The comparison is then "REVELIO post-triage" vs. "baselines pre-triage."

**(H)** Report two FP rates uniformly: **pre-triage** (any non-targeted crash) and **post-triage** (excluding NULL-deref / OOM / memory-leak under a stated maintainer policy). Apply both rules to all systems. Cite the OSS-Fuzz / Chromium maintainer policy you are deferring to.

### B4. Add Recall@k vs. token budget Pareto curves.
**(W)** Tables 8–11 give point estimates at fixed budgets. Reviewers want the **curve**: at what budget does REVELIO match an unaided fuzzer? At what budget does it saturate?

**(H)** Plot Recall vs. cumulative tokens (and vs. wall-clock) for K∈{1,3,5,10} hypotheses and iters∈{1,2,5,10}. Overlay each baseline as a single point. This figure is the strongest possible defense of the four-tier ablation.

### B5. Triage uniqueness of "new" bugs in RQ1.
**(W)** §5.1 reports 19 zero-days / 7 CVEs. Reviewers will ask whether some of the 19 reduce to fewer root causes or to known-but-not-yet-CVE bugs that already existed in ARVO/OSS-Fuzz logs but were not yet triaged.

**(H)** For each of the 19, attach: (i) the smallest-distance commit that introduced the bug, (ii) a check that no public OSS-Fuzz issue exists for the same file × function × bug-class, (iii) maintainer confirmation that the report was treated as new (often visible in the bug tracker).

---

## C. Threats to internal validity

### C1. Post-cutoff CVE freshness is not the same as training-data freshness.
**(W)** Appendix B's "fresh CVE" benchmark relies on CVE *disclosure* date being after the model's knowledge cutoff. But the underlying *commit dates* of vulnerable code (and often the patch commits) precede the cutoff and may be in training data. The model can have memorized the patch without having memorized the CVE assignment.

**(H)** For each of the 10 fresh CVEs, report (a) vulnerable-commit date, (b) patch-commit date, (c) model knowledge-cutoff date, (d) whether the patch text appears in Common-Crawl / GitHub snapshots. The cleanest demonstration is **patch-commit > knowledge-cutoff**, not "CVE > knowledge-cutoff."

### C2. The oracle-file protocol covertly subsidizes REVELIO's multi-pass prompts.
**(W)** Stage 1 makes 4–5 LLM passes *per file* (summary, whole-file feature interactions, unchecked-arg patterns, per-function). Given the oracle file, REVELIO can afford to throw 4× compute at one file. Baselines that decide *where to look* themselves cannot.

**(H)** Either restrict REVELIO to a fixed token budget *per case* equal to the median baseline budget, or grant baselines the same `for f in case_files: multi_pass(f)` driver. Re-measure recall. This will also calibrate A3.

### C3. The Haiku→Sonnet asymmetric routing is a hyperparameter, not a finding.
**(W)** Appendix C shows that Sonnet+Sonnet is more expensive than Haiku+Sonnet for similar recall, and Haiku+Haiku is worse. The right experiment is *not* a 2×2 — it is a per-stage cost-vs-recall sweep over (Haiku, Sonnet, Opus) × (Haiku, Sonnet, Opus) plus the next family (Gemini, GPT, DeepSeek).

**(H)** Add this 3×3×N sweep. Argue from the Pareto frontier, not from a single A/B.

### C4. The 100-case subset's representativeness is asserted, not shown.
**(W)** §5.2 says "100 random cases from ARVO/CyberGym." Reviewers will ask whether the random seed produced a CWE distribution / project-size distribution / language-feature distribution representative of the full 1,507.

**(H)** Show a 2-column comparison: CWE class, project LOC, language (C vs. C++), sanitizer (asan/ubsan/msan) on the 100 vs. on the 1,507. Run an additional 200-case sample if any axis is off by >20%.

---

## D. Design alternatives that should be ablated

### D1. Variant-analysis seeding.
**(W)** Big Sleep's signature win on SQLite came from seeding the agent with a *recent fixed commit* and asking for similar bugs. REVELIO seeds nothing. For a project with N recent patches, variant-analysis seeding is essentially free recall.

**(H)** Add a third RQ: "Does seeding Stage 1 with the project's last K patched CVEs increase recall?" Compare ad-hoc hypothesis generation vs. patch-seeded hypothesis generation on the same 100-case set and on RQ1's 7 OSS-Fuzz projects.

### D2. Replace the heuristic ranker with a learned or LLM-as-judge cross-encoder.
**(W)** §3.1's ranker uses severity + reachability + confidence. This is hand-tuned. Reviewers will ask whether a learned re-ranker would change the top-K composition.

**(H)** Add an ablation that ranks the hypothesis pool with (i) the existing heuristic, (ii) Sonnet-as-judge pairwise, (iii) a tiny BGE cross-encoder fine-tuned on (hypothesis, was-real-CVE) pairs from ARVO history.

### D3. Replace Stage 2 PoV synthesis with coverage-guided directed fuzzing.
**(W)** §3.2's Stage 2 is "ask Sonnet to write a PoV input." An alternative is to take each top-K hypothesis and feed `function_name` to AFL++ as a target, with seed inputs derived from the harness corpus. This is the canonical fuzzing approach and reviewers will demand it be ablated.

**(H)** Add a 4-arm comparison on the 100-case set: (a) REVELIO Stage 2, (b) AFL++ directed for 30 min/case, (c) AFL++ + LLM-suggested input mutations, (d) REVELIO Stage 2 + AFL++ fallback after N failed iterations. The "fall-back" arm is the likely strongest combined system and a natural future-work claim.

### D4. Per-sanitizer breakdown.
**(W)** §4 reports "asan, ubsan, msan" enabled but no per-sanitizer attribution. Some of the 19 zero-days are likely UBSan-only (integer overflow / underflow) and some are ASan-only (heap OOB).

**(H)** Add a per-sanitizer column to Table 5 and a per-sanitizer recall column to Table 8. Discuss whether MSan adds anything (typical answer: a couple of bugs at 2–3× cost).

### D5. Replace symbol-table reachability with real call-graph reachability.
**(W)** "Reachable from harness" is currently computed by *symbol containment* in the harness binary, which over-approximates by including dead code linked in. Reviewers from PL background will pick on this.

**(H)** Use SVF or a clang `-ftime-trace` + `-call-graph` static call-graph traversal (or even a runtime dynamic call-graph from one fuzz run). Re-measure how many of the 19 zero-days were unreachable under the stricter oracle. Discuss the precision/recall trade-off.

### D6. Tree-sitter vs. a stronger preprocessor.
**(W)** Stage 1 uses tree-sitter AST + a hand-rolled parameter-validity pass. A semantically richer signal (clang-static-analyzer warnings, CodeQL "untrusted-source" sink reachability, IRIS-style LLM-inferred taint specs) would likely reduce the hypothesis pool and free token budget.

**(H)** Add a "preprocessing-strength" ablation: (a) raw file → Sonnet, (b) tree-sitter+param-check → Sonnet (current), (c) CodeQL queries → Sonnet, (d) IRIS-style taint specs → Sonnet. This isolates how much of the lift comes from the preprocessor and how much from the LLM.

### D7. Strong-scaling curve of Stage 1.
**(W)** Stage 1 is embarrassingly parallel; the paper reports wall-clock per project but no scaling study.

**(H)** Plot tokens-per-minute as a function of concurrent file workers up to API rate limits. Reviewers care because the practical cost of a "$42 project" is 65 minutes of latency that can be parallelized down to 5 minutes.

---

## E. Cost, reproducibility, and model dependence

### E1. Dollars are not a reproducible cost metric.
**(W)** "$300 total," "$42/project," "$9.10/vuln" all assume specific API price lists. Anthropic, OpenAI, Google all rotate prices quarterly. Reviewers reading the paper 12 months after submission will see prices that don't match.

**(H)** Report **tokens** (input cache-miss, input cache-hit, output, tool-tokens) as the primary cost metric; dollars as derived secondary. Provide a spreadsheet that re-computes dollars from any price list.

### E2. The whole pipeline runs on Anthropic models only.
**(W)** Section 4 fixes Haiku 4.5 + Sonnet 4.6. Reviewers will read this as "REVELIO is Anthropic-coded" — that the design only works because Sonnet 4.6's tool-use loop is particularly good.

**(H)** Re-run the four-tier ablation on at least one other family (Gemini 3 Pro is available; DeepSeek-V3.2 and Qwen3-Coder-A22B are open-weights and free at scale). Show that T3 (the enforced harness) still wins. If T3 *doesn't* win on another family, the contribution narrows and the story has to change — that is also a useful finding.

### E3. Open-source the agent before the camera-ready, not after.
**(W)** "anonymous.4open.science/r/revelio-submission" is good for double-blind, but the artifact must include: (i) Docker images for the 7 OSS-Fuzz targets, (ii) the ARVO subset selection seed and the CWE/project distribution, (iii) full prompts (the appendix is helpful but typos and whitespace matter for LLMs), (iv) sanitizer wrapper scripts and the validate() tool, (v) the 19 PoV inputs minimally with maintainer permission, with sensitive ones redacted under embargo until the disclosure deadline.

**(A)** *"Can a reviewer reproduce one row of Table 8 in under an hour?"*

### E4. Provide a 1-CPU, 8 GB-RAM, free-tier reproduction recipe.
**(W)** Most reviewers cannot spend $300 to spot-check. Give them a 1-case recipe: one ARVO case, one Haiku call (free tier), one Sonnet call ($0.10), one PoV. If this works in 5 minutes, half the rigor concerns evaporate.

---

## F. Threat model, ethics, and disclosure for an S&P submission

### F1. Per-CVE disclosure timeline must be explicit.
**(W)** Table 5 has "Pending" entries with no dates. S&P PCs care about this.

**(H)** Convert Table 5 into a long-form per-CVE timeline: reported on, acknowledged on, fix landed on, CVE assigned on, embargo lifts on. If any CVE will still be embargoed at camera-ready, state your contingency.

### F2. Strengthen the dual-use discussion.
**(W)** §6's PoV-vs.-exploit framing is currently 1 paragraph. Recent papers on offensive LLM agents (Co-RedTeam style, Aardvark, Project Glasswing) have been rejected from top-tier venues over weak ethics sections.

**(H)** Add an explicit ethics subsection that covers: (a) IRB / institutional ethics-review status (or why not applicable), (b) what artifacts are released vs. withheld and under what policy, (c) cost barrier-of-entry for an attacker reusing the harness (≈$42 is *not* a meaningful barrier — say so), (d) interaction with downstream vendors' patching SLAs.

### F3. Address patch-priming.
**(W)** Once a project pushes a security fix, an attacker can run REVELIO on the *pre-fix* tree and the *post-fix* tree to extract a working PoV before users have patched. This is the natural offensive use of variant-analysis tools.

**(H)** Add a paragraph in §6: discuss disclosure-window risk; recommend that maintainers (i) batch fixes, (ii) avoid security keywords in commit messages until end-users patch, (iii) consider security-fix obfuscation (controversial but real). Cite Big Sleep's analogous policy stance.

### F4. Discuss false-positive externalities on maintainers.
**(W)** Even 0% FP under the paper's triage rule pushes work onto maintainers who must read each report. RepoAudit's 78% precision is comparable; the OSS community is increasingly hostile to LLM-generated bug reports.

**(H)** Add a small per-project survey: did maintainers find the reports useful? How many minutes did each report take to triage? Cite the curl maintainer's recent public stance on LLM-generated reports.

---

## G. Presentation

### G1. Fig. 2 headline framing.
**(W)** "Vulnerabilities Found 175 vs 55" is the dominant takeaway from a quick read; recall 69% vs 53% is much smaller and more accurate. Fig. 2 sets reviewer priors that are then read into the rest of the paper.

**(H)** Replace the headline metric with **Targeted-Recall %** and move "discovery count" to a secondary panel with explicit dedup methodology stated in the caption.

### G2. Use "PoV" or "PoC" consistently — not both.
**(W)** Currently both terms appear. Pick one.

### G3. The "we don't need Mythos / Opus 4.7 / Glasswing" framing is unverifiable.
**(W)** Reviewers do not have access to those models. Claims of the form "even without Mythos, REVELIO …" land as defensive.

**(H)** Either (i) demonstrate the claim by running one comparison row with Glasswing, or (ii) delete the framing and let the absolute numbers stand.

### G4. Combine redundant Severity / Class columns in Table 5.
**(W)** CWE already implies class; severity → CVSS.

### G5. Replace the IDManifest case study sidebar.
**(W)** The IDManifest walk-through is a single project. A richer case study would be one where Claude Code, Codex, and Sorcar *all* failed and only REVELIO succeeded — that comparison shows the design contribution most clearly.

### G6. Figures 1 and 3 referenced but not rendered.
**(W)** They appeared blank in my PDF extraction; please verify the camera-ready vector PDF embeds them.

### G7. Move ablation tables into the main body if space allows.
**(W)** Appendix C tables T1/T2/T2.5/T3 are the single most persuasive piece of evidence for the design contribution. They belong in §6 or §7, not in an appendix.

---

## Suggested 4-week rewrite plan

| Week | Deliverable |
|---|---|
| 1 | Implement ARVO patch-differential oracle; rerun the 100 cases × 3 seeds. Drop in RepoAudit and OSS-Fuzz-Gen baselines. Inject `validate()` into Claude Code / Codex / Sorcar variants. |
| 2 | Run Level-0 and Level-1 CyberGym protocols on the same 100 + a full-repo localization study on 30 more. Run a 3×3 model-pairing sweep including one non-Anthropic family. |
| 3 | Variant-analysis seeding ablation (D1); per-sanitizer breakdown (D4); preprocessor-strength ablation (D6); strong-scaling curve (D7). |
| 4 | Rewrite §5/§6/§7; convert dollars to tokens (E1); produce per-CVE timeline (F1); ethics subsection (F2/F3); replace headline metric in Fig. 2 (G1); finalize artifact (E3/E4). |

If the camera-ready can land 8 of the 12 highest-impact items above (A1, A2, A4, A5, B1, B2, B4, C1, C3, D1, D3, E2) the paper goes from "interesting but contestable" to "clearly accept" at S&P 2027.

---

## Sources consulted

1. CyberGym benchmark — Wang et al., arXiv 2506.02548 (v3 March 2026; 1,507 vulns, 188 projects, Level-0/Level-1 protocols, ~20% best-agent success).
2. ARVO — Mei et al., arXiv 2408.02153 (>5,000 reproducible OSS-Fuzz bugs with vulnerable+patched commit pairs; patch-differential oracle).
3. RepoAudit — Guo et al., arXiv 2501.18160 (40 true bugs / 78% precision / $2.54/project / 185 confirmed new bugs in real OSS).
4. Big Sleep / Naptime — Google Project Zero × DeepMind, "From Naptime to Big Sleep" (Oct 2024 SQLite seriesBestIndex zero-day; variant-analysis seeding; co-author K. Sen).
5. IRIS — Li, Dutta, Naik, arXiv 2405.17238 (LLM + CodeQL neuro-symbolic; +28/120 vs. CodeQL on CWE-Bench-Java; github.com/iris-sast/iris).
6. OSS-Fuzz-Gen — Google, github.com/google/oss-fuzz-gen (Apache-2.0; 1.4k stars; 30+ confirmed CVEs; active through Mar 2026).
7. OpenAI Aardvark and Anthropic Frontier Red Team / "Project Glasswing"-class agentic-security efforts — referenced as motivation but not used as baselines (no API/weights publicly available at submission time).

*Reviewer's note:* I have intentionally not assigned a score. The substantive question is whether the authors can complete items A1, A2, A4, A5, B1, B2, C1 by camera-ready. If they can, this paper deserves to be at S&P 2027.
