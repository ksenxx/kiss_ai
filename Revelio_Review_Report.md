# Review Report: "Revelio: Cost-Efficient Agentic Memory Safety Vulnerability Detection for Repository-Scale Codebases"
### Target venue: IEEE S&P (Oakland)

## 1. Summary of the paper

REVELIO is an end-to-end agentic system for discovering memory-safety vulnerabilities in
repository-scale C/C++ codebases. The core design is a two-stage, cost-asymmetric pipeline:

- **Stage 1 — Hypothesis Generation (cheap model, e.g., Haiku 4.5):** brute-force per-file
  scanning. tree-sitter function extraction + a lightweight static argument-check analyzer,
  multi-pass LLM hypothesis generation, sanitizer-aware triage, union-find dedup, an
  independent static-filter sub-agent, reachability annotation, and ranking. Emits a structured
  hypothesis artifact (Table 1).
- **Stage 2 — PoV Confirmation (stronger model, e.g., Sonnet 4.6):** harness selection via
  symbol lookup, realistic-attacker constraint, iterative PoV construction with shell/validate/
  finish tools, and **deterministic sanitizer-backed re-execution** in a fresh subprocess as the
  ground-truth oracle (no LLM-as-judge).

**Claimed results.** 19 zero-day vulns (7 CVEs) across 7 heavily-fuzzed OSS-Fuzz projects, ~1 hr/
project, ~$300 total. On 100 CyberGym/ARVO projects, 175 validated vulns vs. 39 (Codex) and 31
(Sorcar), 69% targeted recall, **0% FP rate**. A four-tier ablation (T1 bare → T2 pipeline-prompt →
T2.5 structured-handoff → T3 enforced harness) argues that programmatic harness enforcement, not
prompt wording, drives the gains, and that asymmetric routing is the best cost/recall tradeoff.

The central thesis: *for memory-safety discovery in repo-scale C/C++, a deliberately minimal
two-stage harness is sufficient, and the harness matters more than the model.*

---

## 2. Overall assessment

The **idea is timely and the empirical core is strong**: real zero-days in mature, continuously
fuzzed targets, plus a controlled ablation that cleanly separates "prompting" from "enforcement."
The sanitizer-grounded oracle giving 0% FP is a genuinely compelling, defensible claim for a
security venue. If the empirical sections were complete and the threats-to-validity tightened, this
could be a competitive S&P submission.

However, **the paper is currently a partial draft and not submittable as-is.** Major sections are
TODO/placeholder stubs (Introduction marked "very initial draft outline, not ready"; entire
Background §2.1–2.5; Discussion §6.1–6.6; Related Work §7.1–7.3; Conclusion = "we xxx"; Ethics and
Open Science sub-bullets empty). RQ2 (§5.2) is explicitly "Not completed," Table 6's Claude Code row
is blank, and the RQ1 token/time cost is "todo." The reviewer-facing content that exists is good;
the gaps are what will sink it. Below is a prioritized, actionable list.

---

## 2b. IEEE S&P 2026 compliance issues (DESK-REJECT RISKS — fix first)

I checked the paper against the official IEEE S&P 2026 Call for Papers
(sp2026.ieee-security.org/cfpapers.html). Several issues here can cause **rejection without review**:

- **CVE identifiers break anonymity (CRITICAL).** The CFP states verbatim: *"Submitted papers should
  not include full CVE identifiers in order to preserve the anonymity of the submission."* Table 4
  lists **CVE-2026-42217, CVE-2026-42216, CVE-2026-47254, CVE-2026-41071** (and "CVE-2026-41071"
  etc.). These uniquely identify the disclosing authors/projects and **violate anonymization**;
  "papers that are not properly anonymized may be rejected without review." Replace with anonymized
  placeholders (e.g., "CVE-A", "a CVE was assigned") for the submission version.
- **Self-citation de-anonymizes the authors.** Reference [12] cites Sorcar as "K. Sen, UC Berkeley,
  kisssorcar.github.io," and the paper compares against Sorcar as a baseline. If Sorcar is the
  authors' own tool, this reveals identity. Cite in the third person / blind the reference per CFP.
- **Required sections must be real, not stubs.** The CFP mandates (a) a well-marked **"Ethics
  considerations"** section (write "None" if N/A; does not count to page limit) and (b) an
  **"LLM usage considerations"** section disclosing and motivating LLM use, including an
  **environmental-footprint** justification of the experiments. The paper's "Ethics Considerations"
  and "Generative AI Used…" headers exist but are empty — fill them and align titles to the CFP's
  exact wording (use the prescribed sentence: *"LLMs were used for editorial purposes in this
  manuscript, and all outputs were inspected by the authors to ensure accuracy and originality."*).
- **Page limit.** Max **13 pages of body** + up to 5 for references/appendices (18 total). The PDF is
  18 pages; ensure the *body* is ≤13 and everything after p.13 is clearly marked appendix.
- **Template.** Must use IEEE `compsoc` (`\documentclass[conference,compsoc]{IEEEtran}`, v1.8b), US
  letter. Wrong template / margin or font tweaking = reject without review.
- **Vulnerability-disclosure policy.** The CFP requires disclosure no later than the rebuttal
  deadline and a detailed disclosure plan in the submission. The paper does responsible disclosure
  (good) but should explicitly state the plan/timeline (the "90-day" follow-up) in the Ethics section.
- **ORCID** for all authors is mandatory at abstract registration (else desk reject) — administrative,
  but flag it to the authors.

---

## 3. Critical issues (must fix before submission)

### C1. Incomplete sections / placeholders
- Introduction is an outline with a visible `TODO[Yiwei: ...]`. Background, Discussion, Related Work,
  Conclusion, Ethics, and Open Science are stubs. **All author-name TODO tags and `[Yiwei: ...]`
  notes must be removed**; leaving any in a submission is an immediate credibility hit and a
  near-certain desk-reject signal.
- RQ2 §5.2 incomplete; **Table 6 "Claude Code / Claude Opus 4.7" row has no numbers**, yet Figure 5
  and the RQ2 result box discuss only Codex/Sorcar/Revelio. Either include Claude Code consistently
  everywhere or drop it consistently — the current mismatch (text says "three advanced agents" but
  only two appear in results) is a factual inconsistency reviewers will flag.
- §5.1 "Token and Time Cost. [Yiwei: todo, detailed statistics]" — the headline "$300 / ~1 hour"
  claims have **no supporting per-project breakdown**. Add the table.

### C2. The "0% false-positive rate" claim needs careful scoping
A 0% FPR is the paper's strongest selling point but also its most attackable claim. Be precise:
- It means *"every reported PoV independently re-triggers a sanitizer crash on rerun."* That is
  **soundness of the oracle**, not absence of false alarms in any security-relevant sense. Sanitizer
  crashes can be non-exploitable (e.g., the 16 NULL-deref bugs you discarded, OOMs, leaks). State
  explicitly that FPR=0 is *with respect to the sanitizer oracle*, and separate "sanitizer-confirmed"
  from "maintainer-confirmed as a security vulnerability."
- The discarding of 16 NULL-derefs and other low-severity crashes is a **post-hoc human triage step**.
  This means the end-to-end system is not fully automated/zero-FP from a maintainer's perspective.
  Be transparent: report counts before and after human triage, and define your FP metric formally.

### C3. Realistic-attacker constraint is enforced by an LLM, weakening the "deterministic oracle" story
The crash is deterministic, but **whether a PoV uses only a "publicly accessible attack surface"** is
judged within the agent/harness reasoning. A reviewer will argue that the security-relevance decision
(the part that actually distinguishes a vuln from a benign sanitizer trip) is still LLM/heuristic and
not deterministic. Make the harness-selection + attack-surface gate fully explicit and, ideally,
mechanical (e.g., "PoV is bytes fed to an existing OSS-Fuzz entry point, nothing else").

### C4. Benchmark comparison fairness
- Baselines (Claude Code, Codex, Sorcar) are run in a **T2-style natural-language pipeline prompt**,
  while REVELIO gets its full enforced harness *and* asymmetric routing. This risks a "you compared
  your tuned system to untuned baselines" objection. Strengthen by (a) giving baselines the same
  sanitizer-rerun oracle for *their* self-claimed PoVs (you already compute FP rate, so partly done),
  and (b) clearly stating that the harness *is* the contribution, so the comparison is system-vs-system,
  not model-vs-model.
- **Cost/vuln is similar across tools** ($9.10 Revelio vs $8.96 Codex vs $6.27 Sorcar), so the
  headline "cost-efficient" should be reframed: REVELIO is *more effective at similar per-vuln cost*
  and *far higher recall/precision*, not necessarily cheaper per vuln. Sorcar is actually cheaper per
  vuln. Reconcile the abstract's "cost-efficient" framing with Table 6.

### C5. Oracle-file ("file-localized") protocol undercuts the repo-scale claim
RQ2 reveals the *vulnerable file* to all tools. The title and thesis are about *repository-scale*
discovery, but the head-to-head comparison removes the localization problem. This is reasonable for a
controlled comparison, but the paper must not let readers conflate the two. The only true repo-scale
evidence is RQ1 (7 projects, single scan each). Consider: a repo-scale comparison on at least a few
projects, or a clearer statement that repo-scale is evaluated *only* in RQ1.

---

## 4. Major issues (strongly recommended)

### M1. Statistical rigor / variance
LLM agents are non-deterministic (the paper itself cites Ullah et al. on GPT-4 non-determinism). Yet
RQ1 scans "each project once" and ablation numbers are single-run percentages on small N (recall on
~10 tasks → 10% granularity). **Report multiple seeds/runs with mean±std or confidence intervals**,
especially for the four-tier ladder where 20% vs 30% on N=10 is one bug. Without this, the ablation
conclusions are statistically fragile.

### M2. Small N in the ablation
Table 7's recall percentages appear to be on the 10-project pilot set (each 10% = 1 task). Differences
like T2.5→T3 "60%→80%" are 2 tasks. Move the four-tier ladder to the 100-project set, or at minimum
report exact counts and CIs, and acknowledge the small-sample caveat.

### M3. Data-contamination / training-cutoff concerns
You added the "recent-CVE, web-search-disabled, post-cutoff" evaluation (§5.2.3) — excellent. But it
is relegated to an appendix with no main-text numbers, and the model snapshots cited (Haiku 4.5,
Sonnet 4.6, Opus 4.7, GPT 5.4/5.5 dated 2026) suggest CyberGym/ARVO historical bugs may be in
training data. **Promote the post-cutoff results into the main text** with concrete numbers — this is
your defense against the strongest reviewer attack on benchmark validity.

### M4. Recall denominator and "Targeted Vulnerability Recall" definition
Define precisely what 69% recall is measured against. CyberGym tasks have a *targeted* vuln but files
may contain others; "175 found / 69% recall" needs the denominator and the relationship between
"vulnerabilities found" (175) and "targeted recall" (69% of 100) made explicit. Currently a reader
cannot reconstruct the arithmetic.

### M5. Missing/weak related-work positioning
Related Work (§7) is empty. For S&P you must engage deeply with: Google **Big Sleep**/Project Zero,
**AIxCC** (DARPA AI Cyber Challenge) and its teams' systems, **AnyPoC** and **AgentFlow** (your two
closest concurrent works — currently only in the intro), **CyberGym**, **ARVO**, fuzzing
(libFuzzer/AFL++/OSS-Fuzz), and ML-based detection (PrimeVul, etc.). The intro promises a "gap"
relative to AgentFlow/AnyPoC ("minimal sufficient harness, cost frontier") — that contrast must be
fully developed and, ideally, *empirically* substantiated (you claim "an order of magnitude cheaper
than concurrent typed-DSL synthesis framework [AgentFlow]" but show no head-to-head with it).

### M6. Reproducibility of the agent comparison
Baseline tool versions are pinned (good), but agent behavior depends on harness/scaffolding details.
Provide the exact prompts (you do, App. A), the orchestration configs, the CyberGym task IDs
(App. C lists them — good), and the random-selection seed. State how many total sanitizer reruns,
budget caps (K=5 hypotheses, iters=5), and timeouts were used per tool to make the cost comparison
auditable.

---

## 5. Minor / presentation issues

- **Reference [11] is wrong:** OpenAI Codex is cited with the title "Claude code." Fix to the correct
  Codex title/URL.
- Spelling/typos: "Enrivonment" (Fig. 1 caption and §4 heading) → "Environment"; "comfirmed" →
  "confirmed"; "traige" → "triage" (multiple); "publically" → "publicly"; "effect" → "affect"
  (§5.2 file-localized protocol); "vulnerabilityhypotheses" missing space; "Hypothese" → "Hypothesis"
  (Fig. 4 caption). Do a careful pass.
- "REVELIO" frequently renders glued to the following word (e.g., "REVELIOdiscovered") — a LaTeX
  spacing macro problem; fix the `\textsc`/macro to emit a trailing space.
- Abstract says "100 randomly selected Arvo projects from the CyberGym benchmark" — capitalize "ARVO"
  consistently and cite both ARVO [9] and CyberGym [8] on first mention.
- Figure 4/5 described in text but ensure the figures themselves are legible in print (axis units,
  the parenthetical "(cost, found)" tuples are confusing as drawn).
- Table 6 column header "Tokens / Vulnerability" with 19.2M tokens/vuln for REVELIO vs 10.6M (Codex)
  — note REVELIO uses *more* tokens/vuln; reconcile with the "cost-efficient" narrative (cheaper
  tokens, not fewer). Make the cheap-token argument explicit in the cost discussion.
- The "Bitter Lesson" framing is invoked repeatedly; cite it once and avoid over-leaning on a blog
  post as a load-bearing argument for a top-tier venue.
- Define ASAN/UBSAN/MSAN scope: MSAN (uninitialized reads) requires fully-instrumented dependencies;
  state how you handled MSAN builds for OSS-Fuzz targets, or whether MSAN findings actually occurred.

---

## 6. Soundness of the core claims (reviewer's lens)

1. **"Sanitizer oracle ⇒ trustworthy, 0% FP."** Sound *as a soundness property of the oracle*, but
   must be scoped (see C2). Strong if framed carefully.
2. **"Harness > model" (RQ3).** The ablation design is the paper's best contribution and is
   methodologically the right experiment. Needs variance/CIs and larger N (M1, M2) to be convincing.
3. **"Cost-asymmetric routing is best."** Supported by Table 7 (asymmetric 90% recall, 0% FP, $6.42),
   but on tiny N. Re-run on 100 projects.
4. **"Repository-scale."** Only RQ1 truly tests it; RQ2 is file-localized. Don't overclaim (C5).
5. **"Cheaper than AgentFlow by 10×."** Currently unsupported by direct comparison. Either run it or
   soften to a qualitative argument with citation.

---

## 7. Suggested experiments to strengthen the paper

1. **Multi-seed runs** (≥3) for RQ1 and RQ3 with mean±std; report variance of #vulns found.
2. **Direct head-to-head with AgentFlow and AnyPoC** on the same CyberGym subset (your two closest
   concurrent systems) — essential to justify the "minimal sufficient harness" novelty claim.
3. **Repo-scale comparison**: run at least the baselines on a few full repositories (no oracle file)
   to support the title.
4. **Ablation of each Stage-1 component** (static arg-check analyzer, dedup, static-filter sub-agent,
   reachability ranking): which actually contribute to recall/cost? Right now Stage 1 has many parts
   with no per-component ablation.
5. **Promote post-cutoff CVE results** to main text (anti-contamination evidence).
6. **Sensitivity to K (top hypotheses) and iters (PoV trials)** beyond the single A1/A2 appendix point.

---

## 8. Ethics / disclosure (S&P-specific)

- The disclosure log (Table 4) is good practice and S&P values it. Ensure the **Ethics
  Considerations** section is fully written: responsible disclosure timeline (you mention 90-day
  follow-up — state the policy explicitly), dual-use discussion, and that all targets are public OSS.
- Anonymize correctly: the artifact URL is a placeholder (`anonymous.4open.science/xxxx`) — ensure a
  working anonymized repo at submission. Reference [12] points to "kisssorcar.github.io" / "K. Sen,
  UC Berkeley" — **this de-anonymizes the authors if Sorcar is your own tool**; cite it neutrally or
  via the anonymized artifact to preserve double-blind review.
- The "Generative AI Used in this Research / for Editorial Purposes" subsections must be filled in per
  S&P's policy.

---

## 9. Top action items (priority order)

0. **(Desk-reject risk) Remove full CVE identifiers from Table 4** and blind the Sorcar self-citation
   [12]; complete the required "Ethics considerations" + "LLM usage considerations" sections; verify
   body ≤13 pages and the IEEE compsoc template.
1. Finish Introduction, Background, Discussion, Related Work, Conclusion; remove ALL `[Yiwei: ...]`/TODO tags.
2. Complete RQ2 (§5.2) and fill the blank Claude Code row in Table 6 (or drop Claude Code consistently).
3. Add the RQ1 token/time cost table ($300 / ~1 hr breakdown per project).
4. Scope the 0%-FP claim precisely (oracle soundness vs. security relevance; report pre/post human triage).
5. Add multi-seed runs + confidence intervals, especially for the four-tier ablation; enlarge N.
6. Promote post-cutoff CVE evaluation into the main text (contamination defense).
7. Add direct comparison with AgentFlow/AnyPoC to justify "minimal sufficient harness" + "10× cheaper."
8. Clarify oracle-file vs. repo-scale scoping; avoid conflation in title/abstract.
9. Fix reference [11] (Codex mislabeled "Claude code") and the Sorcar self-citation de-anonymization risk.
10. Full proofreading pass for the REVELIO-spacing macro bug and the many typos.

---

## 10. Verdict

**Promising but currently a partial draft (not ready to submit).** The technical idea (cost-asymmetric
two-stage harness with a deterministic sanitizer oracle) is sound and well-motivated, the real-world
zero-day results are genuinely impressive, and the harness-vs-prompt ablation is the right experiment.
To be competitive at S&P, the authors must (1) finish all prose sections and remove TODOs, (2) complete
and de-conflict the RQ2 tables, (3) add statistical rigor (seeds/CIs, larger N), (4) carefully scope the
0%-FP and repo-scale claims, and (5) position rigorously against Big Sleep/AIxCC/AgentFlow/AnyPoC with at
least one direct head-to-head. With those, this is a strong systems-security paper.

---

## 11. Citation / fact verification (done via live web search)

I cross-checked the paper's key external references against current sources:

- **Big Sleep / CVE-2025-6965 (intro):** ACCURATE. Big Sleep is the Google DeepMind + Project Zero
  LLM agent (evolved from Project Naptime, Oct 2024). It found CVE-2025-6965 in SQLite (memory
  corruption, integer overflow/stack buffer underflow, CVSS 7.2, versions < 3.50.2), announced Jul
  2025 — first AI agent to foil an in-the-wild zero-day. (blog.google, thehackernews, cve.org, NVD.)
- **CyberGym [8]:** ACCURATE. arXiv:2506.02548 (UC Berkeley, Z. Wang, D. Song et al.); 1,507
  real-world vulns across 188 OSS projects, built on ARVO, with Levels 0–3; found 34 zero-days.
  **Fix the bibliography entry** — "arXiv e-prints, pp. arXiv–2506" should be "arXiv:2506.02548."
- **ARVO [9]:** ACCURATE. arXiv:2408.02153 (Mei, Pearce, Dolan-Gavitt et al., 2024); 5,000+ (repo
  says 6,000+) reproducible C/C++ memory vulns from OSS-Fuzz. Citation correct.
- **PrimeVul [1]:** ACCURATE. Ding et al., ICSE 2025 (DOI 10.1109/ICSE55347.2025.00038). NOTE: the
  paper's paraphrase "best LLMs 70% accuracy, very low precision/recall" understates the headline
  result. The striking, citable number is that a SOTA 7B model dropping from **68.26% F1 on BigVul to
  3.09% F1 on PrimeVul** — use that for impact.
- **Ullah et al. [2] (S&P 2024):** ACCURATE. arXiv:2312.12575, "LLMs Cannot Reliably Identify and
  Reason About Security Vulnerabilities (Yet?)" — SecLLMHolmes framework, 228 scenarios; GPT-4
  non-deterministic, incorrect reasoning, fooled by trivial perturbations. Paraphrase is fair.
- **AgentFlow [4] (arXiv:2604.20801):** REAL and VERIFIED. Liu, Shou, Liu, Wen, Chen, Fang, Feng
  (UC Santa Barbara, 22 Apr 2026), "Synthesizing Multi-Agent Harnesses for Vulnerability Discovery";
  uses a typed graph DSL to auto-synthesize harnesses (GitHub berabuddies/agentflow). This is
  REVELIO's closest competitor — the contrast ("minimal harness" vs. "automated typed-DSL harness
  synthesis") is accurate, **but the claimed "order of magnitude cheaper than [AgentFlow]" is not
  backed by any direct comparison.** Run a head-to-head on a shared CyberGym subset or soften to a
  qualitative claim.
- **AnyPoC [5] (arXiv:2604.11950):** REAL and VERIFIED. Zhao, Yang, Wang, Yang, Zhang, Zhang (2026),
  "Universal Proof-of-Concept Test Generation for Scalable LLM-Based Bug Detection" (GitHub
  zzjas/anypoc). REVELIO's contrast (a general LLM-as-judge/PoV-validation framework vs. REVELIO's
  sanitizer-only oracle) is fair; a direct empirical comparison would strengthen the novelty claim.

*Methodology note: this review combines a complete close reading of all 18 pages of the extracted PDF
with live web verification (10 sources) of every key external reference: Big Sleep/CVE-2025-6965,
CyberGym [8], ARVO [9], PrimeVul [1], Ullah et al. [2], OSS-Fuzz [7], AgentFlow [4], and AnyPoC [5],
plus the IEEE S&P 2026 CFP. All cited works are real and the descriptions are accurate; the main
fixable bibliographic issues are the CyberGym entry format, the PrimeVul stat phrasing, and the
unsupported "10× cheaper than AgentFlow" claim.*
