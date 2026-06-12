# Task: Deep review of ~/Downloads/revelio2.pdf for S&P 2027 — suggest improvements (not a typical review). Use internet search extensively (10 sites, go_to_url, tmp/information-*.md).

## Done so far
1. Extracted PDF text: tmp/revelio2.txt (19 pages, via pypdf in tmp/pdfenv venv). Split into tmp/part_aa..part_ae.
2. Read pages 1-14 (intro through references). Key facts:
   - REVELIO: two-stage agentic memory-safety vuln detection. Stage 1: cheap model (Haiku 4.5) per-file hypothesis generation + tree-sitter static preprocessing + triage/dedup/harness-reachability ranking. Stage 2: stronger model (Sonnet 4.6) iterative PoV construction, sanitizer (ASan/UBSan/MSan) validated, independent re-execution. Zero FP claim.
   - Eval: 7 OSS-Fuzz projects (DNSMasq, OpenEXR, assimp, cairo, Sleuth Kit, libheif, Poppler) -> 19 zero-days, 7 CVEs, ~$42/project, ~65min median, $300 total. 16 NULL-deref omitted from FP accounting.
   - CyberGym/ARVO 100 cases, oracle-file protocol (vulnerable file given!). REVELIO 175 vulns found, 69% recall, 0% FP vs Claude Code (55, 53%, 49% FP), Codex (39), Sorcar (31). Tokens/vuln REVELIO 19.2M (highest!).
   - Recent-CVE eval: 10 post-cutoff CVEs, details in Appendix B.1.
   - Ablation T1/T2/T2.5/T3: prompting alone doesn't help (T2/T2.5 worse than T1 for Haiku); harness enforcement + asymmetric routing -> 90% recall, 0 FP, $6.42/vuln. Ablation on only 10 cases.
   - Config: top-5 hypotheses/file, 5 PoV iters per hypothesis; pilot sweep on 10 cases to pick baselines' models.
   - Limitations acknowledged: oracle-file protocol, sanitizer-observable scope only, triage under maintainer threat models.
   - Related work: RepoAudit, AnyPoC, Co-RedTeam, AgentFlow, IRIS, KNighter, CyberGym, ARVO, Magma, Big-picture fuzzing/SAST comparisons.
   - Fictional-future names in paper (Opus 4.7, GPT 5.5, Mythos, Glasswing, CVE-2026-*) — paper is set in 2026.
3. Read appendix fragment (part_ae start): ablation extra metrics (Hyp->PoV rate 53%, stage-2 cost share ~58%). Have NOT fully read appendices (part_ad = references + Appendix A prompts; part_ae = appendix tables B/C).

## Remaining TODO
1. (Optional) skim part_ad/part_ae appendices cheaply (low priority).
2. Web research: visit 10 sites via go_to_url, log in tmp/information-1.md with counter header. Suggested targets: CyberGym paper/GitHub, ARVO paper, RepoAudit, Google Project Zero Big Sleep, OSS-Fuzz-Gen / AI in OSS-Fuzz blog, Anthropic security research, AnyPoC arXiv, IRIS, Magma benchmark, S&P 2027 CFP (review criteria), recent LLM vuln-agent papers (e.g., Naptime, VulnHuntr, A1), SEC-bench.
3. Synthesize "how to improve the paper and results" review: experimental-design weaknesses (oracle-file vs repo-scale comparison fairness; baselines not given REVELIO's validation tool; 'Vulnerabilities Found' metric counts unknown extra crashes—need ground truth/dedup verification; zero-FP definition excludes 16 null-derefs; ablation N=10; pilot sweep selection bias; single run, no variance; cost accounting fairness re: tokens/vuln being highest; recall vs CyberGym leaderboard comparisons missing; missing baselines: RepoAudit, AnyPoC, fuzzing-with-equal-budget baseline, OSS-Fuzz-Gen; no coverage analysis; no comparison vs Big Sleep; dedup of 175 vulns—are they distinct root causes?; harness reachability via symbol table is weak—suggest call-graph; suggest measuring FN on full repo scan; suggest reporting variance, statistical tests; presentation issues; ethics/disclosure section placement for S&P; artifact availability).
4. Write final improvement-focused review in finish(summary=...). Also save the full review to a git-added file (e.g., reviews/revelio2_review.md) per SORCAR.md artifact rule, git add it.
5. Cleanup tmp files (revelio2.txt, part_*, pdfenv, information-*.md) before finish.
