# TASK: Audit arXiv 2605.19407 "A Bitter Lesson for Data Filtering"

User task: Read https://arxiv.org/pdf/2605.19407 and thoroughly check for **wrong assumptions, cheating, irreproducibility, fraud, potential for cheating in evaluation, AI slop, security vulnerabilities**. Use internet search extensively (MUST visit >=10 sites via go_to_url, tracked in tmp/information-bitterlesson.md). Generate an HTML report in PWD/reports/ and open in user's default browser (use `open reports/<file>.html` on macOS). Fact check everything.

## Status: paper fully read (main text + Appendix A/B partially). Web research: 1/10 sites visited. NEXT: do web verification (9+ more sites), then write report.

## Paper summary (already extracted — do NOT re-read the whole PDF; full text is at tmp/paper.txt, chunks at tmp/chunk_aa..ae; I read chunk_aa(implicitly via paper.txt first 600 lines), chunk_ab, chunk_ac. chunk_ad and chunk_ae remain UNREAD (likely Appendix B figures, Appendix C theory/proofs, MMLU prompt details))

- Authors: Christopher Mohri (xmohri@stanford.edu), John Duchi, Tatsunori Hashimoto — all Stanford. Submitted 19 May 2026, v1 only, CC-BY 4.0, cs.LG/cs.AI. NeurIPS-style "Preprint".
- Claim: with enough compute, "the best data filter is no data filter". Unfiltered Common Crawl (DCLM-Pool, 240T GPT-NeoX tokens) eventually beats filtered versions (RefinedWeb, DCLM-Baseline, English/Repetition/StopWords filters) in best-achievable loss L*(D)=min_{M,N} loss, unconstrained compute.
- Setup: random CC subsets 670M–10B tokens; Llama-style models 15M–7B params (Table 2: hidden 128/512/1024/2048/4096; layers 8/8/18/17/32; LRs 1e-2,5e-3,5e-3,5e-3,1e-3; wd tuned in [0.1,0.5]); Meta Lingua repo; ctx 1024; batch 2^19 tokens; 500-step warmup; steps in powers of 2; eval 5x during training, report BEST checkpoint; H200s, single 8-GPU node (7B uses FSDP); >20,000 H200 GPU-hours total.
- Metrics: avg NLL on C4-English, FineWeb-Edu, Cosmopedia. Benchmarks (ARC-Easy, PIQA, SocialIQA) only in Appendix B, "much noisier", "trends roughly the same".
- Filter retention on their pool (by tokens): English 28.2%, Repetition 45.3%, StopWords 50.4%, RefinedWeb 13%, DCLM-Baseline 2.1%.
- Fig1: 1B model best losses CC 3.37 < SW 3.46 < RP 3.58 < EN 3.59 < RW 3.93 < D-B 5.29. At 15M model the pool is NOT best (EN 4.15 best, CC 4.44).
- Junk injection: +random strings (10k fake words, 3-8 chars) and +shuffled-word CC docs up to +800%. Large models close gap; shuffled even BEATS pool on 330M/1B (e.g. 1B: +400% shuffled 3.36 vs CC 3.40).
- Scaling-law prediction: crossing point for 240T pool vs RefinedWeb at ~1e30 FLOPs (two fits: 600 tok/param "following DeepSeek V4" -> 3.6e30; 4-epoch constraint per Muennighoff -> 9.0e29). Cites xAI Grok-4 model card for frontier ~5e26 FLOPs; cites Epoch AI (David Owen, epoch.ai/files/AI_2030.pdf) for 1e29 by 2030.
- Fig5: crossing-epoch counts up to 121.6 epochs; extrapolations from second-degree polynomial fits in log-log space; hollow points = extrapolated (power-law fit of decaying loss) NOT empirical; 80M model has NO crossing at 10B pool.
- Table 1: GPT5-mini judgments of keyword-matched CC docs for 4 MMLU categories: support >> refute (e.g. world_religions 5.89 vs 0.00) — used to argue CC has little actively-harmful misinformation.
- Sec 6: shuffled data hurts first-token prediction (Fig 7) but they dismiss it.
- Sec 7: Proposition 7.1 low-rank matrix-factorization model: with rank r >= k tasks (orthogonal inputs), junk tasks absorbed w/o penalty; cites Baldi-Hornik 1988, Zhu et al 2018, Lee et al 2016.
- Limitations they admit: dense-only, no curricula/post-training, MoE may differ; dup docs; 1e30 FLOPs needed; AI-generated content post-2023 unclear; factuality edge cases.
- Code: https://github.com/chrismohrii/bitter-lesson-data-filtering (claims config files released).

## My candidate findings so far (TO VERIFY via web)
1. **Evaluation-choice bias (potential cheating in evaluation)**: C4/FineWeb/Cosmopedia NLL favors training distribution closer to raw web; NLL != downstream ability. Their own cited Saada et al. 2025 (2510.00866) makes exactly this point. Benchmarks relegated to appendix and "noisier"; on ARC-Easy/PIQA at small scales DCLM-Baseline is known (from DCLM paper) to be much better — check Fig 8 numbers: CC usually top at 1B but margins tiny (e.g. ARC-Easy 1B: CC 0.409 vs RW 0.344? note RW *worse*) and SocialIQA values ~0.35-0.39 are near random (3-class, random=0.333). ARC-Easy random = 0.25, values 0.30-0.41. PIQA random=0.5, values 0.55-0.64. So benchmarks barely above chance => weak evidence.
2. **Best-checkpoint reporting**: "report the best checkpoint" of 5 evals = mild selection bias on val loss.
3. **Unconstrained-compute objective L*** assumes min over M,N approximated by grid up to 1B/7B params; "best achieved" with no error bars / no seeds mentioned — single runs? No variance reported.
4. **Massive extrapolation**: from <=10B-token pools to 240T (4+ orders of magnitude) and 1e30 FLOPs (~3-4 OOM above frontier); 121.6-epoch crossings; quadratic fits in log-log; they themselves shade regions as unreliable but headline claim relies on extrapolation. R^2>0.99 on a scaling law fitted to ~4-5 points.
5. **Contamination claim**: "pool sizes only up to 10B tokens so no test-set contamination expected" — weak: contamination possible at any size; also val sets (C4, FineWeb-Edu, Cosmopedia-style) overlap CC distribution by construction (C4 IS filtered CC; FineWeb-Edu IS filtered CC) -> train/test distribution overlap inflates pool advantage. Cosmopedia is synthetic (HuggingFace).
6. **"DeepSeek V4" 600:1 ratio** — verify existence/ratio. Possibly unpublished/rumor-based citation (no reference entry given for DeepSeek V4! Check references — none listed). If no citation -> unverifiable claim.
7. **Tokenizer**: GPT-NeoX tokenizer w/ vocab giving -log(1/V)~10.8 => V~e^10.8~49k ≈ 50257/50304 plausible (GPT-NeoX vocab 50277/50304). OK.
8. **DCLM facts to verify**: DCLM-Pool = 240T tokens; DCLM-Baseline = 3.8T tokens / ~1% of CC; resiliparse extraction; Bevendorff et al. 2018 citation is for ChatNoir not resiliparse (common miscitation — actually resiliparse canonical cite IS Bevendorff 2018 ECIR per its docs — verify).
9. **Muennighoff et al. 2023/2025 "Scaling Data-Constrained LMs"**: ~4 epochs diminishing returns — verify; also note repetition value drops to zero ~16 epochs in that paper, while this paper relies on crossings at up to 121 epochs — tension/wrong assumption.
10. **Villalobos et al.: 200-500T tokens internet stock** — verify (Epoch "Will we run out of data": ~300T tokens median, 100T-1000T range? check).
11. **xAI Grok 4 model card 5e26 FLOPs** — verify URL/number.
12. **GPT5-mini judge**: LLM-as-judge for Table 1, no human validation, no calibration; keyword matching for 4 MMLU subjects only; counts per what denominator? "Average judgements" unclear units. Weak factuality audit, support/refute asymmetry could reflect judge bias.
13. **Security vulnerabilities angle**: recommending NO data filtering ignores data-poisoning literature: Anthropic/UK AISI Oct 2025 "Poisoning attacks require near-constant number of documents" (arXiv 2510.07192) showed ~250 docs can backdoor models regardless of size — directly contradicts "junk data harmless" w.r.t. adversarial data; also CC poisoning feasibility (Carlini et al. 2023/2024 "Poisoning Web-Scale Training Datasets is Practical", arXiv 2302.10149 — expired domains in CC). Paper only considers random/shuffled junk, not adversarial. VERIFY both papers.
14. Fig 3 oddity: +20% random BEATS pool on 330M/1B (3.38 vs 3.40) — they hand-wave "regularization effect"; within-noise? no error bars.
15. Check GitHub repo: does it exist? configs only or full pipeline? data subsets released? exact subset seeds? If only configs -> irreproducible w/o 20k H200 hours + DCLM-Pool access (240T tokens, huge download).
16. AI slop check: paper reads coherent, low slop probability; check for hallucinated refs (Saada 2510.00866, Ru 2502.06604, Kim 2509.14786, Li 2505.04741, Fang 2503.07879, Owen AI_2030.pdf, Sardana 2401.00448, Cheng 2206.12041 "How many labelers do you have?" by Cheng/Asi/Duchi — self-cite, verify title matches).
17. Authors legitimacy: Christopher Mohri = Stanford PhD student (son of Mehryar Mohri?), prior work w/ Hashimoto on conformal LM factuality. Verify via Stanford/Google Scholar/openreview.
18. Note: cited "Awasthi, Cortes, Mohri (Christopher)" AISTATS 2023 — self-cite consistent w/ identity.
19. Sinha et al. 2021 — shuffled-word MLM paper, real (2104.06644).
20. Goyal et al. 2024 (2404.07177) "Scaling laws for data filtering" CVPR 2024 — real; check their claim characterization.
21. Allen-Zhu & Li capacity laws (2404.05405): "junk data significantly reduces knowledge capacity" — paper says it "aligns with our findings on sufficient model sizes" — actually A-Z&Li found junk HURTS capacity even for big models unless tokens repeated more — possible mischaracterization; verify abstract.
22. Wrong assumption candidate: "loss on C4/FW-Edu/Cosmo correlates with downstream performance" — contested in literature (e.g., Saada et al; DCLM paper used CORE benchmarks not perplexity).
23. Fraud check: numbers internally consistent so far (Fig1 CC 3.37 on 1B; Fig5 700M pool CC 3.37 vs RW 3.93 consistent w/ Fig1 RW 3.93 ✓). 6ND compute: 670M pool 1B model 100B tokens => 6*1e9*1e11=6e20... Fig2 x-axis only goes to 1e19?? CHECK: Fig2 axis 1e17–1e19; but 1B model @ up to 1e11 tokens = 6e20 FLOPs — axis seems truncated/preview may differ; verify in chunk_ad/ae or HTML version. Could be mis-extraction by pdftotext. Don't accuse without checking figure directly (open https://arxiv.org/html/2605.19407v1).
24. "1 trillion parameter Chinchilla budget" — Chinchilla ~20 tok/param => 20T tokens; DCLM-Baseline 3.8T < 20T ✓ internally fine.

## STATUS UPDATE (step ~29): Research COMPLETE — 12/12 sites visited, all key facts verified and logged in tmp/information-bitterlesson.md. READ THAT FILE for all findings. Appendix chunks ad/ae read (benchmarks near chance; proofs OK).

Key verified findings for report:
1. Authors real (C. Mohri Stanford PhD in Duchi group; Scholar verified). No fraud indicators; numbers internally consistent; theory proofs correct.
2. Repo real, 1 commit, configs+scripts only; data via mutable Google Drive; no LICENSE; seeds present (seed=1 in sweep; data-gen --seed optional default None).
3. Eval-metric bias: headline = NLL on C4/FineWeb-Edu/Cosmopedia (all CC-derived/synthetic); Saada 2510.00866 verified: CQF improves downstream but not HQ-set NLL → metric choice favors "no filter". MMLU commented out in repo eval config; many measured benchmarks (hellaswag, winogrande, race, truthfulqa, gpqa, boolq, commonsense_qa, arc_challenge, openbookqa) unreported; reported 3 are near chance (SocialIQA 0.34-0.39 vs 0.333; ARC-E 0.26-0.42 vs 0.25; PIQA 0.52-0.64 vs 0.5).
4. Best-checkpoint of 5 evals reporting = mild selection. Single runs, no error bars/seeds variance.
5. Extrapolation: crossings up to 121.6 epochs vs Muennighoff (verified) "value of compute decays to zero" beyond ~4-16 epochs; 240T/1e30 FLOPs prediction = 4 OOM extrapolation; R²>0.99 on ~4-5 fitted points (quadratic in log-log).
6. DCLM 240T verified ✓; 3.8T DCLM-Baseline ≈1.6% ("~1%" ok); DeepSeek V4 real (Apr 24 2026, MoE 1.6T/49B active) but 600:1 ratio unreferenced & ambiguous for MoE; xAI Grok-4 model card EXISTS but contains NO 5e26 figure → citation doesn't support claim; Epoch 1e29-by-2030 verified ✓; Villalobos verified ✓; Allen-Zhu&Li verified (knowledge-capacity tension); GPT5-mini real (GPT-5 mini, Aug 2025).
7. Security: paper ignores adversarial poisoning. Verified: Souly et al. 2510.07192 (~250 docs backdoor any size model); Carlini 2302.10149 ($60 poisons 0.01% LAION; split-view/frontrunning attacks on crawled corpora). Unfiltered CC maximizes poisoning surface + PII/CSAM/toxicity concerns not addressed. Grok-4 model card itself says xAI filters for "data quality and safety".
8. AI slop: none — coherent, well-written, real refs (all checked refs exist), real code, honest limitations section. Fig 4 shuffled-text example consistent.
9. Wrong assumptions list: L* unconstrained-compute objective economically unrealistic (their own Fig2 Pareto shows filters win at all practical compute ≤1e19-1e20); "best filter is no filter" depends on NLL metric; "junk≠adversarial" blind spot; assumes loss↔downstream correlation (contested, Saada); contamination dismissal weak (val sets ARE CC-derived → distribution overlap, not contamination per se); +20% random "beats" pool within noise (no error bars); MoE-derived 600:1 ratio applied to dense models.

## Remaining TODO (next session)
1. Read tmp/chunk_ad and tmp/chunk_ae (appendix B figures/numbers, Appendix C theory, MMLU-judge prompt details, Table 1 methodology denominator).
2. Web research via go_to_url, updating tmp/information-bitterlesson.md counter to >=10:
   - GitHub repo chrismohrii/bitter-lesson-data-filtering (existence, contents, license, commit history)
   - arXiv HTML version (check Fig 2 axis / numbers) https://arxiv.org/html/2605.19407v1
   - DCLM paper page (2406.11794 abstract) for 240T/3.8T/1% claims
   - Muennighoff 2305.16264 abstract (4 epochs, 16 epochs->zero value)
   - Villalobos 2211.04325 abstract (token stock estimate)
   - xAI Grok-4 model card PDF (5e26)
   - Epoch AI AI_2030 (1e29 by 2030)
   - Anthropic poisoning blog or arXiv 2510.07192 (250 docs backdoor)
   - Carlini 2302.10149 (poisoning CC practical)
   - Saada 2510.00866 abstract (data-quality illusion)
   - Allen-Zhu & Li 2404.05405 abstract (junk data capacity)
   - DeepSeek V4 600 tok/param — search; likely cite-less
   - Christopher Mohri Stanford page / scholar
   - Goyal 2404.07177 abstract
3. Optionally clone/inspect GitHub repo via Bash for reproducibility + security (e.g., creds, unsafe code).
4. Write reports/bitter_lesson_audit.html (self-contained, professional, with severity ratings per category: wrong assumptions / cheating / irreproducibility / fraud / eval-gaming potential / AI slop / security). Every claim must cite verified source w/ URL. Be fair: paper appears to be legit Stanford work; likely verdict = no fraud, but heavy extrapolation, eval-metric choice favors conclusion, limited reproducibility (compute), and security blind spot re: data poisoning.
5. `open` the HTML in default browser. git add reports dir. Clean tmp files (paper.pdf, paper.txt, chunk_*, information file). Then finish with full summary.
