# Task: Deep review of ~/Downloads/paper.pdf for SIGOPS ATC 2026
Goal: NOT a typical review — suggest improvements to paper+results, judge novelty, check related work via extensive internet search, produce HTML report in PWD/reports and open in default browser.

## Status — COMPLETE
- [x] Extracted PDF text to tmp/paper.txt (pdftotext); split into tmp/chunk_aa..ae
- [x] Read main body fully (chunks aa, ab, ac) + appendix A–D partially (chunk_ad first 400 lines)
- [x] Web research 10/10 sites via go_to_url (ATC'25 CFP, TACCL arXiv, MSCCL++ GitHub, SimAI NSDI'25, ForestColl arXiv, OpenXLA op-semantics docs, AlphaEvolve arXiv, Google search AWS Neuron collectives/NKI/TorchNeuron, KernelBench arXiv, DuckDuckGo Centauri ASPLOS'24 best paper) — logged in tmp/information-overlayccl.md
- [x] Synthesized deep review; wrote reports/overlayccl_review.html (TL;DR verdict, paper summary, novelty judgment, 10 ranked improvement actions, related-work table, writing fixes, reviewer-risk table, experiment ranking); git added; opened in default browser via `open`
- [x] Cleaned my tmp files (paper.txt, chunk_*, information-overlayccl.md)

## Paper summary (detailed — next session can rely on this, no need to re-read paper)
**Title**: "OverlayCCL: Composing Collectives Above a Vendor Library via a Closed-Stack Search Loop" (anonymous, targets ATC-style venue, 2026)

**Problem**: Picking fastest collective-communication strategy for distributed training on CLOSED vendor stacks (AWS Trainium / Google TPU / Meta MTIA). Microbenchmarks mispredict in-training cost ("cross-scope inversion"); end-to-end trials too expensive; existing simulators (SimAI, ASTRA-sim) drift across machines/models/library versions. Two challenges: (C1) no runtime visibility; (C2) no schedule-level control (no MSCCL/TACCL-style IR).

**Approach**: OverlayCCL = LLM-agent closed-stack search loop, 5 phases:
- Phase 1: LLM agent calibrates a 6-term cost-model simulator via on-device probe tools (Table 2: m_collective_lat, m_p2p_transfer, m_xla_op_overhead, m_compilation_cost, m_launch_overhead, m_memcopy_thru, m_b2b_amortization). 8–12 min once per (hw, sw-version). Agent designs probe campaign; tools prevent fabrication.
- Phase 2: score seed library (incl. AWS production baseline written by 5 AWS engineers; LLM not told which is baseline).
- Phase 3: "strategy-enumerate" search: 1 LLM call enumerates K=5 distinct strategy sketches, K calls implement, 2R=4 calls refine top-2 by simulator score. Alternatives ablated: cc-react (Claude Code ReAct), multi-island GA (AlphaEvolve-like). All converge to same winner; strategy-enumerate ~25% less wall time, ~20% fewer tokens. Total 10 LLM calls/problem.
- Phase 4: HW validation gates (20-iter microbench + 10-step TV harness on synthetic 8-layer LM); crashes/OOM/NaN excluded; LLM recovery up to 2 attempts. HW timing NOT the ranking signal.
- Phase 5: deploy lowest-simulator-score survivor passing gates. One-line import swap.

**Search space**: Python functions composing xm.* primitives {all_gather, reduce_scatter, all_reduce, collective_permute, all_to_all} + local XLA ops (reshape/pad/slice). "Strategy layer" vs MSCCL/TACCL "schedule layer". Correctness: byte-equality vs pure-Python ref at ws∈{4,8} in CPU sandbox (monkey-patched xm.*, NumPy), fp32 byte-equal + bf16 tolerance.

**Cost model (Eq.1)**: T_step = T_local + T_bw + T_coll(back-to-back amortization) + T_launch (mark_step tax, AST-derived loop multiplier) + T_NEFF (cache reload amortized) + T_net. Plus reward-hacking-resistant structural terms: fusion-credit pass, smooth quadratic HBM-overrun penalty (16GB/core), primitive-viability +∞ (collective_permute compiler abort at large ws; all_reduce payload limit). NEURON_NUM_RECENT_MODELS_TO_KEEP=1 setting.

**Evaluation**: 7× trn1.32xlarge (224 ranks, 8 EFA×12.5GB/s per node), Claude Sonnet 4.5. 8 collective problems (Table 1): MoE-side AllToAllV, Uniform AllToAll, Ring KV, distributed cross-entropy; Llama-side PP cross-stage, TP MLP all-reduce, FSDP weight prefetch, layer-block AR.
- Headline: OLMoE-10B (~11.5B params) 1.40–1.41× end-to-end steady-step speedup (4407→3126 ms) on 2500-step real OpenWebText run w/ matched loss (baseline 6.843 vs agent 6.945, ±0.15 per 50-step checkpoint, attributed to bf16 non-associativity). Llama-7B 3.24× (71.3→22.0 ms steady) with random-token loop, bit-identical normalized loss 4.970.
- Cross-scope inversion (Fig 2/Table 5): baseline wins 1-node and often 7-node microbench but agent wins per-step in real training for most primitives (per-step contribution ratios ~4–6× on Llama prims due to per-microbatch dispatch tax; bundled-vs-per_mb is the key Llama trick: M dispatches → 1 fused graph; speedup tracks M: 1.19/1.71/2.86/4.99× at M=2/4/8/16).
- Config sweep: OLMoE robust 1.39–1.41× across seqlen/dim/layers; Llama tracks M.
- Cluster-size generalization: redeploy 7-node strategies at 3-node (96 ranks) and 5-node (160): OLMoE speedup widens (1.40→1.50→1.79×); Llama compresses (3.58→2.74→2.49×).
- No-simulator ablation: hide sim from Phase-3/5, LLM-judge ranks by HW microbench → picks structurally-baseline patterns → OLMoE collapses to 1.00×, Llama 1.01×. Central ablation claim.
- Cost: search 56–78 min, ~$19 LLM per style; vs ~$6,000/40h for E2E trials of K=5×8 problems. Table 3.

**Related work cited**: MSCCL/MSCCLang, TACCL, SCCL, AutoCCL (NSDI'25), Blink, CoCoNet; SimAI (NSDI'25), ASTRA-sim 1/2; ZeRO/DeepSpeed/Megatron/FSDP/GPipe/PipeDream/Horovod/BytePS/FlexFlow; FunSearch, AlphaEvolve, ShinkaEvolve, AlphaTensor, AlphaDev, Codex/AlphaCode/Self-Refine/Reflexion/ToT; TVM/AutoTVM/Ansor/Halide/Triton/TASO/MLIR; LLM kernel systems STARK, K-Search, Astra, KernelEvolve; MoE lineage GShard/Switch/expert-choice/OLMoE; anonymized code at github.com/OverlayCCL/OverlayCCL.

**Appendices**: A loss-match; B methodology (probe gating via .item() late-window, add_step_closure for OLMoE; $150.50/h cluster cost calculus); C per-problem ablation tables; D "compositions discovered" (strategy vs algorithm distinction; ~100 LOC to add a problem); E case study AG+RS vs pack-and-gather + reward-hacking; F cost-model details; G tool API; H prompts; I seed code. (Chunks ad tail + ae not fully read — mostly tables/code boxes.)

## My initial critical observations (to merge with web research)
1. Single-platform validation (Trainium only) despite repeated TPU/MTIA generalization claims — no TPU experiment at all; claim "we observe the same pattern on Google TPU and Meta MTIA" unsupported.
2. Llama-7B 3.24× headline is essentially ONE trick (bundle M microbatch collectives into one graph = known fusion/bucketing idea, cf. DDP gradient bucketing, FSDP); baseline "per_mb" may be a weak baseline — AWS NXD recipes; reviewers will ask whether a hand-tuned bundled baseline exists (a one-line dev fix). The paper admits practitioners use per_mb for memory/compile-time reasons; need evidence bundled wasn't already known/rejected.
3. Llama-7B end-to-end run is only 200 steps with random tokens; wall-clock 2.5 vs 5.9 min in Table 6 (agent wall WORSE? — baseline 2.5 min, agent 5.9 min: presumably compile time dominates; paper should explain).
4. Loss gap 6.843 vs 6.945 (agent ~0.1 worse final loss) — claimed within noise but no seed-variance study (need N-seed baseline-vs-baseline noise band to substantiate ±0.15 claim).
5. Cost model has many hand-designed structural terms (fusion credit, HBM penalty, viability regexes, AST loop rules) — "LLM builds the simulator" is overstated; LLM only fits constants; the term structure is hand-engineered & Trainium-specific. Generality unclear.
6. Simulator fidelity never directly evaluated: no scatter plot of predicted vs measured step time across candidates; only the no-sim ablation. Should report rank correlation (Spearman) of Sim vs ground truth.
7. K=5, R=2, 10 LLM calls — tiny search budget; no sensitivity to K/R; no variance across LLM seeds/reruns (stochastic proposer, single run reported?). Should run search N times and report variance.
8. Correctness at ws∈{4,8} only via CPU NumPy sandbox; deployed at 224 ranks — admits a correctness gap; TV gate is only 10 steps. Padding/slicing strategies can have subtle wrong-at-scale behaviors (e.g., non-divisible shapes). Also fixed I/O shapes — dynamic shapes (AllToAllV with varying counts!) — expert-choice sidesteps variable counts; so "AllToAllV" tested under deterministic counts only.
9. Baseline characterization: "internal-AWS-optimized production strategy by 5 experienced AWS developers" — not externally verifiable; anonymity vs AWS-internal claim tension; should compare also vs public NeuronX-Distributed release.
10. trn1 (Trainium1) is aging; trn2 exists (different topology, NeuronLink-v3) — results may not carry; at least discuss.
11. No comparison against simply running each of the K=5 candidates end-to-end once (the $6k figure assumes 1h per candidate; but a 250-step steady measurement may take minutes, not 1h — Table 4 says end-to-end 250 steps; the cost-calculus may be inflated).
12. Cross-scope inversion: interesting but mechanism = mark_step/graph-launch tax + NEFF cache + fusion; mostly XLA/lazy-tensor artifacts, arguably known to PyTorch/XLA practitioners; novelty of the *finding* should be positioned against pjrt/lazy-tensor literature.
13. Eight problems but end-to-end only exercises 2 (OLMoE: AllToAllV+dxe) + 4 (Llama block). Ring KV & Uniform A2A only microbench-level per-step probes.
14. ATC fit: SIGOPS/USENIX ATC likes systems+artifacts: codebase anonymized ✓; but heavy LLM dependency raises reproducibility (Sonnet 4.5 nondeterminism, API cost); should pin prompts+cache responses, report multi-run variance.
15. Statistical rigor: medians reported; no CIs/error bars mentioned for step times.
16. Related work gaps to check online: NCCL tuner plugins, MSCCL++, TE/transformer-engine comm overlap, Centauri, Syndicate, TACOS, "LLM for systems" (e.g., NVIDIA kernel-gen blogs), AI Metropolis?, CommGPT?, anything on Trainium collectives (AWS Neuron "cc-overlap"), Google's GSPMD/XLA collective combiner passes (latency hiding scheduler, async collectives, collective-combine = the SAME bundling trick inside XLA!) — XLA already has CollectiveCombiner passes; check whether Neuron exposes flags; that would undercut novelty of bundling.
17. Also check: ForestColl?, TE-CCL (Microsoft, traffic engineering for collectives), "MultiCCL"?, OpalCCL? etc. And LLM-agent-for-perf papers: "KernelBench", "Mirage", "SakanaAI CUDA engineer" (known reward-hacking cautionary tale → relevant to their anti-reward-hacking design).

## Next session TODO
1. Web research: use go_to_url on ≥10 sites (arxiv/MSCCL++/TACCL/AutoCCL NSDI25/SimAI NSDI25/XLA collective-combiner docs/AWS Neuron docs on collectives & NEFF cache/trn2/KernelBench/AlphaEvolve/Sakana CUDA reward hacking/ATC 2026 CFP for fit+page/format). Maintain tmp/information-<id>.md with counter per protocol.
2. Synthesize: novelty judgment (what's new: LLM-calibrated per-deployment cost model + strategy-layer search above closed API + cross-scope-inversion measurement study; what's not: bundling≈collective-combining, LLM-evolution loop≈AlphaEvolve, cost models≈ASTRA-sim).
3. Write reports/overlayccl_review.html — sections: TL;DR, novelty assessment, strengths, improvement suggestions (experiments to add, baselines, stats, writing), related-work additions w/ links, ATC-fit & reviewer-risk table, actionable checklist. Style it nicely (self-contained CSS).
4. git add reports/; open with `open reports/overlayccl_review.html`; rm tmp/chunk_*, tmp/paper.txt, tmp/information-*.md; finish with full review summary text.
