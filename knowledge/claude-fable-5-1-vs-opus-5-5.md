---
title: Claude Fable 5.1 vs Opus 5.5 comparison (Sep 2026)
uuid: 6e47510b-c2e3-4f38-8a01-bd8df5eb655d
summary: 'When to pick claude-fable-5-1 over claude-opus-5-5: pricing, benchmarks,
  refusals, cache ratios, Mythos sibling; sources and repo MODEL_INFO discrepancies.'
created: '2026-09-26T20:22:07Z'
updated: '2026-09-26T20:22:07Z'
---
Opus 5.5 was released 2026-09-22 and Fable 5.1 on 2026-09-01.
- Pricing per 1M tokens: Opus 5.5 is $4 in / $20 out, $0.20 cache read, $5 cache write. Fable 5.1 is $10 in / $50 out, $0.25 cache read (2.5% of input vs 5% for Opus), $12.50 cache write.
- Benchmarks published by Anthropic: Opus 5.5 beats Fable 5.1 on every shared row (TB4 66.4 vs 55.8, FrontierCode 54.4 vs 50.3, CursorBench 57.8 vs 51.8, HLE 67.7 vs 65.6, OSWorld2 81.8 vs 80.7). On the Artificial Analysis Intelligence Index, Opus 5.5 scores 58 and Fable 5.1 scores 53. Anthropic itself says the real-world gap is narrower than these scores suggest.
- Anthropic's guidance: use Opus 5.5 by default and escalate to Fable 5.1 for the hardest tasks that need more reasoning.
- Reasons to pick Fable: (1) the hardest long-horizon, open-ended reasoning work; (2) Fable 5.1 shares weights with Claude Mythos 5.1, which vetted organizations get through Glasswing, so it gives continuity for cyber and bio work; (3) the cache-read discount is proportionally deeper; (4) its default effort is high, while Opus 5.5 defaults to medium; (5) integrations already tuned to it. Opus 5.5 also brought breaking API changes: thinking can't be disabled, forced tool_choice was removed, and the computer-use tool type changed.
- Downsides of Fable: it is slow (about 66 tok/s vs about 93 for Opus 5.5). It also refuses more often: computingforgeeks saw a stop_reason=refusal on a Terraform security-group task, and the repo's skillopt_sea.py notes that Fable refuses bare ranking prompts.
- Repo discrepancy: src/kiss/core/models/MODEL_INFO.json lists a 500k context for both models, but the articles say 1M. The repo also has no cache prices or thinking flags for fable-5-1.
Sources: datastudios.org/post/claude-opus-5-5-vs-claude-fable-5-1-complete-comparison-..., computingforgeeks.com/claude-opus-5-5-released-features-benchmarks/
