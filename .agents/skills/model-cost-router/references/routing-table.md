# Routing table (researched 2026-09-24)

Prices are USD per 1M tokens, vendor list price, standard tier. "AA" is the
Artificial Analysis Intelligence Index; "TB2" is Terminal-Bench 2.0 (agentic
coding). The local Sorcar catalog `~/.kiss/MODEL_INFO.json` overrides these
numbers when `scripts/route.py` runs; the table explains the ordering in
`assets/tiers.json`.

## Tiers

| Tier | Use for | Candidates in preference order | In / Out $/M | Signal |
|---|---|---|---|---|
| small | reads, greps, summaries, log and test-output triage, renames, boilerplate, format conversions, reviewers against a stated checklist | `zai-org/GLM-5.3-Flash` | 0.15 / 0.50 | AA 42; #1 on OpenRouter by weekly tokens |
| | | `gpt-6-luna` | 0.10 / 0.50 | AA 37; cheapest cost per task on AA |
| | | `deepseek-ai/DeepSeek-V4.1-Flash` | 0.30 / 1.20 | AA 39; cache read $0.006 |
| | | `openrouter/qwen/qwen3.8-flash` | 0.15 / 0.47 | AA 40 |
| | | `gpt-5.6-luna` | 0.20 / 1.20 | previous generation |
| | | `claude-haiku-4-5` | 1.00 / 5.00 | TB2 28%; 7x the price of GLM-5.3-Flash, last resort |
| medium | implement from a clear spec, tests for existing code, known-cause bug fixes, single-module refactors, docs of existing behaviour, diff review | `gemini-3.8-flash` | 0.75 / 3.75 | AA 41; promo price through 2026-12-31 (then 1.50 / 7.50) |
| | | `zai-org/GLM-5.3` | 1.40 / 4.40 | AA 45; GLM 5 TB2 52% |
| | | `deepseek-ai/DeepSeek-V4-Pro-0813` | 1.32 / 3.96 | AA 36; half price off-peak |
| | | `openrouter/x-ai/grok-4.7` | 1.60 / 4.80 | AA 46 |
| | | `gpt-6-sol` | 2.00 / 10.00 | AA 48, best score in tier |
| | | `claude-sonnet-5` | 2.00 / 10.00 | AA 38; pick when an Anthropic model is required |
| | | `kimi-k3` | 3.00 / 15.00 | AA 44; 1M context |
| frontier | root cause after repeated failure, cross-module design, conflicting requirements, security and concurrency review, final acceptance | `claude-opus-5-5` | 4.00 / 20.00 | AA 58, top score and cheapest frontier |
| | | `claude-opus-5` | 5.00 / 25.00 | AA 51 |
| | | `gpt-6-astra` | 10.00 / 50.00 | AA 53 |
| | | `claude-fable-5-1` | 10.00 / 50.00 | AA 53; longest-horizon tasks |

Price ratios at a 3:1 input:output blend: medium/small about 6 to 8x,
frontier/medium about 4 to 5x, frontier/small 33 to 40x. `claude-fable-5-1`
and `gpt-6-astra` cost 2.5x `claude-opus-5-5` for a lower AA score; they pay
off only on units Opus 5.5 has already failed.

## The effort axis

Reasoning effort is a second routing dimension inside one model. Artificial
Analysis measured `claude-opus-5-5` at $0.55 per task at `low` effort (AA 42,
a medium-tier result) against $5.98 per task at `max` (AA 58): an 11x spread
without changing model. `gpt-6-sol` at `medium` effort ($0.25 per task, AA 40)
is cheaper per task than most dedicated small models. In Sorcar, effort
variants are catalog aliases (`gpt-6-sol-low`, `gpt-6-sol-medium`,
`kimi-k3-low`, ...): passing one to `run_parallel(model_name=...)` is the
same as a tier change. Prefer "stronger model, lower effort" over "weaker
model, higher effort" when the unit needs broad knowledge but little
deliberation (API usage questions, idiomatic rewrites). Do not use `max` by
default: the Claude Code docs describe it as prone to overthinking with
diminishing returns.

## Prompt caching

Cache-read prices are 10 to 50x below list input price (`claude-opus-5-5`
$0.20, `deepseek-flash` $0.006, `GLM-5.3-Flash` $0.03 per 1M). Caches are per
model and per provider, so one `set_model` in the middle of a long context
repays the whole context at full input price on the new model. Route at
sub-agent boundaries, where a fresh context starts anyway, and change the
main-loop model only at phase boundaries.

## Where cheap routing fails

- Units whose difficulty is only visible after execution (a "simple" bug that
  turns out to be a design flaw). The fix is escalation on a verified failure,
  not a smarter up-front classifier: RouteLLM's routers dropped to near random
  on out-of-distribution queries, and TwinRouterBench found the information
  needed to route well is produced by execution, not present in the prompt.
- Retry loops. A small model that fails, retries with more context and then
  escalates costs more per accepted task than starting one tier higher.
  Escalate on the first verified failure; never retry the same tier.
- Self-reported success. Small-tier agents over-report; accept only through
  the mechanical check named in the unit.
- Context-heavy phases. When two phases share most of their context, a
  sub-agent must re-read it all; keep such phases in one loop.
- Anything irreversible or security-sensitive, and the final acceptance
  judgement: wrong answers there are expensive to detect, which is exactly
  when a cheap model's error rate matters most.

## Sources

- Anthropic pricing: https://claude.com/pricing#api
- OpenAI pricing: https://developers.openai.com/api/docs/pricing
- Google pricing: https://ai.google.dev/gemini-api/docs/pricing
- DeepSeek pricing: https://api-docs.deepseek.com/quick_start/pricing
- Z.ai pricing: https://docs.z.ai/guides/overview/pricing
- Moonshot pricing: https://platform.kimi.ai/docs/pricing/chat
- xAI pricing: https://docs.x.ai/developers/pricing
- Terminal-Bench 2.0 leaderboard: https://www.tbench.ai/?version=2.0
- Artificial Analysis index and cost per task: https://artificialanalysis.ai/
- OpenRouter rankings by token volume: https://openrouter.ai/rankings
