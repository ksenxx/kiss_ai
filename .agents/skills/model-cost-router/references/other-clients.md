# Applying the protocol outside Sorcar

The seven steps in `SKILL.md` are client-independent. What changes is the
lever used in Step 3 (dispatch) and Step 4 (phase plan), and the classifier
in Step 2: without `decide()`, classify by the unit-type table in the rubric
yourself and still apply "ties go up".

## Claude Code

Levers, in the order Claude Code resolves them (v2.1.251 and later):

1. `model` parameter on the Agent tool call, per invocation.
2. `model:` in the sub-agent's frontmatter (`haiku`, `sonnet`, `opus`,
   `fable`, a full model id, or `inherit`). Optional `effort:` (`low`,
   `medium`, `high`, `xhigh`, `max`) for that sub-agent only.
3. `CLAUDE_CODE_SUBAGENT_MODEL` env var; `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1`
   (v2.1.257+) makes it override frontmatter including the built-in Explore
   and Plan agents.
4. The main conversation's model.

Setup that implements the tiers without any per-call decision:

```markdown
# .claude/agents/explorer.md
---
name: explorer
description: Read-only exploration, grep, log and test-output triage. Returns a summary, never raw output.
model: haiku
effort: low
omitClaudeMd: true
---
```

```markdown
# .claude/agents/implementer.md
---
name: implementer
description: Implements a unit with a stated acceptance check inside the named files only.
model: sonnet
effort: medium
---
```

```json
// .claude/settings.json
{ "env": { "CLAUDE_CODE_SUBAGENT_MODEL": "haiku" } }
```

- Phase plan: `/model opusplan` runs Opus in plan mode and Sonnet in execution
  mode. `ANTHROPIC_DEFAULT_OPUS_MODEL` and `ANTHROPIC_DEFAULT_SONNET_MODEL`
  choose which models those aliases mean; `ANTHROPIC_DEFAULT_HAIKU_MODEL` also
  sets the model used for background summarisation.
- Since v2.1.198 the built-in Explore agent inherits the main model; define
  your own `Explore` agent with `model: haiku` to keep exploration cheap.
- `/model opus` mid-session also moves every inheriting sub-agent to Opus;
  pin `model:` in the agent definition to prevent that.
- Cache warning: `/model` and `/effort` switches invalidate the prompt cache
  and re-bill the history at the new model's input price; switch only at phase
  boundaries.
- Escalation: when an `implementer` run fails its check, re-dispatch with
  `model: opus` and the failure evidence; do not re-run Sonnet at `xhigh`.
- Ledger: the `log` sub-command of `scripts/route.py` works unchanged; run it
  from Bash after each dispatch.

Sources: https://code.claude.com/docs/en/sub-agents and
https://code.claude.com/docs/en/model-config

## OpenAI Codex CLI

- Sub-agents (`spawn_agent`, on by default via `features.multi_agent`): set
  `agents.default_subagent_model` and
  `agents.default_subagent_reasoning_effort` in `~/.codex/config.toml` to the
  small tier so every spawned agent is cheap unless the spawn names a model;
  "an explicit spawn model takes precedence". Declare one role per tier with
  `agents.<name>.config_file` (a TOML layer setting `model` and
  `model_reasoning_effort`) and `agents.<name>.description` so the main agent
  picks the role by unit type.
- Main-loop phase plan: `codex --model <id>` or `model = "<id>"`; profiles
  live next to `config.toml` as `$CODEX_HOME/<profile>.config.toml` and are
  selected with `--profile <profile>`, so one profile per tier gives the
  opusplan pattern across two `codex exec` invocations.
- `model_reasoning_effort` is the effort axis; the same guidance applies (low
  for scoped mechanical units, never the highest level by default).

Source: https://developers.openai.com/codex/config-reference

## OpenRouter, Not Diamond, Martian (API-level routers)

`openrouter/auto`, Not Diamond and Martian route a single request by
predicted quality; they have no view of agent state (files touched, attempts
failed, tests red). Use them, if at all, only for the small tier where the
downside of a wrong pick is bounded, and keep escalation in the agent loop.
RouteLLM's out-of-distribution result (see `evidence.md`) is the reason not to
trust a prompt-level router with medium-vs-frontier decisions in coding work.

## Any Agent Skills client

The frontmatter of this skill is valid under the Agent Skills specification
(https://agentskills.io/specification). Install with
`npx skills add <repo> --skill model-cost-router` or copy the directory into
`~/.agents/skills/`, `~/.claude/skills/`, or `~/.kiss/skills/`.
