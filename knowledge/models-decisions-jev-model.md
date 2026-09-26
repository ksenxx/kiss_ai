---
title: 'Decisions (jev) model and the Sorcar decide tool: noul / choice / score questions'
uuid: 949164af-2ccc-43c1-a4b4-e0ae71064e12
summary: 'openrouter/~typesafe/jev-latest: dec flag, DecisionsModel.decide on /api/alpha/decisions,
  noul/choice/score questions, usage.cost, Sorcar decide tool (validation, availability,
  profiles, cost).'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Decisions (jev) model and the decide tool

## What it is
TypeSafe's `~typesafe/jev-latest` and `typesafe/jev-1.13`, served by OpenRouter, are **not text
generators**. A request carries a `state` (text or JSON) plus named typed `questions`; the answer
is a typed value with calibrated probabilities, produced in one non-generative pass costing a
fraction of an LLM round trip. OpenRouter rejects them on `/chat/completions` (HTTP 400), so they
need their own adapter.

Question types (`QUESTION_TYPES`), with builders in `decisions_model.py`:
- `noul(instructions)` -> answer `{"type": "noul", "noul": p_true}`. No criteria.
- `choice(instructions, criteria)` (`{id: description}` or a list of ids; a list becomes `{id: id}`)
  -> `choice`, `probabilities` per option, `confidence`.
- `score(instructions, criteria)` (list of rubric levels, ascending; a mapping is rejected with
  400, so the builder forces a list) -> fractional `score`, `legend`, `probabilities` per level,
  `confidence`.

## Catalog and routing
Catalog entries: `openrouter/~typesafe/jev-latest` and `openrouter/typesafe/jev-1.13` with
`"dec": true` (`ModelInfo.is_decisions_supported`) and `gen`/`fc`/`emb` false, so they never
appear in the agent model picker (`get_available_models` filters on `gen`). In `model()` the `dec`
check runs **first**, before prefix routing and before a `base_url` override, and calls
`_decisions_model`: `base_url` (default `OPENROUTER_DECISIONS_BASE_URL = "https://openrouter.ai/api"`,
not `/api/v1`) and `api_key` (default `OPENROUTER_API_KEY`) are consumed from `model_config`.

## `DecisionsModel` (`src/kiss/core/models/decisions_model.py`)
- `endpoint_url = base_url + "/alpha/decisions"` (`DECISIONS_PATH`).
- `decide(state, questions)`: validates types, POSTs `{"model": api_model_id(name), "state",
  "questions"}` with `requests` (Bearer auth, `extra_headers` from config, `timeout` default 60 s).
  `api_model_id` strips the `openrouter/` prefix. Errors (bad type, non-2xx, network, non-JSON, no
  `answers` object) all raise `KISSError`. Returns the parsed body: `model`, `answers`, `usage`
  (`input_tokens`, `output_tokens`, `cost`).
- Model-contract methods: `generate()` judges the concatenated conversation text against
  `model_config["questions"]` (raises `KISSError` if none), returns `json.dumps(answers)` and calls
  the token callback once (no streaming). `generate_and_process_with_tools` always raises.
  Attachments are dropped with a warning.
- Accounting: `extract_input_output_token_counts_from_response` -> `(input, output, 0, 0)` (no
  prompt cache); `extract_cost_from_response` -> `usage.cost` via `reported_cost`.

## Sorcar `decide` tool (`src/kiss/agents/sorcar/decide_tool.py`)
`make_decide_tool(agent, model_name=DEFAULT_DECISIONS_MODEL, model_config=None)` with
`DEFAULT_DECISIONS_MODEL = "openrouter/~typesafe/jev-latest"` returns `decide(state, questions)`.
It raises `KISSError` if the model is not a `DecisionsModel`. The pre-run task classifier
(`task_classifier.py`, see `sorcar-task-classifier`) builds its calls through the same module.

**Questions format.** `questions` is a JSON **string** (same convention as `run_parallel`'s
`tasks`): `{"<name>": {"type": "noul"|"choice"|"score", "instructions": "...", "criteria": ...}}`.
`parse_questions` rebuilds each entry with the `noul`/`choice`/`score` builders so the wire shape
matches the endpoint. It raises `KISSError` on invalid JSON, a non-object or empty object, a blank
name, a non-object entry, an unknown type, blank instructions, or badly shaped criteria: `choice`
needs a non-empty dict or list of non-blank strings (a number as an option id would be an HTTP
400); `score` needs a list of **at least two** non-blank levels. The tool catches `KISSError`
(parse or request) and returns `"Error: ..."` instead of raising.

**Output and cost.** `format_decision` returns `(text, cost_usd, total_tokens)`; `text` is JSON
`{"answers", "model" (served id), "usage": {input_tokens, output_tokens, cost_usd}}`. The cost is
OpenRouter's `usage.cost` when present, else `calculate_cost(model, input, output)`. When
`agent` is set and cost or tokens are non-zero, it is charged to the task through
`sorcar_agent._attribute_sub_usage(agent, cost, tokens, 0)`, the same way `talk` TTS is charged;
`agent=None` skips attribution (library use).

**Availability.** `decisions_tool_available()` requires `DEFAULT_CONFIG.OPENROUTER_API_KEY` and a
`MODEL_INFO` entry for the default model with `is_decisions_supported`. Without them the tool is
**not offered at all** (an unusable tool only costs prompt tokens). When available,
`SorcarAgent._get_tools` adds it in the `full` profile and the restricted `review` and
`assistant` profiles (not `shell` or `bash`).

## Usage example
```python
from kiss.core.models.model_info import model
from kiss.core.models.decisions_model import noul, choice
m = model("openrouter/~typesafe/jev-latest")
r = m.decide("Checkout crashes on submit", {
    "is_bug": noul("Does the message report a software bug?"),
    "team": choice("Which team?", ["billing", "support"]),
})
r["answers"]["is_bug"]["noul"]
```

## Sources
- `src/kiss/core/models/decisions_model.py` (`DecisionsModel`, `QUESTION_TYPES`, `noul`, `choice`, `score`, `reported_cost`, `api_model_id`, `OPENROUTER_DECISIONS_BASE_URL`, `DECISIONS_PATH`)
- `src/kiss/core/models/model_info.py` (`model`, `_decisions_model`, `ModelInfo.is_decisions_supported`, `calculate_cost`)
- `src/kiss/agents/sorcar/decide_tool.py` (`make_decide_tool`, `decisions_tool_available`, `parse_questions`, `format_decision`, `DEFAULT_DECISIONS_MODEL`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_get_tools`, tool profiles `review`/`assistant`, `_attribute_sub_usage`)
- `src/kiss/agents/sorcar/task_classifier.py` (imports `make_decide_tool`)
