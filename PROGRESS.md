# Task: Convert `xhigh=True` flag to `thinking="xhigh"` string field

## Goal

Replace boolean `supports_xhigh_reasoning_effort` / `xhigh=True` with a more
general `thinking: str | None` field on `ModelInfo` that stores the
`reasoning_effort` level the model accepts. `OpenAICompatibleModel` then
defaults `reasoning_effort` to that level.

## Done so far

### 1. `src/kiss/core/models/model_info.py` — DONE

- `ModelInfo.__init__`: replaced `supports_xhigh_reasoning_effort: bool = False`
  with `thinking: str | None = None`. Stored on `self.thinking`.
- `_mi(...)` helper: replaced `xhigh: bool = False` with
  `thinking: str | None = None`. Docstring updated.
- All 4 `xhigh=True` call sites converted to `thinking="xhigh"`:
  - `"gpt-5.5"`
  - `"gpt-5.5-2026-04-23"`
  - `"openrouter/openai/gpt-5.5"`
  - `"openrouter/~openai/gpt-latest"`

### 2. `src/kiss/core/models/openai_compatible_model.py` — DONE

- `_model_supports_xhigh(name) -> bool` replaced with
  `_model_thinking_level(name) -> str | None` returning
  `MODEL_INFO[name].thinking`.
- `__init__` now does:
  ```python
  thinking_level = _model_thinking_level(self.model_name)
  if thinking_level is not None and "reasoning_effort" not in self.model_config:
      self.model_config = dict(self.model_config)
      self.model_config["reasoning_effort"] = thinking_level
  ```

### 3. `src/kiss/scripts/update_models.py` — PARTIALLY DONE

- `get_current_model_info`: emits `"thinking": info.thinking` (str|None)
  instead of `"xhigh": bool`. DONE.
- `test_xhigh_reasoning_effort(name) -> bool` replaced with
  `detect_thinking_level(name) -> str | None` that probes each level in
  `_THINKING_LEVELS_TO_PROBE = ("xhigh",)` and returns the first that
  succeeds, else `None`. DONE.

## Remaining work

### `src/kiss/scripts/update_models.py` still to update

Caller of test in `test_model_capabilities` (around line 423) currently does:

```python
results["xhigh"] = test_xhigh_reasoning_effort(model_name)
...
results["xhigh"] = False
```

Need to change to:

```python
results["thinking"] = detect_thinking_level(model_name)
...
results["thinking"] = None
```

Also the `flags` print line uses `'Y' if v else 'N'` — that needs to handle
string|None values, e.g. `v or 'N'`.

`_make_entry_line` at line ~803-830: signature uses `xhigh: bool = False`.
Replace with `thinking: str | None = None`. Body:

```python
if thinking:
    extras.append(f'thinking="{thinking}"')
```

`apply_updates_to_file` at line ~877: `new_xhigh = upd["changes"].get("xhigh", cur.get("xhigh", False))`
and `xhigh=new_xhigh,`. Replace with `thinking` key everywhere.

Line ~930: `xhigh=nm.get("xhigh", False),` → `thinking=nm.get("thinking"),`.

Line ~1087: `nm["xhigh"] = caps["xhigh"]` → `nm["thinking"] = caps["thinking"]`.
Line ~1098: `nm["xhigh"] = False` → `nm["thinking"] = None`.
Lines ~1106-1124: `--test-existing` block — rename `xhigh_changed` /
`changes["xhigh"]` to `thinking_changed` / `changes["thinking"]`; comparison
becomes `caps["thinking"] != cur.get("thinking")`.

### `src/kiss/tests/core/models/test_openai_xhigh_default.py`

- Last test uses `MODEL_INFO["gpt-4o"].supports_xhigh_reasoning_effort`.
  Replace with `.thinking` and assign `"xhigh"` / `None` instead of
  `True` / original-bool.

### `src/kiss/tests/scripts/test_update_models_xhigh.py`

- Import `detect_thinking_level as _probe` (rename).
- `_make_entry_line(..., xhigh=True)` → `thinking="xhigh"`.
- Assert generated line says `thinking="xhigh"` not `xhigh=True`.
- `"fc=False, xhigh=True"` → `'fc=False, thinking="xhigh"'`.
- Replace `_probe_xhigh(...) is False` with
  `_probe(...) is None` for all the short-circuit tests.
- `get_current_model_info` tests: assert `snapshot[name]["thinking"]`
  matches `info.thinking` (str|None). For `gpt-5.5` assert `== "xhigh"`,
  for `gpt-4o` assert `is None`.

### Verification

- `uv run pytest src/kiss/tests/core/models/test_openai_xhigh_default.py src/kiss/tests/scripts/test_update_models_xhigh.py -v`
- `uv run check --full`
