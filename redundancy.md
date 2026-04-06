# Redundancy, Duplication, AI Slop, and Inconsistency Analysis

## Issue 1: Three different `finish()` functions with overlapping purpose

**Files:**
- `src/kiss/core/kiss_agent.py` — `KISSAgent.finish(result: str)` (trivial identity function)
- `src/kiss/core/relentless_agent.py` — `finish(success, is_continue, summary)` (YAML-producing)
- `src/kiss/core/utils.py` — `finish(status, analysis, result)` (YAML-producing)

**Problem:** Three `finish` functions serve different agent tiers but have confusingly similar names and overlapping YAML-production logic. `utils.finish` and `relentless_agent.finish` both do `yaml.dump(dict, sort_keys=False)` with slightly different keys. `KISSAgent.finish` is a trivial identity method that just returns its argument.

**Fix:** No code change — these serve distinct agent contracts (KISSAgent's is an LLM tool, relentless_agent's manages sub-session continuations, utils' is for GEPA benchmarks). However, document the distinction clearly.

**Test plan:** N/A (documentation only).

---

## Issue 2: Dead code in `utils.py` — `get_config_value`, `add_prefix_to_each_line`, `read_project_file`, `read_project_file_from_package`

**Files:** `src/kiss/core/utils.py`

**Problem:** Four functions have zero callers outside tests:
- `get_config_value()` — never imported or called anywhere in production code
- `add_prefix_to_each_line()` — never imported or called
- `read_project_file()` — never imported or called
- `read_project_file_from_package()` — never imported or called

These are dead code that adds maintenance burden and confuses the API surface.

**Fix:** Remove all four functions and their associated tests.

**Test plan:** Verify no imports break after removal. Run full test suite to confirm.

---

## Issue 3: Duplicated `usage_info_for_messages` append logic in `AnthropicModel.add_function_results_to_conversation_and_return`

**Files:**
- `src/kiss/core/models/model.py` — `Model.add_function_results_to_conversation_and_return()` appends `self.usage_info_for_messages` to result content
- `src/kiss/core/models/anthropic_model.py` — `AnthropicModel.add_function_results_to_conversation_and_return()` duplicates the same `usage_info_for_messages` append logic

**Problem:** `AnthropicModel` overrides `add_function_results_to_conversation_and_return` entirely because Anthropic uses `tool_result` blocks with `tool_use_id` matching, vs. the base class's OpenAI-style `tool` role messages. However, the `usage_info_for_messages` appending logic is copy-pasted:

```python
# In model.py line 216-217:
if self.usage_info_for_messages:
    result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

# In anthropic_model.py line 306-307:
if self.usage_info_for_messages:
    result_content = f"{result_content}\n\n{self.usage_info_for_messages}"
```

**Fix:** Extract `_enrich_result_content(result_content: str) -> str` into the base `Model` class that appends `usage_info_for_messages` if set. Both implementations call this helper instead of duplicating the check.

**Test plan:** Integration test that creates an `AnthropicModel` instance (using conversation manipulation, not API calls), sets `usage_info_for_messages`, calls `add_function_results_to_conversation_and_return`, and verifies usage info is appended to the result content in the conversation. Same for base `Model` path (via `OpenAICompatibleModel` or `GeminiModel`).

---

## Issue 4: Inconsistent `__str__`/`__repr__` definitions across Model subclasses

**Files:**
- `src/kiss/core/models/model.py` — `Model` defines `__str__` and `__repr__ = __str__`
- `src/kiss/core/models/openai_compatible_model.py` — `OpenAICompatibleModel` overrides `__str__` and `__repr__ = __str__` (adds base_url)
- `src/kiss/core/models/anthropic_batch_model.py` — `AnthropicBatchModel` overrides `__str__` and `__repr__ = __str__` (custom format)
- `src/kiss/core/models/anthropic_model.py` — No override (inherits base)
- `src/kiss/core/models/gemini_model.py` — No override (inherits base)
- `src/kiss/core/models/claude_code_model.py` — No override (inherits base)

**Problem:** The `__repr__ = __str__` line is duplicated in 3 places. Two subclasses override to include extra info (base_url), while others rely on the base. The `AnthropicBatchModel.__str__` returns `AnthropicBatchModel(name=...)` but inheriting from `AnthropicModel` which inherits from `Model` which already returns `ClassName(name=...)` using `self.__class__.__name__` — so the override is unnecessary.

**Fix:** Remove the redundant `__str__` and `__repr__` from `AnthropicBatchModel` (it already gets the correct class name from `Model.__str__` which uses `self.__class__.__name__`). Keep `OpenAICompatibleModel`'s override since it genuinely adds `base_url`.

**Test plan:** Test that `str(AnthropicBatchModel(...))` returns the correct class name after removing the override. Test all model `__str__` representations.

---

## Issue 5: `_build_create_kwargs` in `AnthropicBatchModel` uses model_name swap hack

**Files:** `src/kiss/core/models/anthropic_batch_model.py`

**Problem:** `AnthropicBatchModel._build_create_kwargs` temporarily swaps `self.model_name` to strip the `batch/` prefix so the parent's thinking-mode detection works:

```python
saved = self.model_name
self.model_name = self._api_model_name
try:
    kwargs = super()._build_create_kwargs(tools=tools)
finally:
    self.model_name = saved
kwargs["model"] = self._api_model_name
```

This is fragile and thread-unsafe. The parent `_build_create_kwargs` should use a method to get the API model name rather than reading `self.model_name` directly.

**Fix:** In `AnthropicModel._build_create_kwargs`, introduce `self._get_api_model_name()` that returns `self.model_name` by default. `AnthropicBatchModel` overrides it to strip the `batch/` prefix. The parent uses this method instead of `self.model_name` for both the `model` key and the thinking-detection check.

**Test plan:** Test that `AnthropicBatchModel._build_create_kwargs()` produces correct kwargs with `model` set to the stripped name and thinking config properly detected, without the swap hack.

---

## Issue 6: `_is_retryable_error` check uses string matching on type names — fragile

**File:** `src/kiss/core/kiss_agent.py`

**Problem:** `_is_retryable_error` checks `type(e).__name__` against string patterns and `str(e).lower()` against phrase lists. This is AI-generated pattern matching that's fragile (class name changes, localized error messages, etc.).

**Fix:** This is acceptable for now as a defense-in-depth heuristic — the patterns cover well-known API client error types. No change needed.

**Test plan:** N/A.

---

## Issue 7: `OpenAICompatibleModel._api_model_name` pattern duplicated in `AnthropicBatchModel._api_model_name`

**Files:**
- `src/kiss/core/models/openai_compatible_model.py` — `self._api_model_name` strips `openrouter/` prefix
- `src/kiss/core/models/anthropic_batch_model.py` — `self._api_model_name` strips `batch/` prefix

**Problem:** Both store a "cleaned" model name as `_api_model_name` for API calls, but the attribute name collision is accidental — they strip different prefixes for different reasons. This is not actually duplication since each is domain-specific.

**Fix:** No change needed.

---

## Issue 8: `config_to_dict` strips API_KEY by name matching, not by field metadata

**File:** `src/kiss/core/utils.py`

**Problem:** `config_to_dict` uses `"API_KEY" not in k` to filter sensitive fields. This is fragile — a field named `SOME_API_KEY_PREFERENCE` would be incorrectly excluded.

**Fix:** This is low priority and works correctly for the current config structure. No change.

---

## Issue 9: Channel agent config persistence has copy-paste boilerplate across 20+ agents

**Files:** All files in `src/kiss/channels/` (slack_agent.py, discord_agent.py, telegram_agent.py, etc.)

**Problem:** Every channel agent defines nearly identical:
- `_config_path()` function
- `_load_config()` function  
- `_save_config()` function
- `_clear_config()` function

These are 4-8 line functions that differ only in the directory name and required keys. Example:

```python
# slack_agent.py
def _config_path() -> Path:
    return _SLACK_DIR / "config.json"

def _load_config() -> dict[str, str] | None:
    return load_json_config(_config_path(), ("access_token",))

# discord_agent.py  
def _config_path() -> Path:
    return _DISCORD_DIR / "config.json"

def _load_config() -> dict[str, str] | None:
    return load_json_config(_config_path(), ("bot_token",))
```

Already partially addressed via `_channel_agent_utils.py` with `load_json_config`/`save_json_config`/`clear_json_config`, but each agent still wraps these with boilerplate.

**Fix:** Create a `ChannelConfig` dataclass or simple helper in `_channel_agent_utils.py`:

```python
class ChannelConfig:
    def __init__(self, channel_name: str, required_keys: tuple[str, ...]):
        self._dir = Path.home() / ".kiss" / "channels" / channel_name
        self._required_keys = required_keys

    @property
    def path(self) -> Path:
        return self._dir / "config.json"

    def load(self) -> dict[str, str] | None:
        return load_json_config(self.path, self._required_keys)

    def save(self, data: dict[str, str]) -> None:
        save_json_config(self.path, data)

    def clear(self) -> None:
        clear_json_config(self.path)
```

Each channel agent replaces 4 functions with `_config = ChannelConfig("slack", ("access_token",))`.

**Test plan:** Integration test that creates a `ChannelConfig`, saves a config to a temp dir, loads it, and clears it. Test with missing keys, empty values, and non-existent paths.

---

## Issue 10: `Model._parse_docstring_params` dead branch — `len(parts) == 2` always True

**File:** `src/kiss/core/models/model.py`

**Problem:** In `_parse_docstring_params`:
```python
parts = stripped.split(":", 1)
if len(parts) == 2:  # pragma: no branch
```

`str.split(":", 1)` with `":"` in the string (already checked by `if ":" in stripped`) always returns exactly 2 parts. The `if len(parts) == 2` check is dead code — it can never be False given the outer condition. The `# pragma: no branch` comment confirms this.

**Fix:** Remove the unnecessary `if len(parts) == 2` check. The code already guarantees 2 parts due to the `if ":" in stripped` condition.

**Test plan:** Test that docstring parsing still works correctly after removing the dead check.

---

## Issue 11: `_OPENAI_PREFIXES` checked inconsistently — `"openai/gpt-oss"` special case

**File:** `src/kiss/core/models/model_info.py`

**Problem:** The model routing uses:
```python
_OPENAI_PREFIXES = ("gpt", "text-embedding", "o1", "o3", "o4", "codex", "computer-use")
```

But then has a special exclusion:
```python
if model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith("openai/gpt-oss"):
```

The `openai/gpt-oss` models don't actually match `_OPENAI_PREFIXES` (they start with "openai/", not "gpt"/"o1"/etc.), so the exclusion check `not model_name.startswith("openai/gpt-oss")` is dead code — it can never trigger because `model_name.startswith(_OPENAI_PREFIXES)` would already be False for "openai/gpt-oss-*".

Wait — `_TOGETHER_PREFIXES` includes `"openai/gpt-oss"`, so those models are routed to Together AI. The OpenAI prefix check is actually fine as-is. But the exclusion in the `_OPENAI_PREFIXES` branch is unnecessary dead code.

**Fix:** Remove `and not model_name.startswith("openai/gpt-oss")` from the OpenAI routing check since `"openai/gpt-oss"` never matches `_OPENAI_PREFIXES` in the first place.

Similarly in `get_available_models()`:
```python
elif name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/gpt-oss"):
```

**Test plan:** Test that `model("openai/gpt-oss-120b")` still routes to Together AI. Test that `model("gpt-4o")` still routes to OpenAI. Verify the dead check removal doesn't change behavior.

---

## Issue 12: `KISSAgent._run_agentic_loop` has unreachable `raise KISSError` at the end

**File:** `src/kiss/core/kiss_agent.py`

**Problem:** After the `for _ in range(self.max_steps)` loop, there's:
```python
raise KISSError(  # pragma: no cover
    f"Agent {self.name} completed {self.max_steps} steps without finishing."
)
```

This is unreachable because `_check_limits()` at the top of each iteration already raises `KISSError` when `self.step_count > self.max_steps`. Since `step_count` is incremented before `_check_limits()`, by the time `step_count` reaches `max_steps + 1`, the check fires. The loop runs `max_steps` times, with `step_count` going from 1 to `max_steps`, then on iteration `max_steps + 1` (which doesn't exist since the range is `max_steps`), the loop would exit... 

Actually, looking more carefully: `for _ in range(self.max_steps)` runs `max_steps` iterations. On the first iteration, `step_count` becomes 1. On the last iteration, `step_count` becomes `max_steps`. Then `_check_limits()` checks `self.step_count > self.max_steps` which is `max_steps > max_steps` = False. So the loop completes all iterations, and if no result was returned, the `raise` at the bottom IS reachable.

Wait — but the `_check_limits` happens at the START of the step before incrementing? No, let me reread:

```python
for _ in range(self.max_steps):
    self.step_count += 1    # Goes 1, 2, ..., max_steps
    self._check_limits()     # Checks step_count > max_steps
```

When step_count = max_steps, the check is `max_steps > max_steps` = False, so it passes. The step executes. After all `max_steps` iterations, the loop exits and the raise is hit. So this is actually reachable in theory — but `pragma: no cover` marks it as expected to be unreachable in practice.

**Fix:** No change — the code is correct. The `pragma: no cover` just means it's hard to trigger in tests.

---

## Issue 13: `_ArtifactDirProxy.__eq__` and `__hash__` in `config.py`

**File:** `src/kiss/core/config.py`

**Problem:** `_ArtifactDirProxy` implements `__eq__` and `__hash__` to behave like a string, but it's a singleton used as a module-level variable. The `__eq__` with other `_ArtifactDirProxy` instances compares their resolved paths, and `__hash__` hashes the resolved path. This is reasonable.

**Fix:** No change needed.

---

## Summary of Planned Changes

| # | Type | File(s) | Change | Risk |
|---|------|---------|--------|------|
| 2 | Dead code | `utils.py` | Remove 4 unused functions | Low |
| 3 | Duplication | `model.py`, `anthropic_model.py` | Extract `_enrich_result_content()` helper | Low |
| 4 | Redundancy | `anthropic_batch_model.py` | Remove unnecessary `__str__`/`__repr__` override | Low |
| 5 | Fragile hack | `anthropic_batch_model.py`, `anthropic_model.py` | Replace model_name swap with `_get_api_model_name()` method | Medium |
| 9 | Boilerplate | `_channel_agent_utils.py`, all channel agents | Introduce `ChannelConfig` helper class | Medium |
| 10 | Dead branch | `model.py` | Remove always-true `len(parts) == 2` check | Low |
| 11 | Dead code | `model_info.py` | Remove dead `openai/gpt-oss` exclusion in OpenAI routing | Low |

## Test Plan for Each Change

### Issue 2 Tests
- Verify that removing the functions doesn't break any imports
- Run full test suite to confirm no tests reference them (except their own tests, which are also removed)

### Issue 3 Tests
- Test `_enrich_result_content("hello")` with empty `usage_info_for_messages` returns "hello"
- Test `_enrich_result_content("hello")` with `usage_info_for_messages = "info"` returns "hello\n\ninfo"
- Integration test: Create `AnthropicModel` (no API call), set conversation with assistant tool_use blocks, call `add_function_results_to_conversation_and_return` with usage info set, verify content contains the info
- Integration test: Same for base `Model.add_function_results_to_conversation_and_return` (via any concrete subclass)

### Issue 4 Tests
- Test `str(AnthropicBatchModel("batch/claude-opus-4-6", api_key="test"))` returns `"AnthropicBatchModel(name=batch/claude-opus-4-6)"`
- Test `repr()` returns the same as `str()` for all model classes

### Issue 5 Tests
- Test `AnthropicBatchModel._build_create_kwargs()` with a model name like `batch/claude-opus-4-6` produces `kwargs["model"] == "claude-opus-4-6"`
- Test thinking config is correctly detected for `batch/claude-opus-4-6` (opus 4.x should get adaptive thinking)
- Test `batch/claude-sonnet-4` gets budget thinking (10000 tokens)
- Test `batch/claude-3-5-haiku` (non-4.x) gets no thinking config

### Issue 9 Tests
- Test `ChannelConfig("test_channel", ("token",))` with:
  - Save and load a config in a temp directory
  - Load from non-existent file returns None
  - Load with missing required key returns None
  - Clear removes the file
  - Verify file permissions on non-Windows

### Issue 10 Tests
- Test `_parse_docstring_params` with docstrings containing "Args:" section
- Test with docstrings containing params with "(type)" format
- Test with empty docstring
- Test with docstring without Args section

### Issue 11 Tests
- Test that `model("openai/gpt-oss-120b")` routes to Together AI (raises or succeeds based on API key)
- Test that `model("gpt-4o")` routes to OpenAI
- Test `get_available_models()` includes/excludes models correctly based on API keys
