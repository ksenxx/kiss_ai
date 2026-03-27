# Redundancy & Duplication in Uncommitted Changes

## 1. `_web_use_callback` duplicates `_wait_for_user_callback` in `sorcar_agent.py`

In `run()`, both attributes store the identical `wait_for_user_callback` value:
```python
self._wait_for_user_callback = wait_for_user_callback   # never read by agent logic
self._web_use_callback = wait_for_user_callback          # used in _get_tools()
```
`_wait_for_user_callback` on SorcarAgent is only ever assigned and cleared—never read by any agent logic. `_web_use_callback` is the one actually consumed in `_get_tools()`.

**Fix:** Remove `_web_use_callback` entirely. Initialize `self._wait_for_user_callback: Callable | None = None` in `__init__` and use it directly in `_get_tools()` (replacing the `getattr` defensive access). This eliminates a redundant attribute and the duplicated cleanup line.

## 2. "Resolve tools_schema or build from function_map" duplicated 3× across model subclasses

The exact same ternary pattern was added to each provider:

**`openai_compatible_model.py`:**
```python
tools = tools_schema if tools_schema is not None else self._build_openai_tools_schema(function_map)
```

**`gemini_model.py`:**
```python
source = tools_schema if tools_schema is not None else self._build_openai_tools_schema(function_map)
```

**`anthropic_model.py` (`_build_anthropic_tools_schema`):**
```python
source = openai_schema if openai_schema is not None else self._build_openai_tools_schema(function_map)
```

**Fix:** Add a single helper in the base `Model` class:
```python
def _resolve_openai_tools_schema(
    self,
    function_map: dict[str, Callable[..., Any]],
    tools_schema: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    return tools_schema if tools_schema is not None else self._build_openai_tools_schema(function_map)
```
Replace all three call-sites with `self._resolve_openai_tools_schema(function_map, tools_schema)`. Also remove the extra `openai_schema` parameter from `_build_anthropic_tools_schema` (it can receive the already-resolved list).
