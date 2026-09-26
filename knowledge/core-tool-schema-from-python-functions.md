---
title: Tool schema generation from Python functions (docstrings and type hints)
uuid: 5f2b0cda-6acf-443d-9347-c587b084a80b
summary: 'How a Python function becomes a tool: Model._function_to_openai_tool uses
  the signature, get_type_hints, the first docstring line and the Args section; the
  schema is cached per run.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Tool schema generation from Python functions

Any callable passed to `KISSAgent.run(tools=[...])` becomes a tool. No decorator is needed.
`KISSAgent._setup_tools` registers the functions by `__name__` and calls
`self.model._build_openai_tools_schema(self.function_map)` **once per run**. It calls it again after a
fallback model swap, because provider formats differ. The cached list is passed to every
`generate_and_process_with_tools(..., tools_schema=...)` call.

## Conversion (`Model._function_to_openai_tool`)
- **Name**: `func.__name__`. Two tools with the same name make `_add_functions` raise `KISSError`.
  Bound methods work (`self` is not in the signature).
- **Description**: **only the first line** of `inspect.getdoc(func)`. With no docstring it is
  `"Function <name>"`. Put the essential instruction in the first line; later paragraphs never reach the model.
- **Parameter descriptions**: `_parse_docstring_params` reads the Google-style `Args:` section. Every line
  that contains `:` is read as `name: description`, and `name (type): desc` also works. Parsing stops
  at `Returns:`, `Raises:` or `Example:`. Continuation lines that contain a colon can be misread as a new param.
- **Types**: `typing.get_type_hints(func)` resolves annotations under `from __future__ import annotations`.
  Before this was added, stringified hints fell back to `string`. `_python_type_to_json_schema` maps
  `str/int/float/bool/None` to string/integer/number/boolean/null. `X | None` and `Optional[X]` become X.
  Other unions become `anyOf`. `list[T]` becomes an array of T and a parameterized `dict[K, V]` becomes
  an object. Bare `list`/`dict` (no type arguments), a missing annotation, or any other type becomes `string`.
- **Required**: params without a default.
- The output is in OpenAI format: `{"type":"function","function":{"name","description","parameters"}}`.
  Provider adapters (Anthropic, Gemini, ...) convert it themselves in their own
  `_build_openai_tools_schema` overrides and generate paths (models area).

## Calling
`_execute_tool` calls `function_map[name](**arguments)` and converts the return value with `str()`.
Exceptions (including `SystemExit`) come back to the model as
`"Failed to call <name> with <args>: <err>\nExpected signature: <name><sig>"`. `BudgetExceededError`
propagates instead.

## Tips
- Keep parameter types JSON-friendly. Models send strings for untyped params.
- The built-in `KISSAgent.finish(result: str)` is plain text. Pass `kiss.core.utils.finish` for the
  structured YAML contract (see `core-finish-contract-and-implicit-finish`).

## Sources
- `src/kiss/core/models/model.py` (`Model._build_openai_tools_schema`, `_function_to_openai_tool`, `_parse_docstring_params`, `_python_type_to_json_schema`)
- `src/kiss/core/kiss_agent.py` (`KISSAgent._setup_tools`, `_add_functions`, `_execute_tool`)
