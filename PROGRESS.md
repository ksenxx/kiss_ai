# Progress

## Task: Sorcar CLI interactive mode — agent output landing in input area

### Bug
In `sorcar` interactive mode, while the anchored bottom input box is
active, agent display output (`bash_stream`, `_flush_newline`,
`_print_tool_result`, `_handle_message`) was appearing **inside the
input area** instead of in the scroll region above it.

### Root cause
`kiss.agents.sorcar.cli_client.run_client()` constructs the
`ConsolePrinter` at line 1232 **before** entering the `AnchoredRepl`
context (which installs a `_StdoutProxy` over `sys.stdout` at line
1264).  `ConsolePrinter.__init__` captured a *reference* to the
original `sys.stdout` into `self._file`:

```python
self._file = file or sys.stdout
```

Rich's `Console(file=None)` resolves `sys.stdout` lazily at write
time, so panels rendered through `self._console.print(...)` correctly
went through the proxy.  But every **direct** `self._file.write(...)`
call (in `print(..., type="bash_stream")`, `_flush_newline`,
`_print_tool_result`, `_handle_message`) bypassed the proxy and
landed at the terminal's current cursor position — which the proxy
parks inside the input box body — so streamed output was written
into the input area at the bottom.

### Test reproducing the issue
Added `TestConsolePrinterStdoutBypass` in
`src/kiss/tests/agents/sorcar/test_cli_steering.py` with two
end-to-end cases:

1. `test_bash_stream_output_routes_through_proxy` — asserts
   `bash_stream` output is wrapped in the proxy's
   `ESC 8 … ESC 7` framing (restore output cursor / re-save), proving
   it landed in the scroll region rather than the box body.
1. `test_flush_newline_routes_through_proxy` — asserts the same for
   the `\n` written by `_flush_newline` when `_mid_line` is set.

Both tests fail on the pre-fix code and pass after the fix.

### Fix
`src/kiss/core/print_to_console.py` — replace the eagerly-captured
`self._file` attribute with a lazy property:

```python
def __init__(self, file: Any = None) -> None:
    self._console = Console(highlight=False, file=file)
    self._explicit_file: Any = file
    ...

@property
def _file(self) -> Any:
    return self._explicit_file if self._explicit_file is not None else sys.stdout
```

When an explicit `file=` was passed (test paths use `io.StringIO()`),
behaviour is unchanged.  When `file=None` (the production code path),
`sys.stdout` is resolved at each access — matching the lazy
resolution Rich's `Console` already does — so a later swap of
`sys.stdout` (the CLI's `_StdoutProxy`) is honoured.

### Verification
* `uv run check --full` — all checks pass (lint, mypy, pyright,
  mdformat, syntax, dependency sync, generate-api-docs).
* Printer-related test suites all pass (159 tests):
  `test_cli_steering`, `test_cli_repl`, `test_cli_client`,
  `test_cli_chat_webview_events`, `test_cli_running_task_history_dot`,
  `test_print_to_console`, `test_printer_parity`,
  `test_tool_result_visible_for_all_tools`.
