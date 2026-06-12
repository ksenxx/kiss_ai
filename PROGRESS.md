# Task: Repeatedly run bug-hunt (find/reproduce/fix bugs in src/kiss/agents/vscode/, src/kiss/agents/sorcar/, src/kiss/core/ via integration tests) until an iteration finds no bugs.

## Result: COMPLETE (loop terminated after iteration 2 reported NO BUGS FOUND)

### Iteration 1 — BUGS FOUND: 3 (fixed in commit 5ddd1398)

1. `src/kiss/agents/sorcar/custom_commands.py` `expand_command()` — positional
   `$1..$9` and `$ARGUMENTS` substituted in two separate `re.sub` passes, so an
   argument value literally containing `$ARGUMENTS`/`$N` got re-expanded by the
   later pass. Fixed with a single-pass combined `_PLACEHOLDER_RE` substitution.
   Test: `src/kiss/tests/agents/sorcar/test_bughunt7_expand_args_reinjection.py`.
2. `src/kiss/core/utils.py` `escape_invalid_template_field_names()` — nested
   format spec `{a:{b}}` with valid `a` but invalid `b` produced `{a:{{b}}}`,
   which crashes `str.format` (ValueError: Invalid format specifier); the
   inverse case double-escaped the nested braces. Fixed: a placeholder is
   escaped whole (original spec preserved) if its field OR any field nested in
   its spec is invalid. Test: `src/kiss/tests/core/test_bughunt7_escape_nested_spec.py`.
3. `src/kiss/core/kiss_agent.py` `_set_prompt()` — sequential per-key
   `str.replace` allowed one argument's value containing another key's
   placeholder to be re-expanded (order-dependent value leakage). Fixed with a
   single-pass regex substitution. Test:
   `src/kiss/tests/core/test_bughunt7_prompt_arg_reinjection.py`.

All 16 tests across the 3 new test files pass; `uv run check --full` green.

### Iteration 2 — NO BUGS FOUND

Deep-probed and ruled out: web_server.py tunnel-metrics parsing &
`_is_pid_alive`, server.py sub-tab parent resolution under `_state_lock`,
relentless_agent.py summarizer/budget/result-broadcast paths, cli_steering.py
escape/paste deferral + registry locking + async-exc interruption,
mcp_servers.py connection lifecycle under `_lock`, useful_tools.py,
diff_merge.py C-quoted path parsing, git_worktree/merge_flow rename parsing
(git C-quotes paths with spaces — unambiguous), config_builder argparse bool
semantics, plus grep sweeps for mutable defaults, bare except, identity-vs-
equality, naive datetimes, and path-prefix startswith misuse — zero hits.

Loop terminated per the task's stop condition.
