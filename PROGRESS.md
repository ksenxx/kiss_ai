# Task: Bug-hunt iteration 3, group E (vscode backend: server.py / commands.py / task_runner.py)

## Session 2 state — ALL 6 BUGS FIXED, tests written failing-first and now passing

### Bugs fixed (file:line, root cause, fix, test)

1. **BUG-A** server.py `_handle_command`: unhashable `type` (`{"type": []}`) raised
   TypeError out of the dispatcher → killed the whole client connection.
   Fix: `self._HANDLERS.get(cmd_type) if isinstance(cmd_type, str) else None` →
   routes to the unknown-command error branch.
   Test: test_bughunt3_dispatch_malformed.py::test_unhashable_type_field_does_not_raise
1. **BUG-B** commands.py: unguarded `int()` in `_cmd_delete_task`, `_cmd_set_favorite`,
   `_cmd_resume_session`, `_cmd_get_frequent_tasks` → ValueError on `"abc"` killed the
   connection. Fix: new module-level `_parse_int(value) -> int | None` helper used by
   all four (limit falls back to 50).
   Tests: test_bughunt3_dispatch_malformed.py (4 tests).
1. **BUG-C** commands.py `_cmd_run`: busy guard `task_thread is not None and is_alive()`
   raced the post-lock `thread.start()` — a concurrent second submit saw a created-but-
   unstarted thread, passed the guard, clobbered stop_event/user_answer_queue/task_thread.
   Fix: guard is now `task_thread is not None` (non-None ⇔ task in flight; `_run_task`'s
   finally always resets it to None).
   Test: test_bughunt3_run_start_race.py (deterministic via blocking broadcast).
1. **BUG-D** task_runner.py `_run_task_inner`: malformed attachment (bad base64 /
   non-dict) raised binascii.Error/AttributeError before the big try; `_run_task` has no
   except → task thread died silently. Fix: per-attachment try/except, skip + log.
   Test: test_bughunt3_bad_attachment.py (2 tests).
1. **BUG-E** server.py `_new_chat` + commands.py `_cmd_select_model`: empty tabId minted a
   permanent phantom registry entry keyed "" (undisposable — `_cmd_close_tab` guards empty
   ids). Fix: `_new_chat` early-returns on empty tab_id; `_cmd_select_model` only touches
   the registry when tab_id is non-empty (still updates `_default_model` when a model is
   supplied). Test: test_bughunt3_newchat_phantom_tab.py (2 tests).
1. **BUG-F** commands.py `_cmd_user_answer`: non-string `answer` (None/number) was put on
   `Queue[str]` → `ask_user_question` returned None where str promised. Fix: coerce
   (None → "", other non-str → str(x)). Test: test_bughunt3_useranswer_nonstring.py (2 tests).

### Verification status

- All 5 new test files: 11 failed pre-fix → 12/12 pass post-fix. (The threaded
  excepthook test originally used `"!!!"` which b64decode silently ignores — data
  strengthened to `"%%%not-b64%%%"`.)
- TODO (this session, in order): run existing impacted tests
  (src/kiss/tests/agents/vscode + src/kiss/tests/agents/sorcar/test_vscode_tabs.py),
  fix regressions; `uv run check --full`; clean tmp/; final per-bug report.
  NOTE: `_cmd_select_model` semantics changed slightly — watch
  test_vscode_tabs.py selectModel tests for regressions.

# Task: Bug-hunt iteration 3, group C (sorcar CLI: cli_repl.py / cli_steering.py / cli_helpers.py / cli_panel.py)

## Final state — ALL 4 NEW BUGS FIXED, tests written failing-first and now passing

### Bugs fixed (file:line, root cause, fix, test)

1. **BUG-A** cli_steering.py `_InputBox.feed`: multi-byte UTF-8 split across
   `os.read` chunks was destroyed (`data.decode("utf-8", "ignore")` per chunk),
   so a pasted emoji/é arriving in two reads silently vanished. Fix: instance
   `codecs.getincrementaldecoder("utf-8")(errors="ignore")` buffers partial
   trailing bytes across feed() calls. Test: test_bughunt3_utf8_split.py (3 tests).
1. **BUG-B** cli_steering.py `_InputBox.feed`: escape sequences split across
   reads typed garbage — `ESC` then `[A` typed literal `[A`; split Shift+Enter
   (`ESC[13;2` + `u`) typed `u` instead of newline; split SS3/Ctrl+arrow typed
   the final byte. Fix: incomplete ESC/CSI/SS3 tails are saved in
   `_pending_esc` and prepended to the next chunk (capped at 64 chars).
   Test: test_bughunt3_split_escape.py (7 tests).
1. **BUG-C** cli_panel.py `clip_buf`/`panel_body`/`body_cursor_col`: width math
   used `len()` (code points), not display columns — CJK/emoji buffers rendered
   a body row wider than the panel (right border pushed off the line) and
   parked the caret in the wrong column. Fix: new `char_width`/`display_width`
   helpers (east_asian_width W/F → 2, combining → 0) + `_clip_pad`; all three
   call sites converted. ASCII geometry unchanged.
   Test: test_bughunt3_wide_chars.py (7 tests).
1. **BUG-D** cli_steering.py: terminal resize never re-anchored the box —
   `start()` emitted the DECSTBM scroll region once; after a resize the box
   drew at the new bottom rows while output kept scrolling in the stale region,
   corrupting the box. Fix: `_InputBox._rows` tracks the region's rows;
   `_draw_locked` re-emits `ESC[1;{rows-3}r` + re-saves the output cursor on
   change; `SteeringSession._loop` polls `_term_size()` each 100 ms select
   timeout and redraws on change. Test: test_bughunt3_resize.py (3 tests).

### Non-bugs verified (do NOT revisit)

resume_chat_by_id never raises on unknown id; `_on_submit` redraw via feed's
changed flag; panel_top/panel_bottom math; empty-line submit ignored; Ctrl+D in
box ignored; mouse/CSI-u modes never enabled; `_StdoutProxy` delegation;
registry pop atomicity; print_outcome contract; `_history_path` digest;
readline prompt \\001/\\002 markers and KeyboardInterrupt race deemed not
deterministically testable. cli_repl.py and cli_helpers.py needed NO changes.

### Verification status

- 20 bughunt3 group-C tests: 15 failed pre-fix → 20/20 pass post-fix.
- Full CLI suite (test_cli_panel/repl/steering, test_bughunt_cli\*,
  test_shift_enter_newline): 74/74 pass, no regressions.
- `uv run check --full`: passes (after mdformat of this file).
