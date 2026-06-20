# PROGRESS

## Task (continuation 2)

> The last task made no difference in the fast complete issue.
> Reproduce the issue by writing an integration test. Then fix the issue.
> Use gpt-5.5 (not codex) for thorough review and use the original
> model for all tasks including coding, bug fixing, and test creation.

## Root cause IDENTIFIED this session

The previous task added Tab-triggered in-place menu, but the user's
expectation of "fast complete" is the **prompt_toolkit-style auto-pop
menu as you type** (`complete_while_typing=True` in
`cli_prompt.py::PtkLineReader`). When sorcar runs the anchored box
code path (`_run_anchored_client` → `AnchoredRepl`), there is NO
auto-pop — the menu only appears on Tab. So the user sees
no menu while typing → "fast complete does not work".

PTY reproduction confirmed:

- `tmp/pty_repro.py` (now deleted) drove `AnchoredRepl` in a child
  process with a fixed completer. After sending `/` + `\t`, the menu
  WAS painted (`/help foo`, `❯` marker present). So the rendering
  layer works; the bug is that nothing happens BEFORE Tab.
- Real `CliCompleter._build_matches` returns reasonable candidates
  for typical inputs (e.g. `h`, `ho`, `/h`, `fix`, `/`), so the
  completer wiring is fine.

The fix is to mirror prompt_toolkit's `complete_while_typing=True`:
on every printable keystroke (and backspace that leaves a non-empty
buffer), call a new `_refresh_typing_menu()` helper that re-queries
the completer and pops the menu with current candidates — without
auto-replacing `buf` (only Tab does that).

## Changes applied this session (cli_steering.py)

1. **Added `_refresh_typing_menu()`** to `_InputBox`. Helper:

   ```python
   def _refresh_typing_menu(self) -> bool:
       if self.completer_fn is None or not self.buf:
           if self._menu_open:
               self._reset_completion_state()
               return True
           return False
       try:
           cands = [c.rstrip("\n") for c in self.completer_fn(self.buf)]
       except Exception:
           logger.debug("completer raised", exc_info=True)
           cands = []
       with self.lock:
           if not cands:
               if not self._menu_open:
                   return False
               self._menu_open = False
               self._menu_items = []
               self._menu_sel = 0
               self._menu_scroll = 0
               return True
           self._menu_items = cands
           self._menu_sel = 0
           self._menu_scroll = 0
           self._menu_open = True
       return True
   ```

   Unlike `_open_completion_menu`, it NEVER auto-replaces `buf` on a
   single-candidate match — typing only previews.

1. **Updated `feed()`** to call `_refresh_typing_menu` on:

   - Printable char branch (replaces the previous
     `self._reset_completion_state()`).
   - Backspace branch when buf becomes non-empty after pop.

1. **Updated Tab branch** so when menu is already open with exactly
   one candidate, Tab accepts it (mirrors the closed-menu single-
   candidate shortcut now that typing pre-opens the menu).

## Remaining work for next session (IN ORDER)

1. **Update existing tests** in
   `src/kiss/tests/agents/sorcar/test_cli_steering.py` to reflect
   new semantics. Failing tests (from `uv run pytest` -x):

   - `test_tab_opens_menu_with_first_selected`: typing "/he" now
     auto-opens menu → Tab advances sel to 1. Either (a) update
     expected sel to 1, OR (b) use a completer that returns []
     except after "/he\\t" (i.e. seed buf without auto-pop).
   - `test_tab_advances_selection_when_menu_open`: similar — sel
     starts at 1 after first Tab, advances to 2.
   - `test_typing_closes_menu_and_appends`: with always-matching
     completer, typing keeps menu open. Change completer to return
     matches only for specific prefixes, OR rename test to
     `test_typing_refreshes_menu`.
   - `test_backspace_with_buffer_closes_menu`: similar.
   - `test_single_candidate_replaces_buf_no_menu`: typing "/h"
     now opens menu with 1 candidate (preview, no replace). Then
     Tab on 1-candidate menu accepts → buf="/help ". Update
     assertion order.
   - `test_menu_renders_candidates_above_box`: typing "/" now
     auto-opens menu. Probably still passes since post-Tab
     assertions match; may need to adjust the "snapshot before
     Tab" logic.
   - Check ALL tests in `TestInputBoxCompletionMenu` + edge cases.

1. **Add new tests** in a new class `TestCompleteWhileTyping`:

   - `test_typing_auto_opens_menu_with_matches` — type "/h", assert
     menu open, items match, buf NOT replaced.
   - `test_typing_keeps_menu_open_when_matches` — type "a" then
     "b", menu refreshes (still open, items reflect "ab").
   - `test_typing_closes_menu_when_no_matches` — completer returns
     [] for "az", menu closes after typing "z".
   - `test_backspace_refreshes_menu` — backspace shrinks buf,
     completer matches reappear.
   - `test_empty_buf_after_backspace_closes_menu`.
   - `test_single_candidate_preview_does_not_replace_buf`.
   - `test_pty_end_to_end_complete_while_typing` — pty subprocess
     repro, validates `/help foo`/`❯` appear in painted bytes
     BEFORE Tab is sent (just by typing `/h`).

1. Run `uv run check --full` and fix lint/type errors.

1. Run targeted tests:

   - `uv run pytest src/kiss/tests/agents/sorcar/test_cli_steering.py`
   - `uv run pytest src/kiss/tests/agents/sorcar/test_cli_client.py`
   - sample sorcar bughunt tests.

1. **Review with gpt-5.5** (set_model + parallel agent) for thorough
   feedback on the new `_refresh_typing_menu` helper and the
   interaction with Tab/Enter/backspace. Apply any
   critical/medium fixes.

1. Final `uv run check --full`.

1. Clean tmp/, call `finish(success=True)`.

## Files modified this session

- `src/kiss/agents/sorcar/cli_steering.py`:
  - Added `_refresh_typing_menu()` (~30 lines).
  - `feed()` printable branch: `_reset_completion_state` →
    `_refresh_typing_menu`.
  - `feed()` backspace-non-empty branch: same.
  - `feed()` Tab branch: 1-candidate-open accepts via
    `_menu_accept` instead of bumping sel.

## Files NOT modified this session

- `src/kiss/tests/agents/sorcar/test_cli_steering.py` — needs
  updates AND new tests next session.
- `src/kiss/agents/sorcar/cli_panel.py` — unchanged.

## tmp/ state

Empty. No scratch files preserved.
