# PROGRESS

## Task

Make Tab fast-complete work in sorcar CLI as an in-place menu above the input box.

## Status

Implementation + tests complete. All gpt-5.1 review fixes (C2, C3, C4, B1, B2, B3, B7, B8, B9) APPLIED. Edge-case tests still TODO.

## Files modified

- `src/kiss/agents/sorcar/cli_panel.py` — `menu_row()` helper with C0/C1 ANSI sanitisation (C2).
- `src/kiss/agents/sorcar/cli_steering.py` — menu state, draw, feed, ask_user_question changes.
- `src/kiss/tests/agents/sorcar/test_cli_steering.py` — TestInputBoxCompletionMenu (16 tests).

## Continuation 3 fixes applied this session

- `_reset_completion_state`: removed dead `_tab_candidates/_idx/_origin` assignments (B9).
- `_draw_locked` shrink branch: use `self._rows or rows` for `prev_top`, clear ALL previous menu rows when `_drawn_menu_h > 0` (C4).
- `_append_paste` call sites (both branches in `feed`): call `_reset_completion_state()` to dismiss menu on paste (B2).
- `SteeringSession.ask_user_question`: calls `self.box._reset_completion_state()` before flipping title (B3).
- `_reset_completion_state`, `_open_completion_menu`, `_menu_move`, `_menu_accept`: wrapped state mutations in `self.lock` (B7).
- `_menu_accept`: now delegates close to `_reset_completion_state` (B8).

## Verification this session

- `uv run pytest src/kiss/tests/agents/sorcar/test_cli_steering.py` → 51/51 pass.

## Remaining for NEXT session (in order)

1. Add `TestInputBoxCompletionMenuEdgeCases` class in test_cli_steering.py covering:
   - candidate with `\x1b[31m...` produces NO ESC bytes in painted output (E4, C2 verification).
   - `stop()` while menu open clears menu rows AND resets `_menu_open=False`, `_drawn_menu_h=0` (E11, C3).
   - paste while menu open dismisses the menu (E8, B2).
   - `_menu_h` auto-dismisses when room=0 (monkeypatch `_term_size` to return (4, 80)) (E2, B1).
   - `SteeringSession.ask_user_question` dismisses an open menu before title flip (E10, B3).
1. `uv run check --full` — must pass.
1. `uv run pytest src/kiss/tests/agents/sorcar/test_cli_steering.py src/kiss/tests/agents/sorcar/test_cli_client.py` — must pass.
1. Quick run of related sorcar tests (bughunt3/4/5/6, panel, shift_enter).
1. Delete `tmp/tmp_diff_for_review.patch` and `tmp/review.md`.
1. Call `finish(success=True)`.
