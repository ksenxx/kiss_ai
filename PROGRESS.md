# Progress

## Task: Sorcar CLI interactive — show completions in contrasting colors with orange (Claude Code style)

### Step 1 — Research (done)

Searched 2 sites (Google AI overview + amanhimself.dev). Findings:

- Claude Code supports `/color orange` (and red/blue/green/yellow/purple/pink/cyan) to color the prompt bar (since v2.1.75).
  Refs: https://code.claude.com/docs/en/commands and https://amanhimself.dev/blog/color-coding-claude-code-sessions/
- Claude brand "Coral" orange is approximately `#D97757`; another common orange used in Claude Code screenshots is around `#FF8800`.
- The completion menu in Claude Code uses ARROW marker (`❯`) for the selected entry, orange highlight on the selected row, and dim for unselected — matching the user's "contrasting colors with orange" request.

### Step 2 — Code map (done)

- Completion menu is rendered by `menu_row()` in
  `src/kiss/agents/sorcar/cli_panel.py` (line 197).
  Currently uses `CYAN` for borders, no special highlight for the
  selected row, and `DIM` for unselected rows.
- Used by `_InputBox._draw_menu_locked()` in
  `src/kiss/agents/sorcar/cli_steering.py` (line 552, with `menu_row(item, idx == self._menu_sel, cols)`).
- Tests for menu rendering live in
  `src/kiss/tests/agents/sorcar/test_cli_steering.py`
  (search hits for `❯` at lines 504, 792). They assert presence of `❯`
  on the selected row but do not pin a specific color, so adding orange
  ANSI is non-breaking.
- `test_cli_panel.py` does not test colors of menu rows.

### Step 3 — Planned change

In `cli_panel.py`:

1. Add `ORANGE = f"{_ESC}[38;5;208m"` (256-color DarkOrange — closest
   universal-terminal match to Claude Code's prompt-bar orange and
   contrasts strongly with the surrounding dim/gray rows).
1. Update `menu_row(text, selected, cols)` so the selected row prints
   the marker `❯ <text>` in `ORANGE` + bold (`\x1b[1m`), and the
   unselected row stays `DIM` for clear contrast. Border glyphs remain
   `CYAN` to keep the menu visually anchored to the input panel.

Add an end-to-end test in `test_cli_panel.py` asserting:

- Selected row contains the ANSI sequence `\x1b[38;5;208m` and `❯`.
- Unselected row contains the `DIM` sequence `\x1b[2m` and **not** the
  orange sequence.
- Both rows still carry the `CYAN` `│` border glyph.

### Next session

Implement the change, run `uv run check --full`, then targeted tests:

- `uv run pytest -v src/kiss/tests/agents/sorcar/test_cli_panel.py src/kiss/tests/agents/sorcar/test_cli_steering.py`.

### Step 4 — Implementation (done)

- Added `BOLD = "\x1b[1m"` and `ORANGE = "\x1b[38;5;208m"` (xterm-256
  DarkOrange ≈ #FF8700, the closest universal-terminal match to
  Claude Code's coral-orange `/color orange` palette) to
  `src/kiss/agents/sorcar/cli_panel.py`.
- Updated `menu_row()` so the highlighted candidate is rendered as
  bold-orange `❯ <text>` and unselected candidates stay dim — the
  same contrast pattern Claude Code uses for its slash-command menu.
  Border glyphs remain cyan to keep the menu visually anchored to the
  input panel below.
- Added `TestMenuRowContrastingOrange` to
  `src/kiss/tests/agents/sorcar/test_cli_panel.py` with 4 tests
  verifying selected vs unselected styling, equal display width, and
  that ESC-injection attempts still get sanitised.

### Step 5 — Verification (done)

- `uv run check --full` → all checks pass.
- `uv run pytest -v src/kiss/tests/agents/sorcar/test_cli_panel.py src/kiss/tests/agents/sorcar/test_cli_steering.py` → 79/79 pass.
- Live render of `menu_row('install pkg', True, 40)` confirms the
  ANSI sequence `\x1b[38;5;208m\x1b[1m` wraps the `❯` row.
