---
title: VS Code theme colour variables for Sorcar webview accents
uuid: 284d0053-48e4-4c88-a611-2fff1b6daa98
summary: Which --vscode-* colour vars suit UI accents (charts.*), their registry defaults,
  and the charts.orange translucency trap.
created: '2026-09-26T20:22:32Z'
updated: '2026-09-26T20:22:32Z'
---
## Facts (verified 2026-09-26 from microsoft/vscode main)
- `charts.green` dark/hcDark #89D185, light #388A34, hcLight #374e06; `charts.purple` dark #B180D7, light #652D90
  (src/vs/platform/theme/common/colors/chartsColors.ts).
- `charts.red` = editorError.foreground (dark #F14C4C, light #E51400, hcDark #F48771);
  `charts.yellow` = editorWarning.foreground (dark #CCA700, light #BF8803, hcDark #FFD370);
  `charts.blue` = editorInfo.foreground (editorColors.ts).
- TRAP: `charts.orange` = minimap.findMatchHighlight = editor.findMatchHighlightBackground = `#EA5C0055`
  (translucent) in dark/light and null in HC. Never use it as a text/icon colour.
- `testing.iconPassed` #73c991 (dark, light, hcDark); hcLight #007100. `testing.iconFailed` = list.errorForeground.
- Terminal ANSI colours are neon in Dark+ (ansiYellow #e5e510, ansiMagenta #bc3fbc); poor as UI accents.

## Sorcar usage (since the 2026-09-26 colour pass)
- `media/main.css` `:root` and `remote-codex.css` `body.remote-chat`: --green/--red/--yellow/--purple use
  `--vscode-charts-*`; --cyan keeps `--vscode-terminal-ansiCyan`; --orange = color-mix(yellow 55%, red).
- The remote webapp injects the charts vars in `web_server.py` `_VSCODE_DARK_MODERN_CSS` / `_VSCODE_LIGHT_MODERN_CSS`;
  `scripts/ui_theme_screenshots.py` THEMES carries them for Dark+/Light+/HC screenshots.
- Stylelint rule `kiss/design-tokens` (src/kiss/agents/vscode/scripts/stylelint-design-tokens.mjs) rejects hex,
  px radii, numeric z-index, unit font sizes outside custom properties of `:root` / `body.remote-chat`.
