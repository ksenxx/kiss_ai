# Live remote-webapp validation of the periodic `summary` tool feature

These screenshots were captured from a real end-to-end run of the KISS Sorcar
remote web app (`kiss.server.web_server.RemoteAccessServer`, the same
`media/main.js` chat webview used by the VS Code extension) with the model
`claude-fable-5` executing a 12-command dependent bash chain (each command
uses the previous command's numeric output, forcing one tool call per step).

- `summary_collapsed_midrun.png` — taken while the task was running (step 15
  of 17). The agent has called `summary(description=...)` after crossing the
  5-step boundary; the chat webview rendered a collapsed `summary` panel whose
  `description` text is fully visible (multi-line, no truncation), with the
  preceding six event panels adopted as hidden sub-panels. Subsequent Bash
  panels (chain steps 9-12) appear after the summary panel as usual.
- `summary_collapsed_final.png` — after task completion (17 steps, "Done").
  The whole activity is condensed to: System Prompt, Prompt, the last
  collapsed `summary` panel (earlier summaries and all intermediate panels
  are nested hierarchically inside it), the `finish` panel, and the Result.
- `task_result.png` — the task's final Result block reporting the correct
  chain output 16383 = 2^14 - 1, confirming the agent genuinely executed all
  12 sequential bash commands while emitting periodic summaries.
