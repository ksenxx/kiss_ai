## User Preferences and Invariants

- Use `uv run check --full` to lint, typecheck, and format code before finishing
- Python line length limit is 100 characters
- Import blocks must be sorted (ruff I001 rule)
- VS Code extension uses prettier for formatting JS/TS files
- Markdown files are formatted with mdformat
- For long-running evaluations, use background processes with output redirected to log files and poll periodically
- GEPA evaluation scripts and results are stored in PWD/tmp/
- VS Code extension config is stored in ~/.kiss/config.json (non-API-key settings)
- API keys are saved to shell RC files and refreshed via os.environ
- New vscode extension features should go in src/kiss/agents/vscode/ with minimal changes to existing files
- Tab bar button order: tabs, +, config, history
- New message types in the VS Code extension must be added to FromWebviewMessage, AgentCommand, and/or ToWebviewMessage in types.ts
- VS Code extension also uses stylelint for CSS linting; run auto-fix if needed
