# Lessons

- When running long bash commands, use `timeout_seconds=600` parameter directly instead of background processes with `nohup` and temp file polling. The `nohup` + background approach causes timeouts on its own due to shell session handling.
- `uv run check --full` runs: clean artifacts, uv sync, generate-api-docs, compileall, ruff, mypy, pyright, and mdformat --check on markdown files. Defined in `src/kiss/scripts/check.py`.
- To fix markdown formatting issues flagged by `mdformat --check`, simply run `uv run mdformat <file>`.
- To build the VSCode extension: `cd src/kiss/agents/vscode && bash copy-kiss.sh && npx tsc -p tsconfig.json && vsce package --no-dependencies -o kiss-sorcar.vsix`. The `vsce package` also runs prepublish which re-runs compile and copy-kiss.
- VS Code CLI is at `/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code`. Use `--install-extension <path>.vsix --force` to install, `--uninstall-extension <id>` to remove, `--list-extensions --show-versions` to list.
- The extension publisher is `ksenxx` (package.json). Old builds may have publisher `kiss`. Clean up old versions after install.
- The installed VS Code extension files live at `~/.vscode/extensions/ksenxx.kiss-sorcar-<version>/`. For quick CSS/JS fixes, copy files directly there AND update the source. Compare source vs installed with `diff` to catch uncommitted-but-installed divergence.
- When debugging VS Code webview UI issues (e.g. elements not appearing), always check for `overflow: hidden` on parent containers — absolutely positioned children (like dropdowns/autocomplete) will be clipped. Compare working tree against HEAD to find uncommitted culprit changes.
