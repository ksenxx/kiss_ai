#!/bin/bash
# Build and install the KISS Sorcar VS Code extension.
# Usage: scripts/build-extension.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXT_DIR="$PROJECT_ROOT/src/kiss/agents/vscode"
CODE="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"

cd "$EXT_DIR"

echo "==> Compiling TypeScript..."
npx tsc -p ./

echo "==> Copying KISS project files..."
npm run copy-kiss

echo "==> Packaging VSIX..."
npm run package

echo "==> Installing extension..."
"$CODE" --install-extension kiss-sorcar.vsix --force

# Write marker so the extension knows a fresh install just happened and
# should show the restart/setup notification (even on the fast-path where
# uv + .venv already exist).
mkdir -p "$HOME/.kiss"
date -u +%Y-%m-%dT%H:%M:%SZ > "$HOME/.kiss/.extension-updated"

echo "==> Cleaning up build artifacts..."
rm -rf "$EXT_DIR/out" "$EXT_DIR/kiss_project" "$EXT_DIR/kiss-sorcar.vsix"

echo "==> Done. KISS Sorcar extension installed successfully."

# If VS Code is running, its built-in mechanism will reload the extension
# host automatically.  If not, open VS Code so the user sees the update.
if pgrep -qx "Code" 2>/dev/null; then
    echo "    VS Code is running — it will auto-reload the extension shortly."
    echo "    If nothing happens, press Cmd+Shift+P → 'Reload Window'."
else
    echo "    Opening VS Code..."
    cd "$PROJECT_ROOT"
    "$CODE" .
fi
