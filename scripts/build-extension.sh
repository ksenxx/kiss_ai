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

echo "==> Uninstalling old extension (if present)..."
"$CODE" --uninstall-extension ksenxx.kiss-sorcar 2>/dev/null || true

echo "==> Installing extension..."
"$CODE" --install-extension kiss-sorcar.vsix --force

echo "==> Cleaning up build artifacts..."
rm -rf "$EXT_DIR/out" "$EXT_DIR/kiss_project" "$EXT_DIR/kiss-sorcar.vsix"

echo "==> Done. KISS Sorcar extension installed successfully."
