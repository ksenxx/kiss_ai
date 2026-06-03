#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Build and install the KISS Sorcar VS Code extension.
# Usage: scripts/build-extension.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXT_DIR="$PROJECT_ROOT/src/kiss/agents/vscode"
CODE="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"

echo "==> Ensuring Claude Code skills are present..."
"$SCRIPT_DIR/fetch-claude-skills.sh"

cd "$EXT_DIR"

echo "==> Compiling TypeScript..."
npx tsc -p ./

echo "==> Copying KISS project files..."
npm run copy-kiss

echo "==> Packaging VSIX..."
npm run package

echo "==> Installing extension..."
"$CODE" --uninstall-extension ksenxx.kiss-sorcar 2>/dev/null || true
"$CODE" --install-extension kiss-sorcar.vsix --force

# Stop the old kiss-web daemon BEFORE the extension auto-reloads.  The
# previous --uninstall-extension call deleted the directory tree the
# running daemon's .venv was loaded from, so the daemon is technically
# broken even if it is still alive (subsequent UDS requests can hit
# stale / missing module paths).  ``pkill -x kiss-web`` is unreliable
# on macOS — kiss-web is a Python shebang script so the kernel's
# ``comm`` field is the (15-char-truncated) interpreter path, NOT
# the literal name.  Kill by listening port instead.  The macOS
# LaunchAgent / Linux systemd unit's ``KeepAlive`` will respawn a
# clean daemon from the freshly-installed VSIX before the extension
# reload fires (~2 s after the marker write below).
if command -v lsof >/dev/null 2>&1; then
    OLD_PIDS=$(lsof -ti :8787 2>/dev/null || true)
    if [ -n "$OLD_PIDS" ]; then
        echo "==> Stopping old kiss-web daemon (PIDs: $OLD_PIDS)..."
        echo "$OLD_PIDS" | xargs kill 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            sleep 0.3
            if ! lsof -i :8787 -t >/dev/null 2>&1; then break; fi
        done
        # Force-kill survivors.
        STRAGGLERS=$(lsof -ti :8787 2>/dev/null || true)
        if [ -n "$STRAGGLERS" ]; then
            echo "$STRAGGLERS" | xargs kill -9 2>/dev/null || true
        fi
    fi
fi
# Remove the stale Unix-domain socket left behind by the now-dead
# daemon.  The new daemon ``_setup_server`` unlinks before binding,
# but pre-emptive cleanup avoids ENOENT/ECONNREFUSED reconnect-loop
# noise from any extension client that is mid-flight.
rm -f "$HOME/.kiss/sorcar.sock"

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
