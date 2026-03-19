#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_BASE="$PROJECT_DIR"

# Detect architecture
if sysctl -n hw.optional.arm64 2>/dev/null | grep -q '1'; then
    ARCH="arm64"
else
    ARCH="x86_64"
fi

mkdir -p "$INSTALL_BASE/bin"

# ---------------------------------------------------------------------------
# 1. Install uv
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
# Symlink uv/uvx into local bin
UV_BIN="$(command -v uv)"
ln -sf "$UV_BIN" "$INSTALL_BASE/bin/uv"
if [ -f "$(dirname "$UV_BIN")/uvx" ]; then
    ln -sf "$(dirname "$UV_BIN")/uvx" "$INSTALL_BASE/bin/uvx"
fi

# ---------------------------------------------------------------------------
# 2. Install code-server
# ---------------------------------------------------------------------------
CS_VERSION="4.111.0"
if [ ! -x "$INSTALL_BASE/code-server/bin/code-server" ]; then
    echo ">>> Installing code-server ${CS_VERSION}..."
    CS_TARBALL="code-server-${CS_VERSION}-macos-${ARCH}.tar.gz"
    CS_URL="https://github.com/coder/code-server/releases/download/v${CS_VERSION}/${CS_TARBALL}"
    CS_TMP="$(mktemp -d)"
    curl -fSL -o "$CS_TMP/$CS_TARBALL" "$CS_URL"
    rm -rf "$INSTALL_BASE/code-server"
    mkdir -p "$INSTALL_BASE/code-server"
    tar xzf "$CS_TMP/$CS_TARBALL" -C "$INSTALL_BASE/code-server" --strip-components=1
    rm -rf "$CS_TMP"
    # Clean up unnecessary files (~190MB of source maps and type declarations)
    find "$INSTALL_BASE/code-server" \( -name '*.map' -o -name '*.d.ts' \) -type f -delete
    echo "   code-server installed"
else
    echo ">>> code-server already installed"
fi
chmod +x "$INSTALL_BASE/code-server/bin/code-server"
ln -sf "$INSTALL_BASE/code-server/bin/code-server" "$INSTALL_BASE/bin/code-server"

# ---------------------------------------------------------------------------
# 3. Install git (symlink system git if available)
# ---------------------------------------------------------------------------
echo ">>> Configuring git..."
SYS_GIT=""
if [ -f "/Library/Developer/CommandLineTools/usr/bin/git" ]; then
    SYS_GIT="/Library/Developer/CommandLineTools/usr/bin/git"
elif command -v git &> /dev/null; then
    SYS_GIT="$(command -v git)"
fi
if [ -n "$SYS_GIT" ]; then
    ln -sf "$SYS_GIT" "$INSTALL_BASE/bin/git"
    echo "   Linked git from $SYS_GIT"
else
    echo "   WARNING: git not found. Install Xcode Command Line Tools: xcode-select --install"
fi

# ---------------------------------------------------------------------------
# 4. Sync Python dependencies
# ---------------------------------------------------------------------------
echo ">>> Syncing Python dependencies..."
cd "$PROJECT_DIR"
uv sync

# Symlink entry-point scripts into bin
for script in sorcar check generate-api-docs; do
    if [ -f "$PROJECT_DIR/.venv/bin/$script" ]; then
        ln -sf "$PROJECT_DIR/.venv/bin/$script" "$INSTALL_BASE/bin/$script"
    fi
done

# ---------------------------------------------------------------------------
# 5. Install Playwright Chromium
# ---------------------------------------------------------------------------
echo ">>> Installing Playwright Chromium..."
PLAYWRIGHT_BROWSERS_PATH="$INSTALL_BASE/playwright-browsers" uv run playwright install chromium

# ---------------------------------------------------------------------------
# 6. Write install_dir marker
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.kiss"
printf '%s\n' "$INSTALL_BASE" > "$HOME/.kiss/install_dir"

# ---------------------------------------------------------------------------
# 7. Create env.sh
# ---------------------------------------------------------------------------
PROFILE_SNIPPET="$INSTALL_BASE/env.sh"
cat > "$PROFILE_SNIPPET" << EOF
# KISS Agent Framework - added by install.sh
export PATH="$INSTALL_BASE/bin:\$PATH"
export UV_PYTHON_INSTALL_DIR="$INSTALL_BASE/python"
export PLAYWRIGHT_BROWSERS_PATH="$INSTALL_BASE/playwright-browsers"
EOF

# Add GIT_EXEC_PATH only if we have a local git with libexec
if [ -d "$INSTALL_BASE/git/libexec/git-core" ]; then
    echo "export GIT_EXEC_PATH=\"$INSTALL_BASE/git/libexec/git-core\"" >> "$PROFILE_SNIPPET"
fi

# ---------------------------------------------------------------------------
# 8. Add source line to shell rc
# ---------------------------------------------------------------------------
_add_to_shell_rc() {
    local rc_file="$1"
    local source_line="source \"$PROFILE_SNIPPET\""
    if [ -f "$rc_file" ]; then
        if ! grep -qF "$source_line" "$rc_file"; then
            printf '\n%s\n' "$source_line" >> "$rc_file"
            echo "   Added to $rc_file"
        else
            echo "   Already in $rc_file"
        fi
    else
        echo "$source_line" > "$rc_file"
        echo "   Created $rc_file with source line"
    fi
}

echo ">>> Configuring shell profile..."
case "${SHELL:-/bin/zsh}" in
    */zsh)  _add_to_shell_rc "$HOME/.zshrc" ;;
    */bash) _add_to_shell_rc "$HOME/.bashrc" ;;
    *)      _add_to_shell_rc "$HOME/.zshrc"
            _add_to_shell_rc "$HOME/.bashrc" ;;
esac

echo ""
echo "=== Installation Complete ==="
echo "Project: $PROJECT_DIR"
echo ""
echo "Open a new terminal or run: source \"$PROFILE_SNIPPET\""
