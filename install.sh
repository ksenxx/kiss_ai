#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_BASE="$PROJECT_DIR"

# Detect OS and architecture
OS="$(uname -s)"
case "$OS" in
    Darwin) OS="macos" ;;
    Linux)  OS="linux" ;;
    *)      echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  ARCH_ALT="amd64" ;;
    arm64|aarch64) ARCH="arm64"; ARCH_ALT="arm64" ;;
    *)       echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

mkdir -p "$INSTALL_BASE/bin"

# Fetch latest release version from GitHub. Falls back to $2 if API fails.
_latest_github_version() {
    local repo="$1" default="$2" version
    version="$(curl -sSf --max-time 10 \
        "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null \
        | grep '"tag_name"' | sed -E 's/.*"v?([^"]+)".*/\1/')" || true
    echo "${version:-$default}"
}

# ---------------------------------------------------------------------------
# 1. Install uv  (https://astral.sh/uv)
# ---------------------------------------------------------------------------
echo ">>> [1/4] Installing uv..."
if command -v uv &> /dev/null; then
    CURRENT_UV="$(uv --version | awk '{print $2}')"
    LATEST_UV="$(_latest_github_version astral-sh/uv "$CURRENT_UV")"
    if [ "$CURRENT_UV" != "$LATEST_UV" ]; then
        echo "   Upgrading uv from $CURRENT_UV to $LATEST_UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
UV_BIN="$(command -v uv)"
ln -sf "$UV_BIN" "$INSTALL_BASE/bin/uv"
if [ -f "$(dirname "$UV_BIN")/uvx" ]; then
    ln -sf "$(dirname "$UV_BIN")/uvx" "$INSTALL_BASE/bin/uvx"
fi
echo "   uv $(uv --version) ready"

# ---------------------------------------------------------------------------
# 2. Install code-server  (https://github.com/coder/code-server/releases)
# ---------------------------------------------------------------------------
CS_FALLBACK_VERSION="4.112.0"
CS_VERSION="$(_latest_github_version coder/code-server "$CS_FALLBACK_VERSION")"
echo ">>> [2/4] Installing code-server ${CS_VERSION}..."

_install_code_server() {
    CS_TARBALL="code-server-${CS_VERSION}-${OS}-${ARCH_ALT}.tar.gz"
    CS_URL="https://github.com/coder/code-server/releases/download/v${CS_VERSION}/${CS_TARBALL}"
    CS_TMP="$(mktemp -d)"
    curl -fSL -o "$CS_TMP/$CS_TARBALL" "$CS_URL"
    rm -rf "$INSTALL_BASE/code-server"
    mkdir -p "$INSTALL_BASE/code-server"
    tar xzf "$CS_TMP/$CS_TARBALL" -C "$INSTALL_BASE/code-server" --strip-components=1
    rm -rf "$CS_TMP"
    # Clean up unnecessary files (~190MB of source maps and type declarations)
    find "$INSTALL_BASE/code-server" \( -name '*.map' -o -name '*.d.ts' \) -type f -delete
    echo "   code-server ${CS_VERSION} installed"
}

if [ -x "$INSTALL_BASE/code-server/bin/code-server" ]; then
    CURRENT_CS="$("$INSTALL_BASE/code-server/bin/code-server" --version 2>/dev/null \
        | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' || echo "")"
    if [ "$CURRENT_CS" != "$CS_VERSION" ]; then
        echo "   Upgrading code-server from ${CURRENT_CS:-unknown} to $CS_VERSION..."
        _install_code_server
    else
        echo "   code-server $CS_VERSION already up to date"
    fi
else
    _install_code_server
fi
chmod +x "$INSTALL_BASE/code-server/bin/code-server"
ln -sf "$INSTALL_BASE/code-server/bin/code-server" "$INSTALL_BASE/bin/code-server"

# ---------------------------------------------------------------------------
# 3. Sync Python dependencies  (via uv)
# ---------------------------------------------------------------------------
echo ">>> [3/4] Syncing Python dependencies..."
cd "$PROJECT_DIR"
uv sync

# Symlink entry-point scripts into bin
for script in sorcar check generate-api-docs; do
    if [ -f "$PROJECT_DIR/.venv/bin/$script" ]; then
        ln -sf "$PROJECT_DIR/.venv/bin/$script" "$INSTALL_BASE/bin/$script"
    fi
done

# ---------------------------------------------------------------------------
# 4. Install Playwright Chromium  (https://playwright.dev)
# ---------------------------------------------------------------------------
echo ">>> [4/4] Installing Playwright Chromium..."
PLAYWRIGHT_BROWSERS_PATH="$INSTALL_BASE/playwright-browsers" uv run playwright install chromium

# ---------------------------------------------------------------------------
# Write install_dir marker
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.kiss"
printf '%s\n' "$INSTALL_BASE" > "$HOME/.kiss/install_dir"

# ---------------------------------------------------------------------------
# Create env.sh
# ---------------------------------------------------------------------------
PROFILE_SNIPPET="$INSTALL_BASE/env.sh"
cat > "$PROFILE_SNIPPET" << EOF
# KISS Agent Framework - added by install.sh
export PATH="$INSTALL_BASE/bin:\$PATH"
export UV_PYTHON_INSTALL_DIR="$INSTALL_BASE/python"
export PLAYWRIGHT_BROWSERS_PATH="$INSTALL_BASE/playwright-browsers"
EOF

if [ -d "$INSTALL_BASE/git/libexec/git-core" ]; then
    echo "export GIT_EXEC_PATH=\"$INSTALL_BASE/git/libexec/git-core\"" >> "$PROFILE_SNIPPET"
fi

# ---------------------------------------------------------------------------
# Add source line to shell rc
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
