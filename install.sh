#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Standard install locations
BIN_DIR="$HOME/.local/bin"
LIB_DIR="$HOME/.local/lib"

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

mkdir -p "$BIN_DIR" "$LIB_DIR"

# Fetch latest release version from GitHub. Falls back to $2 if API fails.
_latest_github_version() {
    local repo="$1" default="$2" version
    version="$(curl -sSf --max-time 10 \
        "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null \
        | grep '"tag_name"' | sed -E 's/.*"v?([^"]+)".*/\1/')" || true
    echo "${version:-$default}"
}

_code_server_ca_cert() {
    for cert in \
        /etc/ssl/cert.pem \
        /opt/homebrew/etc/openssl@3/cert.pem \
        /usr/local/etc/openssl@3/cert.pem \
        /etc/openssl/cert.pem
    do
        if [ -f "$cert" ]; then
            echo "$cert"
            return 0
        fi
    done
    return 1
}

_write_code_server_wrapper() {
    local wrapper="$BIN_DIR/code-server"
    local cert

    cat > "$wrapper" <<'EOF'
#!/bin/bash
set -e

CODE_SERVER_BIN="$HOME/.local/lib/code-server/bin/code-server"

if [ "$(uname -s)" = "Darwin" ]; then
    export NODE_OPTIONS="--use-openssl-ca${NODE_OPTIONS:+ $NODE_OPTIONS}"

    if [ -z "${NODE_EXTRA_CA_CERTS:-}" ]; then
        for cert in \
            /etc/ssl/cert.pem \
            /opt/homebrew/etc/openssl@3/cert.pem \
            /usr/local/etc/openssl@3/cert.pem \
            /etc/openssl/cert.pem
        do
            if [ -f "$cert" ]; then
                export NODE_EXTRA_CA_CERTS="$cert"
                break
            fi
        done
    fi
fi

exec "$CODE_SERVER_BIN" "$@"
EOF

    chmod +x "$wrapper"

    cert="$(_code_server_ca_cert)" || cert=""
    if [ -n "$cert" ]; then
        echo "   code-server wrapper configured with CA bundle: $cert"
    else
        echo "   code-server wrapper installed"
    fi
}

# ---------------------------------------------------------------------------
# 1. Install or upgrade Homebrew  (https://brew.sh)
# ---------------------------------------------------------------------------
echo ">>> [1/4] Installing Homebrew..."
if command -v brew &> /dev/null; then
    echo "   Homebrew already installed, upgrading..."
    brew update
else
    echo "   Installing Homebrew from latest binaries..."
    if [ "$OS" = "macos" ]; then
        # Grab latest Homebrew.pkg from GitHub releases and install non-interactively
        BREW_PKG_URL="$(curl -sSf --max-time 10 \
            "https://api.github.com/repos/Homebrew/brew/releases/latest" \
            | grep '"browser_download_url".*Homebrew\.pkg' \
            | sed -E 's/.*"(https:[^"]+)".*/\1/')"
        BREW_TMP="$(mktemp -d)"
        curl -fSL -o "$BREW_TMP/Homebrew.pkg" "$BREW_PKG_URL"
        sudo installer -pkg "$BREW_TMP/Homebrew.pkg" -target /
        rm -rf "$BREW_TMP"
    fi
fi

# Add Homebrew to PATH for this session
if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [ "$OS" = "macos" ]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi

echo "   Homebrew $(brew --version | head -1) ready"

# ---------------------------------------------------------------------------
# 2. Install code-server from binaries if not already installed
# ---------------------------------------------------------------------------
echo ">>> [2/4] Installing code-server..."
if [ -x "$LIB_DIR/code-server/bin/code-server" ]; then
    echo "   code-server already installed"
else
    CS_FALLBACK_VERSION="4.112.0"
    CS_VERSION="$(_latest_github_version coder/code-server "$CS_FALLBACK_VERSION")"
    CS_TARBALL="code-server-${CS_VERSION}-${OS}-${ARCH_ALT}.tar.gz"
    CS_URL="https://github.com/coder/code-server/releases/download/v${CS_VERSION}/${CS_TARBALL}"
    CS_TMP="$(mktemp -d)"
    curl -fSL -o "$CS_TMP/$CS_TARBALL" "$CS_URL"
    rm -rf "$LIB_DIR/code-server"
    mkdir -p "$LIB_DIR/code-server"
    tar xzf "$CS_TMP/$CS_TARBALL" -C "$LIB_DIR/code-server" --strip-components=1
    rm -rf "$CS_TMP"
    find "$LIB_DIR/code-server" \( -name '*.map' -o -name '*.d.ts' \) -type f -delete
    echo "   code-server ${CS_VERSION} installed"
fi
chmod +x "$LIB_DIR/code-server/bin/code-server"
_write_code_server_wrapper

# ---------------------------------------------------------------------------
# 3. Install uv from binaries if not installed  (https://astral.sh/uv)
# ---------------------------------------------------------------------------
echo ">>> [3/4] Installing uv..."
if command -v uv &> /dev/null; then
    echo "   uv already installed"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$BIN_DIR:$HOME/.cargo/bin:$PATH"
fi
echo "   uv $(uv --version) ready"

# ---------------------------------------------------------------------------
# 4. Create virtual environment and sync Python dependencies
# ---------------------------------------------------------------------------
echo ">>> [4/4] Setting up Python environment..."
cd "$PROJECT_DIR"
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "   Creating virtual environment with Python 3.13..."
    uv venv --python 3.13
fi
uv sync

# Symlink entry-point scripts into bin
for script in sorcar check generate-api-docs; do
    if [ -f "$PROJECT_DIR/.venv/bin/$script" ]; then
        ln -sf "$PROJECT_DIR/.venv/bin/$script" "$BIN_DIR/$script"
    fi
done

# ---------------------------------------------------------------------------
# 5. Install Playwright Chromium  (https://playwright.dev)
# ---------------------------------------------------------------------------
echo ">>> [5] Installing Playwright Chromium..."
uv run playwright install chromium

# ---------------------------------------------------------------------------
# Write install_dir marker (used by env.py for offline-installer compat)
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.kiss"
printf '%s\n' "$PROJECT_DIR" > "$HOME/.kiss/install_dir"

# ---------------------------------------------------------------------------
# Create env.sh
# ---------------------------------------------------------------------------
PROFILE_SNIPPET="$PROJECT_DIR/env.sh"
cat > "$PROFILE_SNIPPET" << EOF
# KISS Agent Framework - added by install.sh
export PATH="$BIN_DIR:\$PATH"
EOF

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
