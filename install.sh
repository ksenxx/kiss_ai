#!/bin/bash
# Install KISS Sorcar from source: downloads binary dependencies, creates a
# Python virtualenv, and installs Playwright Chromium.
# Log saved to ~/.kiss/install.log
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Standard install locations
BIN_DIR="$HOME/.local/bin"
LOG_DIR="$HOME/.kiss"
LOG_FILE="$LOG_DIR/install.log"
NODE_VERSION="v22.16.0"
UV_VERSION="0.11.2"

mkdir -p "$BIN_DIR" "$LOG_DIR"
export PATH="$BIN_DIR:$PATH"

# Detect OS / arch
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Darwin|Linux) ;;
    *)  echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|aarch64|arm64) ;;
    *)  echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Require curl
if ! command -v curl &>/dev/null; then
    echo "ERROR: curl is required but not found. Please install curl first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: install git
# ---------------------------------------------------------------------------
install_git() {
    case "$OS" in
        Darwin)
            if command -v brew &>/dev/null; then
                echo "   Installing git via Homebrew..."
                brew install git
            else
                echo "   Triggering Xcode Command Line Tools (provides git)..."
                xcode-select --install 2>&1 || true
                echo "   NOTE: Complete the Xcode CLT dialog, then re-run this script."
                exit 1
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y git
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y git
            elif command -v yum &>/dev/null; then
                sudo yum install -y git
            elif command -v pacman &>/dev/null; then
                sudo pacman -S --noconfirm git
            elif command -v apk &>/dev/null; then
                sudo apk add git
            else
                echo "   ERROR: No supported package manager found. Install git from https://git-scm.com"
                exit 1
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: install uv from binary tarball
# ---------------------------------------------------------------------------
install_uv() {
    echo "   Downloading uv $UV_VERSION ..."
    local TARGET
    case "$OS" in
        Darwin)
            case "$ARCH" in
                x86_64)         TARGET="x86_64-apple-darwin" ;;
                aarch64|arm64)  TARGET="aarch64-apple-darwin" ;;
            esac
            ;;
        Linux)
            case "$ARCH" in
                x86_64)         TARGET="x86_64-unknown-linux-gnu" ;;
                aarch64|arm64)  TARGET="aarch64-unknown-linux-gnu" ;;
            esac
            ;;
    esac
    local URL="https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz"
    if curl -fsSL "$URL" | tar xz -C "$BIN_DIR" --strip-components=1; then
        echo "   uv $UV_VERSION installed to $BIN_DIR"
    else
        echo "   ERROR: Failed to download uv from $URL"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: install Node.js from binary tarball
# ---------------------------------------------------------------------------
install_node() {
    echo "   Downloading Node.js $NODE_VERSION ..."
    local OS_NODE ARCH_NODE
    OS_NODE="$(echo "$OS" | tr '[:upper:]' '[:lower:]')"   # darwin / linux
    case "$ARCH" in
        x86_64)         ARCH_NODE="x64" ;;
        aarch64|arm64)  ARCH_NODE="arm64" ;;
    esac
    local URL="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-${OS_NODE}-${ARCH_NODE}.tar.gz"
    mkdir -p "$HOME/.local"
    if curl -fsSL "$URL" | tar xz -C "$HOME/.local" --strip-components=1; then
        echo "   Node.js $NODE_VERSION installed to ~/.local/"
    else
        echo "   ERROR: Failed to download Node.js from $URL"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: install VS Code and ensure CLI is on PATH
# ---------------------------------------------------------------------------
install_code_cli() {
    case "$OS" in
        Darwin)
            local VSCODE_APP="/Applications/Visual Studio Code.app"
            if [ ! -d "$VSCODE_APP" ]; then
                echo "   Downloading VS Code for macOS..."
                local ARCH_VS
                case "$ARCH" in
                    aarch64|arm64) ARCH_VS="darwin-arm64" ;;
                    x86_64)        ARCH_VS="darwin" ;;
                esac
                local TMP_ZIP
                TMP_ZIP="$(mktemp /tmp/vscode-XXXXXX.zip)"
                if curl -fsSL "https://update.code.visualstudio.com/latest/${ARCH_VS}/stable" -o "$TMP_ZIP"; then
                    unzip -q "$TMP_ZIP" -d /Applications/
                    rm -f "$TMP_ZIP"
                    echo "   VS Code installed to /Applications/"
                else
                    rm -f "$TMP_ZIP"
                    echo "   ERROR: Failed to download VS Code"
                    return 1
                fi
            fi
            local CODE_BIN="$VSCODE_APP/Contents/Resources/app/bin/code"
            if [ -x "$CODE_BIN" ]; then
                ln -sf "$CODE_BIN" "$BIN_DIR/code"
                echo "   Linked VS Code CLI to $BIN_DIR/code"
            fi
            ;;
        Linux)
            if command -v snap &>/dev/null; then
                sudo snap install --classic code 2>&1 || true
            elif command -v apt-get &>/dev/null; then
                curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
                    | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg 2>&1
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
                    | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null 2>&1
                sudo apt-get update -y && sudo apt-get install -y code 2>&1
            elif command -v dnf &>/dev/null; then
                sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc 2>&1
                sudo tee /etc/yum.repos.d/vscode.repo >/dev/null <<'REPO'
[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc
REPO
                sudo dnf install -y code 2>&1
            else
                echo "   Please install VS Code from https://code.visualstudio.com"
                return 1
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: locate VS Code CLI binary
# ---------------------------------------------------------------------------
find_code_cli() {
    CODE_CLI=""
    for candidate in \
        "$(command -v code 2>/dev/null || true)" \
        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
        "$BIN_DIR/code" \
        "/usr/local/bin/code" \
        "/usr/bin/code" \
        "/snap/bin/code"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            CODE_CLI="$candidate"
            return 0
        fi
    done
    return 1
}

# === Main install (logged to ~/.kiss/install.log) =========================
{
    echo "=== KISS Sorcar Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Directory: $PROJECT_DIR"
    echo "OS: $OS ($ARCH)"
    echo ""

    # --- 1. git ---------------------------------------------------------------
    echo ">>> [1/8] Checking git..."
    if ! command -v git &>/dev/null; then
        install_git
    fi
    echo "   $(git --version) ready"
    echo ""

    # --- 2. uv ----------------------------------------------------------------
    echo ">>> [2/8] Installing uv..."
    if ! command -v uv &>/dev/null; then
        install_uv
    fi
    echo "   $(uv --version) ready"
    echo ""

    # --- 3. Node.js -----------------------------------------------------------
    echo ">>> [3/8] Installing Node.js..."
    if ! command -v node &>/dev/null; then
        install_node || true
    fi
    if command -v node &>/dev/null; then
        echo "   node $(node --version) ready"
    else
        echo "   WARNING: Node.js not installed — some agent tools may be unavailable"
    fi
    echo ""

    # --- 4. VS Code -----------------------------------------------------------
    echo ">>> [4/8] Installing VS Code..."
    if ! find_code_cli; then
        install_code_cli || true
        find_code_cli || true
    fi
    if [ -n "$CODE_CLI" ]; then
        echo "   code CLI ready: $CODE_CLI"
    else
        echo "   WARNING: VS Code not installed — extension install will be skipped"
    fi
    echo ""

    # --- 5. Python environment ------------------------------------------------
    echo ">>> [5/8] Setting up Python environment..."
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
    echo "   Python environment ready"
    echo ""

    # --- 6. Playwright Chromium -----------------------------------------------
    echo ">>> [6/8] Installing Playwright Chromium..."
    uv run playwright install chromium
    echo ""

    # --- 7. Build VS Code extension ------------------------------------------
    echo ">>> [7/8] Building VS Code extension..."
    VSCODE_EXT_DIR="$PROJECT_DIR/src/kiss/agents/vscode"
    VSIX="$VSCODE_EXT_DIR/kiss-sorcar.vsix"
    if [ -f "$VSIX" ]; then
        echo "   kiss-sorcar.vsix already exists — skipping build"
    else
        cd "$VSCODE_EXT_DIR"
        npm ci
        npm run package
        cd "$PROJECT_DIR"
        if [ -f "$VSIX" ]; then
            echo "   Built $VSIX"
        else
            echo "   WARNING: Failed to build VSIX"
        fi
    fi
    echo ""

    # --- 8. Install VS Code extension ----------------------------------------
    echo ">>> [8/8] Installing VS Code extension..."
    if [ -f "$VSIX" ]; then
        if find_code_cli && [ -n "$CODE_CLI" ]; then
            "$CODE_CLI" --install-extension "$VSIX" --force 2>&1
            echo "   Extension installed into VS Code"
        else
            echo "   WARNING: VS Code CLI not found — skipping extension install"
            echo "   To install manually: code --install-extension $VSIX --force"
        fi
    else
        echo "   WARNING: VSIX not found — skipping extension install"
    fi
    echo ""

    # --- Write install_dir marker (used by env.py) ----------------------------
    printf '%s\n' "$PROJECT_DIR" > "$HOME/.kiss/install_dir"

    # --- Persist PATH in shell rc file ----------------------------------------
    echo "--- Persist PATH in shell rc ---"
    _add_path_to_rc() {
        local rc_file="$1"
        local path_line="export PATH=\"$BIN_DIR:\$PATH\""
        if [ -f "$rc_file" ]; then
            if ! grep -qF "$BIN_DIR" "$rc_file"; then
                printf '\n# KISS Agent Framework\n%s\n' "$path_line" >> "$rc_file"
                echo "   Added $BIN_DIR to $rc_file"
            else
                echo "   $BIN_DIR already in $rc_file"
            fi
        else
            printf '# KISS Agent Framework\n%s\n' "$path_line" > "$rc_file"
            echo "   Created $rc_file with PATH"
        fi
    }

    case "${SHELL:-/bin/zsh}" in
        */zsh)  _add_path_to_rc "$HOME/.zshrc" ;;
        */bash) _add_path_to_rc "$HOME/.bashrc" ;;
        */fish)
            _fish_config="$HOME/.config/fish/config.fish"
            mkdir -p "$(dirname "$_fish_config")"
            _fish_line="fish_add_path $BIN_DIR"
            if [ -f "$_fish_config" ] && grep -qF "$BIN_DIR" "$_fish_config"; then
                echo "   $BIN_DIR already in $_fish_config"
            else
                printf '\n# KISS Agent Framework\n%s\n' "$_fish_line" >> "$_fish_config"
                echo "   Added $BIN_DIR to $_fish_config"
            fi
            ;;
        *)      _add_path_to_rc "$HOME/.zshrc"
                _add_path_to_rc "$HOME/.bashrc" ;;
    esac
    echo ""

    echo "=== Installation Complete ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Project: $PROJECT_DIR"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to $LOG_FILE"
echo "Open a new terminal for PATH changes to take effect."
