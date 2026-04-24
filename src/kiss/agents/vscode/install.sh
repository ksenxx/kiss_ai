#!/bin/bash
# Build, package, and install the KISS Sorcar VS Code extension.
# Automatically downloads and installs missing binaries from official sources.
# Generates a log file at ~/.kiss/install.log
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/.kiss"
LOG_FILE="$LOG_DIR/install.log"
NODE_VERSION="v22.16.0"
UV_VERSION="0.11.2"

mkdir -p "$LOG_DIR"

# Ensure ~/.local/bin is in PATH for newly installed binaries
export PATH="$HOME/.local/bin:$PATH"

{
    echo "=== KISS Sorcar Extension Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Directory: $SCRIPT_DIR"
    echo ""

    # --- Installation helpers ---

    install_node() {
        echo "  Downloading Node.js $NODE_VERSION from https://nodejs.org ..."
        local OS ARCH
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64)         ARCH="x64" ;;
            aarch64|arm64)  ARCH="arm64" ;;
            *)  echo "  ERROR: Unsupported architecture: $ARCH"; return 1 ;;
        esac
        local URL="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-${OS}-${ARCH}.tar.gz"
        mkdir -p "$HOME/.local"
        if curl -fsSL "$URL" | tar xz -C "$HOME/.local" --strip-components=1; then
            echo "  Node.js $NODE_VERSION installed to ~/.local/"
        else
            echo "  ERROR: Failed to download/extract Node.js from $URL"
            return 1
        fi
    }

    install_uv() {
        echo "  Downloading uv $UV_VERSION from https://releases.astral.sh ..."
        local OS ARCH TARGET
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m)
        case "$OS" in
            darwin)
                case "$ARCH" in
                    x86_64)         TARGET="x86_64-apple-darwin" ;;
                    aarch64|arm64)  TARGET="aarch64-apple-darwin" ;;
                    *)  echo "  ERROR: Unsupported architecture: $ARCH"; return 1 ;;
                esac
                ;;
            linux)
                case "$ARCH" in
                    x86_64)         TARGET="x86_64-unknown-linux-gnu" ;;
                    aarch64|arm64)  TARGET="aarch64-unknown-linux-gnu" ;;
                    *)  echo "  ERROR: Unsupported architecture: $ARCH"; return 1 ;;
                esac
                ;;
            *)  echo "  ERROR: Unsupported OS: $OS"; return 1 ;;
        esac
        local URL="https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz"
        mkdir -p "$HOME/.local/bin"
        if curl -fsSL "$URL" | tar xz -C "$HOME/.local/bin" --strip-components=1; then
            echo "  uv $UV_VERSION installed to ~/.local/bin/"
        else
            echo "  ERROR: Failed to download/extract uv from $URL"
            return 1
        fi
    }

    install_git() {
        case "$(uname -s)" in
            Darwin)
                if command -v brew >/dev/null 2>&1; then
                    echo "  Installing git via Homebrew (binary bottle)..."
                    brew install git 2>&1
                else
                    echo "  Triggering Xcode Command Line Tools installation (provides git)..."
                    xcode-select --install 2>&1 || true
                    echo "  NOTE: Please complete the Xcode CLT dialog if prompted, then re-run this script."
                fi
                ;;
            Linux)
                if command -v apt-get >/dev/null 2>&1; then
                    echo "  Installing git via apt-get..."
                    sudo apt-get update -y && sudo apt-get install -y git 2>&1
                elif command -v dnf >/dev/null 2>&1; then
                    echo "  Installing git via dnf..."
                    sudo dnf install -y git 2>&1
                elif command -v yum >/dev/null 2>&1; then
                    echo "  Installing git via yum..."
                    sudo yum install -y git 2>&1
                elif command -v pacman >/dev/null 2>&1; then
                    echo "  Installing git via pacman..."
                    sudo pacman -S --noconfirm git 2>&1
                elif command -v apk >/dev/null 2>&1; then
                    echo "  Installing git via apk..."
                    sudo apk add git 2>&1
                else
                    echo "  ERROR: No supported package manager found. Install git from https://git-scm.com"
                fi
                ;;
        esac
    }

    install_code_cli() {
        case "$(uname -s)" in
            Darwin)
                local VSCODE_APP="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"
                if [ -x "$VSCODE_APP" ]; then
                    echo "  Linking VS Code CLI to ~/.local/bin/code ..."
                    mkdir -p "$HOME/.local/bin"
                    ln -sf "$VSCODE_APP" "$HOME/.local/bin/code"
                else
                    echo "  VS Code app not found. Download from https://code.visualstudio.com"
                    return 1
                fi
                ;;
            Linux)
                if command -v snap >/dev/null 2>&1; then
                    echo "  Installing VS Code via snap..."
                    sudo snap install --classic code 2>&1
                elif command -v apt-get >/dev/null 2>&1; then
                    echo "  Installing VS Code via apt (Microsoft repository)..."
                    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
                        | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg 2>&1
                    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
                        | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null 2>&1
                    sudo apt-get update -y && sudo apt-get install -y code 2>&1
                else
                    echo "  Please install VS Code from https://code.visualstudio.com"
                    return 1
                fi
                ;;
            *)
                echo "  Please install VS Code from https://code.visualstudio.com"
                return 1
                ;;
        esac
    }

    find_code_cli() {
        CODE=""
        for candidate in \
            "$(command -v code 2>/dev/null || true)" \
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
            "$HOME/.local/bin/code" \
            "/usr/local/bin/code" \
            "/usr/bin/code" \
            "/snap/bin/code"; do
            if [ -n "$candidate" ] && [ -x "$candidate" ]; then
                CODE="$candidate"
                return 0
            fi
        done
        return 1
    }

    # --- Check prerequisites and auto-install missing ones ---
    echo "--- Check prerequisites ---"

    # curl is needed to download anything
    if ! command -v curl >/dev/null 2>&1; then
        echo "  ERROR: curl is required but not found. Please install curl first."
        exit 1
    fi

    MISSING=""

    # Node.js (provides node, npm, npx)
    if ! command -v node >/dev/null 2>&1; then
        install_node || true
    fi
    for cmd in node npm npx; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "  $cmd: $(command -v "$cmd")"
        else
            echo "  $cmd: NOT FOUND"
            MISSING="$MISSING $cmd"
        fi
    done

    # uv
    if ! command -v uv >/dev/null 2>&1; then
        install_uv || true
    fi
    if command -v uv >/dev/null 2>&1; then
        echo "  uv: $(command -v uv)"
    else
        echo "  uv: NOT FOUND"
        MISSING="$MISSING uv"
    fi

    # git
    if ! command -v git >/dev/null 2>&1; then
        install_git || true
    fi
    if command -v git >/dev/null 2>&1; then
        echo "  git: $(command -v git)"
    else
        echo "  git: NOT FOUND"
        MISSING="$MISSING git"
    fi

    # VS Code CLI
    if ! find_code_cli; then
        install_code_cli || true
        find_code_cli || true
    fi
    if [ -n "$CODE" ]; then
        echo "  code: $CODE"
    else
        echo "  code: NOT FOUND"
        MISSING="$MISSING code"
    fi

    if [ -n "$MISSING" ]; then
        echo ""
        echo "ERROR: Could not install required binaries:$MISSING"
        echo "Please install them manually and re-run this script."
        exit 1
    fi
    echo "All prerequisites found"
    echo ""

    cd "$SCRIPT_DIR"

    echo "--- Compile TypeScript ---"
    npx tsc -p ./ 2>&1
    echo ""

    echo "--- Copy KISS project files ---"
    npm run copy-kiss 2>&1
    echo ""

    echo "--- Package VSIX ---"
    npx vsce package --no-dependencies --allow-missing-repository -o kiss-sorcar.vsix 2>&1
    echo ""

    echo "--- Check Python version ---"
    PYTHON_VERSION=$(uv run python --version 2>&1)
    echo "$PYTHON_VERSION"
    MAJOR=$(echo "$PYTHON_VERSION" | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f2)
    if [ "${MAJOR:-0}" -lt 3 ] || { [ "${MAJOR:-0}" -eq 3 ] && [ "${MINOR:-0}" -lt 13 ]; }; then
        echo "ERROR: Python 3.13+ is required but found $PYTHON_VERSION"
        exit 1
    fi
    echo "Python version OK"
    echo ""

    VSIX="$SCRIPT_DIR/kiss-sorcar.vsix"
    if [ ! -f "$VSIX" ]; then
        echo "ERROR: No .vsix file found after packaging"
        exit 1
    fi
    echo "Built: $VSIX"
    echo ""

    echo "--- Persist PATH in shell rc ---"
    # Ensure ~/.local/bin is in the user's shell rc file for future terminal sessions
    SHELL_RC=""
    case "${SHELL:-}" in
        */zsh*)  SHELL_RC="$HOME/.zshrc" ;;
        */fish*) SHELL_RC="$HOME/.config/fish/config.fish" ;;
        *)       SHELL_RC="$HOME/.bashrc" ;;
    esac
    if [ -n "$SHELL_RC" ]; then
        LOCAL_BIN='$HOME/.local/bin'
        if [ -f "$SHELL_RC" ] && grep -q "$LOCAL_BIN" "$SHELL_RC" 2>/dev/null; then
            echo "  PATH already in $SHELL_RC"
        else
            mkdir -p "$(dirname "$SHELL_RC")"
            if echo "$SHELL_RC" | grep -q "config.fish"; then
                echo "fish_add_path \"$LOCAL_BIN\"" >> "$SHELL_RC"
            else
                echo "export PATH=\"$LOCAL_BIN:\$PATH\"" >> "$SHELL_RC"
            fi
            echo "  Added $LOCAL_BIN to PATH in $SHELL_RC"
        fi
    fi
    echo ""

    echo "--- Install Extension ---"
    "$CODE" --install-extension "$VSIX" --force 2>&1
    echo ""

    echo "--- Clean up build artifacts ---"
    rm -rf "$SCRIPT_DIR/out" "$SCRIPT_DIR/kiss_project"
    echo "  Removed out/ and kiss_project/"
    echo ""

    echo "=== Install complete ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to $LOG_FILE"
