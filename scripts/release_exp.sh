#!/bin/bash

# Build, publish, and install the VS Code extension only.
# Skips version bumps, git operations, and PyPI publishing.
#
# Usage: scripts/release_exp.sh

set -e

# =============================================================================
# Constants
# =============================================================================
VERSION_FILE="src/kiss/_version.py"
README_FILE="README.md"
VSCODE_EXT_DIR="src/kiss/agents/vscode"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

get_version() {
    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "Version file not found: $VERSION_FILE"
        exit 1
    fi
    VERSION=$(grep -oP '__version__\s*=\s*"\K[^"]+' "$VERSION_FILE" 2>/dev/null || \
              grep '__version__' "$VERSION_FILE" | sed 's/.*"\(.*\)".*/\1/')
    if [[ -z "$VERSION" ]]; then
        print_error "Could not extract version from $VERSION_FILE"
        exit 1
    fi
    echo "$VERSION"
}

build_vscode_extension() {
    print_step "Building VS Code extension..."
    cp "$README_FILE" "$VSCODE_EXT_DIR/README.md"
    print_info "Copied $README_FILE to $VSCODE_EXT_DIR/README.md"
    cd "$VSCODE_EXT_DIR"
    npm ci
    npm run package

    if [[ ! -f "kiss-sorcar.vsix" ]]; then
        print_error "VSIX file not found: kiss-sorcar.vsix"
        cd - > /dev/null
        return 1
    fi

    print_info "Built kiss-sorcar.vsix"
    rm -rf out kiss_project
    print_info "Cleaned up build artifacts (out/, kiss_project/)"
    cd - > /dev/null
}

publish_vscode_extension() {
    local version="$1"

    if [[ -z "${VSCE_PAT:-}" ]]; then
        print_error "VSCE_PAT environment variable is not set"
        print_info "Please set it with: export VSCE_PAT='your-personal-access-token'"
        return 1
    fi

    print_step "Publishing VS Code extension..."
    cd "$VSCODE_EXT_DIR"
    npx @vscode/vsce publish --packagePath "kiss-sorcar.vsix" --pat "$VSCE_PAT"
    cd - > /dev/null

    print_info "Successfully published VS Code extension v$version"
    print_info "View at: https://marketplace.visualstudio.com/items?itemName=ksenxx.kiss-sorcar"
}

upload_to_github_release() {
    local version="$1"
    local tag_name="v$version"
    local vsix_asset="${VSCODE_EXT_DIR}/kiss-sorcar.vsix"

    if ! command -v gh &>/dev/null; then
        print_warn "gh CLI not found — skipping GitHub release upload"
        return
    fi

    if ! gh release view "$tag_name" --repo ksenxx/kiss_ai &>/dev/null; then
        print_warn "GitHub release $tag_name not found — skipping upload"
        return
    fi

    print_step "Uploading VSIX to GitHub release $tag_name..."
    gh release upload "$tag_name" "$vsix_asset" --repo ksenxx/kiss_ai --clobber
    print_info "VSIX uploaded to release $tag_name"
}

install_local_extension() {
    local vsix_path="${VSCODE_EXT_DIR}/kiss-sorcar.vsix"

    # Install into VS Code
    local code_cli=""
    for candidate in \
        "$(command -v code 2>/dev/null || true)" \
        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
        "$HOME/.local/bin/code"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            code_cli="$candidate"
            break
        fi
    done
    if [[ -n "$code_cli" ]]; then
        print_step "Installing extension into VS Code..."
        if "$code_cli" --install-extension "$vsix_path" --force 2>&1; then
            print_info "Extension installed into VS Code"
        else
            print_warn "Failed to install extension into VS Code — continuing"
        fi
    else
        print_info "VS Code CLI not found — skipping local VS Code install"
    fi

    # Install into Cursor
    local cursor_cli=""
    if command -v cursor &>/dev/null; then
        cursor_cli="cursor"
    elif [[ -x "/Applications/Cursor.app/Contents/Resources/app/bin/cursor" ]]; then
        cursor_cli="/Applications/Cursor.app/Contents/Resources/app/bin/cursor"
    fi
    if [[ -n "$cursor_cli" ]]; then
        print_step "Installing extension into Cursor IDE..."
        if "$cursor_cli" --install-extension "$vsix_path" --force 2>&1; then
            print_info "Extension installed into Cursor IDE"
        else
            print_warn "Failed to install extension into Cursor IDE — continuing"
        fi
    else
        print_info "Cursor IDE not found — skipping local Cursor install"
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    print_step "Extension-only release"
    echo

    VERSION=$(get_version)
    print_info "Current version: $VERSION"

    # Step 1: Build the extension
    build_vscode_extension

    # Step 2: Publish to VS Code marketplace
    publish_vscode_extension "$VERSION"

    # Step 3: Upload to existing GitHub release (if it exists)
    upload_to_github_release "$VERSION"

    # Step 4: Install locally
    install_local_extension

    echo
    print_info "========================================"
    print_info "Extension release completed!"
    print_info "========================================"
    print_info "Version: $VERSION"
    print_info "VSCode:  https://marketplace.visualstudio.com/items?itemName=ksenxx.kiss-sorcar"
    echo
}

main "$@"
