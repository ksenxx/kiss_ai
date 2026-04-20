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

    # Generate a unique version for experimental builds using a timestamp
    EXP_VERSION="$(echo "$VERSION" | cut -d. -f1).$(echo "$VERSION" | cut -d. -f2).$(date +%s)"

    # Rename extension to "KISS Sorcar Buggy" for experimental builds
    sed -i.bak \
        -e 's/"name": "kiss-sorcar"/"name": "kiss-sorcar-buggy"/' \
        -e 's/"displayName": "KISS Sorcar"/"displayName": "KISS Sorcar Buggy"/' \
        "$VSCODE_EXT_DIR/package.json"
    # Export KISS_EXP_VERSION so copy-kiss.sh (run during vsce package) uses it
    # instead of reading _version.py, which would overwrite our timestamp version.
    export KISS_EXP_VERSION="$EXP_VERSION"
    print_info "Set extension name to 'kiss-sorcar-buggy', displayName to 'KISS Sorcar Buggy', version to '$EXP_VERSION'"

    cd "$VSCODE_EXT_DIR"
    npm ci
    npm run package

    # npm run package hardcodes -o kiss-sorcar.vsix; rename to match the buggy name
    if [[ -f "kiss-sorcar.vsix" ]]; then
        mv kiss-sorcar.vsix kiss-sorcar-buggy.vsix
    fi

    if [[ ! -f "kiss-sorcar-buggy.vsix" ]]; then
        print_error "VSIX file not found: kiss-sorcar-buggy.vsix"
        cd - > /dev/null
        return 1
    fi

    print_info "Built kiss-sorcar-buggy.vsix"
    rm -rf out kiss_project
    print_info "Cleaned up build artifacts (out/, kiss_project/)"
    cd - > /dev/null

    # Restore original package.json
    mv "$VSCODE_EXT_DIR/package.json.bak" "$VSCODE_EXT_DIR/package.json"
    print_info "Restored original package.json"
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
    npx @vscode/vsce publish --packagePath "kiss-sorcar-buggy.vsix" --pat "$VSCE_PAT"
    cd - > /dev/null

    print_info "Successfully published VS Code extension v$version"
    print_info "View at: https://marketplace.visualstudio.com/items?itemName=ksenxx.kiss-sorcar-buggy"
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
    publish_vscode_extension "$EXP_VERSION"

    # Step 3: Remove local VSIX artifact
    rm -f "${VSCODE_EXT_DIR}/kiss-sorcar-buggy.vsix"
    print_info "Removed local kiss-sorcar-buggy.vsix"

    echo
    print_info "========================================"
    print_info "Extension release completed!"
    print_info "========================================"
    print_info "Version: $EXP_VERSION"
    print_info "VSCode:  https://marketplace.visualstudio.com/items?itemName=ksenxx.kiss-sorcar-buggy"
    echo
}

main "$@"
