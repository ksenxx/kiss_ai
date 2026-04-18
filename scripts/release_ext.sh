#!/bin/bash

# Script to build the KISS Sorcar VS Code extension and publish it to the
# Visual Studio Code Marketplace under the name "KISS Sorcar Buggy".
#
# This is intended for publishing a separate "buggy" / experimental build
# alongside the stable "KISS Sorcar" extension. It keeps the same publisher
# ("ksenxx") but uses a different extension id ("kiss-sorcar-buggy") and
# display name ("KISS Sorcar Buggy") so Marketplace treats it as a distinct
# extension that users can install side-by-side with the stable one.
#
# Workflow:
#   1. Verify VSCE_PAT is set.
#   2. Copy the project README into the extension dir (so the Marketplace
#      listing has a description), same as release.sh does.
#   3. Temporarily rewrite package.json's "name" and "displayName" to the
#      "Buggy" variant.
#   4. Build the VSIX (npm ci + npm run package) as kiss-sorcar-buggy.vsix.
#   5. Publish the built VSIX to the Marketplace with `vsce publish`.
#   6. Restore the original package.json (and README cleanup) and remove
#      build artifacts.
#
# IMPORTANT: This script deliberately performs NO git operations — no commits,
# no tags, no pushes to any remote (origin or public). It only touches local
# files (which it restores) and uploads to the VS Code Marketplace.

set -e  # Exit on error

# =============================================================================
# Constants
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VSCODE_EXT_DIR="$PROJECT_ROOT/src/kiss/agents/vscode"
README_FILE="$PROJECT_ROOT/README.md"
VERSION_FILE="$PROJECT_ROOT/src/kiss/_version.py"

BUGGY_NAME="kiss-sorcar-buggy"
BUGGY_DISPLAY_NAME="KISS Sorcar Buggy"
BUGGY_VSIX="${BUGGY_NAME}.vsix"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

PKG_JSON="$VSCODE_EXT_DIR/package.json"
PKG_JSON_BACKUP="$VSCODE_EXT_DIR/package.json.release_ext.bak"
README_COPY="$VSCODE_EXT_DIR/README.md"
README_COPY_EXISTED=false

restore_files() {
    if [[ -f "$PKG_JSON_BACKUP" ]]; then
        mv "$PKG_JSON_BACKUP" "$PKG_JSON"
        print_info "Restored original package.json"
    fi
    if [[ "$README_COPY_EXISTED" == false && -f "$README_COPY" ]]; then
        rm -f "$README_COPY"
        print_info "Removed temporary $README_COPY"
    fi
}

cleanup_artifacts() {
    # Build artifacts produced by `npm run package` (TypeScript out/ and
    # the copied kiss_project/). Keep the .vsix so the user can inspect
    # or re-upload it manually if needed.
    rm -rf "$VSCODE_EXT_DIR/out" "$VSCODE_EXT_DIR/kiss_project"
    print_info "Cleaned up build artifacts (out/, kiss_project/)"
}

on_exit() {
    local exit_code=$?
    restore_files
    if [[ $exit_code -ne 0 ]]; then
        print_error "release_ext.sh failed with exit code $exit_code"
    fi
    exit $exit_code
}

read_project_version() {
    # Read __version__ from src/kiss/_version.py (e.g. '0.2.79'), bump its
    # patch component by 1, write the new value back, and expose it as
    # BUGGY_VERSION. Persisting the bump mirrors what scripts/release.sh does
    # for the stable release and guarantees each Buggy publish uses a fresh,
    # unique version (the Marketplace rejects duplicates).
    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "Version file not found: $VERSION_FILE"
        exit 1
    fi
    local current major minor patch
    current=$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$VERSION_FILE" | head -1)
    if [[ -z "$current" ]]; then
        print_error "Could not extract __version__ from $VERSION_FILE"
        exit 1
    fi
    IFS='.' read -r major minor patch <<< "$current"
    if [[ -z "$major" || -z "$minor" || -z "$patch" ]]; then
        print_error "Unexpected version format in $VERSION_FILE: '$current' (need MAJOR.MINOR.PATCH)"
        exit 1
    fi
    BUGGY_VERSION="${major}.${minor}.$((patch + 1))"
    sed -i.bak "s/__version__ = \"${current}\"/__version__ = \"${BUGGY_VERSION}\"/" "$VERSION_FILE"
    rm -f "${VERSION_FILE}.bak"
    print_info "Bumped $VERSION_FILE: $current -> $BUGGY_VERSION"
}

rewrite_package_json() {
    # Rewrite "name", "displayName" and "version" in package.json using Node
    # so we preserve exact JSON formatting semantics (vsce validates the file).
    #
    # The version comes from read_project_version(), which bumps the patch
    # component of __version__ in src/kiss/_version.py so every Buggy publish
    # uses a fresh, unique version (the Marketplace rejects duplicates).
    node - "$PKG_JSON" "$BUGGY_NAME" "$BUGGY_DISPLAY_NAME" "$BUGGY_VERSION" <<'NODE'
const fs = require('fs');
const [, , path, name, displayName, version] = process.argv;
const pkg = JSON.parse(fs.readFileSync(path, 'utf8'));
pkg.name = name;
pkg.displayName = displayName;
pkg.version = version;
fs.writeFileSync(path, JSON.stringify(pkg, null, 2) + '\n');
NODE
    print_info "Rewrote package.json: name=$BUGGY_NAME, displayName=\"$BUGGY_DISPLAY_NAME\", version=$BUGGY_VERSION"
}

# =============================================================================
# Main
# =============================================================================
main() {
    print_step "Starting KISS Sorcar Buggy extension release"

    if [[ -z "${VSCE_PAT:-}" ]]; then
        print_error "VSCE_PAT environment variable is not set"
        print_info "Please set it with: export VSCE_PAT='your-personal-access-token'"
        exit 1
    fi

    if [[ ! -d "$VSCODE_EXT_DIR" ]]; then
        print_error "VS Code extension directory not found: $VSCODE_EXT_DIR"
        exit 1
    fi

    if [[ ! -f "$PKG_JSON" ]]; then
        print_error "package.json not found: $PKG_JSON"
        exit 1
    fi

    # Install restore trap BEFORE any mutation so failures always roll back.
    trap on_exit EXIT

    # Copy README for Marketplace listing (mirrors release.sh behaviour).
    if [[ -f "$README_COPY" ]]; then
        README_COPY_EXISTED=true
    fi
    if [[ -f "$README_FILE" ]]; then
        cp "$README_FILE" "$README_COPY"
        print_info "Copied $README_FILE to $README_COPY"
    else
        print_warn "Root README not found at $README_FILE — skipping copy"
    fi

    # Back up and rewrite package.json for the Buggy variant.
    read_project_version
    cp "$PKG_JSON" "$PKG_JSON_BACKUP"
    rewrite_package_json

    # Build the VSIX.
    print_step "Building VS Code extension..."
    cd "$VSCODE_EXT_DIR"
    npm ci
    npx vsce package --no-dependencies --allow-star-activation --allow-missing-repository -o "$BUGGY_VSIX"
    if [[ ! -f "$BUGGY_VSIX" ]]; then
        print_error "VSIX file not found: $BUGGY_VSIX"
        exit 1
    fi
    print_info "Built $BUGGY_VSIX"

    # Publish the pre-built VSIX to the Marketplace.
    print_step "Publishing $BUGGY_DISPLAY_NAME to VS Code Marketplace..."
    npx @vscode/vsce publish --packagePath "$BUGGY_VSIX" --pat "$VSCE_PAT"

    # Clean up TypeScript/kiss_project artifacts (but keep the .vsix).
    cleanup_artifacts
    cd - > /dev/null

    print_info "========================================"
    print_info "$BUGGY_DISPLAY_NAME published successfully!"
    print_info "========================================"
    print_info "Marketplace: https://marketplace.visualstudio.com/items?itemName=ksenxx.${BUGGY_NAME}"
    print_info "VSIX:        $VSCODE_EXT_DIR/$BUGGY_VSIX"
}

main "$@"
