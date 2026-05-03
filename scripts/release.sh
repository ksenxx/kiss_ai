#!/bin/bash

# Script to release to public GitHub repository and publish to PyPI.
# Repository: https://github.com/ksenxx/kiss_ai
# PyPI: https://pypi.org/project/kiss-agent-framework/
#
# Workflow:
# 1. Stash any uncommitted changes
# 2. Check if origin is ahead of kiss_ai repo
# 3. If ahead, bump version in _version.py, README.md, SYSTEM.md, package.json, package-lock.json
# 4. Download official Claude Code skills (bundled into the extension)
# 5. Build VS Code extension (.vsix) so it's included in the commit
# 6. Commit changes with "Version bumped" (includes vsix)
# 7. Push to origin
# 8. Push to kiss_ai repo and tag with version
# 9. Create GitHub release and upload VSIX asset
# 10. Publish to PyPI
# 11. Publish VS Code extension to marketplace
# 12. Install extension into local VS Code and Cursor IDE (if installed)
# 13. Restore stashed changes

set -e  # Exit on error

# =============================================================================
# Constants
# =============================================================================
PUBLIC_REMOTE="public"
PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"
VERSION_FILE="src/kiss/_version.py"
README_FILE="README.md"
SYSTEM_FILE="src/kiss/SYSTEM.md"
PYPI_PACKAGE_NAME="kiss-agent-framework"
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

bump_version() {
    local current_version="$1"
    local cur_year cur_month cur_minor
    IFS='.' read -r cur_year cur_month cur_minor <<< "$current_version"

    local now_year now_month
    now_year=$(date +%Y)
    now_month=$(date +%-m)  # no leading zero

    if [[ "$cur_year" == "$now_year" && "$cur_month" == "$now_month" ]]; then
        echo "${now_year}.${now_month}.$(( cur_minor + 1 ))"
    else
        echo "${now_year}.${now_month}.0"
    fi
}

update_version_file() {
    local new_version="$1"
    sed -i.bak "s/__version__ = \".*\"/__version__ = \"${new_version}\"/" "$VERSION_FILE"
    rm -f "${VERSION_FILE}.bak"
    print_info "Updated $VERSION_FILE to version $new_version"
}

update_readme_version() {
    local version="$1"
    if [[ ! -f "$README_FILE" ]]; then
        print_warn "README file not found: $README_FILE - skipping"
        return
    fi
    local old_version
    old_version=$(grep -oP 'badge/version-\K[0-9][0-9.]*(?=-blue)' "$README_FILE" 2>/dev/null || \
                  grep 'badge/version-' "$README_FILE" | sed 's/.*badge\/version-\([0-9][0-9.]*\)-blue.*/\1/' | head -1)
    if [[ -n "$old_version" && "$old_version" != "$version" ]]; then
        sed -i.bak "s/${old_version}/${version}/g" "$README_FILE"
        rm -f "${README_FILE}.bak"
        print_info "Updated all occurrences of $old_version to $version in $README_FILE"
    elif [[ -z "$old_version" ]]; then
        print_warn "Version badge not found in $README_FILE - skipping"
    else
        print_info "README already at version $version"
    fi
}

update_system_md_version() {
    local version="$1"
    if [[ ! -f "$SYSTEM_FILE" ]]; then
        print_warn "SYSTEM file not found: $SYSTEM_FILE - skipping"
        return
    fi
    local old_version updated=false

    # Replace "Your version is X.Y.Z"
    old_version=$(grep -oP 'Your version is \K[0-9][0-9.]*' "$SYSTEM_FILE" 2>/dev/null || \
                  grep 'Your version is ' "$SYSTEM_FILE" | sed 's/.*Your version is \([0-9][0-9.]*\).*/\1/' | head -1)
    if [[ -n "$old_version" && "$old_version" != "$version" ]]; then
        sed -i.bak "s/Your version is ${old_version}/Your version is ${version}/g" "$SYSTEM_FILE"
        rm -f "${SYSTEM_FILE}.bak"
        print_info "Updated 'Your version is' from $old_version to $version in $SYSTEM_FILE"
        updated=true
    fi

    # Replace "ksenxx.kiss-sorcar-X.Y.Z"
    old_version=$(grep -oP 'ksenxx\.kiss-sorcar-\K[0-9][0-9.]*' "$SYSTEM_FILE" 2>/dev/null || \
                  grep 'ksenxx\.kiss-sorcar-' "$SYSTEM_FILE" | sed 's/.*ksenxx\.kiss-sorcar-\([0-9][0-9.]*\).*/\1/' | head -1)
    if [[ -n "$old_version" && "$old_version" != "$version" ]]; then
        sed -i.bak "s/ksenxx\.kiss-sorcar-${old_version}/ksenxx.kiss-sorcar-${version}/g" "$SYSTEM_FILE"
        rm -f "${SYSTEM_FILE}.bak"
        print_info "Updated 'ksenxx.kiss-sorcar-' from $old_version to $version in $SYSTEM_FILE"
        updated=true
    fi

    if [[ "$updated" == false ]]; then
        if [[ -z "$old_version" ]]; then
            print_warn "Version not found in $SYSTEM_FILE - skipping"
        else
            print_info "SYSTEM.md already at version $version"
        fi
    fi
}

update_vscode_package_version() {
    local version="$1"
    local pkg_json="${VSCODE_EXT_DIR}/package.json"
    if [[ ! -f "$pkg_json" ]]; then
        print_warn "VS Code package.json not found: $pkg_json - skipping"
        return
    fi
    sed -i.bak "s/\"version\": \"[^\"]*\"/\"version\": \"${version}\"/" "$pkg_json"
    rm -f "${pkg_json}.bak"
    print_info "Updated $pkg_json to version $version"
}

update_vscode_package_lock_version() {
    local version="$1"
    local lock_json="${VSCODE_EXT_DIR}/package-lock.json"
    if [[ ! -f "$lock_json" ]]; then
        print_warn "VS Code package-lock.json not found: $lock_json - skipping"
        return
    fi
    # Only the first 15 lines contain the project's own version (lines 3 and 9);
    # dependency versions deeper in the file must not be touched.
    sed -i.bak "1,15s/\"version\": \"[^\"]*\"/\"version\": \"${version}\"/" "$lock_json"
    rm -f "${lock_json}.bak"
    print_info "Updated $lock_json to version $version"
}

ensure_remote() {
    if ! git remote get-url "$PUBLIC_REMOTE" &>/dev/null; then
        print_info "Adding remote '$PUBLIC_REMOTE'..."
        git remote add "$PUBLIC_REMOTE" "$PUBLIC_REPO_SSH"
    fi
}

publish_to_pypi() {
    local version="$1"
    
    print_step "Building package for PyPI..."
    rm -rf dist/*.tar.gz dist/*.whl
    uv build
    
    if [[ -z "$(ls dist/*.tar.gz dist/*.whl 2>/dev/null)" ]]; then
        print_error "Build failed - no .tar.gz or .whl files in dist/"
        return 1
    fi
    
    print_info "Built packages:"
    ls -la dist/*.tar.gz dist/*.whl
    
    print_step "Uploading to PyPI..."
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
        print_error "UV_PUBLISH_TOKEN environment variable is not set"
        print_info "Please set it with: export UV_PUBLISH_TOKEN='pypi-your-token-here'"
        return 1
    fi
    
    uv publish
    
    print_info "Successfully published version $version to PyPI"
    print_info "View at: https://pypi.org/project/${PYPI_PACKAGE_NAME}/${version}/"
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
    npx @vscode/vsce publish \
        --packagePath "kiss-sorcar.vsix" \
        --pat "$VSCE_PAT" \
        --allow-proposed-apis contribSourceControlInputBoxMenu
    cd - > /dev/null

    print_info "Successfully published VS Code extension v$version"
    print_info "View at: https://marketplace.visualstudio.com/items?itemName=ksenxx.kiss-sorcar"
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
# Main Release Process
# =============================================================================
main() {
    print_step "Starting release process"
    echo "Public repo: $PUBLIC_REPO_URL"
    echo

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Get current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_info "Current branch: $CURRENT_BRANCH"

    # Ensure public remote exists
    ensure_remote

    # Step 1: Stash uncommitted changes, sync with origin, then check against public
    print_step "Syncing with origin and checking kiss_ai repo..."
    STASHED=false
    if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        print_info "Stashing uncommitted changes..."
        git stash push --include-untracked -m "release-script: pre-release stash"
        STASHED=true
    fi
    trap 'if [[ "$STASHED" == true ]]; then print_warn "Restoring stashed changes after failure..."; git stash pop; fi' EXIT
    git fetch origin
    git fetch "$PUBLIC_REMOTE"
    git pull --rebase origin "$CURRENT_BRANCH"

    ORIGIN_HEAD=$(git rev-parse HEAD)
    PUBLIC_HEAD=$(git rev-parse "$PUBLIC_REMOTE/main" 2>/dev/null || echo "")

    if [[ -z "$PUBLIC_HEAD" ]]; then
        print_info "Public repo has no main branch yet - will create it"
    elif [[ "$ORIGIN_HEAD" == "$PUBLIC_HEAD" ]]; then
        print_info "Origin and kiss_ai are in sync - nothing to release"
        exit 0
    elif git merge-base --is-ancestor "$PUBLIC_HEAD" "$ORIGIN_HEAD"; then
        print_info "Origin is ahead of kiss_ai - proceeding with release"
    else
        print_warn "Origin and kiss_ai have diverged - will force-push to sync"
    fi

    # Step 2: Bump version in _version.py and README.md
    CURRENT_VERSION=$(get_version)
    VERSION=$(bump_version "$CURRENT_VERSION")
    TAG_NAME="v$VERSION"
    
    print_info "Current version: $CURRENT_VERSION"
    print_info "New version: $VERSION (tag: $TAG_NAME)"
    
    print_step "Bumping version..."
    update_version_file "$VERSION"
    update_readme_version "$VERSION"
    update_system_md_version "$VERSION"
    update_vscode_package_version "$VERSION"
    update_vscode_package_lock_version "$VERSION"

    # Step 3: Download official Claude Code skills (before building extension)
    print_step "Downloading official Claude Code skills..."
    CLAUDE_SKILLS_DIR="src/kiss/agents/claude_skills"
    if [ -d "$CLAUDE_SKILLS_DIR" ] && [ "$(ls -d "$CLAUDE_SKILLS_DIR"/*/ 2>/dev/null)" ]; then
        print_info "Claude skills already present — skipping download"
    else
        mkdir -p "$CLAUDE_SKILLS_DIR"
        SKILLS_TMP="$(mktemp -d)"
        print_info "Cloning anthropics/claude-code plugins..."
        if git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/anthropics/claude-code.git "$SKILLS_TMP/claude-code" 2>&1; then
            cd "$SKILLS_TMP/claude-code"
            git sparse-checkout set plugins 2>&1
            for plugin_dir in plugins/*/; do
                if [ -d "$plugin_dir" ]; then
                    plugin_name="$(basename "$plugin_dir")"
                    cp -R "$plugin_dir" "$CLAUDE_SKILLS_DIR/$plugin_name"
                fi
            done
            cd - > /dev/null
            SKILL_COUNT="$(ls -d "$CLAUDE_SKILLS_DIR"/*/ 2>/dev/null | wc -l | tr -d ' ')"
            print_info "Installed $SKILL_COUNT Claude skills to $CLAUDE_SKILLS_DIR"
        else
            print_warn "Failed to download Claude Code skills"
        fi
        rm -rf "$SKILLS_TMP"
    fi

    # Step 4: Build VS Code extension (before commit so vsix is included)
    build_vscode_extension

    # Step 5: Commit changes (includes version bump + fresh vsix)
    print_step "Committing version bump..."
    git add -A
    git commit -m "Version bumped to $VERSION"
    print_info "Committed version bump"

    # Step 6: Pull latest from origin (rebase), then push (with retry)
    print_step "Syncing with origin..."
    for attempt in 1 2 3; do
        git pull --rebase origin "$CURRENT_BRANCH"
        if git push origin "$CURRENT_BRANCH"; then
            break
        fi
        if [[ $attempt -eq 3 ]]; then
            print_error "Failed to push to origin after 3 attempts"
            exit 1
        fi
        print_warn "Push to origin failed (attempt $attempt/3), retrying in 2s..."
        sleep 2
    done
    print_info "Pushed to origin"

    # Step 7: Push to kiss_ai repo (mirror from origin, force to ensure sync)
    print_step "Pushing to kiss_ai repo..."
    git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force
    print_info "Pushed to kiss_ai repo"

    print_step "Creating and pushing tag..."
    git tag -a "$TAG_NAME" -m "Release $VERSION"
    git push "$PUBLIC_REMOTE" "$TAG_NAME"
    print_info "Created and pushed tag: $TAG_NAME"

    # Step 8: Create GitHub release and upload VSIX
    print_step "Creating GitHub release..."
    gh release create "$TAG_NAME" \
        --repo ksenxx/kiss_ai \
        --title "KISS $VERSION" \
        --notes "Release $VERSION"
    print_info "GitHub release created: https://github.com/ksenxx/kiss_ai/releases/tag/$TAG_NAME"

    local vsix_asset="${VSCODE_EXT_DIR}/kiss-sorcar.vsix"
    if [[ -f "$vsix_asset" ]]; then
        print_step "Uploading VSIX to GitHub release..."
        gh release upload "$TAG_NAME" "$vsix_asset" --repo ksenxx/kiss_ai
        print_info "VSIX uploaded to release"
    fi

    # Step 9: Publish to PyPI
    print_step "Publishing to PyPI..."
    publish_to_pypi "$VERSION"

    # Step 10: Publish VS Code extension (already built in step 4)
    publish_vscode_extension "$VERSION"

    # Step 11: Install extension into local VS Code and Cursor IDE if available
    install_local_extension

    # Clean up source claude_skills now that they are bundled in the extension
    if [ -d "$CLAUDE_SKILLS_DIR" ]; then
        rm -rf "$CLAUDE_SKILLS_DIR"
        print_info "Cleaned up $CLAUDE_SKILLS_DIR (bundled in extension)"
    fi

    # Restore stashed changes
    trap - EXIT
    if [[ "$STASHED" == true ]]; then
        print_step "Restoring stashed changes..."
        git stash pop
        print_info "Stashed changes restored"
    fi

    echo
    print_info "========================================"
    print_info "Release completed successfully!"
    print_info "========================================"
    print_info "GitHub:  $PUBLIC_REPO_URL"
    print_info "PyPI:    https://pypi.org/project/${PYPI_PACKAGE_NAME}/"
    print_info "VSCode:  https://marketplace.visualstudio.com/items?itemName=ksenxx.kiss-sorcar"
    print_info "Version: $VERSION"
    print_info "Tag:     $TAG_NAME"
    echo
}

main "$@"
