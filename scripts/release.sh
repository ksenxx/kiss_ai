#!/bin/bash

# Script to release to public GitHub repository and publish to PyPI.
# Repository: https://github.com/ksenxx/kiss_ai
# PyPI: https://pypi.org/project/kiss-agent-framework/
#
# Workflow:
# 1. Check if origin is ahead of kiss_ai repo
# 2. If ahead, bump version in _version.py and README.md
# 3. Commit changes with "Version bumped"
# 4. Push to origin
# 5. Push to kiss_ai repo and tag with version
# 6. Build offline .pkg and create GitHub release (with .pkg asset)
# 7. Publish to PyPI

set -e  # Exit on error

# =============================================================================
# Constants
# =============================================================================
PUBLIC_REMOTE="public"
PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"
VERSION_FILE="src/kiss/_version.py"
README_FILE="README.md"
PYPI_PACKAGE_NAME="kiss-agent-framework"

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
    local major minor patch
    IFS='.' read -r major minor patch <<< "$current_version"
    patch=$((patch + 1))
    echo "${major}.${minor}.${patch}"
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
    if grep -q 'img.shields.io/badge/version-' "$README_FILE"; then
        sed -i.bak "s|badge/version-[0-9][0-9.]*-blue|badge/version-${version}-blue|g" "$README_FILE"
        rm -f "${README_FILE}.bak"
        print_info "Updated version badge in $README_FILE to $version"
    else
        print_warn "Version badge not found in $README_FILE - skipping"
    fi
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

    # Step 1: Sync local with origin, then check against public
    print_step "Syncing with origin and checking kiss_ai repo..."
    git add -A
    git diff --cached --quiet || git commit -m "Pre-release sync"
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

    # Step 3: Commit changes
    print_step "Committing version bump..."
    git add -A
    git commit -m "Version bumped to $VERSION"
    print_info "Committed version bump"

    # Step 4: Pull latest from origin (rebase), then push (with retry)
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

    # Step 5: Push to kiss_ai repo (mirror from origin, force to ensure sync)
    print_step "Pushing to kiss_ai repo..."
    git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force
    print_info "Pushed to kiss_ai repo"

    print_step "Creating and pushing tag..."
    git tag -a "$TAG_NAME" -m "Release $VERSION"
    git push "$PUBLIC_REMOTE" "$TAG_NAME"
    print_info "Created and pushed tag: $TAG_NAME"

    # Step 6: Build offline .pkg and create GitHub release
    print_step "Building offline installer package..."
    OFFLINE_PKG="$PWD/dist/kiss-offline-installer.pkg"
    bash scripts/build_offline_pkg.sh
    if [[ -f "$OFFLINE_PKG" ]]; then
        print_step "Creating GitHub release with offline installer..."
        gh release create "$TAG_NAME" "$OFFLINE_PKG" \
            --repo ksenxx/kiss_ai \
            --title "KISS $VERSION" \
            --notes "Release $VERSION

## Downloads
- **kiss-offline-installer.pkg** — macOS offline installer (bundles uv, code-server, Python 3.13, git, and all dependencies)

## macOS Installation

The installer package is not signed with an Apple Developer certificate. macOS Gatekeeper will block it by default. To install, use **one** of these methods:

### Method 1: System Settings (recommended)
1. Double-click \`kiss-offline-installer.pkg\` — you'll see a warning dialog
2. Open **System Settings → Privacy & Security**
3. Scroll down to find *\"kiss-offline-installer.pkg\" was blocked*
4. Click **Open Anyway** and confirm

### Method 2: Remove quarantine attribute
\`\`\`bash
xattr -d com.apple.quarantine ~/Downloads/kiss-offline-installer.pkg
open ~/Downloads/kiss-offline-installer.pkg
\`\`\`

### Method 3: Command-line install
\`\`\`bash
xattr -d com.apple.quarantine ~/Downloads/kiss-offline-installer.pkg
sudo installer -pkg ~/Downloads/kiss-offline-installer.pkg -target /
\`\`\`"
        print_info "GitHub release created: https://github.com/ksenxx/kiss_ai/releases/tag/$TAG_NAME"
    else
        print_warn "Offline installer .pkg not found at $OFFLINE_PKG — skipping GitHub release asset upload"
        gh release create "$TAG_NAME" \
            --repo ksenxx/kiss_ai \
            --title "KISS $VERSION" \
            --notes "Release $VERSION"
        print_info "GitHub release created (without .pkg): https://github.com/ksenxx/kiss_ai/releases/tag/$TAG_NAME"
    fi

    # Step 7: Publish to PyPI
    print_step "Publishing to PyPI..."
    publish_to_pypi "$VERSION"

    echo
    print_info "========================================"
    print_info "Release completed successfully!"
    print_info "========================================"
    print_info "GitHub:  $PUBLIC_REPO_URL"
    print_info "PyPI:    https://pypi.org/project/${PYPI_PACKAGE_NAME}/"
    print_info "Version: $VERSION"
    print_info "Tag:     $TAG_NAME"
    echo
}

main "$@"
