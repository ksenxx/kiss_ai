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
# 6. Publish to PyPI

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
    if grep -q '^\*\*Version:\*\*' "$README_FILE"; then
        sed -i.bak "s/^\*\*Version:\*\* .*/\*\*Version:\*\* $version/" "$README_FILE"
        rm -f "${README_FILE}.bak"
        print_info "Updated version in $README_FILE to $version"
    else
        print_warn "Version line not found in $README_FILE - skipping"
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
    rm -rf dist/ build/
    
    if ! uv run python -c "import build" &>/dev/null; then
        print_info "Installing build package..."
        uv pip install build twine
    fi
    
    uv run python -m build
    
    if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
        print_error "Build failed - no files in dist/"
        return 1
    fi
    
    print_info "Built packages:"
    ls -la dist/
    
    print_step "Checking package..."
    uv run python -m twine check dist/*
    
    print_step "Uploading to PyPI..."
    if [[ -z "$UV_PUBLISH_TOKEN" ]]; then
        print_error "UV_PUBLISH_TOKEN environment variable is not set"
        print_info "Please set it with: export UV_PUBLISH_TOKEN='pypi-your-token-here'"
        return 1
    fi
    
    uv run python -m twine upload dist/* -u __token__ -p "$UV_PUBLISH_TOKEN"
    
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

    # Step 1: Check if origin is ahead of kiss_ai repo
    print_step "Checking if origin is ahead of kiss_ai repo..."
    git fetch origin
    git fetch "$PUBLIC_REMOTE"

    ORIGIN_HEAD=$(git rev-parse "origin/$CURRENT_BRANCH")
    PUBLIC_HEAD=$(git rev-parse "$PUBLIC_REMOTE/main" 2>/dev/null || echo "")

    if [[ -z "$PUBLIC_HEAD" ]]; then
        print_info "Public repo has no main branch yet - will create it"
    elif [[ "$ORIGIN_HEAD" == "$PUBLIC_HEAD" ]]; then
        print_info "Origin and kiss_ai are in sync - nothing to release"
        exit 0
    elif git merge-base --is-ancestor "$PUBLIC_HEAD" "$ORIGIN_HEAD"; then
        print_info "Origin is ahead of kiss_ai - proceeding with release"
    else
        print_error "Origin and kiss_ai have diverged - please resolve manually"
        exit 1
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
    git add "$VERSION_FILE" "$README_FILE"
    git commit -m "Version bumped to $VERSION"
    print_info "Committed version bump"

    # Step 4: Push to origin
    print_step "Pushing to origin..."
    git push origin "$CURRENT_BRANCH"
    print_info "Pushed to origin"

    # Step 5: Push to kiss_ai repo and tag
    print_step "Pushing to kiss_ai repo..."
    git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force-with-lease
    print_info "Pushed to kiss_ai repo"

    print_step "Creating and pushing tag..."
    git tag -a "$TAG_NAME" -m "Release $VERSION"
    git push "$PUBLIC_REMOTE" "$TAG_NAME"
    print_info "Created and pushed tag: $TAG_NAME"

    # Step 6: Publish to PyPI
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
