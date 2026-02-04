#!/bin/bash

# Script to release to public GitHub repository with file filtering and version tagging
# Repository: https://github.com/ksenxx/kiss_ai

set -e  # Exit on error

# =============================================================================
# CONFIGURATION: Files and directories to EXCLUDE from public release
# =============================================================================
# Add paths relative to repo root that should NOT be pushed to public repo
PRIVATE_FILES=(
    # Add private files/directories here, one per line
    # Example:
    # "private_config.yaml"
    # "internal_docs/"
    # "secrets/"
)

# =============================================================================
# Constants
# =============================================================================
PUBLIC_REMOTE="public"
PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"
VERSION_FILE="src/kiss/_version.py"
README_FILE="README.md"
RELEASE_BRANCH="release-staging"

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

# Get version from _version.py
get_version() {
    if [[ ! -f "$VERSION_FILE" ]]; then
        print_error "Version file not found: $VERSION_FILE"
        exit 1
    fi
    # Extract version string from __version__ = "x.y.z"
    VERSION=$(grep -oP '__version__\s*=\s*"\K[^"]+' "$VERSION_FILE" 2>/dev/null || \
              grep '__version__' "$VERSION_FILE" | sed 's/.*"\(.*\)".*/\1/')
    if [[ -z "$VERSION" ]]; then
        print_error "Could not extract version from $VERSION_FILE"
        exit 1
    fi
    echo "$VERSION"
}

# Check if remote exists, add if not
ensure_remote() {
    if ! git remote get-url "$PUBLIC_REMOTE" &>/dev/null; then
        print_info "Adding remote '$PUBLIC_REMOTE'..."
        git remote add "$PUBLIC_REMOTE" "$PUBLIC_REPO_SSH"
    else
        print_info "Remote '$PUBLIC_REMOTE' exists"
    fi
}

# Check if tag exists on remote
tag_exists_on_remote() {
    local tag="$1"
    git fetch "$PUBLIC_REMOTE" --tags &>/dev/null || true
    if git ls-remote --tags "$PUBLIC_REMOTE" | grep -q "refs/tags/$tag$"; then
        return 0  # Tag exists
    else
        return 1  # Tag does not exist
    fi
}

# Update version in README.md
update_readme_version() {
    local version="$1"
    if [[ ! -f "$README_FILE" ]]; then
        print_warn "README file not found: $README_FILE - skipping version update"
        return
    fi
    
    # Update the **Version:** line in README.md
    if grep -q '^\*\*Version:\*\*' "$README_FILE"; then
        sed -i.bak "s/^\*\*Version:\*\* .*/\*\*Version:\*\* $version/" "$README_FILE"
        rm -f "${README_FILE}.bak"
        print_info "Updated version in $README_FILE to $version"
    else
        print_warn "Version line not found in $README_FILE - skipping update"
    fi
}

# =============================================================================
# Main Release Process
# =============================================================================
main() {
    print_step "Starting release to public repository"
    echo "Repository: $PUBLIC_REPO_URL"
    echo

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Get current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_info "Current branch: $CURRENT_BRANCH"

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_error "You have uncommitted changes. Please commit or stash them first."
        exit 1
    fi

    # Get version
    VERSION=$(get_version)
    TAG_NAME="v$VERSION"
    print_info "Version: $VERSION (tag: $TAG_NAME)"

    # Ensure remote exists
    ensure_remote

    # Check if there are private files to exclude
    if [[ ${#PRIVATE_FILES[@]} -eq 0 ]]; then
        print_info "No private files configured - pushing entire repo"
        
        # Simple push without filtering
        print_step "Pushing to public remote..."
        git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force-with-lease
        
    else
        print_info "Private files to exclude:"
        for file in "${PRIVATE_FILES[@]}"; do
            echo "  - $file"
        done
        echo

        # Create a temporary branch for the filtered release
        print_step "Creating filtered release branch..."
        
        # Delete release branch if it exists
        git branch -D "$RELEASE_BRANCH" 2>/dev/null || true
        
        # Create new branch from current HEAD
        git checkout -b "$RELEASE_BRANCH"

        # Remove private files from the release branch
        print_step "Removing private files from release..."
        for file in "${PRIVATE_FILES[@]}"; do
            if [[ -e "$file" ]]; then
                git rm -rf --cached "$file" 2>/dev/null || true
                print_info "Removed: $file"
            else
                print_warn "File not found (skipping): $file"
            fi
        done

        # Check if there are changes to commit
        if ! git diff-index --quiet HEAD --; then
            git commit -m "Release $VERSION - remove private files"
        fi

        # Push the filtered branch to public remote
        print_step "Pushing filtered branch to public remote..."
        git push "$PUBLIC_REMOTE" "$RELEASE_BRANCH:main" --force-with-lease

        # Return to original branch
        print_step "Cleaning up..."
        git checkout "$CURRENT_BRANCH"
        git branch -D "$RELEASE_BRANCH"
    fi

    # Handle version tagging
    print_step "Checking version tag..."
    if tag_exists_on_remote "$TAG_NAME"; then
        print_info "Tag '$TAG_NAME' already exists on public remote - skipping"
    else
        print_info "Version bump detected - creating tag '$TAG_NAME'..."
        
        # Update version in README.md
        update_readme_version "$VERSION"
        
        # Commit README version update if there are changes
        if ! git diff --quiet "$README_FILE" 2>/dev/null; then
            git add "$README_FILE"
            git commit -m "Update version to $VERSION in README.md"
            print_info "Committed README version update"
            
            # Re-push the branch with the version update
            print_step "Re-pushing branch with version update..."
            if [[ ${#PRIVATE_FILES[@]} -eq 0 ]]; then
                git push "$PUBLIC_REMOTE" "$CURRENT_BRANCH:main" --force-with-lease
            else
                # Need to recreate the filtered branch with the new commit
                git branch -D "$RELEASE_BRANCH" 2>/dev/null || true
                git checkout -b "$RELEASE_BRANCH"
                for file in "${PRIVATE_FILES[@]}"; do
                    if [[ -e "$file" ]]; then
                        git rm -rf --cached "$file" 2>/dev/null || true
                    fi
                done
                if ! git diff-index --quiet HEAD --; then
                    git commit -m "Release $VERSION - remove private files"
                fi
                git push "$PUBLIC_REMOTE" "$RELEASE_BRANCH:main" --force-with-lease
                git checkout "$CURRENT_BRANCH"
                git branch -D "$RELEASE_BRANCH"
            fi
        fi
        
        # Create tag locally if it doesn't exist
        if ! git tag -l "$TAG_NAME" | grep -q "$TAG_NAME"; then
            git tag -a "$TAG_NAME" -m "Release $VERSION"
            print_info "Created local tag: $TAG_NAME"
        fi
        
        # Push tag to public remote
        git push "$PUBLIC_REMOTE" "$TAG_NAME"
        print_info "Pushed tag '$TAG_NAME' to public remote"
    fi

    echo
    print_info "========================================"
    print_info "Release completed successfully!"
    print_info "========================================"
    print_info "Repository: $PUBLIC_REPO_URL"
    print_info "Version:    $VERSION"
    print_info "Tag:        $TAG_NAME"
    echo
}

# Run main function
main "$@"
