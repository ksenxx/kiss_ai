#!/bin/bash
# Public Release Script for KISS Agent Framework (Option 4: Orphan Branch)
#
# This script creates a clean public release by:
# 1. Creating an orphan branch (no commit history)
# 2. Removing private directories
# 3. Pushing to a separate public repository
#
# Your private development repo remains unchanged.
#
# Usage: ./scripts/public_release.sh [--push]
#   --push: Actually push to the public repository (default: dry run)

set -e

# Configuration - MODIFY THESE FOR YOUR SETUP
PUBLIC_REPO_URL="git@github.com:ksenxx/kiss_ai.git"  # Your PUBLIC repo URL
PUBLIC_BRANCH="main"
VERSION=$(grep -o '__version__ = "[^"]*"' src/kiss/_version.py | cut -d'"' -f2)

# Directories and files to exclude from public release
EXCLUDED_DIRS=(
    "src/kiss/agents/vulnerability_detector"
    "src/kiss/agents/advanced_coding_agent"
    "logs"
    "artifacts"
)

# Files to exclude (glob patterns)
EXCLUDED_FILES=(
    "gemini-3-flash-preview_sample_0.kiss_swebench_verified.json"
    "gemini-3-pro-preview_sample_0.kiss_swebench_verified.json"
    "*.kiss_swebench_verified.json"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}KISS Public Release Script v${VERSION}${NC}"
echo -e "${GREEN}(Orphan Branch Method)${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/kiss" ]; then
    echo -e "${RED}Error: Must run from the KISS repository root directory${NC}"
    exit 1
fi

# Parse arguments
PUSH_MODE=false
if [ "$1" == "--push" ]; then
    PUSH_MODE=true
    echo -e "${YELLOW}Mode: PUSH (will push to public repository)${NC}"
else
    echo -e "${YELLOW}Mode: DRY RUN (use --push to actually push)${NC}"
fi

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "\n${GREEN}Current branch: ${CURRENT_BRANCH}${NC}"

# Stash any uncommitted changes
echo -e "\n${GREEN}Stashing any uncommitted changes...${NC}"
git stash push -m "public-release-temp-stash" 2>/dev/null || true

# Create orphan branch for public release
RELEASE_BRANCH="public-release-temp-$$"
echo -e "\n${GREEN}Creating orphan branch: ${RELEASE_BRANCH}${NC}"
git checkout --orphan "$RELEASE_BRANCH"

# Remove excluded directories from git index and working tree
echo -e "\n${GREEN}Removing private directories...${NC}"
for dir in "${EXCLUDED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  Removing directory: $dir"
        git rm -rf --cached "$dir" 2>/dev/null || true
        rm -rf "$dir"
    fi
done

# Remove excluded files
echo -e "\n${GREEN}Removing private files...${NC}"
for pattern in "${EXCLUDED_FILES[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            echo "  Removing file: $file"
            git rm -f --cached "$file" 2>/dev/null || true
            rm -f "$file"
        fi
    done
done

# Remove caches and build artifacts
echo -e "\n${GREEN}Cleaning up build artifacts...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .ruff_cache/ 2>/dev/null || true
rm -rf logs/ artifacts/ workspace/ 2>/dev/null || true

# Remove from git index
git rm -rf --cached __pycache__ dist build .pytest_cache .mypy_cache .ruff_cache logs artifacts workspace 2>/dev/null || true

# Stage all remaining files
echo -e "\n${GREEN}Staging files for release...${NC}"
git add .

# Show what will be committed
echo -e "\n${GREEN}Files to be released:${NC}"
git status --short | head -30
FILE_COUNT=$(git status --short | wc -l)
echo -e "... (total: ${FILE_COUNT} files)"

# Verify excluded directories are not staged
echo -e "\n${GREEN}Verifying exclusions:${NC}"
for dir in "${EXCLUDED_DIRS[@]}"; do
    if git ls-files --cached "$dir" 2>/dev/null | grep -q .; then
        echo -e "${RED}ERROR: $dir is still staged!${NC}"
        git checkout "$CURRENT_BRANCH"
        git branch -D "$RELEASE_BRANCH"
        git stash pop 2>/dev/null || true
        exit 1
    else
        echo -e "  ✓ $dir excluded"
    fi
done

# Commit
echo -e "\n${GREEN}Creating release commit...${NC}"
git commit -m "KISS Agent Framework v${VERSION}

A simple and portable AI agent framework for building and evolving LLM agents.

Features:
- Simple ReAct architecture with native function calling
- Support for multiple LLM providers (OpenAI, Anthropic, Gemini, Together AI, OpenRouter)
- GEPA prompt optimization
- KISSEvolve evolutionary algorithm discovery
- Docker integration for isolated execution
- Trajectory visualization"

# Create tag
git tag -a "v${VERSION}" -m "Release v${VERSION}" 2>/dev/null || true

if [ "$PUSH_MODE" = true ]; then
    echo -e "\n${GREEN}Pushing to public repository...${NC}"
    echo -e "${YELLOW}Target: ${PUBLIC_REPO_URL} (branch: ${PUBLIC_BRANCH})${NC}"
    
    # Add public remote if not exists
    git remote add public "$PUBLIC_REPO_URL" 2>/dev/null || git remote set-url public "$PUBLIC_REPO_URL"
    
    # Force push to public repo
    git push -f public "${RELEASE_BRANCH}:${PUBLIC_BRANCH}"
    git push -f public "v${VERSION}" 2>/dev/null || true
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Public release v${VERSION} pushed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Public repo: ${PUBLIC_REPO_URL}"
    echo -e "Branch: ${PUBLIC_BRANCH}"
    echo -e "Tag: v${VERSION}"
else
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}DRY RUN COMPLETE${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "Release commit created on branch: ${RELEASE_BRANCH}"
    echo -e "To inspect: git log --oneline -1"
    echo -e "To see files: git ls-files"
    echo -e "\nTo push for real, run: $0 --push"
fi

# Return to original branch
echo -e "\n${GREEN}Returning to original branch: ${CURRENT_BRANCH}${NC}"
git checkout "$CURRENT_BRANCH"

# Delete temporary branch
git branch -D "$RELEASE_BRANCH"

# Restore stashed changes
git stash pop 2>/dev/null || true

echo -e "\n${GREEN}Done! Your private repo is unchanged.${NC}"
