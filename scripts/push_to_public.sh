#!/bin/bash

# Script to push commits with full history to the public GitHub repository
# Repository: https://github.com/ksenxx/kiss_ai

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PUBLIC_REMOTE="public"
PUBLIC_REPO_URL="https://github.com/ksenxx/kiss_ai.git"
PUBLIC_REPO_SSH="git@github.com:ksenxx/kiss_ai.git"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if remote exists
check_remote() {
    if ! git remote get-url "$PUBLIC_REMOTE" &>/dev/null; then
        print_warn "Remote '$PUBLIC_REMOTE' not found. Adding it..."
        git remote add "$PUBLIC_REMOTE" "$PUBLIC_REPO_SSH"
        print_info "Remote '$PUBLIC_REMOTE' added successfully"
    else
        REMOTE_URL=$(git remote get-url "$PUBLIC_REMOTE")
        print_info "Remote '$PUBLIC_REMOTE' exists: $REMOTE_URL"
        
        # Verify it points to the correct repository
        if [[ "$REMOTE_URL" != *"kiss_ai"* ]]; then
            print_warn "Remote '$PUBLIC_REMOTE' doesn't point to kiss_ai repository"
            read -p "Do you want to update it? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git remote set-url "$PUBLIC_REMOTE" "$PUBLIC_REPO_SSH"
                print_info "Remote URL updated"
            fi
        fi
    fi
}

# Function to fetch from public remote
fetch_public() {
    print_info "Fetching from public remote..."
    git fetch "$PUBLIC_REMOTE" || {
        print_error "Failed to fetch from public remote"
        exit 1
    }
}

# Function to push current branch
push_current_branch() {
    local BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_info "Current branch: $BRANCH"
    
    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        print_warn "You have uncommitted changes"
        read -p "Do you want to commit them before pushing? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add -A
            read -p "Enter commit message: " COMMIT_MSG
            git commit -m "$COMMIT_MSG"
        else
            print_warn "Proceeding with uncommitted changes (they won't be pushed)"
        fi
    fi
    
    print_info "Pushing branch '$BRANCH' to '$PUBLIC_REMOTE' with full history..."
    git push "$PUBLIC_REMOTE" "$BRANCH" --force-with-lease || {
        print_error "Failed to push branch '$BRANCH'"
        print_warn "If you need to force push (overwrites remote), use: git push $PUBLIC_REMOTE $BRANCH --force"
        exit 1
    }
    
    print_info "Successfully pushed branch '$BRANCH' to '$PUBLIC_REMOTE'"
}

# Function to push all branches
push_all_branches() {
    print_info "Pushing all branches to '$PUBLIC_REMOTE'..."
    git push "$PUBLIC_REMOTE" --all --force-with-lease || {
        print_error "Failed to push all branches"
        exit 1
    }
    print_info "Successfully pushed all branches to '$PUBLIC_REMOTE'"
}

# Function to push tags
push_tags() {
    print_info "Pushing tags to '$PUBLIC_REMOTE'..."
    git push "$PUBLIC_REMOTE" --tags || {
        print_warn "Failed to push tags (this is optional)"
    }
    print_info "Tags pushed successfully"
}

# Main execution
main() {
    print_info "Starting push to public repository: $PUBLIC_REPO_URL"
    echo
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
    
    # Check and setup remote
    check_remote
    
    # Fetch from public remote to see what's there
    fetch_public
    
    # Ask user what to push
    echo
    print_info "What would you like to push?"
    echo "1) Current branch only (default)"
    echo "2) All branches"
    echo "3) Current branch + tags"
    echo "4) All branches + tags"
    read -p "Enter choice [1-4] (default: 1): " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            push_current_branch
            ;;
        2)
            push_all_branches
            ;;
        3)
            push_current_branch
            push_tags
            ;;
        4)
            push_all_branches
            push_tags
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
    
    echo
    print_info "Push completed successfully!"
    print_info "Repository: $PUBLIC_REPO_URL"
}

# Run main function
main "$@"
