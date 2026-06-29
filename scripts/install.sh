#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
set -e

# Capture the user's shell PWD *before* any ``cd`` so VS Code can later open
# this directory as its workspace root.  Agents launched inside VS Code use
# this as their PWD (see ``kiss.agents.vscode.server``).
USER_PWD="$PWD"

# Returns 0 only if git is actually runnable. On macOS, /usr/bin/git is a stub
# that exits non-zero with an xcode-select message when the Command Line Tools
# are not installed, so `command -v git` is not sufficient.
have_working_git() {
  command -v git &> /dev/null || return 1
  local out
  out=$(git --version 2>&1) || return 1
  case "$out" in
    *"no developer tools were found"*|*"xcode-select"*|*"command line tools"*|*"CommandLineTools"*)
      return 1
      ;;
  esac
  return 0
}

install_git() {
  if have_working_git; then
    return 0
  fi
  echo "git not found (or not runnable) in PATH; attempting to install git..."
  case "$(uname -s)" in
    Darwin)
      if command -v brew &> /dev/null; then
        brew install git
      elif command -v xcode-select &> /dev/null; then
        # xcode-select --install opens a GUI dialog to install Command Line Tools (which include git)
        xcode-select --install 2> /dev/null || true
        echo "A GUI dialog should have appeared to install the Xcode Command Line Tools."
        echo "After it finishes, re-run this script."
      else
        echo "Neither 'brew' nor 'xcode-select' is available on this macOS system."
        echo "Attempting to install Homebrew non-interactively..."
        if command -v curl &> /dev/null; then
          NONINTERACTIVE=1 /bin/bash -c \
            "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
            && eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null)" \
            && brew install git \
            || echo "Homebrew bootstrap failed. Please install git manually from https://git-scm.com/download/mac"
        else
          echo "curl is also unavailable; please install git manually from https://git-scm.com/download/mac"
        fi
      fi
      ;;
    Linux)
      if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y git
      elif command -v dnf &> /dev/null; then
        sudo dnf install -y git
      elif command -v yum &> /dev/null; then
        sudo yum install -y git
      elif command -v pacman &> /dev/null; then
        sudo pacman -Sy --noconfirm git
      elif command -v zypper &> /dev/null; then
        sudo zypper install -y git
      elif command -v apk &> /dev/null; then
        sudo apk add git
      else
        echo "No supported package manager found; cannot install git automatically."
      fi
      ;;
    *)
      echo "Unsupported OS for automatic git install."
      ;;
  esac
  have_working_git
}

install_git || true

cd
if [ -d ~/kiss_ai ]; then
  if [ -d ~/kiss_ai/.git ]; then
    cd ~/kiss_ai
    # Try a fast-forward pull; if the branch diverged (e.g. upstream was
    # force-pushed), stash any local edits and reset hard to upstream so the
    # bootstrap doesn't abort.  The main ./install.sh below will restore the
    # stash via its own helpers if it can.
    if ! git pull --ff-only; then
      echo "git pull --ff-only failed; attempting to reset to upstream..."
      git stash push --include-untracked -m "scripts/install.sh auto-stash" || true
      git fetch --tags --prune origin || true
      if git rev-parse --abbrev-ref '@{upstream}' &>/dev/null; then
        git reset --hard '@{upstream}' \
          || echo "WARNING: reset to upstream failed; continuing with current checkout."
      else
        echo "WARNING: no upstream tracking branch; continuing with current checkout."
      fi
    fi
  else
    rm -rf ~/kiss_ai
    if have_working_git; then
      git clone https://github.com/ksenxx/kiss_ai.git ~/kiss_ai
    else
      curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
      unzip main.zip
      rm main.zip
      mv kiss_ai-main ~/kiss_ai
    fi
  fi
else
  if have_working_git; then
    git clone https://github.com/ksenxx/kiss_ai.git ~/kiss_ai
  else
    curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
    unzip main.zip
    rm main.zip
    mv kiss_ai-main ~/kiss_ai
  fi
fi
cd ~/kiss_ai
./install.sh
