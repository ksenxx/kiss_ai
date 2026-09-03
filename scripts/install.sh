#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
set -e

# Capture the user's shell PWD *before* any ``cd`` so VS Code can later open
# this directory as its workspace root.  Agents launched inside VS Code use
# this as their PWD (see ``kiss.server.server``).
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
        xcode-select --install 2> /dev/null 9>&- || true
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

# ---------------------------------------------------------------------------
# Cross-process update lock.
#
# Two installers on one ~/.kiss/kiss_ai tree -- the Update button in two
# VS Code windows, or a window and the kiss-web daemon's update endpoint --
# race each other's git reset, uv sync and daemon restart.  The lock is a
# kernel advisory lock (flock(2)) on $HOME/.kiss/.update.lock: the tree it
# protects (~/.kiss/kiss_ai, the global extension install) follows $HOME,
# so the lock does too, deliberately NOT $KISS_HOME.
#
# Bash keeps the lock file open on fd 9 for its whole lifetime, including
# the ./install.sh it hands over to at the end; perl (a hard dependency of
# ./install.sh, shipped by macOS and Linux -- flock(1) is not on macOS)
# takes the lock on that very open file description, so it persists after
# perl exits and the kernel releases it when this process dies, however it
# dies.  There is no stale state to recover, hence nothing to break.  The
# pid in the file is informational (for the refusal message).
#
# fd 9 must not leak into long-lived children (the daemon ./install.sh
# restarts, VS Code): every launch line closes it with ``9>&-``, and the
# handover to ./install.sh below does too -- KISS_UPDATE_LOCK_HELD=1 tells
# that nested run an ancestor already holds the lock, so it neither locks
# again nor is refused.
# ---------------------------------------------------------------------------
KISS_UPDATE_LOCK_FILE="$HOME/.kiss/.update.lock"

acquire_update_lock() {
  local holder attempt
  mkdir -p "$HOME/.kiss"
  exec 9>>"$KISS_UPDATE_LOCK_FILE"
  if ! perl -e 'use Fcntl qw(:flock); open(my $f, ">&=", 9) or exit 2; exit(flock($f, LOCK_EX | LOCK_NB) ? 0 : 1)'; then
    # The winner writes its pid right after locking; give it a moment.
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      holder=$(cat "$KISS_UPDATE_LOCK_FILE" 2>/dev/null || true)
      [ -n "$holder" ] && break
      sleep 0.05
    done
    echo "another KISS update is already running (pid ${holder:-unknown}); exiting." >&2
    exit 1
  fi
  echo "$$" > "$KISS_UPDATE_LOCK_FILE"
  # A stray INT/TERM must not drop the lock while ./install.sh is still
  # running underneath: bash runs the trap (and exits) only once its
  # foreground child has returned.
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export KISS_UPDATE_LOCK_HELD=1
}

if [ -z "${KISS_UPDATE_LOCK_HELD:-}" ]; then
  acquire_update_lock
fi

install_git || true

mkdir -p ~/.kiss
cd ~/.kiss
if [ -d ~/.kiss/kiss_ai ]; then
  if [ -d ~/.kiss/kiss_ai/.git ]; then
    cd ~/.kiss/kiss_ai
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
    rm -rf ~/.kiss/kiss_ai
    if have_working_git; then
      git clone https://github.com/ksenxx/kiss_ai.git ~/.kiss/kiss_ai
    else
      curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
      unzip main.zip
      rm main.zip
      mv kiss_ai-main ~/.kiss/kiss_ai
    fi
  fi
else
  if have_working_git; then
    git clone https://github.com/ksenxx/kiss_ai.git ~/.kiss/kiss_ai
  else
    curl -L -o main.zip https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip
    unzip main.zip
    rm main.zip
    mv kiss_ai-main ~/.kiss/kiss_ai
  fi
fi
cd ~/.kiss/kiss_ai
./install.sh 9>&-
