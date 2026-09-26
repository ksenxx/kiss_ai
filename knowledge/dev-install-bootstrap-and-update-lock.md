---
title: scripts/install.sh bootstrap (curl | bash), ~/.kiss/kiss_ai checkout, cross-process
  update lock
uuid: 0beb6c1f-6ff2-4835-ae87-5c750f8a6bea
summary: 'scripts/install.sh curl one-liner: clone ksenxx/kiss_ai into ~/.kiss/kiss_ai,
  ff-only pull with stash+reset recovery, install.sh repair, flock lock ~/.kiss/.update.lock,
  KISS_UPDATE_LOCK_HELD.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# scripts/install.sh bootstrap and the update lock

## Entry point
The README's install command is
`curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash`.
It fetches from the public kiss_ai repo, so it installs the last release, not the private
origin. The Update button runs the same script from `~/.kiss/kiss_ai`.

## What it does
1. Ensures a working git (`have_working_git`). On macOS `/usr/bin/git` can be a stub that fails
   until the Command Line Tools exist, so `command -v git` is not enough. It installs git with
   the platform package manager (Homebrew, apt, dnf, yum, pacman, ...) when possible.
2. Takes the update lock (below) unless `KISS_UPDATE_LOCK_HELD` is already set.
3. Checkout at `~/.kiss/kiss_ai`:
   - it exists with `.git`: `git pull --ff-only`. If that fails (for example kiss_ai history
     was rewritten by the release purge), it restores `kiss-sorcar.vsix` from HEAD (clearing a
     skip-worktree bit), stashes local changes with `--include-untracked`, runs
     `git fetch --tags --prune --force origin` and `git reset --hard @{upstream}`. If the stash
     fails, the reset is skipped so local edits are not lost;
   - it exists without `.git`: delete it and re-clone;
   - it is missing: `git clone https://github.com/ksenxx/kiss_ai.git`, or without git, download
     `archive/refs/heads/main.zip` and unzip it.
   A stale rebuilt VSIX must never enter the stash: popping it back over a newer tracked VSIX
   left bricked clones.
4. If the root `install.sh` is missing or not a regular file, restore it with
   `git checkout HEAD -- install.sh`, or re-clone.
5. Hand over: `./install.sh 9>&-` (see `dev-install-sh-flow`). Pop the stash afterwards, so
   the install runs on a pristine tree. A conflicted pop keeps the stash.
6. Exit with install.sh's exit code.

## The update lock
Two installers on one tree (Update clicked in two VS Code windows, or a window plus the
kiss-web daemon's update endpoint) raced each other's git reset, npm build and extension
install. Both scripts share one lock:
- The file is `$HOME/.kiss/.update.lock`. It follows `$HOME`, deliberately not `$KISS_HOME`,
  because the protected resources (`~/.kiss/kiss_ai`, the global extension install) follow
  `$HOME`.
- Bash opens it on fd 9, and perl calls `flock(LOCK_EX|LOCK_NB)` on that same open file
  description, since `flock(1)` is not on macOS. The lock lasts as long as the bash process and
  the kernel releases it however the process dies, so there is no stale lock to break. The pid
  written into the file only feeds the refusal message
  "another KISS update is already running (pid N); exiting.".
- A loser waits briefly (up to 20 x 50 ms) for the winner to write its pid, because the file
  may still name a dead previous holder.
- fd 9 must not leak into long-lived children (VS Code, the daemon), so every launch line uses
  `9>&-`.
- `scripts/install.sh` exports `KISS_UPDATE_LOCK_HELD=1` before handing over. The root
  `install.sh` (`acquire_update_lock`, between `BEGIN/END: kiss-update-lock` markers) then
  skips locking, and unsets the marker afterwards so later runs from long-lived children do not
  skip the lock. Direct callers of `install.sh` get the same lock.
- INT/TERM/HUP traps keep the lock held until the foreground `./install.sh` returns. When
  VS Code disposes the Update terminal it sends SIGHUP to bash while the detached installer is
  still running.
- In the root `install.sh` the lock is taken after the setsid re-exec (so the detached bash
  records its own pid) and after `set -eo pipefail`.

## Sources
- `scripts/install.sh` (`have_working_git`, lock block, update/clone block, handover)
- `install.sh` (`acquire_update_lock`, `kiss-update-lock` block)
- `README.md` (Installation)
- `src/kiss/tests/scripts/test_bootstrap_restores_missing_install_sh.py`, `test_bootstrap_rewritten_history_backcompat.py`, `test_audit0902_fix_root_install_lock.py`
