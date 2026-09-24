#!/bin/bash
# End-to-end test for the local install step at the end of ./rsorcar.
# Run: bash scripts/test_rsorcar_local_install.sh
#
# Runs the real rsorcar against a fake remote: ssh/scp/curl are PATH stubs
# that answer exactly what a healthy deploy would see (a reachable host, no
# running task, a live tunnel URL), and the checkout-local helpers that do the
# heavy lifting (sync-repo.sh, sync-task-db.sh) are no-ops.  install.sh in the
# checkout is a recorder, so the test observes the final step:
#   * rsorcar runs install.sh on the LOCAL machine, non-interactively and
#     without launching an editor, after printing the deploy summary,
#   * when started from a linked git worktree, it runs the MAIN repository's
#     install.sh (a worktree is reclaimed after its task, and install.sh
#     points the persistent launchers at its own directory),
#   * and a failing local install fails the script without hiding the
#     remote URL and password printed just before it.
# The ~/.ssh/ copy of step 2 runs for real through the ssh stub: the fake
# HOME has an ~/.ssh with a key, an authorized_keys and an agent socket, and
# the stub hands the tar stream rsorcar sends to the real
# scripts/install-ssh-identity.sh with HOME set to the fake remote home — so
# the test sees the key arrive, the excluded files stay behind, and the
# remote's authorized_keys survive.  No rsync anywhere: the remote may not
# have it.
# Step 1a runs for real too: the stub runs the scripts/install-remote-prereqs.sh
# it is fed on ``bash -s`` (against this machine, which has every tool), and
# the test checks that it runs before the git sync — a fresh image has no git
# — and before the running-task probe and the ~/.ssh copy, and that a remote where the tools cannot be installed stops the deploy
# there (FAKE_PREREQS_FAIL=1 makes the stub fail that one script).
# Step 1c runs for real as well: the stub runs scripts/check-remote-disk-space.sh
# fed on ``NEED_BYTES=... bash -s`` against this machine's disk, and the test
# checks that it runs after the task probe and before the ~/.ssh copy and the
# sync, that it removes a stale ~/.kiss/sorcar.db.incoming on the remote, and
# that a remote without room stops the deploy there (SORCAR_DISK_HEADROOM_GB
# set to more than any disk holds), while SORCAR_SKIP_DISK_CHECK=1 skips it.
# Step 4 runs for real as far as the stubs allow: the ssh stub runs the key
# probe rsorcar feeds to ``bash -s`` against the fake remote HOME and the scp
# stub records what it is asked to copy, so the test sees this machine's keys
# travel to a remote without any (tests 1 and 10) and stay home when the
# remote already has keys of its own — in ~/.kiss/api_keys.env (test 8), in
# its ~/.bashrc or fish config (test 12) — even from a machine without any
# key of its own (test 9); a key store the probe cannot read stops the
# deploy (test 13).
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The physical path: rsorcar resolves the main repository with cd && pwd, and
# on macOS mktemp answers under /var, a symlink to /private/var.
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

FAKE_URL="https://fake-tunnel.example"
FAKE_PW="pw-for-test"

# --- Populate one checkout with rsorcar, stub helpers, recording install.sh --
# $1: checkout dir, $2: exit code of the fake install.sh, $3: fixture dir
# (where the marker is written)
populate_checkout() {
    local dir="$1" install_rc="$2" fix="$3"
    mkdir -p "$dir/scripts" "$dir/src/kiss/scripts"
    cp "$REPO_ROOT/rsorcar" "$dir/rsorcar"
    chmod +x "$dir/rsorcar"
    cat > "$dir/install.sh" <<EOF
#!/bin/bash
{
    echo "self=\$0"
    echo "args=\$*"
    echo "skip_launch=\${KISS_SKIP_LAUNCH:-}"
} > "$fix/install-marker.txt"
exit $install_rc
EOF
    chmod +x "$dir/install.sh"

    # The deploy helpers rsorcar insists on: the two it runs locally are
    # no-ops, the rest only need to exist (they travel to the "remote").
    printf '#!/bin/bash\nexit 0\n' > "$dir/scripts/sync-repo.sh"
    printf '#!/bin/bash\nexit 0\n' > "$dir/scripts/sync-task-db.sh"
    # The memory sync (step 4c) records what it was asked to sync with.
    printf '#!/bin/bash\nprintf "%%s\\n" "$*" >> "%s/sync-memory-args.txt"\nexit 0\n' "$fix" \
        > "$dir/scripts/sync-memory.sh"
    printf '#!/bin/bash\nexit 0\n' > "$dir/scripts/install-api-keys.sh"
    # Steps 1a, 1c and 4 feed these three to the remote's ``bash -s``; the ssh
    # stub runs what it is fed, so the real scripts run (against this machine,
    # which has every tool the first looks for and room for the second, and
    # against the fake remote HOME for the third's key count).
    cp "$REPO_ROOT/scripts/install-remote-prereqs.sh" \
       "$REPO_ROOT/scripts/check-remote-disk-space.sh" \
       "$REPO_ROOT/scripts/count-api-keys.sh" "$dir/scripts/"
    local helper
    for helper in scripts/collect-github-auth.sh scripts/install-github-auth.sh \
                  scripts/install-ssh-identity.sh scripts/move-home-to-disk.sh \
                  scripts/wait-for-public-url.sh \
                  src/kiss/scripts/sync_db.py src/kiss/scripts/relocate_work_dir.py \
                  src/kiss/scripts/carry_over_tables.py \
                  src/kiss/scripts/db_fingerprint.py \
                  src/kiss/scripts/running_tasks.py \
                  src/kiss/scripts/merge_memory_pages.py \
                  src/kiss/scripts/remote_config.py; do
        : > "$dir/$helper"
    done
}

# --- Fixture: fake HOME, fake remote HOME and the ssh/scp/curl stubs ---------
# $1: fixture dir
make_env() {
    local fix="$1"
    mkdir -p "$fix/home/.kiss" "$fix/home/.ssh/agent" "$fix/rhome/.ssh" "$fix/rhome/.kiss" "$fix/bin"
    # The key store the deploy ships (no ~/*rc file in the fake HOME).
    echo 'export FAKE_API_KEY=x' > "$fix/home/.kiss/api_keys.env"
    # The ssh identity the deploy copies (step 2), with the files it must
    # leave behind: authorized_keys, a socket, a .DS_Store.
    echo 'local private key' > "$fix/home/.ssh/id_test"
    echo 'local public key' > "$fix/home/.ssh/id_test.pub"
    echo 'local authorized_keys' > "$fix/home/.ssh/authorized_keys"
    : > "$fix/home/.ssh/.DS_Store"
    # Bound by a relative name: a Unix socket path is limited to about 100
    # bytes, and a long TMPDIR would push the absolute path past that.
    (cd "$fix/home/.ssh/agent" \
        && python3 -c 'import socket; socket.socket(socket.AF_UNIX).bind("x.sock")')
    # The remote's own authorized_keys: the file that lets the deploy in.
    echo 'remote authorized_keys' > "$fix/rhome/.ssh/authorized_keys"
    # What an upload that ran out of disk left behind on the remote.
    head -c 3072 /dev/zero > "$fix/rhome/.kiss/sorcar.db.incoming"
    # The scp stub copies nothing, so the remote half of the copy is put where
    # rsorcar's scp would have put it.
    cp "$REPO_ROOT/scripts/install-ssh-identity.sh" "$fix/rhome/.kiss/"

    # ssh stub: answers each probe rsorcar sends a healthy deploy's answer.
    cat > "$fix/bin/ssh" <<EOF
#!/bin/bash
while [[ "\${1:-}" == -* ]]; do
    if [[ "\$1" == -o ]]; then shift 2; else shift; fi
done
shift  # the user@host target
cmd="\$*"
case "\$cmd" in
    'echo ok') echo ok ;;
    *'printf %s "\$HOME"'*) printf '%s' "$fix/rhome" ;;
    'python3 - '*) cat >/dev/null; echo 0 ;;                  # running-task probe
    REMOTE_DIR=*) cat >/dev/null; echo "SORCAR_PUBLIC_URL=$FAKE_URL" ;;
    *remote-url.json*) echo "$FAKE_URL" ;;
    *remote_password*) echo "$FAKE_PW" ;;
    *'git log -1'*) printf 'abc123 fake commit\n0\n' ;;
    *install-ssh-identity.sh*) HOME="$fix/rhome" bash -c "\$cmd" ;;   # the real remote half, fed the tar stream
    'bash -s'|*' bash -s')                      # a helper fed on standard input (steps 1a, 1c and 4): run it
        script="\$(cat)"
        printf '%s\n' "\$script" >> "$fix/bash-s-stdin.txt"
        if [[ -n "\${FAKE_PREREQS_FAIL:-}" ]] && grep -q install-remote-prereqs.sh <<< "\$script"; then
            echo "[ERR]  fakehost is missing git (fake failure)" >&2; exit 1
        fi
        # The variables rsorcar puts before ``bash -s`` (NEED_BYTES=...) are
        # assignments for the remote shell; HOME is the fake remote's.
        HOME="$fix/rhome" bash -c "\$cmd" <<< "\$script" ;;
    *) [ -t 0 ] || cat >/dev/null; exit 0 ;;
esac
EOF
    # scp stub: copies nothing, records what rsorcar asked it to copy.
    cat > "$fix/bin/scp" <<SCP
#!/bin/bash
while [[ "\${1:-}" == -* ]]; do shift; done
printf '%s\n' "\$*" >> "$fix/scp-args.txt"
exit 0
SCP
    printf '#!/bin/bash\nprintf 200\n' > "$fix/bin/curl"
    chmod +x "$fix/bin/"*
}

# $1: fixture dir, $2: the rsorcar to run
run_rsorcar() {
    local fix="$1" rsorcar="$2"
    HOME="$fix/home" KISS_HOME='' PATH="$fix/bin:$PATH" \
    SORCAR_NO_BROWSER=1 SORCAR_SKIP_GITHUB_AUTH=1 \
        bash "$rsorcar" user@fakehost 2>&1
}

# --- Test 1: the deploy ends by running install.sh locally -------------------
make_env "$WORK/ok"
populate_checkout "$WORK/ok/checkout" 0 "$WORK/ok"
OUT=$(run_rsorcar "$WORK/ok" "$WORK/ok/checkout/rsorcar") || fail "rsorcar failed:
$OUT"
[[ -f "$WORK/ok/install-marker.txt" ]] || fail "rsorcar never ran install.sh locally"
grep -qx 'args=--non-interactive' "$WORK/ok/install-marker.txt" \
    || fail "local install.sh not run with --non-interactive: $(cat "$WORK/ok/install-marker.txt")"
grep -qx 'skip_launch=1' "$WORK/ok/install-marker.txt" \
    || fail "local install.sh not run with KISS_SKIP_LAUNCH=1"
pass "rsorcar runs install.sh locally, non-interactively, no editor launch"

RSSH="$WORK/ok/rhome/.ssh"
[[ "$(cat "$RSSH/id_test")" == "local private key" ]] || fail "the ssh key did not reach the remote ~/.ssh"
[[ "$(cat "$RSSH/id_test.pub")" == "local public key" ]] || fail "the public key did not reach the remote ~/.ssh"
[[ "$(cat "$RSSH/authorized_keys")" == "remote authorized_keys" ]] \
    || fail "the remote's authorized_keys was overwritten: $(cat "$RSSH/authorized_keys")"
[[ ! -e "$RSSH/.DS_Store" && ! -e "$RSSH/agent" ]] \
    || fail "excluded files were copied: $(ls -A "$RSSH")"
[[ -z "$(find "$WORK/ok/rhome/.kiss" -maxdepth 1 -name 'ssh-replaced-*')" ]] \
    || fail "a backup directory was created although nothing was replaced"
echo "$OUT" | grep -q "SSH identity copied to $WORK/ok/rhome/.ssh (2 files)" \
    || fail "the ssh copy was not reported with its file count:
$OUT"
pass "the ssh identity travels as a tar stream: key arrives, authorized_keys and excluded files stay put"

echo "$OUT" | grep -q "Copying API keys (1 exported variables) to user@fakehost" \
    || fail "this machine's keys were not copied to a remote without any:
$OUT"
grep -qE '/api_keys\.env user@fakehost:\.kiss/$' "$WORK/ok/scp-args.txt" \
    || fail "api_keys.env was never handed to scp: $(cat "$WORK/ok/scp-args.txt")"
echo "$OUT" | grep -q "already has API keys (" && fail "a remote without keys was reported as having some:
$OUT"
pass "a remote without API keys gets this machine's"

grep -q "install-remote-prereqs.sh" "$WORK/ok/bash-s-stdin.txt" \
    || fail "scripts/install-remote-prereqs.sh was never fed to the remote's bash -s"
echo "$OUT" | grep -q "The tools the deploy needs are all installed" \
    || fail "the prerequisite check did not run (or did not report) on the remote:
$OUT"
PREREQ_LINE=$(echo "$OUT" | grep -n "Checking that git, curl, tar, python3 and ssh are installed" | cut -d: -f1 | head -1)
TASK_LINE=$(echo "$OUT" | grep -n "Checking whether a task is running" | cut -d: -f1 | head -1)
SSH_LINE=$(echo "$OUT" | grep -n "Copying ~/.ssh/ to" | cut -d: -f1 | head -1)
SYNC_LINE=$(echo "$OUT" | grep -n "through origin (branch" | cut -d: -f1 | head -1)
[[ -n "$PREREQ_LINE" && -n "$TASK_LINE" && -n "$SSH_LINE" && -n "$SYNC_LINE" ]] \
    || fail "a step is missing from the output (lines $PREREQ_LINE / $TASK_LINE / $SSH_LINE / $SYNC_LINE)"
# The running-task probe runs python3 on the remote and the ~/.ssh copy runs
# tar there, so the install must come before both, not only before the sync.
[[ "$PREREQ_LINE" -lt "$TASK_LINE" && "$PREREQ_LINE" -lt "$SSH_LINE" && "$PREREQ_LINE" -lt "$SYNC_LINE" ]] \
    || fail "the prerequisite install does not run first on the remote (lines $PREREQ_LINE / $TASK_LINE / $SSH_LINE / $SYNC_LINE)"
pass "git and the other tools are checked/installed first on the remote: before the task probe, the ssh copy and the git sync"

grep -q "check-remote-disk-space.sh" "$WORK/ok/bash-s-stdin.txt" \
    || fail "scripts/check-remote-disk-space.sh was never fed to the remote's bash -s"
echo "$OUT" | grep -q "INFO.*free in $WORK/ok/rhome on .*the deploy needs about" \
    || fail "the room check did not run (or did not report) on the remote:
$OUT"
DISK_LINE=$(echo "$OUT" | grep -n "Checking that user@fakehost has room for the deploy" | cut -d: -f1 | head -1)
[[ -n "$DISK_LINE" && "$TASK_LINE" -lt "$DISK_LINE" && "$DISK_LINE" -lt "$SSH_LINE" && "$DISK_LINE" -lt "$SYNC_LINE" ]] \
    || fail "the room check does not run after the task probe and before the ssh copy and the sync (lines $TASK_LINE / $DISK_LINE / $SSH_LINE / $SYNC_LINE)"
echo "$OUT" | grep -q "Removed ~/.kiss/sorcar.db.incoming (3.0 KiB)" \
    || fail "the stale sorcar.db.incoming on the remote was not removed (or not reported):
$OUT"
[[ ! -e "$WORK/ok/rhome/.kiss/sorcar.db.incoming" ]] || fail "the stale sorcar.db.incoming is still on the remote"
pass "the room check runs on the remote before anything of size travels, and removes a stale upload"

grep -qx 'user@fakehost' "$WORK/ok/sync-memory-args.txt" 2>/dev/null \
    || fail "scripts/sync-memory.sh was not run with the target: $(cat "$WORK/ok/sync-memory-args.txt" 2>/dev/null)"
DB_SYNC_LINE=$(echo "$OUT" | grep -n "Syncing the task database with user@fakehost" | cut -d: -f1 | head -1)
MEM_SYNC_LINE=$(echo "$OUT" | grep -n "Syncing the agent's memory with user@fakehost" | cut -d: -f1 | head -1)
INSTALL_LINE=$(echo "$OUT" | grep -n "Installing KISS Sorcar on user@fakehost" | cut -d: -f1 | head -1)
[[ -n "$DB_SYNC_LINE" && -n "$MEM_SYNC_LINE" && -n "$INSTALL_LINE" ]] \
    || fail "a sync step is missing from the output (lines $DB_SYNC_LINE / $MEM_SYNC_LINE / $INSTALL_LINE)"
[[ "$DB_SYNC_LINE" -lt "$MEM_SYNC_LINE" && "$MEM_SYNC_LINE" -lt "$INSTALL_LINE" ]] \
    || fail "the memory sync does not run after the task database sync and before the remote install (lines $DB_SYNC_LINE / $MEM_SYNC_LINE / $INSTALL_LINE)"
pass "the agent's memory is synced with the remote, after the task database and before the remote install"

URL_LINE=$(echo "$OUT" | grep -n "URL:.*$FAKE_URL" | cut -d: -f1 | head -1)
STEP_LINE=$(echo "$OUT" | grep -n "Running install.sh on the local machine" | cut -d: -f1 | head -1)
DONE_LINE=$(echo "$OUT" | grep -n "Done." | cut -d: -f1 | head -1)
[[ -n "$URL_LINE" ]] || fail "deploy summary (URL) missing from output"
echo "$OUT" | grep -q "Password:.*$FAKE_PW" || fail "deploy summary (password) missing"
[[ -n "$STEP_LINE" && "$URL_LINE" -lt "$STEP_LINE" ]] \
    || fail "local install did not run after the deploy summary"
[[ -n "$DONE_LINE" && "$STEP_LINE" -lt "$DONE_LINE" ]] \
    || fail "local install did not run before rsorcar finished"
pass "local install runs after the summary box and before Done."

# --- Test 2: from a linked worktree, the MAIN repo's install.sh runs ---------
make_env "$WORK/wt"
populate_checkout "$WORK/wt/main" 0 "$WORK/wt"
git -C "$WORK/wt/main" init -q -b main
git -C "$WORK/wt/main" config user.email "test@test.com"
git -C "$WORK/wt/main" config user.name "Test"
git -C "$WORK/wt/main" add -A
git -C "$WORK/wt/main" commit -q -m "initial"
git -C "$WORK/wt/main" worktree add -q "$WORK/wt/linked" -b kiss/wt-test
OUT=$(run_rsorcar "$WORK/wt" "$WORK/wt/linked/rsorcar") || fail "rsorcar failed from a worktree:
$OUT"
[[ -f "$WORK/wt/install-marker.txt" ]] || fail "worktree deploy never ran install.sh locally"
grep -qx "self=$WORK/wt/main/install.sh" "$WORK/wt/install-marker.txt" \
    || fail "install.sh did not run from the main repo: $(cat "$WORK/wt/install-marker.txt")"
pass "from a linked worktree, the main repository's install.sh is run"

# --- Test 3: a failing local install fails rsorcar, summary already shown ----
make_env "$WORK/bad"
populate_checkout "$WORK/bad/checkout" 1 "$WORK/bad"
if OUT=$(run_rsorcar "$WORK/bad" "$WORK/bad/checkout/rsorcar"); then
    fail "rsorcar succeeded although the local install.sh failed"
fi
[[ -f "$WORK/bad/install-marker.txt" ]] || fail "failing install.sh never ran"
echo "$OUT" | grep -q "install.sh failed on the local machine" \
    || fail "no clear error message for the failed local install:
$OUT"
echo "$OUT" | grep -q "URL:.*$FAKE_URL" \
    || fail "the failed local install hid the remote URL"
echo "$OUT" | grep -q "Password:.*$FAKE_PW" \
    || fail "the failed local install hid the remote password"
pass "a failing local install fails the deploy without hiding URL and password"

# --- Test 4: a remote where git cannot be installed stops the deploy ---------
# Before the sync, which is git from its first command, and before anything
# else is shipped.
make_env "$WORK/nogit"
populate_checkout "$WORK/nogit/checkout" 0 "$WORK/nogit"
if OUT=$(FAKE_PREREQS_FAIL=1 run_rsorcar "$WORK/nogit" "$WORK/nogit/checkout/rsorcar"); then
    fail "rsorcar went on although git could not be installed on the remote"
fi
echo "$OUT" | grep -q "Could not install the tools the deploy needs on user@fakehost" \
    || fail "no clear error message when the remote's tools cannot be installed:
$OUT"
echo "$OUT" | grep -q "through origin (branch" \
    && fail "the git sync was attempted although git could not be installed"
[[ ! -f "$WORK/nogit/install-marker.txt" ]] || fail "install.sh ran although the deploy had stopped"
pass "a remote where git cannot be installed stops the deploy before the git sync"

# --- Test 5: a remote without room stops the deploy ---------------------------
# No disk holds a million gigabytes, so the real check fails against this
# machine's own filesystem; the deploy must stop before the ~/.ssh copy, the
# sync and the install.
make_env "$WORK/full"
populate_checkout "$WORK/full/checkout" 0 "$WORK/full"
if OUT=$(SORCAR_DISK_HEADROOM_GB=1000000 run_rsorcar "$WORK/full" "$WORK/full/checkout/rsorcar"); then
    fail "rsorcar went on although the remote has no room for the deploy"
fi
echo "$OUT" | grep -q "Not enough room on .*the deploy needs about 976.6 TiB" \
    || fail "the room check did not explain the shortfall:
$OUT"
echo "$OUT" | grep -q "user@fakehost does not have room for the deploy" \
    || fail "no clear error message when the remote has no room:
$OUT"
echo "$OUT" | grep -q "Copying ~/.ssh/ to" && fail "the ssh copy ran although the remote has no room"
echo "$OUT" | grep -q "through origin (branch" && fail "the git sync was attempted although the remote has no room"
[[ ! -f "$WORK/full/install-marker.txt" ]] || fail "install.sh ran although the deploy had stopped"
pass "a remote without room for the deploy stops it before anything of size travels"

# --- Test 6: SORCAR_SKIP_DISK_CHECK=1 deploys anyway ----------------------------
make_env "$WORK/skip"
populate_checkout "$WORK/skip/checkout" 0 "$WORK/skip"
OUT=$(SORCAR_DISK_HEADROOM_GB=1000000 SORCAR_SKIP_DISK_CHECK=1 \
      run_rsorcar "$WORK/skip" "$WORK/skip/checkout/rsorcar") || fail "rsorcar failed with the room check skipped:
$OUT"
echo "$OUT" | grep -q "has room for the deploy" && fail "the room check ran although SORCAR_SKIP_DISK_CHECK=1"
[[ -f "$WORK/skip/install-marker.txt" ]] || fail "the deploy did not finish with the room check skipped"
pass "SORCAR_SKIP_DISK_CHECK=1 skips the room check"

# --- Test 7: a headroom that is not a number is refused -------------------------
if OUT=$(SORCAR_DISK_HEADROOM_GB=lots run_rsorcar "$WORK/skip" "$WORK/skip/checkout/rsorcar"); then
    fail "SORCAR_DISK_HEADROOM_GB=lots was accepted"
fi
echo "$OUT" | grep -q "SORCAR_DISK_HEADROOM_GB must be a whole number of gigabytes, got 'lots'" \
    || fail "no clear error for a bad SORCAR_DISK_HEADROOM_GB:
$OUT"
pass "a SORCAR_DISK_HEADROOM_GB that is not a number is refused"

# --- Test 8: a remote that already has API keys keeps them ----------------------
# Its ~/.kiss/api_keys.env holds a key, so this machine's keys are not copied
# (neither the distilled file nor the rc file), while the helpers still travel
# and ~/.bashrc is still wired to the remote's own key store.
make_env "$WORK/haskeys"
populate_checkout "$WORK/haskeys/checkout" 0 "$WORK/haskeys"
echo 'export REMOTE_API_KEY=r' > "$WORK/haskeys/rhome/.kiss/api_keys.env"
echo 'export LOCAL_RC_TOKEN=t' > "$WORK/haskeys/home/.zshrc"
OUT=$(run_rsorcar "$WORK/haskeys" "$WORK/haskeys/checkout/rsorcar") || fail "rsorcar failed against a remote with keys:
$OUT"
echo "$OUT" | grep -q "user@fakehost already has API keys (1 found)" \
    || fail "the remote's existing keys were not reported:
$OUT"
echo "$OUT" | grep -q "Copying API keys" && fail "this machine's keys were copied although the remote has its own"
grep -qE 'api_keys\.env|\.zshrc' "$WORK/haskeys/scp-args.txt" \
    && fail "a key file was handed to scp although the remote has its own keys: $(cat "$WORK/haskeys/scp-args.txt")"
grep -q 'remote_config.py' "$WORK/haskeys/scp-args.txt" \
    || fail "the remote helpers were not shipped: $(cat "$WORK/haskeys/scp-args.txt")"
echo "$OUT" | grep -q "API keys installed (~/.kiss/api_keys.env, sourced from ~/.bashrc)" \
    || fail "install-api-keys.sh did not run for the remote's own keys:
$OUT"
[[ "$(cat "$WORK/haskeys/rhome/.kiss/api_keys.env")" == "export REMOTE_API_KEY=r" ]] \
    || fail "the remote's key store was changed: $(cat "$WORK/haskeys/rhome/.kiss/api_keys.env")"
[[ -f "$WORK/haskeys/install-marker.txt" ]] || fail "the deploy did not finish"
pass "a remote that already has API keys keeps them; this machine's are not copied"

# --- Test 9: ... even when this machine has no keys at all ---------------------
# Local key discovery is skipped, so its "No API keys found" error cannot fire.
make_env "$WORK/nolocal"
populate_checkout "$WORK/nolocal/checkout" 0 "$WORK/nolocal"
echo 'export REMOTE_API_KEY=r' > "$WORK/nolocal/rhome/.kiss/api_keys.env"
rm "$WORK/nolocal/home/.kiss/api_keys.env"
OUT=$(run_rsorcar "$WORK/nolocal" "$WORK/nolocal/checkout/rsorcar") || fail "rsorcar failed from a machine without keys against a remote with keys:
$OUT"
echo "$OUT" | grep -q "already has API keys (1 found)" || fail "the remote's keys were not found:
$OUT"
[[ -f "$WORK/nolocal/install-marker.txt" ]] || fail "the deploy did not finish"
pass "a machine without keys can re-deploy to a remote that has them"

# --- Test 10: a remote key store without any key does not count ------------------
# The file exists but holds no FOO_API_KEY= / FOO_TOKEN= line, so this
# machine's keys are copied as to a fresh remote.
make_env "$WORK/emptykeys"
populate_checkout "$WORK/emptykeys/checkout" 0 "$WORK/emptykeys"
cat > "$WORK/emptykeys/rhome/.kiss/api_keys.env" <<'KEYS'
# export REMOTE_API_KEY=deleted
note=API_KEY=
export REMOTE_API_KEY=
export QUOTED_API_KEY=''
export DQUOTED_API_KEY=""
export COMMENTED_API_KEY= # deleted
export NOT_A_KEY=1
KEYS
mkdir -p "$WORK/emptykeys/rhome/.config/fish"
printf "set -gx FISH_API_KEY ''\nset -gx FISH_TOKEN # deleted\n" > "$WORK/emptykeys/rhome/.config/fish/config.fish"
printf '[ -f "$HOME/.kiss/api_keys.env" ] && . "$HOME/.kiss/api_keys.env"\n' > "$WORK/emptykeys/rhome/.bashrc"
OUT=$(run_rsorcar "$WORK/emptykeys" "$WORK/emptykeys/checkout/rsorcar") || fail "rsorcar failed against a remote with an empty key store:
$OUT"
echo "$OUT" | grep -q "already has API keys (" && fail "a key store without keys counted as having some"
echo "$OUT" | grep -q "Copying API keys (1 exported variables) to user@fakehost" \
    || fail "this machine's keys were not copied to a remote whose key store holds no key:
$OUT"
grep -qE '/api_keys\.env user@fakehost:\.kiss/$' "$WORK/emptykeys/scp-args.txt" \
    || fail "api_keys.env was never handed to scp: $(cat "$WORK/emptykeys/scp-args.txt")"
pass "a remote key store that holds no key gets this machine's keys"

# --- Test 11: no keys anywhere is still an error ------------------------------------
make_env "$WORK/nokeys"
populate_checkout "$WORK/nokeys/checkout" 0 "$WORK/nokeys"
rm "$WORK/nokeys/home/.kiss/api_keys.env"
if OUT=$(run_rsorcar "$WORK/nokeys" "$WORK/nokeys/checkout/rsorcar"); then
    fail "rsorcar went on with no API keys on either machine"
fi
echo "$OUT" | grep -q "No API keys found — neither a ~/\*rc file with keys nor" \
    || fail "no clear error when neither machine has keys:
$OUT"
[[ ! -f "$WORK/nokeys/install-marker.txt" ]] || fail "install.sh ran although the deploy had stopped"
pass "no API keys on either machine stops the deploy"

# --- Test 12: keys living only in the remote's shell rc count too -------------------
# An install older than ~/.kiss/api_keys.env kept its keys as export lines
# in ~/.bashrc (or ``set -gx`` lines in fish's config), which the kiss-web
# daemon migrates into the store at startup; they must not be overwritten.
make_env "$WORK/rckeys"
populate_checkout "$WORK/rckeys/checkout" 0 "$WORK/rckeys"
printf 'export PATH=$PATH:/x\nexport REMOTE_API_KEY="r"\n' > "$WORK/rckeys/rhome/.bashrc"
mkdir -p "$WORK/rckeys/rhome/.config/fish"
printf 'set -gx REMOTE_TOKEN r\n' > "$WORK/rckeys/rhome/.config/fish/config.fish"
OUT=$(run_rsorcar "$WORK/rckeys" "$WORK/rckeys/checkout/rsorcar") || fail "rsorcar failed against a remote with rc keys:
$OUT"
echo "$OUT" | grep -q "already has API keys (2 found)" \
    || fail "keys in the remote's ~/.bashrc and fish config were not found:
$OUT"
echo "$OUT" | grep -q "Copying API keys" && fail "this machine's keys were copied over the remote's rc keys"
grep -q 'api_keys\.env' "$WORK/rckeys/scp-args.txt" \
    && fail "api_keys.env was handed to scp although the remote has rc keys: $(cat "$WORK/rckeys/scp-args.txt")"
[[ -f "$WORK/rckeys/install-marker.txt" ]] || fail "the deploy did not finish"
pass "keys that live only in the remote's shell rc are kept; this machine's are not copied"

# --- Test 13: a key store the probe cannot read stops the deploy ------------------
# Unreadable is not the same as empty: the copy would overwrite keys nobody
# could see.  (root reads everything, so the case cannot be staged as root.)
if [[ "$(id -u)" != "0" ]]; then
    make_env "$WORK/unreadable"
    populate_checkout "$WORK/unreadable/checkout" 0 "$WORK/unreadable"
    echo 'export REMOTE_API_KEY=r' > "$WORK/unreadable/rhome/.kiss/api_keys.env"
    chmod 000 "$WORK/unreadable/rhome/.kiss/api_keys.env"
    if OUT=$(run_rsorcar "$WORK/unreadable" "$WORK/unreadable/checkout/rsorcar"); then
        fail "rsorcar went on although the remote's key store cannot be read"
    fi
    chmod 600 "$WORK/unreadable/rhome/.kiss/api_keys.env"
    echo "$OUT" | grep -q "Could not tell whether user@fakehost already has API keys" \
        || fail "no clear error for an unreadable remote key store:
$OUT"
    echo "$OUT" | grep -q "Copying API keys" && fail "this machine's keys were copied over an unreadable key store"
    if [[ -e "$WORK/unreadable/scp-args.txt" ]] && grep -q 'api_keys\.env' "$WORK/unreadable/scp-args.txt"; then
        fail "api_keys.env was handed to scp although the remote's key store is unreadable"
    fi
    [[ ! -f "$WORK/unreadable/install-marker.txt" ]] || fail "install.sh ran although the deploy had stopped"
    pass "a remote key store the probe cannot read stops the deploy"
fi

echo
echo "ALL TESTS PASSED"
