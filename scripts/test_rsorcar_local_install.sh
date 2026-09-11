#!/bin/bash
# End-to-end test for the local install step at the end of ./rsorcar.
# Run: bash scripts/test_rsorcar_local_install.sh
#
# Runs the real rsorcar against a fake remote: ssh/scp/rsync/curl are PATH
# stubs that answer exactly what a healthy deploy would see (a reachable host,
# no running task, a live tunnel URL), and the checkout-local helpers that do
# the heavy lifting (sync-repo.sh, sync-task-db.sh) are no-ops.  install.sh in
# the checkout is a recorder, so the test observes the new final step:
#   * rsorcar runs install.sh on the LOCAL machine, non-interactively and
#     without launching an editor, after printing the deploy summary,
#   * when started from a linked git worktree, it runs the MAIN repository's
#     install.sh (a worktree is reclaimed after its task, and install.sh
#     points the persistent launchers at its own directory),
#   * and a failing local install fails the script without hiding the
#     remote URL and password printed just before it.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
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
    printf '#!/bin/bash\nexit 0\n' > "$dir/scripts/install-api-keys.sh"
    local helper
    for helper in scripts/collect-github-auth.sh scripts/install-github-auth.sh \
                  src/kiss/scripts/sync_db.py src/kiss/scripts/relocate_work_dir.py \
                  src/kiss/scripts/carry_over_tables.py \
                  src/kiss/scripts/db_fingerprint.py \
                  src/kiss/scripts/running_tasks.py \
                  src/kiss/scripts/remote_config.py; do
        : > "$dir/$helper"
    done
}

# --- Fixture: fake HOME, fake remote HOME and the ssh/scp/rsync/curl stubs ---
# $1: fixture dir
make_env() {
    local fix="$1"
    mkdir -p "$fix/home/.kiss" "$fix/rhome" "$fix/bin"
    # The key store the deploy ships (no ~/*rc file in the fake HOME).
    echo 'export FAKE_API_KEY=x' > "$fix/home/.kiss/api_keys.env"

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
    *) [ -t 0 ] || cat >/dev/null; exit 0 ;;
esac
EOF
    printf '#!/bin/bash\nexit 0\n' > "$fix/bin/scp"
    printf '#!/bin/bash\nexit 0\n' > "$fix/bin/rsync"
    printf '#!/bin/bash\nprintf 200\n' > "$fix/bin/curl"
    chmod +x "$fix/bin/"*
}

# $1: fixture dir, $2: the rsorcar to run
run_rsorcar() {
    local fix="$1" rsorcar="$2"
    HOME="$fix/home" KISS_HOME= PATH="$fix/bin:$PATH" \
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

echo
echo "ALL TESTS PASSED"
