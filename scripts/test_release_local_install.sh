#!/bin/bash
# End-to-end test for the local install step at the end of scripts/release.sh.
# Run: bash scripts/test_release_local_install.sh
#
# Builds a scratch repo with bare "origin" and "public" remotes, then runs the
# real main() of release.sh with only the outward-facing pieces stubbed
# (PyPI/marketplace publishing, gh, the vsix build, and the public push URL,
# which must point at the local bare repo instead of github.com).  install.sh
# is a recorder at the fixture root, so the test observes exactly how and when
# the release runs it:
#   * with --non-interactive and KISS_SKIP_LAUNCH=1,
#   * strictly after gh, PyPI and marketplace publishing, while the
#     pre-release stash is still stashed (the uncommitted wip.txt must be
#     absent when install.sh runs),
#   * also on the "nothing to release" exit,
#   * and a failing install.sh aborts the release with the stash restored.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_SH="$REPO_ROOT/scripts/release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# --- Fixture: repo + origin + public remotes, recording install.sh ----------
# $1: fixture dir, $2: exit code of the fake install.sh
make_fixture() {
    local fix="$1" install_rc="$2"
    rm -rf "$fix"
    mkdir -p "$fix"
    git init -q -b main --bare "$fix/origin.git"
    git init -q -b main --bare "$fix/public.git"
    git init -q -b main "$fix/repo"
    cd "$fix/repo"
    git config user.email "test@test.com"
    git config user.name "Test"
    mkdir -p scripts src/kiss/core
    echo '[]' > scripts/exclude.json
    echo '__version__ = "2020.1.0"' > src/kiss/core/_version.py
    echo '*.vsix' > .gitignore
    cat > install.sh <<EOF
#!/bin/bash
echo "install" >> "$fix/events.log"
{
    echo "args=\$*"
    echo "skip_launch=\${KISS_SKIP_LAUNCH:-}"
    if [ -e wip.txt ]; then echo "wip=present"; else echo "wip=absent"; fi
} > "$fix/install-marker.txt"
exit $install_rc
EOF
    chmod +x install.sh
    git add -A
    git commit -q -m "initial"
    git remote add origin "$fix/origin.git"
    git remote add public "$fix/public.git"
    git push -q -u origin main
    # Uncommitted work forces the pre-release stash, so the marker can prove
    # install.sh runs before that stash is restored.
    echo "uncommitted work" > wip.txt
}

# --- Driver: run the real main() with external services stubbed -------------
# A separate bash process (not a subshell inside an `if`), so release.sh's own
# `set -e` semantics stay intact when the install step fails.  Every stub
# appends to events.log, so the test can assert the publish -> install order.
cat > "$WORK/driver.sh" <<EOF
#!/bin/bash
FIX="\$1"
cd "\$FIX/repo"
source "$RELEASE_SH"
# The only stub that redirects git traffic: push to the local bare "public"
# repo instead of github.com.
public_push_url() { git remote get-url public; }
# No real npm build; the release only needs the vsix file to exist.
build_vscode_extension() {
    mkdir -p "\$(dirname "\$VSIX_FILE")"
    printf 'PK\003\004 fake vsix' > "\$VSIX_FILE"
}
publish_to_pypi() { echo "pypi" >> "\$FIX/events.log"; }
publish_vscode_extension() { echo "vsce" >> "\$FIX/events.log"; }
gh() { echo "gh" >> "\$FIX/events.log"; }
main
EOF

# --- Test 1: the release runs install.sh before restoring the stash ---------
make_fixture "$WORK/ok" 0
OUT=$(bash "$WORK/driver.sh" "$WORK/ok" 2>&1) || fail "release main() failed:
$OUT"
[[ -f "$WORK/ok/install-marker.txt" ]] || fail "release never ran install.sh"
grep -qx 'args=--non-interactive' "$WORK/ok/install-marker.txt" \
    || fail "install.sh not run with --non-interactive: $(cat "$WORK/ok/install-marker.txt")"
grep -qx 'skip_launch=1' "$WORK/ok/install-marker.txt" \
    || fail "install.sh not run with KISS_SKIP_LAUNCH=1"
pass "release runs install.sh non-interactively without launching an editor"

grep -qx 'wip=absent' "$WORK/ok/install-marker.txt" \
    || fail "install.sh saw uncommitted work: it ran after the stash was restored"
[[ -f "$WORK/ok/repo/wip.txt" ]] || fail "stash was not restored after the install"
pass "install.sh runs on the released tree, before the stash is restored"

# gh runs twice (release create + vsix upload), then PyPI, then the
# marketplace, and only then the local install.
[[ "$(cat "$WORK/ok/events.log")" == "gh
gh
pypi
vsce
install" ]] || fail "wrong publish/install order: $(tr '\n' ' ' < "$WORK/ok/events.log")"
echo "$OUT" | grep -q "Release completed successfully" \
    || fail "release did not finish after the local install"
pass "install.sh runs after gh, PyPI and marketplace publishing, before success"

# --- Test 2: nothing to release still ends with a local install -------------
rm -f "$WORK/ok/install-marker.txt" "$WORK/ok/events.log"
OUT=$(bash "$WORK/driver.sh" "$WORK/ok" 2>&1) || fail "no-op release run failed:
$OUT"
echo "$OUT" | grep -q "nothing to release" || fail "second run was not a no-op release:
$OUT"
[[ -f "$WORK/ok/install-marker.txt" ]] || fail "the no-op release never ran install.sh"
grep -qx 'wip=absent' "$WORK/ok/install-marker.txt" \
    || fail "no-op release ran install.sh after the stash was restored"
[[ "$(cat "$WORK/ok/events.log")" == "install" ]] \
    || fail "the no-op release published something: $(tr '\n' ' ' < "$WORK/ok/events.log")"
[[ -f "$WORK/ok/repo/wip.txt" ]] || fail "no-op release did not restore the stash"
pass "a nothing-to-release run still installs locally and restores the stash"

# --- Test 3: a failing install.sh aborts the release, stash restored --------
make_fixture "$WORK/bad" 1
if OUT=$(bash "$WORK/driver.sh" "$WORK/bad" 2>&1); then
    fail "release succeeded although install.sh failed"
fi
[[ -f "$WORK/bad/install-marker.txt" ]] || fail "failing install.sh never ran"
echo "$OUT" | grep -q "Release completed successfully" \
    && fail "release claimed success although install.sh failed"
[[ -f "$WORK/bad/repo/wip.txt" ]] \
    || fail "stash was not restored after the failed install (EXIT trap broken)"
pass "a failing install.sh aborts the release and the stash is restored"

echo
echo "ALL TESTS PASSED"
