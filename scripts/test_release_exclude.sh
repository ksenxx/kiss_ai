#!/bin/bash
# End-to-end test for the exclude.json filtering in scripts/release.sh.
# Run: bash scripts/test_release_exclude.sh
# Builds a scratch repo + bare "public" remote, sources release.sh functions,
# and verifies excluded paths never reach the public repo.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_SH="$REPO_ROOT/scripts/release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# --- Set up scratch repo with secrets and a bare public remote ---
git init -q -b main "$WORK/repo"
# -b main on the bare remote too: with init.defaultBranch unset its HEAD would
# point at refs/heads/master and "git clone" below could not check anything out.
git init -q -b main --bare "$WORK/public.git"
cd "$WORK/repo"
git config user.email "test@test.com"
git config user.name "Test"

mkdir -p secrets keep scripts
echo "public code" > keep/app.py
echo "top secret" > secrets/key.txt
echo "another secret" > secrets/deep.txt
echo "private notes" > private_notes.md
echo "readme" > README.md
cat > scripts/exclude.json <<'EOF'
[
    "scripts/exclude.json",
    "secrets",
    "private_notes.md"
]
EOF
git add -A
git commit -q -m "initial"

# Source release.sh functions without running main (BASH_SOURCE guard).
source "$RELEASE_SH"

# --- Test 1: filtered_tree removes excluded paths, keeps the rest ---
TREE=$(filtered_tree HEAD)
LISTING=$(git ls-tree -r --name-only "$TREE")
echo "$LISTING" | grep -q "keep/app.py" || fail "keep/app.py missing from filtered tree"
echo "$LISTING" | grep -q "README.md" || fail "README.md missing from filtered tree"
echo "$LISTING" | grep -q "secrets" && fail "secrets/ leaked into filtered tree"
echo "$LISTING" | grep -q "private_notes.md" && fail "private_notes.md leaked into filtered tree"
echo "$LISTING" | grep -q "exclude.json" && fail "exclude.json leaked into filtered tree"
pass "filtered_tree strips excluded paths and keeps others"

# --- Test 2: real index and working tree untouched ---
git diff --quiet && git diff --cached --quiet || fail "working tree/index modified by filtered_tree"
[[ -f secrets/key.txt ]] || fail "secrets/key.txt removed from working tree"
pass "real index and working tree untouched"

# --- Test 3: create_public_commit + verify + push; clone has history, no secrets ---
create_public_commit "$(git rev-parse HEAD)"
verify_no_excluded_paths "$PUBLIC_COMMIT" || fail "verify_no_excluded_paths rejected clean commit"
verify_no_excluded_paths_in_history "$PUBLIC_COMMIT" ||
    fail "verify_no_excluded_paths_in_history rejected clean history"
[[ "$PUBLIC_COMMIT" == "$FILTERED_HEAD" ]] || fail "public commit is not the filtered history tip"
[[ "$PUBLIC_COMMIT" != "$(git rev-parse HEAD)" ]] || fail "public commit is origin's own commit"
git remote add public "$WORK/public.git"
git push -q public "$PUBLIC_COMMIT:refs/heads/main" --force
git clone -q "$WORK/public.git" "$WORK/clone"
[[ -f "$WORK/clone/keep/app.py" ]] || fail "keep/app.py missing in public clone"
[[ -e "$WORK/clone/secrets" ]] && fail "secrets/ present in public clone"
[[ -e "$WORK/clone/private_notes.md" ]] && fail "private_notes.md present in public clone"
[[ -e "$WORK/clone/scripts/exclude.json" ]] && fail "exclude.json present in public clone"
git -C "$WORK/clone" log --format='%s' | grep -qx "initial" ||
    fail "development commit 'initial' missing from public history"
git diff --quiet && git diff --cached --quiet || fail "create_public_commit touched the real index"
[[ -z "$(git status --porcelain)" ]] || fail "create_public_commit left the working tree dirty"
pass "public clone carries the filtered development history and no excluded paths"

# --- Test 4: excluded blobs are not even reachable in the public object db ---
SECRET_BLOB=$(git rev-parse HEAD:secrets/key.txt)
if git -C "$WORK/public.git" cat-file -e "$SECRET_BLOB" 2>/dev/null; then
    fail "secret blob object was pushed to public repo"
fi
pass "secret blob objects absent from public object database"

# --- Test 5: second release publishes every dev commit; secrets-only commits
# are pruned and secrets stay out of the whole public history ---
echo "only secret" > secrets/only.txt
git add -A
git commit -q -m "secrets only"
echo "v2" >> keep/app.py
echo "new secret" > secrets/key2.txt
git add -A
git commit -q -m "second"
create_public_commit "$(git rev-parse HEAD)"
verify_no_excluded_paths "$PUBLIC_COMMIT" || fail "verify rejected second commit"
verify_no_excluded_paths_in_history "$PUBLIC_COMMIT" || fail "verify rejected second history"
git push -q public "$PUBLIC_COMMIT:refs/heads/main" --force
[[ "$PUBLIC_COMMIT" == "$FILTERED_HEAD" ]] || fail "second public commit is not the filtered tip"
# initial + second survive; "secrets only" (excluded content only) is pruned.
[[ "$(git -C "$WORK/public.git" rev-list --count main)" == "2" ]] ||
    fail "public main should hold 2 dev commits, got $(git -C "$WORK/public.git" rev-list --count main)"
for subject in "initial" "second"; do
    git -C "$WORK/public.git" log --format='%s' main | grep -qx "$subject" ||
        fail "development commit '$subject' missing from public history"
done
git -C "$WORK/public.git" log --format='%s' main | grep -qx "secrets only" &&
    fail "commit whose only content was excluded paths survived in public history"
for c in $(git -C "$WORK/public.git" rev-list main); do
    git -C "$WORK/public.git" ls-tree -r --name-only "$c" | grep -q "secrets" && fail "secrets in public history commit $c"
done
pass "second release republishes the filtered history with no secrets anywhere"

# --- Test 6: sync check equivalence — filtered tree of HEAD == public tree ---
PUB_TREE=$(git -C "$WORK/public.git" rev-parse 'main^{tree}')
[[ "$(filtered_tree HEAD)" == "$PUB_TREE" ]] ||
    fail "filtered tree of HEAD != public tree (sync check would misfire)"
pass "in-sync detection: filtered tree matches the public tree after release"

# --- Test 6e: release_needed drives the step-3 decision ---
PUBLIC_MAIN=$(git -C "$WORK/public.git" rev-parse main)
# Rebuilding the filtered history must yield the same shas (deterministic
# rewrite), or the in-sync check below could never hold across releases.
build_filtered_history "$(git rev-parse HEAD)" > /dev/null
if release_needed "$FILTERED_HEAD" "$PUBLIC_MAIN" >/dev/null; then
    fail "release_needed wants a release although kiss_ai already has this history"
fi
pass "release_needed: in sync (same history published) -> nothing to release"

# The tip shape of earlier releases: a "Release X.Y.Z" commit (it used to add
# the packaged VS Code extension) on top of the very same filtered history.
# The source is current, but the tip is not the rewritten head: re-release so
# the extra commit disappears from public main.
LEGACY=$(git commit-tree "$(filtered_tree HEAD)" -p "$FILTERED_HEAD" -m "Release 1.2.4")
OUT=$(release_needed "$FILTERED_HEAD" "$LEGACY") ||
    fail "release_needed skipped a public main topped by an old-style release commit"
echo "$OUT" | grep -q "has the current source but not the filtered development history" ||
    fail "old-style tip reason not reported: $OUT"
pass "release_needed: same source under an old-style release tip -> release"

# History-only changes leave origin's tree identical but must still be
# published: a change plus its revert, and an intentionally empty commit
# (which --prune-empty auto preserves).
echo "temp" >> keep/app.py
git add -A && git commit -q -m "tempchange"
git revert --no-edit HEAD > /dev/null
git commit -q --allow-empty -m "empty note"
[[ "$(filtered_tree HEAD)" == "$(git rev-parse "${FILTERED_HEAD}^{tree}")" ]] ||
    fail "setup: history-only commits were expected to leave the tree unchanged"
build_filtered_history "$(git rev-parse HEAD)" > /dev/null
OUT=$(release_needed "$FILTERED_HEAD" "$PUBLIC_MAIN") ||
    fail "release_needed skipped history-only commits (tree unchanged)"
echo "$OUT" | grep -q "has the current source but not the filtered development history" ||
    fail "history-only reason not reported: $OUT"
create_public_commit "$(git rev-parse HEAD)" > /dev/null
git push -q public "$PUBLIC_COMMIT:refs/heads/main" --force
for subject in "tempchange" 'Revert "tempchange"' "empty note"; do
    git -C "$WORK/public.git" log --format='%s' main | grep -qFx "$subject" ||
        fail "history-only commit '$subject' missing from public history"
done
PUBLIC_MAIN=$(git -C "$WORK/public.git" rev-parse main)
if release_needed "$FILTERED_HEAD" "$PUBLIC_MAIN" >/dev/null; then
    fail "release_needed not in sync right after publishing the history-only commits"
fi
pass "release_needed: history-only commits (revert pair, empty commit) are released"

echo "v3" >> keep/app.py
git add -A && git commit -q -m "third"
build_filtered_history "$(git rev-parse HEAD)" > /dev/null
OUT=$(release_needed "$FILTERED_HEAD" "$PUBLIC_MAIN") ||
    fail "release_needed missed a source change"
echo "$OUT" | grep -q "Origin differs" || fail "source-change reason not reported: $OUT"
pass "release_needed: source changed -> release"

# A squashed snapshot (the pre-history-publication release shape): current
# tree, but parented on another snapshot instead of being the rewritten
# development history. It must be re-released so the history gets published.
OLD_SNAP=$(git commit-tree "$(filtered_tree HEAD^)" -m "Release 0.0.1")
SQUASHED=$(git commit-tree "$(filtered_tree HEAD)" -p "$OLD_SNAP" -m "Release 0.0.2")
OUT=$(release_needed "$FILTERED_HEAD" "$SQUASHED") ||
    fail "release_needed kept a squashed-snapshot main"
echo "$OUT" | grep -q "has the current source but not the filtered development history" ||
    fail "squashed-snapshot reason not reported: $OUT"
pass "release_needed: squashed snapshot main -> release (history gets published)"

# A public tip that cannot be inspected (a sha this repo does not have) is
# released rather than skipped.
OUT=$(release_needed "$FILTERED_HEAD" "0000000000000000000000000000000000000000") ||
    fail "release_needed kept an uninspectable public tip"
echo "$OUT" | grep -q "cannot be inspected" || fail "uninspectable reason not reported: $OUT"
pass "release_needed: uninspectable public tip -> release"

git reset -q --hard HEAD^

# --- Test 7: tagging the filtered commit works ---
git tag -a "v1.2.4-test" -m "Release 1.2.4" "$PUBLIC_COMMIT"
git push -q public "v1.2.4-test"
git -C "$WORK/public.git" rev-parse "v1.2.4-test^{commit}" >/dev/null || fail "tag not on public"
pass "tag points at filtered commit and pushes to public"

# --- Test 7b: a squashed pre-history-publication main is replaced by the full
# filtered development history in one leased, atomic, non-fast-forward push ---
PUBLIC_REPO_URL="$WORK/public.git"
PUBLIC_REPO_SSH="$WORK/public.git"
OLD_SNAP2=$(git commit-tree "$(filtered_tree HEAD^)" -m "Release 0.1.0")
SQUASHED2=$(git commit-tree "$(filtered_tree HEAD)" -p "$OLD_SNAP2" -m "Release 0.2.0")
git push -q public "+$SQUASHED2:refs/heads/main"
build_filtered_history "$(git rev-parse HEAD)" > /dev/null
release_needed "$FILTERED_HEAD" "$SQUASHED2" >/dev/null ||
    fail "release_needed skipped the migration off squashed snapshots"
create_public_commit "$(git rev-parse HEAD)" > /dev/null
git tag -a "v2.0.0-mig" -m "Release 2.0.0" "$PUBLIC_COMMIT"
push_public_snapshot "$PUBLIC_COMMIT" "v2.0.0-mig" "$SQUASHED2" > /dev/null ||
    fail "migration push (non-fast-forward, leased) failed"
[[ "$(git -C "$WORK/public.git" rev-parse main)" == "$PUBLIC_COMMIT" ]] ||
    fail "public main was not replaced by the filtered history"
git -C "$WORK/public.git" log --format='%s' main | grep -qx "initial" ||
    fail "development history missing after the migration push"
git -C "$WORK/public.git" log --format='%s' main | grep -qx "Release 0.2.0" &&
    fail "squashed snapshot still reachable from public main"
pass "squashed main is replaced by the filtered development history under a lease"

# --- Test 7c: the very first release goes into an empty public repo through
# push_public_snapshot with an empty expected-main lease ---
git init -q -b main --bare "$WORK/fresh.git"
git remote add freshpub "$WORK/fresh.git"
OLD_PUBLIC_REMOTE="$PUBLIC_REMOTE"
PUBLIC_REMOTE=freshpub
PUBLIC_REPO_URL="$WORK/fresh.git"
PUBLIC_REPO_SSH="$WORK/fresh.git"
git tag -a "v2.0.0-first" -m "Release 2.0.0" "$PUBLIC_COMMIT"
push_public_snapshot "$PUBLIC_COMMIT" "v2.0.0-first" "" > /dev/null ||
    fail "first release into an empty public repo failed"
[[ "$(git -C "$WORK/fresh.git" rev-parse main)" == "$PUBLIC_COMMIT" ]] ||
    fail "first release did not create main in the empty public repo"
[[ "$(git -C "$WORK/fresh.git" cat-file -t refs/tags/v2.0.0-first)" == "tag" ]] ||
    fail "first release did not publish its annotated tag"
git -C "$WORK/fresh.git" log --format='%s' main | grep -qx "initial" ||
    fail "first release did not publish the development history"
PUBLIC_REMOTE="$OLD_PUBLIC_REMOTE"
PUBLIC_REPO_URL="$WORK/public.git"
PUBLIC_REPO_SSH="$WORK/public.git"
pass "first release into an empty public repo publishes history, main and tag"

# --- Test 8: missing exclude.json is a hard error (no fail-open) ---
git rm -q scripts/exclude.json
git commit -q -m "remove exclude"
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "missing exclude.json did not fail (fail-open)"
fi
pass "missing exclude.json is a hard error (no fail-open)"

# --- Test 9: empty list => full tree, and history is published as is ---
mkdir -p scripts
echo "[]" > scripts/exclude.json
git add scripts/exclude.json && git commit -q -m "empty exclude"
[[ "$(filtered_tree HEAD)" == "$(git rev-parse 'HEAD^{tree}')" ]] || fail "empty exclude list changed tree"
build_filtered_history "$(git rev-parse HEAD)" > /dev/null
[[ "$FILTERED_HEAD" == "$(git rev-parse HEAD)" ]] ||
    fail "empty exclude list rewrote history instead of publishing origin's commits as is"
pass "empty exclude list keeps full tree and origin history unrewritten"

# --- Test 10: malformed JSON fails loudly ---
echo "{ not json" > scripts/exclude.json
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "malformed JSON did not fail"
fi
pass "malformed JSON causes failure"

# --- Test 11: non-list JSON fails loudly ---
echo '{"exclude": ["a"]}' > scripts/exclude.json
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "non-list JSON did not fail"
fi
pass "non-list JSON causes failure"

# --- Test 12: verify_no_excluded_paths catches a leak ---
echo '["keep"]' > scripts/exclude.json
if verify_no_excluded_paths "$PUBLIC_COMMIT" >/dev/null 2>&1; then
    fail "verify_no_excluded_paths missed a leaked path"
fi
pass "verify_no_excluded_paths detects leaked paths"

# --- Test 13: dirty working tree does not break filtering (git rm -f path) ---
echo '["secrets", "private_notes.md", "scripts/exclude.json"]' > scripts/exclude.json
git add -A && git commit -q -m "restore excludes"
echo "dirty edit" >> secrets/key.txt   # working tree differs from HEAD blob
TREE=$(filtered_tree HEAD)
git ls-tree -r --name-only "$TREE" | grep -q "secrets" && fail "dirty secrets leaked"
git checkout -q -- secrets/key.txt
pass "filtering works with dirty working tree"

# --- Test 14: literal semantics — "foo[1]" must not remove "foo1" ---
printf 'secret' > 'foo[1]'
printf 'public' > foo1
echo '["foo[1]", "secrets", "private_notes.md", "scripts/exclude.json"]' > scripts/exclude.json
git add -A && git commit -q -m "bracket file"
TREE=$(filtered_tree HEAD)
LISTING=$(git ls-tree -r --name-only "$TREE")
echo "$LISTING" | grep -qxF 'foo1' || fail "foo1 wrongly removed (glob semantics)"
echo "$LISTING" | grep -qxF 'foo[1]' && fail "literal foo[1] leaked"
verify_no_excluded_paths "$TREE" >/dev/null || fail "verifier rejected clean bracket tree"
echo '["foo1", "secrets", "private_notes.md", "scripts/exclude.json"]' > scripts/exclude.json
if verify_no_excluded_paths "$TREE" >/dev/null 2>&1; then
    fail "verifier missed leaked foo1 with literal matching"
fi
pass "literal path semantics: no glob expansion, verifier matches literally"

# --- Test 15: trailing slash and spaces in folder names ---
mkdir -p "sec ret"
printf 'x' > "sec ret/a.txt"
echo '["sec ret/", "foo[1]", "secrets", "private_notes.md", "scripts/exclude.json"]' > scripts/exclude.json
git add -A && git commit -q -m "space dir"
TREE=$(filtered_tree HEAD)
git ls-tree -r --name-only "$TREE" | grep -q "sec ret" && fail "'sec ret/' dir leaked"
verify_no_excluded_paths "$TREE" >/dev/null || fail "verifier rejected clean space-dir tree"
pass "trailing-slash entries and folder names with spaces are excluded"

# --- Test 16: path containing newline is rejected ---
printf '["a\\nb"]' > scripts/exclude.json
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "newline-containing path not rejected"
fi
pass "paths containing newlines are rejected"

# --- Test 16b: path containing a tab is rejected (leak reports are tab-framed) ---
printf '["a\\tb"]' > scripts/exclude.json
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "tab-containing path not rejected"
fi
pass "paths containing tabs are rejected"

# --- Test 17b: absolute and '..' paths are rejected ---
echo '["/etc/passwd"]' > scripts/exclude.json
if OUT=$(filtered_tree HEAD 2>&1); then fail "absolute path not rejected"; fi
echo '["a/../b"]' > scripts/exclude.json
if OUT=$(filtered_tree HEAD 2>&1); then fail "'..' path not rejected"; fi
pass "absolute and '..' paths are rejected"

# --- Test 17c: verifier matches unusual (quoted-by-git) filenames verbatim ---
printf 'x' > 'wei"rd.txt'
echo '["wei\"rd.txt", "foo[1]", "sec ret", "secrets", "private_notes.md", "scripts/exclude.json"]' > scripts/exclude.json
git add -A && git commit -q -m "weird name"
if verify_no_excluded_paths HEAD >/dev/null 2>&1; then
    fail "verifier missed quoted filename leak"
fi
TREE=$(filtered_tree HEAD)
verify_no_excluded_paths "$TREE" >/dev/null || fail "verifier rejected clean filtered tree with weird names"
git rm -q 'wei"rd.txt' && git commit -q -m "drop weird"
pass "verifier handles git-quoted filenames verbatim (NUL-delimited)"

# --- Test 17: broken exclude.json aborts main-style flow before side effects ---
printf '{bad' > scripts/exclude.json
if OUT=$(bash -c 'set -e; source "'"$RELEASE_SH"'"; F=$(filtered_tree HEAD); echo "REACHED"' 2>&1); then
    fail "set -e did not abort on broken exclude.json in assignment"
fi
echo "$OUT" | grep -q "REACHED" && fail "flow continued past broken exclude.json"
pass "broken exclude.json aborts before any side effects (set -e assignment)"

echo
echo "ALL TESTS PASSED"
