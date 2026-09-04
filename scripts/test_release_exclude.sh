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
# Same rule as the real repo's .gitignore: the built extension is never
# tracked in origin, only injected into the public snapshot.
echo "*.vsix" > .gitignore
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

# The real repo must ignore the vsix too, or "git add -A" in the release would
# commit it to origin.
git -C "$REPO_ROOT" check-ignore -q "$VSIX_FILE" ||
    fail "$VSIX_FILE is not gitignored in the real repo (would be committed to origin)"
pass "real repo gitignores $VSIX_FILE"

# A stand-in for the packaged extension that build_vscode_extension produces.
mkdir -p "$(dirname "$VSIX_FILE")"
printf 'PK\003\004 fake vsix v1' > "$VSIX_FILE"
[[ -z "$(git status --porcelain -- "$VSIX_FILE")" ]] || fail "vsix shows up as untracked (not ignored)"

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

# --- Test 3: create_public_commit (root) + verify + push; clone has no secrets ---
create_public_commit "$(git rev-parse HEAD)" "1.2.3" ""
verify_no_excluded_paths "$PUBLIC_COMMIT" || fail "verify_no_excluded_paths rejected clean commit"
git remote add public "$WORK/public.git"
git push -q public "$PUBLIC_COMMIT:refs/heads/main" --force
git clone -q "$WORK/public.git" "$WORK/clone"
[[ -f "$WORK/clone/keep/app.py" ]] || fail "keep/app.py missing in public clone"
[[ -e "$WORK/clone/secrets" ]] && fail "secrets/ present in public clone"
[[ -e "$WORK/clone/private_notes.md" ]] && fail "private_notes.md present in public clone"
[[ -e "$WORK/clone/scripts/exclude.json" ]] && fail "exclude.json present in public clone"
pass "public clone contains no excluded paths"

# --- Test 3b: the vsix is in the public clone but in no commit of origin ---
cmp -s "$VSIX_FILE" "$WORK/clone/$VSIX_FILE" || fail "vsix missing or different in public clone"
[[ "$(git ls-tree -r "$PUBLIC_COMMIT" -- "$VSIX_FILE" | cut -d' ' -f1)" == "100644" ]] ||
    fail "vsix not stored as a regular file in the public commit"
git ls-tree -r --name-only HEAD | grep -qxF "$VSIX_FILE" && fail "vsix committed to origin HEAD"
git diff --quiet && git diff --cached --quiet || fail "tree_with_vsix touched the real index"
[[ -z "$(git status --porcelain)" ]] || fail "tree_with_vsix left the working tree dirty"
pass "vsix is part of the public snapshot and never added or committed to origin"

# --- Test 4: excluded blobs are not even reachable in the public object db ---
SECRET_BLOB=$(git rev-parse HEAD:secrets/key.txt)
if git -C "$WORK/public.git" cat-file -e "$SECRET_BLOB" 2>/dev/null; then
    fail "secret blob object was pushed to public repo"
fi
pass "secret blob objects absent from public object database"

# --- Test 5: second release parents on public main; secrets stay out of history ---
echo "v2" >> keep/app.py
echo "new secret" > secrets/key2.txt
printf 'PK\003\004 fake vsix v2' > "$VSIX_FILE"   # rebuilt extension
git add -A
git commit -q -m "second"
PUBLIC_HEAD=$(git rev-parse public/main 2>/dev/null || git ls-remote "$WORK/public.git" refs/heads/main | cut -f1)
create_public_commit "$(git rev-parse HEAD)" "1.2.4" "$PUBLIC_HEAD"
verify_no_excluded_paths "$PUBLIC_COMMIT" || fail "verify rejected second commit"
git push -q public "$PUBLIC_COMMIT:refs/heads/main" --force
[[ "$(git rev-parse "$PUBLIC_COMMIT^")" == "$PUBLIC_HEAD" ]] || fail "second public commit not parented on public main"
for c in $(git -C "$WORK/public.git" rev-list main); do
    git -C "$WORK/public.git" ls-tree -r --name-only "$c" | grep -q "secrets" && fail "secrets in public history commit $c"
done
pass "second release chains on public main with no secrets anywhere in history"

# --- Test 5b: every public release carries the vsix built for it, origin none ---
[[ "$(git -C "$WORK/public.git" show "main:$VSIX_FILE")" == "$(cat "$VSIX_FILE")" ]] ||
    fail "public main does not carry the rebuilt vsix"
[[ "$(git -C "$WORK/public.git" show "main^:$VSIX_FILE")" == 'PK'$'\003'$'\004'' fake vsix v1' ]] ||
    fail "first public release lost its vsix"
for c in $(git rev-list HEAD); do
    git ls-tree -r --name-only "$c" | grep -qxF "$VSIX_FILE" && fail "vsix in origin commit $c"
done
pass "each public release ships its own vsix while origin history has none"

# --- Test 6: sync check equivalence — filtered tree of HEAD == public tree sans vsix ---
PUB_TREE=$(git -C "$WORK/public.git" rev-parse 'main^{tree}')
[[ "$(filtered_tree HEAD)" != "$PUB_TREE" ]] || fail "public tree unexpectedly lacks the vsix"
[[ "$(filtered_tree HEAD)" == "$(tree_without_vsix "$PUB_TREE")" ]] ||
    fail "filtered tree of HEAD != public tree minus vsix (sync check would misfire)"
pass "in-sync detection: filtered tree matches public tree minus vsix after release"

# --- Test 6b: tree_without_vsix is the identity on a tree with no vsix ---
[[ "$(tree_without_vsix "$(filtered_tree HEAD)")" == "$(filtered_tree HEAD)" ]] ||
    fail "tree_without_vsix changed a tree that had no vsix"
pass "tree_without_vsix leaves vsix-free trees unchanged"

# --- Test 6d: a locally rebuilt vsix that differs from the published one must
# not break stripping (git rm --cached without -f refuses such an index entry) ---
printf 'PK\003\004 fake vsix v3 local only' > "$VSIX_FILE"
STRIPPED=$(tree_without_vsix "$PUB_TREE") || fail "tree_without_vsix failed with a differing local vsix"
[[ "$STRIPPED" == "$(filtered_tree HEAD)" ]] || fail "stripped tree wrong with a differing local vsix"
pass "tree_without_vsix works when the local vsix differs from the published blob"

# --- Test 6e: release_needed drives the step-3 decision ---
PUBLIC_MAIN=$(git -C "$WORK/public.git" rev-parse main)
if release_needed "$(filtered_tree HEAD)" "$PUBLIC_MAIN" >/dev/null; then
    fail "release_needed wants a release although kiss_ai already has this source and a vsix"
fi
pass "release_needed: in sync (same source, vsix published) -> nothing to release"

# A snapshot published before the vsix was bundled: same source, no vsix.
LEGACY=$(git commit-tree "$(filtered_tree HEAD)" -m "legacy snapshot without vsix")
OUT=$(release_needed "$(filtered_tree HEAD)" "$LEGACY") ||
    fail "release_needed skipped a public snapshot that lacks the vsix"
echo "$OUT" | grep -q "has no $VSIX_FILE yet" || fail "missing-vsix reason not reported: $OUT"
pass "release_needed: same source but no vsix in kiss_ai -> release"

echo "v3" >> keep/app.py
git add -A && git commit -q -m "third"
OUT=$(release_needed "$(filtered_tree HEAD)" "$PUBLIC_MAIN") ||
    fail "release_needed missed a source change"
echo "$OUT" | grep -q "Origin differs" || fail "source-change reason not reported: $OUT"
pass "release_needed: source changed -> release"

# Not covered: release_needed's "tree_without_vsix failed -> release" fallback.
# It needs a commit whose vsix entry resolves but whose tree cannot be read,
# which only a corrupt object store or a failing mktemp can produce.
git reset -q --hard HEAD^
printf 'PK\003\004 fake vsix v2' > "$VSIX_FILE"

# --- Test 6c: tree_with_vsix / tree_without_vsix fail loudly on bad input ---
mv "$VSIX_FILE" "$WORK/vsix.bak"
if OUT=$(tree_with_vsix "$(filtered_tree HEAD)" 2>&1); then
    fail "tree_with_vsix succeeded without a built vsix"
fi
echo "$OUT" | grep -q "VSIX not found" || fail "missing-vsix error not reported: $OUT"
if OUT=$(bash -c 'set -e; source "'"$RELEASE_SH"'"; create_public_commit HEAD 9.9.9 ""; echo REACHED' 2>&1); then
    fail "create_public_commit succeeded without a built vsix"
fi
echo "$OUT" | grep -q "REACHED" && fail "create_public_commit continued past a missing vsix"
mv "$WORK/vsix.bak" "$VSIX_FILE"
if tree_with_vsix "0000000000000000000000000000000000000000" >/dev/null 2>&1; then
    fail "tree_with_vsix accepted a nonexistent tree"
fi
if tree_without_vsix "0000000000000000000000000000000000000000" >/dev/null 2>&1; then
    fail "tree_without_vsix accepted a nonexistent tree"
fi
pass "tree_with_vsix requires a built vsix; both helpers reject nonexistent trees"

# --- Test 7: tagging the filtered commit works ---
git tag -a "v1.2.4-test" -m "Release 1.2.4" "$PUBLIC_COMMIT"
git push -q public "v1.2.4-test"
git -C "$WORK/public.git" rev-parse "v1.2.4-test^{commit}" >/dev/null || fail "tag not on public"
pass "tag points at filtered commit and pushes to public"

# --- Test 8: missing exclude.json is a hard error (no fail-open) ---
git rm -q scripts/exclude.json
git commit -q -m "remove exclude"
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "missing exclude.json did not fail (fail-open)"
fi
pass "missing exclude.json is a hard error (no fail-open)"

# --- Test 9: empty list => full tree ---
mkdir -p scripts
echo "[]" > scripts/exclude.json
git add scripts/exclude.json && git commit -q -m "empty exclude"
[[ "$(filtered_tree HEAD)" == "$(git rev-parse 'HEAD^{tree}')" ]] || fail "empty exclude list changed tree"
pass "empty exclude list keeps full tree"

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
