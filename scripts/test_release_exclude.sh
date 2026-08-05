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
git init -q --bare "$WORK/public.git"
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

# --- Test 4: excluded blobs are not even reachable in the public object db ---
SECRET_BLOB=$(git rev-parse HEAD:secrets/key.txt)
if git -C "$WORK/public.git" cat-file -e "$SECRET_BLOB" 2>/dev/null; then
    fail "secret blob object was pushed to public repo"
fi
pass "secret blob objects absent from public object database"

# --- Test 5: second release parents on public main; secrets stay out of history ---
echo "v2" >> keep/app.py
echo "new secret" > secrets/key2.txt
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

# --- Test 6: sync check equivalence — filtered tree of HEAD == public tree ---
PUB_TREE=$(git -C "$WORK/public.git" rev-parse 'main^{tree}')
[[ "$(filtered_tree HEAD)" == "$PUB_TREE" ]] || fail "filtered tree of HEAD != public tree (sync check would misfire)"
pass "in-sync detection: filtered tree matches public tree after release"

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
