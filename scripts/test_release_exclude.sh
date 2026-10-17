#!/bin/bash
# End-to-end test for the public history built by scripts/release.sh.
# Run: bash scripts/test_release_exclude.sh
# Builds a scratch repo + bare "public" remote, sources release.sh functions,
# and verifies that excluded paths never reach the public repo, that public
# main consists of "Release X" commits only (no development commits), that
# release_needed publishes exactly when the public tip is not that chain with
# origin's current tree, and that a public repo of another shape is converted.
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

# --- Test 3: create_public_commit + verify + push; clone has one release
# commit, no development commits and no secrets ---
git remote add public "$WORK/public.git"
PUBLIC_REPO_URL="$WORK/public.git"
PUBLIC_REPO_SSH="$WORK/public.git"
create_public_commit "$(git rev-parse HEAD)" 1.0.0 ""
verify_no_excluded_paths "$PUBLIC_COMMIT" || fail "verify_no_excluded_paths rejected clean commit"
verify_no_excluded_paths_in_history "$PUBLIC_COMMIT" ||
    fail "verify_no_excluded_paths_in_history rejected clean history"
[[ "$PUBLIC_COMMIT" != "$(git rev-parse HEAD)" ]] || fail "public commit is origin's own commit"
[[ -z "$(commit_parents "$PUBLIC_COMMIT")" ]] || fail "first release commit is not a root commit"
[[ "$(git log -1 --format=%s "$PUBLIC_COMMIT")" == "Release 1.0.0" ]] ||
    fail "release commit has the wrong subject: $(git log -1 --format=%s "$PUBLIC_COMMIT")"
FIRST_RELEASE="$PUBLIC_COMMIT"
TAG_OBJECT=$(release_tag v1.0.0 1.0.0 "$PUBLIC_COMMIT")
push_public_snapshot "$PUBLIC_COMMIT" v1.0.0 "$TAG_OBJECT" "" > /dev/null ||
    fail "first release into the empty public repo failed"
git clone -q "$WORK/public.git" "$WORK/clone"
[[ -f "$WORK/clone/keep/app.py" ]] || fail "keep/app.py missing in public clone"
[[ -e "$WORK/clone/secrets" ]] && fail "secrets/ present in public clone"
[[ -e "$WORK/clone/private_notes.md" ]] && fail "private_notes.md present in public clone"
[[ -e "$WORK/clone/scripts/exclude.json" ]] && fail "exclude.json present in public clone"
[[ "$(git -C "$WORK/clone" log --format='%s')" == "Release 1.0.0" ]] ||
    fail "public history is not exactly the release commit: $(git -C "$WORK/clone" log --format='%s')"
[[ "$(git -C "$WORK/public.git" cat-file -t refs/tags/v1.0.0)" == "tag" ]] ||
    fail "release tag missing or not annotated"
[[ "$(git -C "$WORK/public.git" rev-parse 'v1.0.0^{commit}')" == "$PUBLIC_COMMIT" ]] ||
    fail "release tag does not point at the release commit"
git rev-parse --verify --quiet refs/tags/v1.0.0 > /dev/null &&
    fail "release_tag created a local tag ref"
git diff --quiet && git diff --cached --quiet || fail "create_public_commit touched the real index"
[[ -z "$(git status --porcelain)" ]] || fail "create_public_commit left the working tree dirty"
pass "public clone holds exactly one release commit and no excluded paths"

# --- Test 4: excluded blobs are not even reachable in the public object db ---
SECRET_BLOB=$(git rev-parse HEAD:secrets/key.txt)
if git -C "$WORK/public.git" cat-file -e "$SECRET_BLOB" 2>/dev/null; then
    fail "secret blob object was pushed to public repo"
fi
pass "secret blob objects absent from public object database"

# --- Test 5: second release adds one release commit on top of the first;
# development commits stay private and secrets stay out of the whole history ---
echo "only secret" > secrets/only.txt
git add -A
git commit -q -m "secrets only"
echo "v2" >> keep/app.py
echo "new secret" > secrets/key2.txt
git add -A
git commit -q -m "second"
create_public_commit "$(git rev-parse HEAD)" 1.0.1 "$FIRST_RELEASE"
verify_no_excluded_paths "$PUBLIC_COMMIT" || fail "verify rejected second commit"
verify_no_excluded_paths_in_history "$PUBLIC_COMMIT" || fail "verify rejected second history"
[[ "$(commit_parents "$PUBLIC_COMMIT")" == "$FIRST_RELEASE" ]] ||
    fail "second release commit is not parented on the first"
TAG_OBJECT=$(release_tag v1.0.1 1.0.1 "$PUBLIC_COMMIT")
push_public_snapshot "$PUBLIC_COMMIT" v1.0.1 "$TAG_OBJECT" "$FIRST_RELEASE" > /dev/null ||
    fail "second release push failed"
[[ "$(git -C "$WORK/public.git" log --format='%s' main)" == $'Release 1.0.1\nRelease 1.0.0' ]] ||
    fail "public main is not the two release commits: $(git -C "$WORK/public.git" log --format='%s' main)"
for c in $(git -C "$WORK/public.git" rev-list --all); do
    git -C "$WORK/public.git" ls-tree -r --name-only "$c" | grep -q "secrets" && fail "secrets in public commit $c"
done
pass "second release appends one release commit; no development commits, no secrets"

# --- Test 6: sync check equivalence: filtered tree of HEAD == public tree ---
PUB_TREE=$(git -C "$WORK/public.git" rev-parse 'main^{tree}')
[[ "$(filtered_tree HEAD)" == "$PUB_TREE" ]] ||
    fail "filtered tree of HEAD != public tree (sync check would misfire)"
pass "in-sync detection: filtered tree matches the public tree after release"

# --- Test 6e: build_release_chain + release_needed drive the step-3 decision ---
fetch_public_refs > /dev/null
PUBLIC_MAIN=$(git -C "$WORK/public.git" rev-parse main)
build_release_chain > /dev/null || fail "build_release_chain failed on a release-only public repo"
[[ "$RELEASE_CHAIN_TIP" == "$PUBLIC_MAIN" ]] ||
    fail "release-only public main was not recognised as the chain tip"
[[ ${#RETAG_REFSPECS[@]} -eq 0 && ${#RETAG_LEASES[@]} -eq 0 ]] ||
    fail "release-only public repo would have its tags re-pointed"
if release_needed "$(filtered_tree HEAD)" "$PUBLIC_MAIN" >/dev/null; then
    fail "release_needed wants a release although kiss_ai already has this tree"
fi
pass "release_needed: in sync (release-only history, same tree) -> nothing to release"

# A release tag moved back onto a development commit rebuilds to the very same
# release commit (release_commit is deterministic), so the chain tip still
# equals public main; the pending re-pointing alone must trigger a release.
git update-ref "${PUBLIC_TAG_NS}/v1.0.1" "$(git rev-parse HEAD)"
build_release_chain > /dev/null
[[ "$RELEASE_CHAIN_TIP" == "$PUBLIC_MAIN" && ${#RETAG_REFSPECS[@]} -eq 1 ]] ||
    fail "setup: moved tag was expected to rebuild to the published commit with one retag"
OUT=$(release_needed "$(filtered_tree HEAD)" "$PUBLIC_MAIN") ||
    fail "release_needed skipped a release tag that points at a development commit"
echo "$OUT" | grep -q "1 kiss_ai release tag(s) do not point at release-only commits" ||
    fail "pending re-pointing reason not reported: $OUT"
fetch_public_refs > /dev/null
build_release_chain > /dev/null
[[ ${#RETAG_REFSPECS[@]} -eq 0 ]] || fail "setup: public tags were expected to be restored"
pass "release_needed: a release tag off the chain -> release (tag gets re-pointed)"

# History-only changes (a change plus its revert, an empty commit) leave the
# tree identical; a release-only public repo has nothing to publish for them.
echo "temp" >> keep/app.py
git add -A && git commit -q -m "tempchange"
git revert --no-edit HEAD > /dev/null
git commit -q --allow-empty -m "empty note"
[[ "$(filtered_tree HEAD)" == "$PUB_TREE" ]] ||
    fail "setup: history-only commits were expected to leave the tree unchanged"
if release_needed "$(filtered_tree HEAD)" "$PUBLIC_MAIN" >/dev/null; then
    fail "release_needed wants a release for history-only commits (tree unchanged)"
fi
pass "release_needed: history-only commits (revert pair, empty commit) -> nothing to release"

echo "v3" >> keep/app.py
git add -A && git commit -q -m "third"
OUT=$(release_needed "$(filtered_tree HEAD)" "$PUBLIC_MAIN") ||
    fail "release_needed missed a source change"
echo "$OUT" | grep -q "Origin differs" || fail "source-change reason not reported: $OUT"
pass "release_needed: source changed -> release"

# A public main that is not the release chain (development history, or a
# snapshot outside the chain) must be replaced even when its tree is current.
STRAY=$(git commit-tree "$(filtered_tree HEAD)" -p "$PUBLIC_MAIN" -m "Release 9.9.9")
OUT=$(release_needed "$(filtered_tree HEAD)" "$STRAY") ||
    fail "release_needed kept a main outside the release chain"
echo "$OUT" | grep -q "not the release-only history" || fail "chain reason not reported: $OUT"
OUT=$(release_needed "$(filtered_tree HEAD)" "$(git rev-parse HEAD)") ||
    fail "release_needed kept a main made of development commits"
echo "$OUT" | grep -q "not the release-only history" || fail "chain reason not reported: $OUT"
pass "release_needed: main outside the release chain -> release"

# A public tip that cannot be inspected (a sha this repo does not have) is
# released rather than skipped.
OUT=$(release_needed "$(filtered_tree HEAD)" "0000000000000000000000000000000000000000") ||
    fail "release_needed kept an uninspectable public tip"
echo "$OUT" | grep -q "cannot be inspected" || fail "uninspectable reason not reported: $OUT"
pass "release_needed: uninspectable public tip -> release"

git reset -q --hard HEAD^

# --- Test 7: a public repo in the legacy shape (main = development history,
# release tags on development commits) is converted into the release-only
# chain: one "Release X" commit per tag in version order, every tag
# re-pointed, in one leased atomic push together with the new release ---
git init -q -b main --bare "$WORK/legacy.git"
git remote add legacypub "$WORK/legacy.git"
OLD_PUBLIC_REMOTE="$PUBLIC_REMOTE"
PUBLIC_REMOTE=legacypub
PUBLIC_REPO_URL="$WORK/legacy.git"
PUBLIC_REPO_SSH="$WORK/legacy.git"
INITIAL=$(git rev-list --max-parents=0 HEAD)
SECOND=$(git rev-parse HEAD~3)
[[ "$(git log -1 --format=%s "$SECOND")" == "second" ]] || fail "setup: HEAD~3 is not 'second'"
git push -q legacypub "+$(git rev-parse HEAD):refs/heads/main" \
    "+$INITIAL:refs/tags/v0.9.0" "+$SECOND:refs/tags/v0.10.0" "+$FIRST_RELEASE:refs/tags/experimental"
# Filtered trees are what the release commits must hold; a secret-only commit
# ("secrets only", HEAD~4) is simply never a release.
fetch_public_refs > /dev/null
build_release_chain > "$WORK/chain.log" || fail "build_release_chain failed on the legacy repo: $(cat "$WORK/chain.log")"
grep -q "2 public release tag(s) do not point at release-only commits yet" "$WORK/chain.log" ||
    fail "re-pointing of the legacy tags not announced: $(cat "$WORK/chain.log")"
[[ ${#RETAG_LEASES[@]} -eq 2 && ${#RETAG_REFSPECS[@]} -eq 2 ]] ||
    fail "expected 2 re-pointed tags, got ${#RETAG_LEASES[@]} leases / ${#RETAG_REFSPECS[@]} refspecs"
[[ "${RETAG_LEASES[0]}" == "--force-with-lease=refs/tags/v0.9.0:$INITIAL" ]] ||
    fail "lease for v0.9.0 is wrong: ${RETAG_LEASES[0]}"
[[ "${RETAG_LEASES[1]}" == "--force-with-lease=refs/tags/v0.10.0:$SECOND" ]] ||
    fail "tags are not ordered by version (v0.10.0 must follow v0.9.0): ${RETAG_LEASES[1]}"
[[ "$(git log -1 --format=%s "$RELEASE_CHAIN_TIP")" == "Release 0.10.0" ]] ||
    fail "chain tip is not Release 0.10.0: $(git log -1 --format=%s "$RELEASE_CHAIN_TIP")"
[[ "$(git rev-parse "${RELEASE_CHAIN_TIP}^{tree}")" == "$(filtered_tree "$SECOND")" ]] ||
    fail "rebuilt release commit does not hold the tag's filtered tree"
[[ "$(git log -1 --format='%an <%ae> %ad' --date=raw "$RELEASE_CHAIN_TIP")" == \
    "$(git log -1 --format='%an <%ae> %ad' --date=raw "$SECOND")" ]] ||
    fail "rebuilt release commit did not keep the released commit's author and date"
OUT=$(release_needed "$(filtered_tree HEAD)" "$(git rev-parse HEAD)") ||
    fail "release_needed skipped the conversion of a legacy public repo"
echo "$OUT" | grep -q "not the release-only history" || fail "legacy reason not reported: $OUT"
# Rebuilding is deterministic: the same inputs give the same commit ids.
REBUILT_TIP="$RELEASE_CHAIN_TIP"
build_release_chain > /dev/null
[[ "$RELEASE_CHAIN_TIP" == "$REBUILT_TIP" ]] || fail "rebuilding the chain changed its commit ids"

create_public_commit "$(git rev-parse HEAD)" 0.11.0 "$RELEASE_CHAIN_TIP" > /dev/null
TAG_OBJECT=$(release_tag v0.11.0 0.11.0 "$PUBLIC_COMMIT")
push_public_snapshot "$PUBLIC_COMMIT" v0.11.0 "$TAG_OBJECT" "$(git rev-parse HEAD)" \
    "${RETAG_LEASES[@]}" "${RETAG_REFSPECS[@]}" > /dev/null ||
    fail "conversion push (non-fast-forward, leased, with re-pointed tags) failed"
[[ "$(git -C "$WORK/legacy.git" rev-parse main)" == "$PUBLIC_COMMIT" ]] ||
    fail "public main was not replaced by the release chain"
[[ "$(git -C "$WORK/legacy.git" log --format='%s' main)" == $'Release 0.11.0\nRelease 0.10.0\nRelease 0.9.0' ]] ||
    fail "public main is not the release chain: $(git -C "$WORK/legacy.git" log --format='%s' main)"
[[ "$(git -C "$WORK/legacy.git" rev-parse 'v0.9.0^{commit}')" == "$(git -C "$WORK/legacy.git" rev-parse 'main~2')" ]] ||
    fail "v0.9.0 was not re-pointed to its release commit"
[[ "$(git -C "$WORK/legacy.git" rev-parse 'v0.10.0^{commit}')" == "$(git -C "$WORK/legacy.git" rev-parse 'main~1')" ]] ||
    fail "v0.10.0 was not re-pointed to its release commit"
for t in v0.9.0 v0.10.0 v0.11.0; do
    [[ "$(git -C "$WORK/legacy.git" cat-file -t "refs/tags/$t")" == "tag" ]] ||
        fail "$t is not an annotated tag after the conversion"
    [[ "$(git -C "$WORK/legacy.git" cat-file tag "refs/tags/$t" | tail -1)" == "Release ${t#v}" ]] ||
        fail "$t has the wrong tag message"
done
[[ "$(git -C "$WORK/legacy.git" rev-parse experimental)" == "$FIRST_RELEASE" ]] ||
    fail "a non-release tag was touched by the conversion"
for c in $INITIAL $SECOND; do
    git -C "$WORK/legacy.git" rev-list main --tags=v\* | grep -q "$c" &&
        fail "development commit $c is still reachable from main or a release tag"
done
[[ -z "$(git -C "$WORK/legacy.git" ls-tree -r --name-only 'v0.9.0^{tree}' | grep secrets)" ]] ||
    fail "re-pointed v0.9.0 still carries secrets"
pass "legacy public repo converted to the release-only chain with re-pointed tags"

# The converted repo is in sync: the chain rebuilds to its own commits and
# nothing is left to release.
fetch_public_refs > /dev/null
build_release_chain > /dev/null
[[ "$RELEASE_CHAIN_TIP" == "$PUBLIC_COMMIT" && ${#RETAG_REFSPECS[@]} -eq 0 ]] ||
    fail "converted repo is not recognised as release-only (tip $RELEASE_CHAIN_TIP, ${#RETAG_REFSPECS[@]} retags)"
if release_needed "$(filtered_tree HEAD)" "$PUBLIC_COMMIT" > /dev/null; then
    fail "release_needed wants another release right after the conversion"
fi
pass "converted repo is recognised as in sync (conversion is idempotent)"

# A tag that moved on the public repo since the fetch fails its lease and
# aborts the whole push: main, the new tag and the other re-pointed tags.
git push -q legacypub "+$INITIAL:refs/tags/v0.8.0"
fetch_public_refs > /dev/null
build_release_chain > /dev/null
# v0.8.0 sorts first, so it becomes the new root and the whole chain (3 tags)
# is rebuilt behind it.
[[ ${#RETAG_REFSPECS[@]} -eq 4 ]] || fail "expected the whole chain to be rebuilt behind v0.8.0, got ${#RETAG_REFSPECS[@]} re-pointed tags"
git push -q legacypub "+$SECOND:refs/tags/v0.8.0"
create_public_commit "$(git rev-parse HEAD)" 0.12.0 "$RELEASE_CHAIN_TIP" > /dev/null
TAG_OBJECT=$(release_tag v0.12.0 0.12.0 "$PUBLIC_COMMIT")
if push_public_snapshot "$PUBLIC_COMMIT" v0.12.0 "$TAG_OBJECT" "$(git -C "$WORK/legacy.git" rev-parse main)" \
    "${RETAG_LEASES[@]}" "${RETAG_REFSPECS[@]}" > "$WORK/stale.log" 2>&1; then
    fail "a tag moved since the fetch was overwritten"
fi
[[ "$(git -C "$WORK/legacy.git" rev-parse v0.8.0)" == "$SECOND" ]] || fail "moved tag was clobbered"
[[ "$(git -C "$WORK/legacy.git" log --format='%s' main)" == $'Release 0.11.0\nRelease 0.10.0\nRelease 0.9.0' ]] ||
    fail "main changed although the push was rejected"
git -C "$WORK/legacy.git" rev-parse --verify --quiet refs/tags/v0.12.0 > /dev/null &&
    fail "new tag published although the push was rejected"
pass "a release tag moved since the fetch aborts the whole atomic push"

PUBLIC_REMOTE="$OLD_PUBLIC_REMOTE"
PUBLIC_REPO_URL="$WORK/public.git"
PUBLIC_REPO_SSH="$WORK/public.git"

# --- Test 8: missing exclude.json is a hard error (no fail-open) ---
git rm -q scripts/exclude.json
git commit -q -m "remove exclude"
if OUT=$(filtered_tree HEAD 2>&1); then
    fail "missing exclude.json did not fail (fail-open)"
fi
pass "missing exclude.json is a hard error (no fail-open)"

# --- Test 9: empty list => full tree, still published as a release commit ---
mkdir -p scripts
echo "[]" > scripts/exclude.json
git add scripts/exclude.json && git commit -q -m "empty exclude"
[[ "$(filtered_tree HEAD)" == "$(git rev-parse 'HEAD^{tree}')" ]] || fail "empty exclude list changed tree"
create_public_commit "$(git rev-parse HEAD)" 3.0.0 "" > /dev/null
[[ "$PUBLIC_COMMIT" != "$(git rev-parse HEAD)" ]] ||
    fail "empty exclude list published origin's own commit instead of a release commit"
[[ "$(git rev-parse "${PUBLIC_COMMIT}^{tree}")" == "$(git rev-parse 'HEAD^{tree}')" ]] ||
    fail "release commit with an empty exclude list does not hold the full tree"
pass "empty exclude list keeps the full tree in a release commit"

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
