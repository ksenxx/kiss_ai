#!/bin/bash
# End-to-end test for the installer step of scripts/release.sh: the Mac
# installer package and the Windows setup program.
# Run: bash scripts/test_release_macos_pkg.sh
#
# Part 1 (macOS only: the builder needs pkgbuild) builds the real package of
# this repository's HEAD the way the release does (build_macos_package: the
# tree minus scripts/exclude.json, with that tree's own builder) and checks
# that its payload holds the app with its version and commit, no path of
# scripts/exclude.json and not the package of the release before. Where
# makensis and 7-Zip are installed it does the same for the Windows setup
# program (build_windows_setup).
#
# Part 2 runs the real main() of release.sh on a scratch repo with bare
# "origin" and "public" remotes; only the outward-facing pieces are stubbed
# (PyPI publishing, gh, and the public push URL). The fixture's
# scripts/build-macos-pkg.sh and scripts/build-windows-exe.sh are a recorder:
# the "package" it writes lists the arguments and the tree it was run on, so
# the pushed commits show what the release built and when:
#   * the package and the setup program of the bumped version are in the
#     "Version bumped" commit that reaches origin and in the release commit
#     that reaches the public repo,
#   * it is built from the committed tree minus the excluded paths,
#   * KISS_PKG_SIGN_IDENTITY, KISS_PKG_KEYCHAIN and KISS_PKG_NOTARIZE_PROFILE
#     reach the builder,
#   * a commit that reaches origin after the build is in the pushed package,
#   * a failing build releases nothing and leaves the checkout as it was,
#   * neither does a failing rebuild after origin moved: no version bump stays,
#   * a commit that lands on top of an already pushed version bump stops the
#     release: its package would lack that commit.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_SH="$REPO_ROOT/scripts/release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# --- Part 1: the real package holds the public tree --------------------------
if [[ "$(uname -s)" == Darwin ]]; then
    cd "$REPO_ROOT"
    bash -c 'source "$1"; build_macos_package HEAD HEAD "$2"' _ "$RELEASE_SH" "$WORK/real.pkg" \
        > "$WORK/build.log" 2>&1 || { cat "$WORK/build.log"; fail "build_macos_package failed"; }
    pkgutil --expand-full "$WORK/real.pkg" "$WORK/expanded"
    PAYLOAD=$(echo "$WORK"/expanded/*.pkg/Payload)
    for required in install.sh scripts/install.sh pyproject.toml uv.lock src/kiss/__init__.py; do
        [[ -f "$PAYLOAD/$required" ]] || fail "package lacks $required"
    done
    cmp -s "$PAYLOAD/src/kiss/core/_version.py" <(git show HEAD:src/kiss/core/_version.py) \
        || fail "package holds another version than HEAD"
    [[ "$(cat "$PAYLOAD/src/kiss/core/_commit.txt")" == "$(git rev-parse --short HEAD)" ]] \
        || fail "package names commit $(cat "$PAYLOAD/src/kiss/core/_commit.txt"), not HEAD"
    pass "the real package holds the app, the version and the commit of HEAD"
    while IFS= read -r excluded; do
        [[ ! -e "$PAYLOAD/$excluded" ]] || fail "package contains excluded path $excluded"
    done < <(bash -c 'source "$1"; read_exclude_paths' _ "$RELEASE_SH")
    # The tree holds the package of the release before (installers/): a
    # package that carried it would grow by its own size with every release.
    [[ ! -e "$PAYLOAD/installers" ]] || fail "package contains the package of the release before"
    pass "the real package carries no path of scripts/exclude.json and no earlier package"
else
    echo "SKIP: the real package is built on macOS only"
fi

SEVEN_ZIP="$(command -v 7zz || command -v 7z || true)"
if command -v makensis >/dev/null && [[ -n "$SEVEN_ZIP" ]]; then
    cd "$REPO_ROOT"
    bash -c 'source "$1"; build_windows_setup HEAD HEAD "$2"' _ "$RELEASE_SH" "$WORK/real.exe" \
        > "$WORK/build.log" 2>&1 || { cat "$WORK/build.log"; fail "build_windows_setup failed"; }
    "$SEVEN_ZIP" x -y -o"$WORK/exe" "$WORK/real.exe" > /dev/null || fail "the setup program does not unpack"
    for required in install.sh scripts/windows-exe/setup.ps1 pyproject.toml uv.lock src/kiss/__init__.py; do
        [[ -f "$WORK/exe/$required" ]] || fail "setup program lacks $required"
    done
    cmp -s "$WORK/exe/src/kiss/core/_version.py" <(git show HEAD:src/kiss/core/_version.py) \
        || fail "setup program holds another version than HEAD"
    [[ "$(cat "$WORK/exe/src/kiss/core/_commit.txt")" == "$(git rev-parse --short HEAD)" ]] \
        || fail "setup program names commit $(cat "$WORK/exe/src/kiss/core/_commit.txt"), not HEAD"
    while IFS= read -r excluded; do
        [[ ! -e "$WORK/exe/$excluded" ]] || fail "setup program contains excluded path $excluded"
    done < <(bash -c 'source "$1"; read_exclude_paths' _ "$RELEASE_SH")
    [[ ! -e "$WORK/exe/installers" ]] || fail "setup program contains the installers of the release before"
    pass "the real setup program holds the app, the version and the commit of HEAD and no excluded path"
else
    echo "SKIP: the real setup program needs makensis and 7-Zip"
fi

# --- Part 2: the release builds, commits and pushes the package -------------
# The recorder standing in for both builders. FAIL_BUILD (a file
# name) makes it fail on a tree that holds that file; MOVE_ORIGIN (a fixture
# dir) makes it push a commit adding the file $LATE to that fixture's origin,
# once, as another machine would between the build and the push.
cat > "$WORK/recorder.sh" <<'EOF'
#!/bin/bash
tree="$(cd "$(dirname "$0")/.." && pwd)"
[[ -z "${FAIL_BUILD:-}" || ! -e "$tree/$FAIL_BUILD" ]] || exit 1
if [[ -n "${MOVE_ORIGIN:-}" && ! -e "$MOVE_ORIGIN/other-$LATE" ]]; then
    git clone -q "$MOVE_ORIGIN/origin.git" "$MOVE_ORIGIN/other-$LATE"
    (cd "$MOVE_ORIGIN/other-$LATE" && echo late > "$LATE" && git add "$LATE" \
        && git -c user.name=Other -c user.email=other@test.com commit -q -m "late" \
        && git push -q origin main)
fi
[[ "$1" == -o ]] || exit 1
mkdir -p "$(dirname "$2")"
{
    echo "version=$(cat "$tree/src/kiss/core/_version.py")"
    echo "commit=$(cat "$tree/src/kiss/core/_commit.txt")"
    (cd "$tree" && find . -type f | sort | sed 's/^/file=/')
    for arg in "${@:3}"; do echo "arg=$arg"; done
} > "$2"
EOF

FIX="$WORK/fix"
mkdir -p "$FIX"
git init -q -b main --bare "$FIX/origin.git"
git init -q -b main --bare "$FIX/public.git"
git init -q -b main "$FIX/repo"
cd "$FIX/repo"
git config user.email "test@test.com"
git config user.name "Test"
mkdir -p scripts src/kiss/core
echo '["private.txt"]' > scripts/exclude.json
echo '__version__ = "2020.1.0"' > src/kiss/core/_version.py
echo "never public" > private.txt
printf '#!/bin/bash\necho install >> "%s/events.log"\n' "$FIX" > install.sh
cp "$WORK/recorder.sh" scripts/build-macos-pkg.sh
cp "$WORK/recorder.sh" scripts/build-windows-exe.sh
git add -A
git commit -q -m "initial"
git remote add origin "$FIX/origin.git"
git remote add public "$FIX/public.git"
git push -q -u origin main
# Uncommitted work: the release stashes it, so it must not reach the package.
echo "uncommitted work" > wip.txt

# A separate bash process, so release.sh's own `set -e` semantics stay intact.
cat > "$WORK/driver.sh" <<EOF
#!/bin/bash
cd "$FIX/repo"
source "$RELEASE_SH"
public_push_url() { git remote get-url public; }
publish_to_pypi() { echo "pypi" >> "$FIX/events.log"; }
gh() { echo "gh" >> "$FIX/events.log"; }
main
EOF

PKG=installers/KISS-Sorcar.pkg
EXE=installers/KISS-Sorcar-Setup.exe
pushed_pkg() { git -C "$FIX/origin.git" show "main:$PKG"; }
pushed_exe() { git -C "$FIX/origin.git" show "main:$EXE"; }

# --- Test 1: the package is committed with the version bump and pushed ------
OUT=$(bash "$WORK/driver.sh" 2>&1) || fail "release main() failed:
$OUT"
VERSION="$(date +%Y).$(date +%-m).0"
[[ "$(git -C "$FIX/origin.git" log -1 --format=%s main)" == "Version bumped to $VERSION" ]] \
    || fail "origin's tip is not the version bump: $(git -C "$FIX/origin.git" log -1 --format=%s main)"
[[ "$(git -C "$FIX/origin.git" show --format= --name-only main | LC_ALL=C sort | tr '\n' ' ')" \
    == "$EXE $PKG src/kiss/core/_version.py " ]] \
    || fail "the version bump commit holds: $(git -C "$FIX/origin.git" show --format= --name-only main | tr '\n' ' ')"
pushed_pkg | grep -qx "version=__version__ = \"$VERSION\"" \
    || fail "the pushed package is not of version $VERSION: $(pushed_pkg | head -1)"
pushed_pkg | grep -qx "commit=$(git rev-parse --short HEAD~1)" \
    || fail "the pushed package does not name the commit it was cut from: $(pushed_pkg | grep '^commit=')"
pushed_exe | grep -qx "version=__version__ = \"$VERSION\"" \
    || fail "the pushed setup program is not of version $VERSION: $(pushed_exe | head -1)"
pushed_exe | grep -qx "commit=$(git rev-parse --short HEAD~1)" \
    || fail "the pushed setup program does not name the commit it was cut from: $(pushed_exe | grep '^commit=')"
pass "the package of the bumped version is committed with the version bump and pushed to origin"

pushed_pkg | grep -qx "file=./src/kiss/core/_version.py" || fail "the builder did not run on the released tree"
if { pushed_pkg; pushed_exe; } | grep -q -e "private.txt" -e "wip.txt" -e "^arg="; then
    fail "an installer was built with an excluded path, uncommitted work or arguments:
$(pushed_pkg; pushed_exe)"
fi
pass "the package is built from the committed tree minus the excluded paths, unsigned by default"

cmp -s <(pushed_pkg) <(git -C "$FIX/public.git" show "main:$PKG") \
    || fail "the public release commit does not hold the package pushed to origin"
cmp -s <(pushed_exe) <(git -C "$FIX/public.git" show "main:$EXE") \
    || fail "the public release commit does not hold the setup program pushed to origin"
[[ "$(tr '\n' ' ' < "$FIX/events.log")" == "gh pypi install " ]] \
    || fail "wrong publish order: $(tr '\n' ' ' < "$FIX/events.log")"
[[ -f wip.txt ]] || fail "the stash was not restored after the release"
pass "the public release commit holds the same package"

# --- Test 2: signing variables; origin moves between the build and the push -
echo "more" > more.txt
git add more.txt
git commit -q -m "more"
OUT=$(KISS_PKG_SIGN_IDENTITY="Developer ID Installer: T (X)" KISS_PKG_KEYCHAIN=/k.keychain \
    KISS_PKG_NOTARIZE_PROFILE=profile MOVE_ORIGIN="$FIX" LATE=late.txt bash "$WORK/driver.sh" 2>&1) \
    || fail "second release failed:
$OUT"
[[ "$(pushed_pkg | grep '^arg=' | tr '\n' ' ')" \
    == "arg=--sign arg=Developer ID Installer: T (X) arg=--keychain arg=/k.keychain arg=--notarize arg=profile " ]] \
    || fail "the signing variables did not reach the builder: $(pushed_pkg | grep '^arg=' | tr '\n' ' ')"
[[ -z "$(pushed_exe | grep '^arg=')" ]] || fail "the Mac signing variables reached the Windows builder"
pass "the signing variables reach the builder as --sign, --keychain and --notarize"

echo "$OUT" | grep -q "Origin moved - rebuilding the installers" || fail "the release did not rebuild the package:
$OUT"
[[ "$(git -C "$FIX/origin.git" log -3 --format=%s main | tr '\n' ' ')" \
    == "Version bumped to $(date +%Y).$(date +%-m).1 more late " ]] \
    || fail "origin's history is: $(git -C "$FIX/origin.git" log -3 --format=%s main | tr '\n' ' ')"
pushed_pkg | grep -qx "file=./late.txt" || fail "the pushed package lacks the commit that reached origin after the build"
pushed_exe | grep -qx "file=./late.txt" || fail "the pushed setup program lacks the commit that reached origin after the build"
pushed_pkg | grep -qx "commit=$(git rev-parse --short HEAD~1)" \
    || fail "the rebuilt package does not name the commit below the bump: $(pushed_pkg | grep '^commit=')"
cmp -s <(pushed_pkg) <(git -C "$FIX/public.git" show "main:$PKG") \
    || fail "the public release commit does not hold the rebuilt package"
pass "a commit that reaches origin after the build is in the pushed package"

# --- Test 3: a failing build releases nothing -------------------------------
echo "even more" > more.txt
git commit -q -a -m "even more"
BEFORE="$(git rev-parse HEAD) $(git -C "$FIX/origin.git" rev-parse main) $(git -C "$FIX/public.git" rev-parse main)"
: > "$FIX/events.log"
if OUT=$(FAIL_BUILD=more.txt bash "$WORK/driver.sh" 2>&1); then
    fail "the release succeeded although the package build failed"
fi
echo "$OUT" | grep -q "Failed to build the installers - nothing was released" || fail "no clear error:
$OUT"
[[ "$(git rev-parse HEAD) $(git -C "$FIX/origin.git" rev-parse main) $(git -C "$FIX/public.git" rev-parse main)" == "$BEFORE" ]] \
    || fail "a failed build moved the checkout, origin or the public repo"
[[ -z "$(git -C "$FIX/public.git" tag --list "v$(date +%Y).$(date +%-m).2")" && ! -s "$FIX/events.log" ]] \
    || fail "a failed build published something: $(tr '\n' ' ' < "$FIX/events.log")"
[[ "$(git status --porcelain)" == "?? wip.txt" ]] \
    || fail "a failed build left the checkout changed: $(git status --porcelain)"
pass "a failing build releases nothing and leaves the checkout as it was"

# --- Test 4: the rebuild after origin moved fails ----------------------------
# The first build passes and the bump is committed; the pull then brings
# late2.txt, and the build of the tree that holds it fails.
PUBLIC_BEFORE="$(git -C "$FIX/public.git" rev-parse main)"
if OUT=$(MOVE_ORIGIN="$FIX" LATE=late2.txt FAIL_BUILD=late2.txt bash "$WORK/driver.sh" 2>&1); then
    fail "the release succeeded although the rebuild of the package failed"
fi
echo "$OUT" | grep -q "Failed to rebuild the installers - nothing was released" || fail "no clear error:
$OUT"
[[ "$(git log -2 --format=%s | tr '\n' ' ')" == "even more late " ]] \
    || fail "a failed rebuild left the checkout at: $(git log -3 --format=%s | tr '\n' ' ')"
[[ "$(git -C "$FIX/origin.git" log -1 --format=%s main)" == "late" \
    && "$(git -C "$FIX/public.git" rev-parse main)" == "$PUBLIC_BEFORE" && ! -s "$FIX/events.log" ]] \
    || fail "a failed rebuild pushed or published something"
[[ "$(git status --porcelain)" == "?? wip.txt" ]] \
    || fail "a failed rebuild left the checkout changed: $(git status --porcelain)"
pass "a failing rebuild releases nothing and leaves no version bump behind"

# --- Test 5: a push that reached origin reports failure, and origin moves on -
# A pre-push hook plays the network: it lets the push through, has another
# machine push late3.txt on top of the version bump, and reports failure. The
# retry's pull then fast-forwards past the bump, whose package lacks late3.txt.
cat > .git/hooks/pre-push <<EOF
#!/bin/bash
[[ ! -e "$FIX/lost-ack" ]] || exit 0
: > "$FIX/lost-ack"
git push -q --no-verify origin main
unset \$(git rev-parse --local-env-vars)
git clone -q "$FIX/origin.git" "$FIX/other-ack"
cd "$FIX/other-ack" && echo late > late3.txt && git add late3.txt \\
    && git -c user.name=Other -c user.email=other@test.com commit -q -m "late3" \\
    && git push -q origin main
exit 1
EOF
chmod +x .git/hooks/pre-push
if OUT=$(bash "$WORK/driver.sh" 2>&1); then
    fail "the release succeeded with a package that lacks a released commit"
fi
rm .git/hooks/pre-push
echo "$OUT" | grep -q "origin moved on after it took the version bump" || fail "no clear error:
$OUT"
[[ "$(git log -2 --format=%s | tr '\n' ' ')" == "late3 Version bumped to $(date +%Y).$(date +%-m).2 " \
    && "$(git rev-parse HEAD)" == "$(git -C "$FIX/origin.git" rev-parse main)" ]] \
    || fail "the checkout is not at origin's commits: $(git log -3 --format=%s | tr '\n' ' ')"
[[ "$(git -C "$FIX/public.git" rev-parse main)" == "$PUBLIC_BEFORE" && ! -s "$FIX/events.log" ]] \
    || fail "the release published a package that lacks a released commit"
[[ "$(git status --porcelain)" == "?? wip.txt" ]] \
    || fail "the checkout was left changed: $(git status --porcelain)"
pass "a commit that lands on a pushed version bump stops the release and stays in the checkout"

echo
echo "ALL TESTS PASSED"
