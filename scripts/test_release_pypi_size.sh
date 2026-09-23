#!/bin/bash
# End-to-end test for the PyPI file-size guard in scripts/release.sh.
# Run: bash scripts/test_release_pypi_size.sh
#
# Part 1 builds the real sdist and wheel of this repository with `uv build`
# and checks that both stay under PyPI's 100 MiB per-file limit, that the
# sdist carries only the wheel's packages (no benchmark results, papers,
# reports or node_modules), and that the wheel can still be built from it.
#
# Part 2 sources release.sh and runs publish_to_pypi against a stub `uv` on
# PATH: an oversize sdist must abort before `uv publish` runs, and a small
# build must reach `uv publish`.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_SH="$REPO_ROOT/scripts/release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

LIMIT=$((100 * 1024 * 1024))

# --- Part 1: the real build fits PyPI ---------------------------------------
cd "$REPO_ROOT"
uv build --out-dir "$WORK/dist" > "$WORK/build.log" 2>&1 || { cat "$WORK/build.log"; fail "uv build failed"; }
SDIST=$(ls "$WORK"/dist/*.tar.gz)
WHEEL=$(ls "$WORK"/dist/*.whl)
for f in "$SDIST" "$WHEEL"; do
    size=$(wc -c < "$f")
    (( size <= LIMIT )) || fail "$(basename "$f") is $size bytes, over PyPI's $LIMIT-byte limit"
done
pass "sdist ($(( $(wc -c < "$SDIST") / 1024 / 1024 )) MiB) and wheel ($(( $(wc -c < "$WHEEL") / 1024 / 1024 )) MiB) are under 100 MiB"

tar tzf "$SDIST" > "$WORK/sdist-files.txt"
# Members are "<name>-<version>/<path>"; the repository-level directories are
# matched at the root so src/kiss/tests/benchmarkings/ does not trip the check.
for banned in benchmarkings/ papers/ assets/ reports/ .kiss-worktrees/; do
    if grep -q "^[^/]*/$banned" "$WORK/sdist-files.txt"; then
        fail "sdist contains $banned:"$'\n'"$(grep "^[^/]*/$banned" "$WORK/sdist-files.txt" | head -3)"
    fi
done
# node_modules is a symlink in agent worktrees, so it is banned anywhere.
if grep -q "/node_modules/" "$WORK/sdist-files.txt"; then
    fail "sdist contains node_modules:"$'\n'"$(grep "/node_modules/" "$WORK/sdist-files.txt" | head -3)"
fi
pass "sdist carries no benchmarkings/, papers/, assets/, reports/, node_modules/ or worktrees"

for required in src/kiss/core/_version.py src/kiss/agents/sorcar/sorcar_agent.py projects/swedefend/ pyproject.toml README.md LICENSE; do
    grep -q "^[^/]*/$required" "$WORK/sdist-files.txt" || fail "sdist lacks $required"
done
pass "sdist carries src/kiss, projects/swedefend, pyproject.toml, README.md and LICENSE"

# The build log proves the wheel came out of the sdist, not the source tree.
grep -q "Building wheel from source distribution" "$WORK/build.log" || fail "uv build did not build the wheel from the sdist"
pass "wheel builds from the sdist"

# --- Part 2: publish_to_pypi refuses oversize files before uploading --------
# Stub uv: `uv build` copies a prepared dist/ into place, `uv publish` records
# the call. $1: directory holding the prepared dist files.
make_stub_uv() {
    local prepared="$1" bindir="$WORK/bin"
    mkdir -p "$bindir"
    cat > "$bindir/uv" <<EOF
#!/bin/bash
case "\$1" in
    build) mkdir -p dist && cp "$prepared"/* dist/ ;;
    publish) echo "publish \$*" >> "$WORK/uv-calls.log" ;;
    *) echo "unexpected uv \$*" >&2; exit 2 ;;
esac
EOF
    chmod +x "$bindir/uv"
}

# $1: label, $2: byte size of the prepared sdist. Runs publish_to_pypi in a
# scratch project with the stub uv first on PATH and prints its output; the
# exit status is that of publish_to_pypi.
run_publish() {
    local label="$1" sdist_bytes="$2" prepared="$WORK/prepared-$label" project="$WORK/project-$label"
    rm -rf "$prepared" "$project" "$WORK/uv-calls.log" "$WORK/bin"
    mkdir -p "$prepared" "$project"
    echo wheel > "$prepared/pkg-1.0-py3-none-any.whl"
    # A sparse file: the guard reads the size, not the content.
    dd if=/dev/null of="$prepared/pkg-1.0.tar.gz" bs=1 count=0 seek="$sdist_bytes" 2>/dev/null
    make_stub_uv "$prepared"
    (
        cd "$project"
        PATH="$WORK/bin:$PATH" UV_PUBLISH_TOKEN=dummy bash -c 'source "$1"; publish_to_pypi 1.0' _ "$RELEASE_SH"
    )
}

if OUT=$(run_publish oversize $(( LIMIT + 1 )) 2>&1); then
    fail "publish_to_pypi accepted an oversize sdist:"$'\n'"$OUT"
fi
echo "$OUT" | grep -q "PyPI rejects files over 100 MiB" || fail "size error not reported: $OUT"
echo "$OUT" | grep -q "pkg-1.0.tar.gz is 100 MiB" || fail "offending file not named: $OUT"
[[ ! -e "$WORK/uv-calls.log" ]] || fail "uv publish ran despite the oversize sdist: $(cat "$WORK/uv-calls.log")"
pass "oversize sdist aborts publish_to_pypi before uv publish"

OUT=$(run_publish exact "$LIMIT" 2>&1) || fail "publish_to_pypi rejected a sdist of exactly 100 MiB:"$'\n'"$OUT"
grep -q "^publish" "$WORK/uv-calls.log" || fail "uv publish did not run for a sdist at the limit"
echo "$OUT" | grep -q "Successfully published version 1.0" || fail "success not reported: $OUT"
pass "sdist of exactly 100 MiB is uploaded"

echo "ALL TESTS PASSED"
