#!/bin/bash
# End-to-end test for scripts/install-remote-prereqs.sh, the step of ./rsorcar
# that installs git (and curl, tar, python3, ssh) on a remote before the
# checkout is there.
# Run: bash scripts/test_install_remote_prereqs.sh
#
# Runs the real script the way ./rsorcar runs it (``bash -s`` with the script
# on standard input) and checks:
#   * a host that has every tool is left alone: nothing is run, exit 0;
#   * as root in a real debian:13 container (when docker is usable): git,
#     curl, python3 and the ssh client are really installed with apt-get;
#   * as a non-root user with passwordless sudo, against a PATH holding only
#     the tools this test provides: the missing tool is installed through
#     ``sudo -n``, apt-get is run non-interactively with the dpkg-lock
#     timeout, ``update`` first and ``install`` with the right packages;
#   * a failing ``apt-get update`` is a warning and the install still runs;
#   * a failing install, or an install after which the tool is still not
#     found, fails with a clear message;
#   * a sudo that wants a password, or no sudo at all, fails naming the exact
#     command to run (with the ``apt-get update`` a fresh image needs first);
#   * a host without a known package manager fails with a clear message;
#   * the package names differ where they must: openssh-clients on dnf,
#     openssh and python on pacman, openssh-client on apk.
#
# The package managers are PATH stubs that record their arguments: this test
# cannot install packages on the machine running it, and the real install is
# covered by the container run.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/install-remote-prereqs.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# Runs the script the way rsorcar does: fed to ``bash -s`` on standard input.
run_script() {
    bash -s < "$SCRIPT" 2>&1
}

# --- Test 1: a host that has everything ---------------------------------------
OUT=$(run_script) || fail "the script failed on a host that has every tool:
$OUT"
echo "$OUT" | grep -q "are all installed" || fail "unexpected output on a complete host:
$OUT"
pass "a host with git, curl, tar, python3 and ssh is left alone (exit 0)"

# --- Test 2: root in a real debian:13 container --------------------------------
if docker info >/dev/null 2>&1; then
    OUT=$(docker run --rm -i debian:13 bash -c \
            'cat > /tmp/prereqs.sh; command -v git && exit 99; bash /tmp/prereqs.sh && command -v git curl python3 ssh' \
            < "$SCRIPT" 2>&1) || fail "the real install in debian:13 failed (rc $?):
$OUT"
    echo "$OUT" | grep -q "Installing git curl python3 ssh on .* with apt-get (git curl python3 openssh-client)" \
        || fail "the container run did not report the expected install:
$OUT"
    echo "$OUT" | grep -q "^/usr/bin/git$" || fail "git is not installed in the container afterwards:
$OUT"
    pass "as root in debian:13, apt-get really installs git, curl, python3 and openssh-client"
else
    echo "SKIP: docker is not usable here; the real root install in debian:13 is not run"
fi

# --- A PATH of this test's own for the stubbed runs ---------------------------
# $1: directory; the rest: package-manager stubs to create there.  The real
# tools the script (and bash) needs are linked in; git is deliberately absent.
make_bin() {
    local bin="$1"; shift
    mkdir -p "$bin"
    local tool real
    # ``true`` is what the script's ``sudo -n true`` probe asks sudo to run;
    # ``type -P`` finds the binary where ``command -v`` would name the builtin.
    for tool in bash sh true chmod hostname id env cat grep sed curl tar python3 ssh; do
        real="$(type -P "$tool")" || fail "this machine has no $tool to link into the test PATH"
        ln -sf "$real" "$bin/$tool"
    done
    : > "$bin/log"
    # sudo -n records that it was asked, then runs the command itself.
    cat > "$bin/sudo" <<EOF
#!/bin/bash
[[ "\$1" == -n ]] || { echo "sudo called without -n: \$*" >> "$bin/log"; exit 1; }
shift
echo "sudo \$*" >> "$bin/log"
[[ "\${SUDO_MODE:-}" == "password" ]] && exit 1
exec "\$@"
EOF
    chmod +x "$bin/sudo"
    local pm
    for pm in "$@"; do
        cat > "$bin/$pm" <<EOF
#!/bin/bash
echo "$pm \$* [DEBIAN_FRONTEND=\${DEBIAN_FRONTEND:-unset}]" >> "$bin/log"
case " \$* " in
    *" update "*) [[ "\${PM_MODE:-}" == "update-fails" ]] && exit 100 ;;
    *" install "*|*" -S "*|*" add "*)
        [[ "\${PM_MODE:-}" == "install-fails" ]] && exit 100
        if [[ "\${PM_MODE:-}" != "install-lies" ]]; then
            printf '#!/bin/bash\necho stub git\n' > "$bin/git"; chmod +x "$bin/git"
        fi ;;
esac
exit 0
EOF
        chmod +x "$bin/$pm"
    done
}

# $1: bin dir; environment assignments may precede the call.
run_stubbed() {
    local bin="$1"
    PATH="$bin" bash -s < "$SCRIPT" 2>&1
}

# --- Test 3: non-root, sudo -n, apt-get installs the missing git --------------
make_bin "$WORK/t3" apt-get
OUT=$(run_stubbed "$WORK/t3") || fail "the stubbed apt-get install failed:
$OUT"
echo "$OUT" | grep -q "Installing git on .* with apt-get (git)" || fail "did not announce the git install:
$OUT"
echo "$OUT" | grep -q "Installed git on" || fail "did not report success:
$OUT"
grep -q "^sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 update -q$" "$WORK/t3/log" \
    || fail "apt-get update was not run through sudo -n as expected:
$(cat "$WORK/t3/log")"
grep -q "^apt-get -o DPkg::Lock::Timeout=300 install -y -q git \[DEBIAN_FRONTEND=noninteractive\]$" "$WORK/t3/log" \
    || fail "apt-get install was not run with the expected arguments:
$(cat "$WORK/t3/log")"
# Real sudo resets the environment, so the variable has to ride on ``env``
# through it for the install too (the stub would pass it on either way).
grep -q "^sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y -q git$" "$WORK/t3/log" \
    || fail "apt-get install was not run through sudo -n env DEBIAN_FRONTEND=noninteractive:
$(cat "$WORK/t3/log")"
[[ -x "$WORK/t3/git" ]] || fail "the stub install did not produce git"
pass "non-root: git is installed through sudo -n and a non-interactive apt-get (update, then install)"

# --- Test 4: a failing apt-get update is a warning; the install still runs ----
make_bin "$WORK/t4" apt-get
OUT=$(PM_MODE=update-fails run_stubbed "$WORK/t4") || fail "a failed apt-get update failed the install:
$OUT"
echo "$OUT" | grep -q "apt-get update failed on .*; trying the install" || fail "no warning about the failed update:
$OUT"
echo "$OUT" | grep -q "Installed git on" || fail "the install did not run after the failed update:
$OUT"
pass "a failing apt-get update is a warning and the install still runs"

# --- Test 5: a failing install is fatal ----------------------------------------
make_bin "$WORK/t5" apt-get
if OUT=$(PM_MODE=install-fails run_stubbed "$WORK/t5"); then
    fail "a failing apt-get install did not fail the script:
$OUT"
fi
echo "$OUT" | grep -q "apt-get could not install git on" || fail "unexpected message for a failed install:
$OUT"
pass "a failing install fails with a clear message"

# --- Test 6: an install that reports success but leaves git missing ----------
make_bin "$WORK/t6" apt-get
if OUT=$(PM_MODE=install-lies run_stubbed "$WORK/t6"); then
    fail "an install that left git missing did not fail the script:
$OUT"
fi
echo "$OUT" | grep -q "reported success on .*, yet git still cannot be found on PATH" \
    || fail "unexpected message when git is still missing after the install:
$OUT"
pass "an install after which git is still missing fails with a clear message"

# --- Test 7: sudo wants a password ------------------------------------------
make_bin "$WORK/t7" apt-get
if OUT=$(SUDO_MODE=password run_stubbed "$WORK/t7"); then
    fail "a sudo that wants a password did not fail the script:
$OUT"
fi
echo "$OUT" | grep -q "sudo wants a password. Run this on .*, then deploy again: sudo apt-get update && sudo apt-get -o DPkg::Lock::Timeout=300 install -y -q git" \
    || fail "the message does not name the command to run by hand (with the apt-get update a fresh image needs):
$OUT"
! grep -q "^apt-get" "$WORK/t7/log" || fail "apt-get was run although sudo refused"
pass "a sudo that wants a password fails naming the exact command to run by hand"

# --- Test 7b: no sudo at all, not root --------------------------------------
make_bin "$WORK/t7b" apt-get
rm -f "$WORK/t7b/sudo"
if OUT=$(run_stubbed "$WORK/t7b"); then
    fail "a host without sudo did not fail the script:
$OUT"
fi
echo "$OUT" | grep -q "sudo is not installed. Run this on .* as root, then deploy again: apt-get update && apt-get -o DPkg::Lock::Timeout=300 install -y -q git" \
    || fail "the message for a host without sudo does not name the command to run as root:
$OUT"
[[ ! -s "$WORK/t7b/log" ]] || fail "something was run although there is no sudo: $(cat "$WORK/t7b/log")"
pass "a host without sudo fails naming the command to run as root"

# --- Test 8: no known package manager ----------------------------------------
make_bin "$WORK/t8"
if OUT=$(run_stubbed "$WORK/t8"); then
    fail "a host without a package manager did not fail the script:
$OUT"
fi
echo "$OUT" | grep -q "is missing git and has no package manager this script knows" \
    || fail "unexpected message for a host without a package manager:
$OUT"
pass "a host without a known package manager fails with a clear message"

# --- Test 9: package names on dnf, pacman and apk -----------------------------
# ssh and python3 are the tools whose package name differs; both are hidden
# from the PATH here, and the stub's install creates git only, so the script
# is expected to stop at its final check — after the install command has been
# recorded, which is what this test reads.
for pm in dnf pacman apk; do
    make_bin "$WORK/t9-$pm" "$pm"
    rm -f "$WORK/t9-$pm/ssh" "$WORK/t9-$pm/python3"
    OUT=$(run_stubbed "$WORK/t9-$pm") && fail "$pm: expected the final check to fail (ssh and python3 are stubs-less):
$OUT"
    case "$pm" in
        dnf)    want="dnf install -y -q git python3 openssh-clients" ;;
        pacman) want="pacman -S --noconfirm --needed git python openssh" ;;
        apk)    want="apk add git python3 openssh-client" ;;
    esac
    grep -q "^$want \[" "$WORK/t9-$pm/log" \
        || fail "$pm: expected '$want' to be run, log was:
$(cat "$WORK/t9-$pm/log")"
    grep -q "^sudo $want$" "$WORK/t9-$pm/log" || fail "$pm: the install did not go through sudo -n"
done
pass "dnf, pacman and apk are asked for their own package names (openssh-clients / openssh, python / openssh-client)"

echo "ALL TESTS PASSED"
