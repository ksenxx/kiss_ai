#!/bin/bash
# End-to-end test for scripts/install-ssh-identity.sh, the remote half of the
# ~/.ssh/ copy in ./rsorcar.
# Run: bash scripts/test_install_ssh_identity.sh
#
# Runs the real script with HOME pointing at a scratch "remote" home and a tar
# stream on standard input, the way ./rsorcar runs it over ssh, and checks:
#   * every file in the stream lands in ~/.ssh/ under its relative path, and a
#     nested path gets its directories created;
#   * a file of the remote's own that the stream replaces with different
#     content is kept in $SSH_BACKUP under the same relative path, listed in
#     the output, and the backup is unreadable by others;
#   * a file whose content is identical is replaced without a backup, and a
#     copy that replaces nothing leaves no backup directory at all;
#   * authorized_keys — never in the stream — is untouched;
#   * a symlink the stream replaces with a different target is kept too, and
#     listed; one with the same target is not;
#   * permissions afterwards: 700 on ~/.ssh, 600 on files, 644 on *.pub and
#     known_hosts*;
#   * the staging directory is gone afterwards;
#   * a missing SSH_BACKUP or a machine without tar fails with a clear message.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/install-ssh-identity.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# stat's mode flag differs between GNU (-c %a) and BSD (-f %Lp) stat.
mode_of() {
    stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1"
}

# $1: directory to archive, rest: relative paths inside it.  Prints the tar
# stream the way ./rsorcar builds it.
stream() {
    local dir="$1"; shift
    COPYFILE_DISABLE=1 tar -cf - -C "$dir" -- "$@"
}

# --- Test 1: a copy over an existing ~/.ssh ---------------------------------
mkdir -p "$WORK/t1/local/agent" "$WORK/t1/rhome/.ssh"
echo "new private key"   > "$WORK/t1/local/id_rsa"
echo "new public key"    > "$WORK/t1/local/id_rsa.pub"
echo "same config"       > "$WORK/t1/local/config"
echo "hosts"             > "$WORK/t1/local/known_hosts"
echo "brand new key"     > "$WORK/t1/local/id_ed25519"
echo "nested"            > "$WORK/t1/local/agent/sub"
ln -s id_rsa               "$WORK/t1/local/link"
echo "remote's own key"  > "$WORK/t1/rhome/.ssh/id_rsa"
echo "same config"       > "$WORK/t1/rhome/.ssh/config"
echo "lets the sender in" > "$WORK/t1/rhome/.ssh/authorized_keys"
ln -s config               "$WORK/t1/rhome/.ssh/link"
chmod 644 "$WORK/t1/rhome/.ssh/id_rsa"
BACKUP1="$WORK/t1/rhome/.kiss/ssh-replaced-test"

OUT=$(stream "$WORK/t1/local" ./id_rsa ./id_rsa.pub ./config ./known_hosts ./id_ed25519 ./agent/sub ./link \
        | HOME="$WORK/t1/rhome" SSH_BACKUP="$BACKUP1" bash "$SCRIPT" 2>&1) \
    || fail "install-ssh-identity.sh failed:
$OUT"

R="$WORK/t1/rhome/.ssh"
[[ "$(cat "$R/id_rsa")" == "new private key" ]] || fail "id_rsa was not replaced"
[[ "$(cat "$R/id_ed25519")" == "brand new key" ]] || fail "new file id_ed25519 not copied"
[[ "$(cat "$R/agent/sub")" == "nested" ]] || fail "nested file agent/sub not copied"
[[ "$(cat "$R/config")" == "same config" ]] || fail "config changed content"
[[ -L "$R/link" && "$(readlink "$R/link")" == "id_rsa" ]] || fail "symlink not replaced"
pass "every file in the stream lands in ~/.ssh under its relative path"

[[ "$(cat "$R/authorized_keys")" == "lets the sender in" ]] || fail "authorized_keys was touched"
pass "authorized_keys is untouched"

[[ "$(cat "$BACKUP1/id_rsa")" == "remote's own key" ]] || fail "the replaced id_rsa was not kept"
[[ -L "$BACKUP1/link" && "$(readlink "$BACKUP1/link")" == "config" ]] || fail "the replaced symlink was not kept"
[[ ! -e "$BACKUP1/config" ]] || fail "an identical config was backed up needlessly"
[[ ! -e "$BACKUP1/authorized_keys" ]] || fail "authorized_keys ended up in the backup"
echo "$OUT" | grep -q "kept the ssh files this copy replaced in ~/.kiss/ssh-replaced-test" \
    || fail "backup not announced in the output:
$OUT"
echo "$OUT" | grep -qE '^ +id_rsa$' || fail "kept file not listed in the output:
$OUT"
echo "$OUT" | grep -qE '^ +link$' || fail "kept symlink not listed in the output:
$OUT"
[[ "$(mode_of "$BACKUP1")" =~ ^7?00$ ]] || fail "backup dir readable by others: $(mode_of "$BACKUP1")"
pass "replaced files with different content are kept in SSH_BACKUP, identical ones are not"

# The same stream again: everything is identical now — the symlink included,
# which is compared by its target — so a second deploy keeps nothing.
BACKUP1B="$WORK/t1/rhome/.kiss/ssh-replaced-again"
OUT=$(stream "$WORK/t1/local" ./id_rsa ./config ./link \
        | HOME="$WORK/t1/rhome" SSH_BACKUP="$BACKUP1B" bash "$SCRIPT" 2>&1) \
    || fail "second run failed:
$OUT"
[[ ! -e "$BACKUP1B" ]] || fail "a second, identical copy backed up: $(ls -A "$BACKUP1B")"
[[ -z "$OUT" ]] || fail "a second, identical copy printed something:
$OUT"
[[ -L "$R/link" && "$(readlink "$R/link")" == "id_rsa" ]] || fail "symlink lost on the second run"
pass "a second run with identical files and symlinks keeps nothing"

[[ "$(mode_of "$R")" == "700" ]] || fail "~/.ssh mode is $(mode_of "$R"), want 700"
[[ "$(mode_of "$R/id_rsa")" == "600" ]] || fail "id_rsa mode is $(mode_of "$R/id_rsa"), want 600"
[[ "$(mode_of "$R/agent/sub")" == "600" ]] || fail "agent/sub mode is $(mode_of "$R/agent/sub"), want 600"
[[ "$(mode_of "$R/id_rsa.pub")" == "644" ]] || fail "id_rsa.pub mode is $(mode_of "$R/id_rsa.pub"), want 644"
[[ "$(mode_of "$R/known_hosts")" == "644" ]] || fail "known_hosts mode is $(mode_of "$R/known_hosts"), want 644"
pass "permissions fixed: 700 dir, 600 files, 644 public keys and known_hosts"

[[ -z "$(find "$WORK/t1/rhome/.kiss" -maxdepth 1 -name 'ssh-incoming.*')" ]] \
    || fail "staging directory left behind: $(ls "$WORK/t1/rhome/.kiss")"
pass "staging directory removed"

# --- Test 2: a copy onto a machine with no ~/.ssh at all --------------------
mkdir -p "$WORK/t2/local" "$WORK/t2/rhome"
echo "key" > "$WORK/t2/local/id_rsa"
BACKUP2="$WORK/t2/rhome/.kiss/ssh-replaced-test"
OUT=$(stream "$WORK/t2/local" ./id_rsa \
        | HOME="$WORK/t2/rhome" SSH_BACKUP="$BACKUP2" bash "$SCRIPT" 2>&1) \
    || fail "install-ssh-identity.sh failed on an empty home:
$OUT"
[[ "$(cat "$WORK/t2/rhome/.ssh/id_rsa")" == "key" ]] || fail "id_rsa not created on an empty home"
[[ ! -e "$BACKUP2" ]] || fail "a backup directory was created although nothing was replaced"
[[ -z "$OUT" ]] || fail "unexpected output when nothing was replaced:
$OUT"
pass "a copy that replaces nothing creates ~/.ssh and no backup directory"

# --- Test 3: SSH_BACKUP unset --------------------------------------------------
mkdir -p "$WORK/t3/rhome"
if OUT=$(stream "$WORK/t2/local" ./id_rsa 2>/dev/null | HOME="$WORK/t3/rhome" bash "$SCRIPT" 2>&1); then
    fail "the script ran without SSH_BACKUP"
fi
echo "$OUT" | grep -q "SSH_BACKUP is not set" || fail "no clear message for a missing SSH_BACKUP:
$OUT"
[[ ! -e "$WORK/t3/rhome/.ssh/id_rsa" ]] || fail "files were installed although SSH_BACKUP was unset"
pass "a missing SSH_BACKUP fails with a clear message before touching anything"

# --- Test 4: a machine without tar -------------------------------------------
# An empty PATH hides tar; bash itself is started by its full path.
mkdir -p "$WORK/t4/rhome" "$WORK/t4/bin"
if OUT=$(stream "$WORK/t2/local" ./id_rsa 2>/dev/null \
        | HOME="$WORK/t4/rhome" SSH_BACKUP="$WORK/t4/rhome/.kiss/b" PATH="$WORK/t4/bin" "$BASH" "$SCRIPT" 2>&1); then
    fail "the script ran without tar"
fi
echo "$OUT" | grep -q "tar is not installed" || fail "no clear message for a missing tar:
$OUT"
pass "a machine without tar fails with a clear message"

echo
echo "ALL TESTS PASSED"
