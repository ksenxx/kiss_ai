# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the shell guards shared by every Sorcar shell tool.

The destructive-command guard and the install-timeout lift came from the
HarnessTax container harness (``coding_sea.py``); these tests exercise them
through the host ``UsefulTools.Bash`` / ``run_commands_parallel`` tools and,
when a Docker daemon is running, through ``DockerManager``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType

import docker
import pytest

from kiss.agents.seas.coding import coding_sea
from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, DockerManager
from kiss.agents.sorcar.shell_guards import (
    DESTRUCTIVE_VERDICT,
    INSTALL_TIMEOUT_SECONDS,
    destructive_command_guard,
    lift_install_timeout,
    refuse_or_lift,
)
from kiss.agents.sorcar.shell_scan import TOO_DEEP_VERDICT, UNREAD_VALUES_VERDICT
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.processes import find_bash
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)
from kiss.tests.agents.sorcar.test_audit1101_s2_shell_guards import guards_stay_linear

BLOCKED = [
    "kill -9 -1",
    "kill -1",
    "echo cleanup; kill -TERM -1",
    "pkill -f .",
    "pkill -9 -f '.*'",
    "sudo rm -rf /",
    "rm -rf {wd}",
    "rm -rf {wd}/",
    "cd / && rm -r {wd}",
    "echo cleanup\nkill -9 -1",
    "echo cleanup\nrm -rf {wd}",
    "rm -rf /tmp/one {wd}",
    "rm -rf /*",
    "rm -rf {wd}*",
    "kill -9 -1 1234",
    "kill -s TERM -1",
    "pkill -f -9 .",
    "sudo -n rm -rf {wd}",
    "sudo -u root rm -rf /",
    "cd {wd} && sudo -n kill -9 -1",
    "pkill -ef .",
    "env DEBUG=1 rm -rf {wd}",
    "sudo env DEBUG=1 rm -rf /",
    "sudo -n sudo -n rm -rf {wd}",
    'rm -rf "{wd}"',
    "rm -rf \"some dir\" {wd}",
    # spellings the coding SEA's harness used to pin on its own
    "sleep 1; kill -1",
    "pkill -9 -f .",
    "pkill -f '.' && ls",
    "cd /tmp && rm -rf {wd}/*",
    "rm -rf /",
    "(rm -r {wd})",
    "/bin/kill -9 -1",
    "kill -9 -1 >/dev/null",
    "kill -9 -1 # stop all",
    "kill -s KILL -1",
    "/usr/bin/pkill -f .",
    "pkill -f . 2>/dev/null",
    "pkill --full .",
    "pkill -f '.*'",
    "rm -rf '{wd}'",
    "/bin/rm -rf {wd}",
    "rm -rf {wd} >/dev/null",
    "rm -rf {wd} /tmp/x",
    "sudo rm -rf {wd}/",
    "FOO=1 kill -9 -1",
]

ALLOWED = [
    # FX05c: a quoted glob is no glob; bash deletes at most a file named ``*`` in the work dir
    "rm -rf '{wd}/*'",
    "kill -1 1234",
    "kill -9 $PID",
    "kill -15 %1",
    "pkill -f myserver",
    "pkill -f 'python app.py'",
    "rm -rf {wd}/build",
    "rm -rf ./tmp/x",
    "rm -rf /tmp/kiss-scratch",
    "grep -R 'kill -1' .",
    "echo 'rm -rf /'",
    "rm -rf build/ {wd}/sub",
    "rm -rf {wd}er",
    "git rm -r --cached {wd}/sub",
    "rm -rf ./build && cd {wd}",
    "rm file.txt; ls {wd}",
    "rm file.txt\nprintf '%s' {wd}",
    "rm -f out.log > {wd}/log.txt",
    "rm -rf ./build | tee {wd}/log.txt",
    "kill -0 $$; printf %s -1",
    "kill -0 $$ && echo -1",
    "rm -f missing # {wd}",
    "rm -f missing # /",
    "pkill -f myserver; echo .",
    "sudo -n true; echo -1",
    'kill -1 "1234"',
    "kill -1 '$PID'",
    'rm -rf "{wd} backup"',
    "rm -rf '/ backup'",
    "pkill -f '. server'",
    "env DEBUG=1 rm -rf {wd}/build",
    # spellings the coding SEA's harness used to pin on its own
    "kill -9 1234",
    "kill -1 $(cat /tmp/pid)",
    "kill -1 %1",
    "pkill -f python3",
    "rm -rf {wd}/*.o",
    "ls {wd}",
    "printf '%s\\n' 'kill -9 -1'",
    'echo "pkill -f ."',
    "grep -F 'rm -rf {wd}' README.md",
    "echo ok # rm -rf {wd}",
]


@pytest.fixture(autouse=True)
def no_inherited_cdpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test without a ``CDPATH`` of the developer's: one redirects a bare ``cd``."""
    monkeypatch.delenv("CDPATH", raising=False)


def test_destructive_guard_protects_root_and_work_dir(tmp_path: Path) -> None:
    """Root and the work dir (any spelling) are refused; everything else runs."""
    wd = str(tmp_path)
    named = tmp_path.as_posix()  # as a command names it: bash eats a Windows path's backslashes
    for command in BLOCKED:
        verdict = destructive_command_guard(command.format(wd=named), wd)
        assert verdict == DESTRUCTIVE_VERDICT, command
    for command in ALLOWED:
        assert destructive_command_guard(command.format(wd=named), wd) is None, command
    # The trailing-slash spelling and the resolved path of a symlinked work dir
    # are protected too, and ``None`` protects the root only.
    link = tmp_path.parent / (tmp_path.name + "-link")
    link.symlink_to(tmp_path)
    assert destructive_command_guard(f"rm -rf {named}", str(link)) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {link.as_posix()}", str(link)) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {named}", f"{tmp_path}/") == DESTRUCTIVE_VERDICT
    assert destructive_command_guard("rm -rf /", None) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {named}", None) is None
    assert destructive_command_guard("rm -rf *", None) is None
    assert destructive_command_guard(f"rm -rf ../{tmp_path.name}", str(link)) == DESTRUCTIVE_VERDICT
    if sys.platform == "win32":
        # Git Bash takes a quoted Windows path as written, a ``\`` in it as a separator and
        # ``C:`` as the root of the drive (full run 2026-11-22: the guard read each of them
        # as a name in the work dir); unquoted, bash eats the backslashes: ``C:Usersx``.
        for command in (f'rm -rf "{wd}"', f"rm -rf '{wd}'/*", f"rm -rf '..\\{tmp_path.name}'",
                        f"rm -rf {wd[:2]}", f"D='{wd}'; rm -rf \"$D\""):
            assert destructive_command_guard(command, wd) == DESTRUCTIVE_VERDICT, command
        assert destructive_command_guard(f"rm -rf {wd}", wd) is None


@pytest.mark.parametrize("command, refused", [
    # the wrappers a model reaches for when it means "stop my server"
    ("exec kill -9 -1", True), ("nohup kill -9 -1", True), ("timeout 5 kill -9 -1", True),
    ("nice kill -9 -1", True), ("nice -n 5 kill -1", True), ("time kill -1", True),
    ("command kill -1", True), ("builtin kill -1", True), ("xargs kill -1", True),
    ("echo x | xargs rm -rf /", True), ("nohup sudo kill -1", True),
    # a shell's -c text, a backquoted substitution
    ("bash -c 'kill -9 -1'", True), ('sh -c "rm -rf /"', True), ("bash -ec 'pkill -f .'", True),
    ("echo `kill -1`", True), ("echo $(kill -1)", True),
    # a wrapper's option value spelled like a command, a directory prefix after the
    # wrapper's options, a command name among the operands of the command
    ("timeout -s kill 5 rm -rf /", True), ("sudo -u kill rm -rf /", True),
    ("exec -a kill rm -rf /", True), ("xargs -I rm rm -rf /", True),
    ("sudo -n /bin/rm -rf /", True), ("sudo -u root /usr/bin/kill -9 -1", True),
    ("rm -rf kill /", True), ("kill -9 rm -1", True), ("sudo rm -rf a kill -1", True),
    ("bash -x -c 'kill -1'", True), ("bash --norc -c 'kill -1'", True),
    # options after the ``-c`` cluster, ``-c`` first in its cluster
    ("bash -c -- 'kill -9 -1'", True), ("bash -c -e 'kill -9 -1'", True),
    ("bash -co posix 'kill -9 -1'", True), ("bash -cl 'kill -1'", True),
    ('sh -c -o posix "rm -rf /"', True), ("bash -c -e 'kill -1 1234'", False),
    ("bash -co posix 'rm -rf ./build'", False),
    ("sudo echo rm -rf x rm ok", False), ("sudo echo /bin/rm /bin/rm ok", False),
    ("rm -rf a kill -1", False), ("timeout -s kill 5 sleep 1", False),
    # several option values spelled like guarded commands before the real one
    ("env -u kill -u pkill rm -rf /tmp/w", True), ("env -u kill -u pkill rm -rf /", True),
    ("sudo -u kill -g pkill kill -9 -1", True), ("env -u rm -u kill pkill -f .", True),
    ("env -u kill -u pkill -u rm rm -rf /tmp/w", True),
    # FX05c: the text of ``-c`` is the first operand, here ``kill``; the rest are its ``$0``...
    ("bash -c -e -u kill -u pkill 'rm -rf /tmp/w'", False),
    ("bash -c -e -u 'rm -rf /tmp/w' kill", True),
    ("env -u kill -u pkill rm -rf /tmp/w/build", False), ("env -u kill -u pkill true", False),
    # the work dir through its parent
    ("rm -rf /tmp/w/..", True), ("rm -rf /tmp/w/../", True), ("rm -rf /tmp/w/../w", True),
    ("rm -rf /tmp/w/../w/", True), ("rm -rf /tmp/w/../*", True), ("rm -rf /..", True),
    # a signal, a pid, a sibling or a subdirectory, a quoted mention in an argument
    ("kill -1 1234", False), ("nice kill -1 1234", False), ("rm -rf /tmp/w/../other", False),
    ("rm -rf /tmp/w/w", False), ("rm -rf /tmp/w/build", False), ("grep -rn 'kill -1' src", False),
    ("xargs grep 'kill -1'", False), ("command -v rm", False), ("timeout 5 sleep 1", False),
    # the same commands where a compound command puts them, with the operand quoted,
    # with the work dir (where every call starts) named from inside it
    ("{ kill -9 -1; }", True), ("if true; then kill -9 -1; fi", True),
    ("for i in 1; do kill -9 -1; done", True), ("true || ! kill -9 -1", True),
    ("if kill -9 -1; then :; fi", True), ("while ! kill -9 -1; do :; done", True),
    ("until kill -9 -1; do :; done", True), ("if false; then :; elif kill -9 -1; then :; fi", True),
    ("if false; then :; else rm -rf /; fi", True), ("stop() { kill -9 -1; }; stop", True),
    ('kill -9 "-1"', True), ("kill -s KILL '-1'", True), ("kill '-1'", True),
    ("rm -rf ./*", True), ("rm -rf *", True), ('rm -rf "$PWD"', True), ("rm -rf ${PWD}/", True),
    ("rm -rf $(pwd)/*", True), ("rm -rf `pwd`", True), ("rm -rf ../w", True),
    ("rm -rf ../w/*", True), ("rm -rf * && cd ..", True), ("ls; { rm -rf *; }", True),
    ("rm -rf ./build/*", False), ("rm -rf *.o", False), ("rm -rf ../w2", False),
    ("rm -rf ../other", False), ('kill -9 "$pid"', False), ('kill "-1" 1234', False),
    ("if true; then kill -15 1234; fi", False), ("echo {} done", False),
    ("echo x | xargs -I{} kill -9 -1", True), ("rm -rf ${HOME}/x /", True),
    ("function stop { kill -9 -1; }; stop", True), ("coproc kill -9 -1", True),
    ("function wipe { rm -rf *; }; wipe", True), ("function hello { echo hi; }; hello", False),
])
def test_destructive_guard_sees_through_command_wrappers(command: str, refused: bool) -> None:
    """Audit 2026-11-07 A12_claude-2: the guard absorbed one ``sudo``/``env`` wrapper and so
    missed ``nohup kill -9 -1``, ``timeout 5 kill -9 -1``, ``bash -c 'kill -9 -1'`` and a
    backquoted substitution, and ``rm -rf <workdir>/..``; one wrapper of any of these kinds,
    a shell's ``-c`` text and the work dir named through its parent are refused.  Audit
    2026-11-08 S8: the shell's options may continue after the ``-c`` cluster (``bash -c --
    '…'``, ``bash -c -e '…'``, ``bash -co posix '…'``) and ``-c`` may lead its cluster.
    Review 2026-11-08 R7 S7/F2: a wrapper's options may hold any number of values spelled
    like guarded commands (``env -u kill -u pkill rm -rf <workdir>``) before the real one."""
    verdict = destructive_command_guard(command, "/tmp/w")
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command
    assert verdict is None or verdict == DESTRUCTIVE_VERDICT


@pytest.mark.parametrize("command, refused", [
    # before the options, before the operand, inside a word, after a wrapper, in a ``-c`` text
    ("rm -rf \\\n/tmp/w", True), ("rm \\\n  -rf /tmp/w", True), ("rm -rf /tmp/\\\nw", True),
    ("r\\\nm -rf /tmp/w", True), ("sudo \\\nrm -rf /", True), ("rm -rf a \\\n b \\\n /tmp/w", True),
    ("kill -9 \\\n-1", True), ("kill \\\n-9 -1", True), ("pkill -f \\\n.", True),
    ("rm -rf \\\n*", True), ("cd . && rm -rf \\\n./*", True), ("bash -c 'rm -rf \\\n/tmp/w'", True),
    # a backslash that another one escapes continues nothing: its newline ends the command
    ("echo \\\\\nrm -rf /tmp/w", True), ("echo \\\\\nrm -rf \\\n/tmp/w", True),
    # nor does a comment's: the next line is a command, which may itself continue
    ("# note \\\nrm -rf /tmp/w", True), ("ls # note \\\nrm -rf \\\n/tmp/w", True),
    ('echo "a # b"; rm -rf \\\n/tmp/w', True),
    # a continued command that deletes or signals something else, a quoted mention
    ("rm -rf \\\n/tmp/w/build", False), ("rm -rf /tmp/w\\\n2", False), ("kill -9 \\\n1234", False),
    ("rm -rf \\\n*.o", False), ("cd sub && rm -rf \\\n*", False), ("echo one \\\ntwo", False),
    ("echo 'rm -rf \\\n/tmp/w'", False),
])
def test_destructive_guard_reads_through_line_continuations(command: str, refused: bool) -> None:
    """Audit 2026-11-21 S12/F4: bash drops a backslash-newline before it reads the command, so
    ``rm -rf \\<newline><workdir>`` deleted the work dir and ``kill -9 \\<newline>-1`` passed:
    the patterns stop at a newline.  The guard reads the command with its continuations
    removed too; a newline after an escaped backslash, or after a comment, still separates."""
    verdict = destructive_command_guard(command, "/tmp/w")
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command


@pytest.mark.parametrize("command, refused", [
    # a quoted or escaped word before the operand: an assignment's value, an argument, a wrapper's
    ('X="a b" rm -rf /tmp/w', True), ("X='a;b' Y=2 rm -rf /tmp/w", True),
    ("X=a\\ b kill -9 -1", True), ('X="a b" pkill -f .', True), ('X="a b" rm -rf *', True),
    ('rm -rf "x # y" /tmp/w', True), ("rm -rf 'x #' /tmp/w", True),
    ('rm -rf "file (1).txt" *', True), ("rm -f 'a;b' 'c|d' \"e>f\" ./*", True),
    ('rm -rf "a\nb" /tmp/w', True), ("rm -rf a\\;b a\\ #c a\\(1\\) /tmp/w", True),
    ("rm -rf $'don\\'t # x' /tmp/w", True), ('env X="a # b" rm -rf /tmp/w', True),
    ('sudo -u "my;user" rm -rf /', True), ('kill -s "a # b" -1', True),
    ('pkill -u "a # b" -f .', True), ('rm -rf "x # y" "$PWD"', True),
    ('rm -rf "x # y" "/tmp/w"', True), ('rm -rf "x # y" \\\n/tmp/w', True),
    # the operand, or the command's name, spelled with quotes
    ('rm -rf "$PWD"/*', True), ('rm -rf "${PWD}"/', True), ('rm -rf "/tmp/w"/*', True),
    ("rm -rf '/tmp/w'/", True), ('rm -rf /tmp/"w"', True), ('rm -rf ""/tmp/w', True),
    ("rm -rf /tmp/\\w", True), ("\\rm -rf /tmp/w", True), ('"rm" -rf /tmp/w', True),
    ("'kill' -9 -1", True), ('kill -9 -"1"', True), ('pkill "-f" .', True),
    ("pkill -f \\.", True), ('rm -rf "$(pwd)"/*', True), ('rm -rf "x # y" "`pwd`"', True),
    # a ``$`` reads as an expansion in single quotes too, as ``rm -rf '$PWD'`` always did
    ("rm -rf '$PWD'/*", True),
    # what the quotes hold is no command, no operand and no comment of its own
    ('rm -rf "x # y" /tmp/w/build', False), ('rm -rf "x # /tmp/w"', False),
    ('rm -rf "x; rm -rf /tmp/w2"', False), ('X="a b" rm -rf /tmp/w/build', False),
    ('X="rm -rf /tmp/w" true', False), ('echo "x # y" /tmp/w', False),
    ('rm -rf "$PWD"/build', False), ('rm -rf "$PWD"/*.o', False), ('rm -rf "/tmp/w"2', False),
    ("rm -rf \\*", False), ('rm -rf "\\$PWD"', False), ('rm -f "a # b" c # /tmp/w', False),
    # FX05c: a word comes without its quotes, so an escaped ``$PWD`` beside a glob reads as
    # ``$PWD``: refused, though bash would delete in a directory named ``$PWD``
    ("rm -rf \\$PWD/*", True), ("rm -rf 'x y' \\`pwd\\`", False),
    # ``-1`` stays the signal before a quoted target, whatever the quotes hold
    ('kill -1 "`cat app.pid`"', False), ('kill -1 "$(cat app.pid)"', False),
    ("kill -s HUP '1 2'", False), ('pkill -f "my app" -u "$USER"', False),
    ("rm -f x # don't\nrm -rf 'y z' /tmp/w/build # isn't /tmp/w", False),
    ('git commit -m "drop (old); rm -rf nothing" && ls /tmp/w', False),
    ('cd sub && rm -rf "a (1)" *', False), ('cd "sub" && rm -rf "$PWD"/*', False),
])
def test_destructive_guard_reads_the_command_without_its_quotes(
    command: str, refused: bool,
) -> None:
    """Review 2026-11-21 of S12 (FX05): the patterns read the command as written and end a word
    at a blank, a separator or a ``#``, inside quotes too, so ``X="a b" rm -rf <workdir>`` and
    ``rm -rf "x # y" <workdir>`` deleted the work dir; so did an operand, or a command name,
    that the quotes cut in two (``rm -rf "$PWD"/*``, ``\\rm``).  The guard also reads the
    command as bash does once it removed the quoting."""
    verdict = destructive_command_guard(command, "/tmp/w")
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command


#: FX05c: what two review rounds still got past the patterns (``{wd}`` is a work directory
#: whose name holds a blank, a ``#`` and parentheses), and what must keep running.
READ_AS_BASH_READS = [
    # the two quote gaps, with a work directory the masked reading could not name
    ('X="a b" rm -rf "{wd}"', True), ('rm -rf "x # y" "{wd}"', True),
    ("rm -rf '{wd}'/*", True), ('cd "{wd}" && rm -rf "x # y" *', True),
    ('X="a b" rm -rf "{wd}/build"', False), ('rm -rf "x # y" "{wd}/a (1)"', False),
    ('rm -rf "{wd} backup"', False),
    # a wrapper before a shell's -c, a wrapper the scanner does not know, an option it misreads
    ("sudo bash -c 'rm -rf /'", True), ("sudo -n bash -c 'rm -rf \"{wd}\"'", True),
    ("env X=1 nohup sh -c 'kill -9 -1'", True), ('setsid rm -rf "{wd}"', True),
    ("strace -f bash -c 'kill -9 -1'", True), ("sudo --user root rm -rf /", True),
    ("su -c 'rm -rf /' root", True), ("sudo -D /tmp rm -rf *", True),
    ("sudo bash -c 'rm -rf \"{wd}/build\"'", False), ("setsid sleep 5", False),
    ("su -c 'make install' root", False), ("cd sub && sudo rm -rf *", False),
    # a quoted word inside a -c text, a text within a text, a text handed on by a pipe
    ("bash -c 'rm -rf \"x # y\" \"{wd}\"'", True), ("X=\"a b\" bash -c 'rm -rf \"{wd}\"'", True),
    ("bash -c \"bash -c 'rm -rf \\\"{wd}\\\"'\"", True), ("echo 'kill -9 -1' | bash", True),
    ("cat <<'EOF' | sh\npkill -f .\nEOF", True), ("trap 'rm -rf \"{wd}\"' EXIT", True),
    ("eval 'rm -rf /'", True), ("echo 'kill -9 -1'", False), ('echo "$greeting" | bash', False),
    ('grep -n "rm -rf /" notes.md', False), ("git commit -m 'rm -rf / is refused'", False),
    # a shell's standard input: a here-string, a here-document its parent expands, a pipeline
    ("sh <<< 'rm -rf \"{wd}\"'", True), ("bash <<EOF\nrm -rf $UNSET/*\nEOF", True),
    ("cat <<EOF | bash\nrm -rf $UNSET/*\nEOF", True), ("printf 'kill -9 -1\\n' | sh", True),
    ("{{ echo kill -9 -1; }} | sh", True), ("echo kill -9 -1 | tee log | sudo sh", True),
    ("bash <<EOF\necho $HOME\nEOF", False), ("sh <<< 'echo hi'", False),
    ("ls run_*.sh | xargs -n 1 bash", False), ("cat <<EOF > notes.txt\nrm -rf /\n$X\nEOF", False),
    ("echo kill -9 -1 | a | a | a | a | a | a | a | a | a", False),
    ("bash -c 'cat' <<< 'rm -rf /'", False),
    # an expansion as an operand, or beside one: a value nobody set is empty
    ('rm -rf $(echo x) "{wd}"', True), ('rm -rf `echo x y` "{wd}"', True),
    ('rm -rf "$(pwd)"', True), ("rm -rf $UNSET/*", True), ('rm -rf "$UNSET"/', True),
    ("rm -rf ${{UNSET:-/}}", True), ('bash -c "rm -rf $UNSET/*"', True),
    ('eval "rm -rf $UNSET/*"', True), ('trap "rm -rf $UNSET/*" EXIT', True),
    ("a=r; b=m; $a$b -rf /", True), ("rm -rf /{{bin,}}", True), ("rm -rf $'\\x2f'", True),
    ("echo $'it\\'s'; rm -rf /", True), ("echo \"a\" $'it\\'s'; rm -rf /", True),
    ('d=$(pwd); rm -rf "$d"', True), ('ROOT="`pwd`"; cd build; rm -rf "$ROOT"/*', True),
    ('ROOT=$(pwd); rm -rf "$ROOT"/build', False),
    ('rm -rf "$d"', False), ("rm -rf $OUT/x", False),
    ('T=$(mktemp -d); rm -rf "$T"/* "$T"', False),
    ('for d in a b; do rm -rf "$d"/*; done', False), ('read -r d; rm -rf "$d"/*', False),
    ('f() {{ rm -rf "$1"/*; }}; f build', False), ("rm -rf ${{UNSET:?}}/*", False),
    ("rm -rf /tmp/{{a,b}}", False), ("echo $'it\\'s'; rm -rf ./build", False),
    # past 16 directories asked about, the rest is read as written; more than 64 brace
    # alternatives are words the guard cannot bound
    ("rm -rf /" + "{{a,b}}" * 7, True),
    ("rm -rf " + " ".join(f"d{n}/x" for n in range(20)), False),
    # the home directory
    ("rm -rf ~", True), ("rm -rf ~/*", True), ('rm -rf "$HOME"', True), ("rm -rf ~/.*", True),
    ("rm -rf ~/.cache/x", False), ('rm -rf "$HOME"/.cache/x', False),
    ("rm -rf ~+", True), ("rm -rf ~+/*", True), ("rm -rf ~nosuchuser/x", False),
    ("export HOME=$(mktemp -d); trap 'rm -rf \"$HOME\"' EXIT", False),
    # a directory is proven by && alone, and by nothing the scanner does not show
    ("cd nosuch; ((1)) && rm -rf *", True), ("cd nosuch; time && rm -rf *", True),
    ("cd sub && a || b && rm -rf *", True), ('cd nosuch; rm -rf "$PWD"/*', True),
    ("cd sub && cd ../.. && rm -rf *", True), ("cd a && b; cd c && rm -rf *", True),
    # a group after ``&&`` runs whole or not at all (review of FX05c, item 3: was refused)
    ("! cd sub && rm -rf *", True), ("cd sub && {{ ls; rm -rf *; }}", False),
    ('cd "$D" && rm -rf "gap 1 (work) #x"', True), ('cd "$D" && rm -rf build', False),
    ("cd /tmp/x && rm -rf * && python3 -c 'print((1, 2))'", False),
    ("cd /tmp/x && echo $((1 + 1)) && rm -rf *", False), ("cd sub && {{ rm -rf *; }}", False),
    ("cd sub && cd deep && rm -rf *", False), ("cd a && b; cd /tmp/x && rm -rf *", False),
    # every process
    ("kill -s KILL -1", True), ("kill -- -1", True), ("pkill .", True), ("pkill -u ksen", True),
    ('killall -u "$USER"', True), ("pkill ''", True), ("pkill --signal KILL '.*'", True),
    ("pkill -f '[a-z]'", True), ("pkill -f '\\w+'", True), ("pkill -v nonexistent", True),
    ("sudo killall5 -9", True), ("kill -9 -01", True), ("$'rm' -rf /", True),
    ("flock /tmp/l -c 'rm -rf /'", True), ("killall -v node", False),
    ('pkill -f "[p]ython server"', False),
    ('pkill -f "my server"', False), ('pkill -u "$USER" -f myserver', False),
    ("killall node", False), ('kill -1 "$pid"', False), ('kill -9 -- "-$PGID"', False),
    # a variable no command of the call set is empty, and ``pkill -f ""`` selects every
    # process (review of FX05c, item 1: was allowed)
    ("pkill -P 1234", False), ('pkill -f "$NAME"', True),
    # programs that only share a name, or delete by their own means
    ("find . -name '*.pyc' -exec rm -rf {{}} +", False), ("git rm -r --cached .", False),
    ('find . -name x -exec rm -rf "{wd}" \\;', True),
    ("docker rm -f $(docker ps -aq)", False), ("rm", False), ("> out.txt", False),
]


@pytest.mark.parametrize("command, refused", READ_AS_BASH_READS)
def test_destructive_guard_reads_commands_not_text(
    command: str, refused: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviews 2026-11-21 of FX05 and FX05b: patterns over the command text were bypassed by a
    quoted word before the operand whenever the work directory's own name held a blank, a ``#``
    or a parenthesis, by a wrapper before ``bash -c``, by a quoted word inside a ``-c`` text
    and by a ``$(...)`` operand.  The guard judges the words of each simple command instead."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work = tmp_path / "gap 1 (work) #x"
    work.mkdir()
    verdict = destructive_command_guard(command.format(wd=work), str(work))
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command


def _any_directory(path: str) -> bool:
    """Answer that *path* is a directory: the oracle of a test that reads the text alone."""
    return True


def _unresolved(path: str) -> str:
    """Return *path* as written: no link is resolved."""
    return path


def _unset_cdpath() -> str:
    """Return no ``CDPATH``."""
    return ""


def _guard_calls(command: str) -> tuple[str | None, int]:
    """Return the destructive guard's verdict on *command* in ``/tmp/w`` and its cost in calls.

    The Python calls the guard made to reach the verdict, the oracles answering by the
    text alone: the same number on an idle machine and on a busy one, which no time is.
    """
    calls = 0

    def count(frame: FrameType, event: str, arg: object) -> None:
        nonlocal calls
        calls += event == "call"

    previous = sys.getprofile()
    sys.setprofile(count)
    try:
        verdict = destructive_command_guard(
            command, "/tmp/w", _any_directory, _unresolved, _unset_cdpath)
    finally:
        sys.setprofile(previous)
    return verdict, calls


#: Review 2026-11-21 of FX05c: (command, refused) in the work directory ``/tmp/w`` with the
#: home directory ``/home/u``, the oracles answering by the text alone.
READ_BY_VALUE = [
    # an unset variable is empty: as a ``pkill`` pattern it selects every process, as does a
    # pattern that matches the empty string
    ('pkill -9 -f "${SERVER_PATTERN}"', True), ('pkill -f "node|$OTHER"', True),
    ("pkill -f 'node|'", True), ("pkill -f 'x*'", True), ("pkill -f '(qqz)?'", True),
    ('NAME=node; pkill -f "$NAME"', False), ("pkill -f 'node|python'", False),
    ("pkill -f 'a{2,3}[[:alpha:]]('", False),
    # the positional parameters of a -c text and of a function call are the words given
    ("bash -c 'rm -rf \"$1\"/*'", True), ("bash -c 'rm -rf \"$1\"/*' _ build", False),
    ("bash -c 'shift; rm -rf \"$1\"/*' _ build", True),  # bash deletes /*: no $1 is left
    ("bash -c 'rm -rf \"${99999999999}\"/x' _ a", False),
    ("sh -c 'rm -rf \"$1\"/*' sh \"$PWD\"", True), ("bash -c 'rm -rf \"$0\"' \"$HOME\"", True),
    ("bash -c 'eval \"$1\"' _ 'rm -rf ~'", True),
    ("find build -maxdepth 0 -exec sh -c 'rm -rf \"$1\"/*' {} \\;", True),
    ("find build -maxdepth 0 -exec sh -c 'rm -rf \"$1\"/*' _ {} \\;", False),
    ('f() { rm -rf "$1"/*; }; f', True), ('f() { rm -rf "$1"/*; }; f build', False),
    ('f() { rm -rf "$1"/*; }; f "$PWD"', True), ('f(){ kill -9 "$@"; }; f -1', True),
    ('f(){ kill -9 "$@"; }; f 1234', False),
    # a compound command after a proven cd runs whole or not at all
    ("cd /tmp/x && for i in 1 2; do rm -rf ./*; ./run.sh; done", False),
    ("cd build && if [ -f CMakeCache.txt ]; then rm -rf *; fi", False),
    ('cd /tmp/t && case "$1" in clean) rm -rf *;; esac', False),
    ("cd sub; { ls; rm -rf *; }", True), ("cd sub && ls; { rm -rf *; }", True),
    # ``cd DIR || exit``: the shell goes on only in DIR; any other exit proves nothing
    ("cd build || exit 1; rm -rf *", False), ("cd build || exit; rm -rf ./*", False),
    ("cd build || exit $?; rm -rf *", False), ("cd build || true; rm -rf *", True),
    ("cd build && exit 1; rm -rf *", True), ("cd build || exit $(rm -rf *)", True),
    ("cd build || exit 1 > /nonexistent/x; rm -rf *", True),
    ("cd build || exit 1 2; rm -rf *", True), ("true || cd build || exit 1; rm -rf *", True),
    ("cd build || exit 1; cd sub; rm -rf *", True),
    # a cd into a directory the command computes, or below the home directory
    ('tmp=$(mktemp -d) && cd "$tmp" && rm -rf ./*', False), ("cd ~/proj && rm -rf ./*", False),
    ('cd "$HOME/proj" && rm -rf ./*', False), ('cd "$UNSET" && rm -rf ./*', True),
    ('cd "$HOME" && rm -rf ./*', True), ("cd ~ && rm -rf *", True),
    ('tmp=$(mktemp -d) && cd "$tmp" && for i in 1 2; do rm -rf ./*; done', False),
    ('tmp=$(mktemp -d); cd "$tmp" && for i in 1 2; do rm -rf ./*; cd ..; done', True),
    # an operator of a parameter expansion, as bash reads it
    ('rm -rf "${HOME:?}"/*', True), ('rm -rf "${HOME%/}"', True),
    ('rm -rf "${HOME:-/nonexistent}"', True), ('rm -rf "${X:-$HOME}"', True),
    ('rm -rf "${OUT_DIR:-$PWD}"/*', True), ("rm -rf ${X:-~}/*", True),
    ('rm -rf ../"${PWD##*/}"', True), ('rm -rf "/tmp/${PWD##*/}"', True),
    ('rm -rf "${PWD%/*}"', True), ('rm -rf "/tmp/x/${PWD##*/}-build"', False),
    ('rm -rf "${OUT_DIR:-build}"/*', False),
    # a value the command assigns is the value the guard reads; the last one
    ('D=$HOME; rm -rf "$D"', True), ('D=~; rm -rf "$D"/*', True),
    ('x=$HOME/; rm -rf "$x"*', True), ('ROOT=$(pwd -P); rm -rf "$ROOT"/*', True),
    ('declare ROOT="$PWD"; rm -rf "$ROOT"/*', True), ('D="$(pwd)/"; rm -rf "$D"*', True),
    ('d=$(mktemp -d); d=$HOME; rm -rf "$d"', True), ("D=$HOME rm -rf build", False),
    ('D=$HOME/.cache/x; rm -rf "$D"', False), ('d=$(mktemp -d); rm -rf "$d" "$d"/*', False),
    # text handed to a shell: a decoded $'...', a shell in a group, printf's format
    ("eval $'rm -rf ~/*'", True), ("bash -c $'cd /\\nrm -rf ~'", True),
    ("sh <<< $'rm -rf ~'", True), ("trap $'rm -rf \"$PWD\"' EXIT", True),
    ("echo 'rm -rf ~' | (bash)", True), ("echo 'rm -rf ~' | { bash; }", True),
    ("printf \"%s\" 'rm -rf ~' | bash", True), ("printf 'rm -rf %s\\n' \"$HOME\" | sh", True),
    ("printf -- 'rm -rf ~\\n' | bash", True), ("printf 'rm -rf %s\\n' build out | sh", False),
    ("echo 'rm -rf ~' | busybox sh", True),
    # more commands that run their operands, or one word of shell text
    ("sudo su -c 'rm -rf ~'", True), ('setpriv rm -rf "$PWD"', True),
    ("uv run rm -rf /tmp/w", True), ("npx rm -rf /", True), ("tmux new -d 'rm -rf ~'", True),
    ("screen -dm bash -c 'rm -rf /'", True), ("tmux new -d 'make test'", False),
    ('uv run pytest -k "rm or kill"', False),
    # the directory before a cd, a pid with blanks, a number spelled as arithmetic, braces
    ('cd build && rm -rf "$OLDPWD"', True), ("cd build && rm -rf ~-", True),
    ("kill -9 ' -1'", True), ("kill -9 $((-1))", True), ("kill -9 -$((1))", True),
    ("kill -9 ' 1234'", False), ("kill -9 $((pid + 1))", False), ('{rm,-rf,"$PWD"}', True),
    # an arithmetic for is no command a list passes through unseen; an arithmetic command is
    ("for ((i=0; i<3; i++)); do echo $i; done; cd ./build && rm -rf *", False),
    ("cd build && ((x++)) && rm -rf *", True),
    # a NUL in a tilde word names no user (it raised ValueError)
    ("rm -rf ~a\x00b/x", False), ("rm -rf ~\x00/x", False),
    # review of FX05d: an assignment that may not run leaves the value before it possible
    ('ROOT=$PWD; if [ -d sub ]; then ROOT=sub; fi; rm -rf "$ROOT"/*', True),
    ('D=$HOME; ( D=build ); rm -rf "$D"', True), ('D=$HOME; D=build; rm -rf "$D"', False),
    ('D=$PWD; false && D=/tmp/x; cd "$D" && rm -rf *', True),
    ('D=build; unset D; rm -rf "$D"/*', True),
    # ``$PWD`` in an argument is the caller's directory
    ('f(){ cd /tmp/x && rm -rf "$1"/*; }; f "$PWD"', True),
    ('f(){ cd /tmp/x && rm -rf "$1"/*; }; f build', False),
    # ``${D:=word}`` assigns; without its colon an operator asks whether the variable is set
    (': "${BUILD_DIR:=$HOME}"; rm -rf "$BUILD_DIR"', True),
    (': "${OUT:=build}"; rm -rf "$OUT"', False), ('rm -rf "${X-$HOME}"', True),
    ('X=; rm -rf "${X+$HOME}"', True), ('X=; rm -rf "${X-$HOME}"', False),
    # the escapes of a printf format and of $'...', a shell fed through its group
    ("printf '\\162\\155 -rf %s\\n' \"$HOME\" | sh", True),
    ("printf '%s\\t-rf\\t/home/u\\n' rm | bash", True),
    ("bash -c $'echo ok\\cJrm -rf /home/u'", True), ("rm -rf $'\\x00'/", True),
    ("printf '%s\\n' 'rm -rf /home/u' | ( cat | bash )", True),
]


@pytest.mark.parametrize("command, refused", READ_BY_VALUE)
def test_destructive_guard_reads_values_arguments_and_exits(
    command: str, refused: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2026-11-21 of FX05c: an unset variable as a ``pkill`` pattern, a missing
    positional parameter, an operator on ``$HOME``, an assigned ``$HOME`` and text handed on
    as ``$'...'`` passed; a delete in a loop after ``cd DIR &&``, after ``cd DIR || exit`` and
    in a directory the command computed was refused; a NUL after a ``~`` raised."""
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("USERPROFILE", "/home/u")  # what expanduser reads on Windows
    verdict = destructive_command_guard(
        command, "/tmp/w", _any_directory, _unresolved, _unset_cdpath)
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command


FAIL_CLOSED = [
    # the review's repros: a value that may be one of several is each of them
    ('D=$HOME; false && D=build; E=$D; rm -rf "$E"', True),
    ('D=$HOME; false && D=build; D=$D; rm -rf "$D"', True),
    ('A=$HOME; B=; false && A=build; false && B=out; rm -rf "$A$B"', True),
    ('f(){ rm -rf "$2"; }; D=$HOME; false && D=build; f unused "$D"', True),
    ("D=$HOME; false && D=build; bash -c 'rm -rf \"$1\"' _ \"$D\"", True),
    ('ROOT=$HOME; : "${BUILD_DIR:=$ROOT}"; rm -rf "$BUILD_DIR"', True),
    ('BUILD_DIR=; : "${BUILD_DIR:=$HOME}"; rm -rf "$BUILD_DIR"', True),
    ('X=build; unset X; rm -rf "${X-$HOME}"', True),
    ('X=build; unset X; rm -rf "${X=$HOME}"', True),
    ('X=build; unset X; rm -rf "${X+$HOME}"', False),
    ("printf '\\162\\155 -rf /home/u\\n' | sh", True),
    ("printf '\\x72\\x6d -rf /home/u\\n' | bash", True),
    ("echo -e '\\x72m -rf /home/u' | bash", True),
    ("printf '%b -rf %s\\n' '\\0162\\0155' \"$HOME\" | bash", True),
    ("printf '%s\\n' 'echo ok\\nrm -rf /home/u' | bash", False),
    ('false && D=build; rm -rf "$D"/*', True),
    ("D=build; bash -c 'rm -rf \"$D\"/*'", True),
    ("D=$HOME bash -c 'rm -rf \"$D\"'", True),
    ('dirs=("$HOME" build); rm -rf "${dirs[@]}"', True),
    ('cleanup(){ rm -rf "$1"; }; trap "cleanup $HOME" EXIT', True),
    ('f(){ shift; rm -rf "$1"; }; f x "$HOME"', True),
    ('set -- "$HOME"; rm -rf "$1"', True),
    ('for i in 1 2; do rm -rf "$D"; D=$HOME; done', True),
    ("$SUDO rm -rf /", True), ("{r,}m -rf /", True),
    # what the everyday forms of the same commands keep
    ('D=build; rm -rf "$D"', False), ("kill $(cat x.pid)", False),
    ("S=tmp/x && rm -rf $S/png $S/*.pdf", False), ("cd /tmp/x && rm -rf *", False),
    ('pids=""; for i in 1 2; do sleep 1 & pids="$pids $!"; done; kill $pids', False),
    # the branches of an if, a list and a condition that stop midway
    ('if true; then D=$HOME; else D=out; fi; rm -rf "$D"', True),
    ('true && D=$HOME && false && D=build; rm -rf "$D"', True),
    ('if D=$HOME; false; then D=build; fi; rm -rf "$D"', True),
    ('if true; then D=a; rm -rf "$D"/*; else D=b; fi', False),
    ('case $x in a) D=$HOME ;; *) D=out ;; esac; rm -rf "$D"', True),
    # the parameters a for loops over, an indirect value, a ~ after a colon
    ('set -- "$HOME"; for a; do rm -rf "$a"; done', True),
    ('A=$HOME; n=A; rm -rf "${!n}"', True), ('A=build; n=A; rm -rf "${!n}"', False),
    ("d=x:~; rm -rf ${d#x:}", True),
    # text the call writes for a shell, a source or xargs
    ("printf 'r%s -rf ~\\n' m | sh", True), ("{ echo -n r; echo 'm -rf ~'; } | sh", True),
    ('rm -rf "$(echo "$HOME")"', True), ("source /dev/stdin <<< 'rm -rf ~'", True),
    ('echo "$HOME" | xargs rm -rf', True),
    ("echo '; rm -rf ~' | xargs -I{} sh -c 'echo {}'", True),
    ("printf '%s\\n' a b | xargs -P 4 -I{} sh -c 'node test/{}.test.js > logs/{}.log'", False),
    ('read X <<< "rm -rf $HOME"; $X', True), ("alias x='rm -rf ~'; x", True),
    # what a program prints that was told a protected directory; else it is unknown
    ('rm -rf "$(cd ~ && pwd)"', True), ('X=$(printenv HOME); rm -rf "$X"', True),
    ('X=`cd ~ && pwd`; cd "$X" && rm -rf ./*', True), ('rm -rf "$(dirname "$HOME/x")"', True),
    ('T=$(mktemp -d "$PWD/x.XXXXXX"); rm -rf "$T"', False),
    ('PY="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/bin"; "$PY/python" -V', False),
    ("sh -c \"out=\\$(docker exec c sh -c 'hostname; pwd'); echo x >> $S/a.txt\"", False),
    ("kill $(cat ~/x.pid)", False),
    # review round 1 of FX05e: what bash does that the guard did not model
    ('D=build; declare -n R=D; R=$HOME; rm -rf "$D"', True),
    ('f(){ local -n R=$1; R=$HOME; }; f D; rm -rf "$D"', True),
    ('set -o posix; D=build; D=$HOME :; rm -rf "$D"', True),
    ('X=$HOME; unset -f X; rm -rf "$X"', True),
    ('readonly X=$HOME; unset X; rm -rf "$X"', True),
    ('IFS=:; D="build:$HOME"; rm -rf $D', True),
    ('IFS=:; D="build:$HOME"; bash -c \'rm -rf "$2"\' _ $D', True),
    ('bash -c \'IFS=:; D="build:$HOME"; rm -rf $D\'', True),
    ("HOME=/tmp/x/..; rm -rf ~/w", True),
    ('X=.; rm -rf "${X,}"/*', True),
    ('X=\'$HOME\'; rm -rf "${X@P}"', True),
    (': "${B:=${X:-${HOME}}}"; rm -rf "$B"', True),
    ('for i in 1 2; do [ $i = 2 ] && f "$HOME"; f(){ rm -rf "$1"; }; done', True),
    ("printf 'r%sm -rf /home/u\\n' '' | /bin/ba?h", True),
    ("printf '\\162\\155 -rf /home/u\\n' | rbash", True),
    ("$SHELL -c 'r''m -rf /home/u'", True),
    ("printf 'r%sm -rf /home/u\\n' '' | $BASH", True),
    ('exec 3<<<"$HOME"; read -r D <&3; rm -rf "$D"', True),
    ("BASH_ENV=<(printf 'r%sm -rf %s\\n' '' /home/u) bash -c :", True),
    ("BASH_ENV=/dev/stdin bash -c : <<< 'rm -rf /home/u'", True),
    (". <(printf 'r%sm -rf /home/u\\n' '')", True),
    ("PS4='$(rm -rf ~)'; set -x; :", True),
    ("env PS4='$(rm -rf ~)' bash -xc :", True),
    ("X='a[$(rm -rf ~)]'; : $((X))", True),
    ("read -r X <<< 'a[$(rm -rf ~)]'; : $((X))", True),
    ("shopt -s extglob\nrm -rf /tmp/w/@(*)", True),
    ('shopt -s extglob\nD=/tmp/w; rm -rf "$D"/@(*)', True),
    ("cd /tmp/w/..; shopt -s nocaseglob; rm -rf [W]", True),
    ('read X <<< "rm -rf $HOME"; bash <<EOF\n$X\nEOF', True),
    ('X=\'$HOME\'; bash <<EOF\nrm -rf "$X"\nEOF', True),
    # a lone surrogate is a name like any other, not a crash
    ("rm -rf $'\\ud800'/x", False),
    # their everyday counterparts, and shapes of real commands those rules once refused
    ("IFS=, read -r a b <<< 'x,y'; rm -rf build", False),
    ('while IFS= read -r line; do rm -f "$line.tmp"; done < list.txt', False),
    ("declare -a parts=(a b); rm -rf build", False),
    ("shopt -s extglob; rm -rf build/*.o", False),
    ('case "$c" in *"(deleted)") rm -rf build/*;; esac', False),
    ("exec 3>log.txt; echo hi >&3; rm -rf build", False),
    ("find build -maxdepth 1 -name '*.o' -exec rm -f {} +", False),
    ("find /home/u -maxdepth 4 -not -path '/home/u/w/*' -not -path '/home/u'", False),
    ("unset -f helper; rm -rf build", False),
    ('readonly OUT=build; rm -rf "$OUT"', False),
    ("PS4='+ '; set -x; rm -rf build", False),
    ("python3 tool.py '`kill -1`' 300", False),
    ('bash -c \'IFS=,; echo x\'; D=build; rm -rf "$D"', False),
    ("for c in 'IFS=marker; echo x' 'declare -u T; T=x'; do rm -rf ./build/*; bash -c \"$c\"; done",
     False),
    ("BASH_ENV=$PWD/startup.sh bash -c 'echo hi'; rm -rf build", False),
    ("PY=python3; echo 'print(1)' | $PY; rm -rf build", False),
    # review round 2 of FX05e: more of what bash does behind the guard's back
    ("pushd /tmp/x >/dev/null; rm -rf ~1/*", True),
    ("rm -rf ../[[.w.]]", True),
    ("rm -rf ../[^a-v]", True),
    ('rm -rf $"$HOME"', True),
    ('[[ $HOME =~ .* ]]; rm -rf "$BASH_REMATCH"', True),
    ('getopts d: o -d "$HOME"; rm -rf "$OPTARG"', True),
    (': ${A[0]:=$HOME}; rm -rf "$A"', True),
    ('E=; ${E:-r""m} -rf ~', True),
    ("set -- rm; ${@:1} -rf ~", True),
    ("printf 'r%sm -rf %s\\n' '' /home/u | ${S:-bash}", True),
    ("exec <<< 'r\"\"m -rf ~'; bash", True),
    ("xargs -a <(echo ~) rm -rf", True),
    ("echo ~ | xargs sh -c 'rm -rf \"$1\"' _", True),
    ('command_not_found_handle(){ rm -rf "$2"; }; zzzq "$HOME"', True),
    ("X=$(printf 'a[$(r%sm -rf ~)]' ''); : $((X))", True),
    ("printf -v PS4 '$(r%sm -rf ~)' ''; set -x; :", True),
    ("env 'BASH_FUNC_f%%=() { r\"\"m -rf ~; }' bash -c f", True),
    ("kill -9 $[-1]", True),
    ("rm -rf build/[[:digit:]]*.log", False),
    ('[[ $f =~ ^(.*)\\.o$ ]] && echo "${BASH_REMATCH[1]}"; rm -rf build', False),
    ("exec 3< list.txt; bash tools/run.sh; rm -rf build", False),
    ("printf '%s\\n' a b | xargs -I{} sh -c 'f={}; s=$(date +%s); echo \"$f $s\"'", False),
    ("for p in 'local `install.sh`' VSIX; do grep -c \"$p\" notes.md; done; rm -rf build", False),
    ('echo "cost: 5$"; rm -rf build', False),
    ("printf '%s\\n' 'n=$(awk \"NR % 2\" list.txt)' > run.sh; rm -rf build", False),
    # a name of the call's own for a program: hash -p, an alias, a handler, an imported function
    ('hash -p "$(type -P rm)" zz; zz -rf ~', True),
    ("shopt -s expand_aliases; alias e=env; e rm -rf ~", True),
    ('mapfile -C "rm -rf ~ #" -c 1 A < /etc/hostname', True),
    ('command_not_found_handle(){ rm -rf "$2"; }; nosuch -x ~', True),
    ("env 'BASH_FUNC_f%%=() { rm -rf ~; }' bash -c f", True),
    ('T="rm -rf /home/u"; env -S "$T"', True),
    # text the call writes, on the standard input of a shell or of xargs
    ("{ bash; } <<'EOF'\nrm -rf ~\nEOF", True),
    ("f(){ bash; }; f <<'EOF'\nrm -rf ~\nEOF", True),
    ("( cat | bash ) <<'EOF'\nrm -rf ~\nEOF", True),
    ("{ bash; } <<< 'rm -rf ~'", True),
    ("cat <<< 'rm -rf ~' | bash", True),
    ("xargs rm -rf < <(echo ~)", True),
    ("while read -r l; do :; done <<'EOF'\nrm -rf ~\nEOF\nrm -rf build", False),
    ("{ python3 -; } <<'EOF'\nprint(1)\nEOF\nrm -rf build", False),
    # arithmetic assigns: a command, an expansion, a subscript, a here-document, an integer
    ("(( P=-1 )); kill -9 $P", True),
    ("P=0; (( P-- )); kill -9 $P", True),
    ("let P=-1; kill -9 $P", True),
    ("declare -i P; P=0-1; kill -9 $P", True),
    ("A=P; B==-1; (( $A$B )); kill -9 $P", True),
    (": ${X[P=-1]}; kill -9 $P", True),
    ("[[ -v A[P=-1] ]]; kill -9 $P", True),
    ("printf -v 'A[P=-1]' x; kill -9 $P", True),
    (": <<EOF\n$((P=-1))\nEOF\nkill -9 $P", True),
    ("B=([P=-1]=x); kill -9 $P", True),
    ("X='P=-1'; (( X )); kill -9 $P", True),
    ("OPTIND='P=-1'; kill -9 $P", True),
    ("kill -9 -$[1]", True),
    ("P=1; kill -9 -$((P))", True),
    ("bash <<'EOF'\n(( P=-1 )); kill -9 $P\nEOF", True),
    ("cat <<'EOF' >/dev/null\n(( P=-1 ))\nEOF\n(( P=-1 ))\nkill -9 $P", True),
    ("cat <<'EOF' >/dev/null\n(( P=-1 ))\nEOF\nkill -9 $P", False),
    ("X=$(echo ']) ~ fr- mr($[a' | rev); : $((X))", True),
    ('declare -i count=0; rm -rf "build-$count"', False),
    ('(( i++ )) || true; D=build; rm -rf "$D"', False),
    ('v=$(cat VERSION); IFS=. read -r major minor patch <<< "$v"; echo "$((minor + 1))"; '
     "rm -rf build", False),
    # what looks like arithmetic in a string or in a here-document that is data
    ("awk '{ if (($1 + $2) > 3) n++ } END { print n }' f; PID=$(cat x.pid); kill $PID", False),
    ("cat > f.py <<'EOF'\nif ((a) and b):\n    x[i] = `y`\nEOF\n"
     "PID=$(cat x.pid); kill $PID", False),
    ("python3 - <<'EOF'\nprint('cat <<EOF')\nx = '$((('\nEOF\n"
     "PID=$(cat x.pid); kill $PID", False),
    # values: a substitution that prints itself, basename, a nested string, an element
    ('X=\'$(echo "$X")\'; rm -rf "$(echo "$X")"', True),
    ('cd ..; D=$(basename /x/w); rm -rf "$D"', True),
    ('rm -rf "${E:-$"$HOME"}"', True),
    ('A[${#A[@]}]=$HOME; rm -rf "${A[0]}"', True),
    ('rm -rf "$(git rev-parse --show-toplevel 2>/dev/null || echo .)/build"', False),
    ('files=(a.o b.o); rm -f "${files[@]}"', False),
    # review round 3: a number bash counts itself, a slice, an empty pkill pattern
    ("false; kill -9 -$?", True), ("X=a; kill -9 -${#X}", True),
    ("set -- a; kill -9 -$#", True), ("kill -9 -$LINENO", True),
    ("X=-1; kill -9 ${X:0}", True), ('rm -rf "/home/u${?:1}"', True),
    ('pkill -9 "${?:1}"', True), ("r${?:1}m -rf ~", True),
    ('echo "$? $# ${#HOME} ${HOME:0:3}"; kill $!', False),
    # text on an input: read by a substitution, a function or a shell given as an argument
    ('echo ~ | rm -rf "$(cat)"', True), ('{ rm -rf "$(cat)"; } <<< ~', True),
    ('f(){ rm -rf "$(cat)"; }; echo ~ | f', True),
    ('f(){ local d; read -r d; rm -rf "$d"; }; echo ~ | f', True),
    ("f(){ \"$@\"; }; f bash <<< 'rm -rf ~'", True),
    ("{ bash 0<&3; } 3< <(echo 'rm -rf ~')", True),
    ("echo 'rm -rf ~' | sh -c 'eval \"$(cat)\"'", True),
    ('exec 3<<<"$HOME"; rm -rf $(cat /dev/fd/3)', True),
    ('exec <<<"$HOME"; rm -rf "$(cat)"', True),
    ('exec 3<<<"$HOME"; xargs rm -rf <&3', True),
    # bash expands a command's words before it makes the command's own redirections
    ('"$(sed -n p conf)python" - <<\'EOF\'\nprint(1)\nEOF', False),
    ('echo x | rm -rf "build/$(date +%F)"', False),
    # values: the name of the directory, a pwd under another spelling, what it sets itself
    ('rm -rf "../$(basename "$PWD")"', True), ('rm -rf "$(cd; p""wd)"', True),
    ('rm -rf "$(basename "$PWD").tar.gz"', False),
    ('rm -rf "../$(basename "$PWD")-backup"', False),
    ('rm -rf "$(for d in /home/u; do echo $d; done)"', True),
    ('rm -rf "$(set -- /home/u; echo $1)"', True),
    ('X=$HOME; unset -n X; rm -rf "$X"', True),
    (': ${A:-${B:-${C:=$HOME}}}; rm -rf "$C"', True),
    ("\"$(which bash)\" -c 'rm -rf ~'", True),
    # a text that follows several ``-c`` is read once (audit 2026-11-22), and every text is read
    ("\"$(which bash)\" -c 'echo hi' -c 'echo hi' -c 'rm -rf ~'", True),
    ("\"$(which bash)\" -c 'echo hi' -c 'echo hi'", False),
]


@pytest.mark.parametrize("command, refused", FAIL_CLOSED)
def test_destructive_guard_fails_closed(
    command: str, refused: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2026-11-21 of FX05d (second round): every repro had a close variant that
    still deleted ``$HOME``, because a value the guard could not tell was read as one of
    its values.  A delete or a signal now runs only when every text its words may be is
    safe: several values are each of them, and what the guard cannot bound is refused.
    Review of FX05e (round 1): so is what bash does that the guard does not model (a name
    reference, a changed ``IFS``, a case or transform operator, an extended pattern, a
    startup file or an input the call writes, a quoted substitution bash expands later).
    Round 2: a directory-stack tilde, a bracket class, ``$".."`` and ``$[..]``, the variables
    bash sets itself, a command word in a default word, what ``xargs`` adds to a text.
    Round 3: a number bash counts itself, a slice, text on an input that a substitution, a
    function or a shell given as an argument reads, the name of the working directory."""
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("USERPROFILE", "/home/u")  # what expanduser reads on Windows
    verdict = destructive_command_guard(
        command, "/tmp/w", _any_directory, _unresolved, _unset_cdpath)
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command


def test_destructive_guard_reads_here_documents_and_unclosed_expansions_once() -> None:
    """Review 2026-11-21 of FX05c (item 12): each here-document its shell expands made the
    guard scan the whole command again: 1.5 s for 150 of them (2 kB), quadratic from there.
    They are read once, and still read.  Review of FX05d: an unclosed ``${X:-`` before n
    ``${a}`` took 2**n steps (4 s for 22 of them); it is an expansion the guard cannot
    read, so the delete is refused.

    The bounds are on the guard's own time, CPU time as in :func:`guards_stay_linear`: the
    wall clock also counts the wait for a core, and in a parallel run of the suite it
    showed more than 10 s for the 1500 here-documents (2 to 6 s of work on the machines
    of the full run 2026-11-22).  Scanned again for each one, they would take minutes.

    Run 2026-11-22: the scan still looked every here-document up in the list of the shells
    that read one, as long as the command has such shells: 200 of them cost 2.6 times the
    calls of 100.  They cost twice the calls now, which is counted: a number the load of
    the machine does not change."""
    started = time.thread_time()
    assert destructive_command_guard(
        'rm -rf "${X:-' + "${a}" * 12000 + '"', "/tmp/w",
        _any_directory, _unresolved, _unset_cdpath) == DESTRUCTIVE_VERDICT
    assert time.thread_time() - started < 10
    blocks = "bash <<E\n$x\nE\n" * 1500
    started = time.thread_time()
    assert destructive_command_guard(
        blocks, "/tmp/w", _any_directory, _unresolved, _unset_cdpath) is None
    assert time.thread_time() - started < 20
    (few_verdict, few), (many_verdict, many) = (
        _guard_calls("bash <<E\n$x\nE\n" * count) for count in (100, 200))
    assert few_verdict is None and many_verdict is None
    assert many < 2.2 * few, (few, many)
    assert destructive_command_guard(
        blocks[:700] + "bash <<E\nrm -rf $HOME\nE\n", "/tmp/w",
        _any_directory, _unresolved, _unset_cdpath) == DESTRUCTIVE_VERDICT


def test_bash_keeps_a_work_dir_whose_name_needs_quotes(tmp_path: Path) -> None:
    """The reviewers' repros, end to end: the ``Bash`` tool refuses both quote gaps for a work
    directory with a blank in its name, and the directory keeps its file."""
    work = tmp_path / "gap 1 work"
    work.mkdir()
    (work / "keep.txt").write_text("kept")
    tools = UsefulTools(work_dir=str(work))
    for command in (f'X="a b" rm -rf {shlex.quote(str(work))}',
                    f'rm -rf "x # y" {shlex.quote(str(work))}'):
        assert tools.Bash(command, "wipe") == DESTRUCTIVE_VERDICT, command
    assert (work / "keep.txt").read_text() == "kept"


def test_destructive_guard_refuses_what_nests_too_deep_to_read() -> None:
    """A text within texts past what the guard rescans is refused, not waved through.

    Nor is one whose values the scanner does not read to their end (audit 2026-11-22):
    the verdict is the scanner's own.
    """
    command = "rm -rf $X/*"
    for _ in range(6):
        command = "eval " + shlex.quote(command + ' "$Y"')
    assert destructive_command_guard(command, "/tmp/w") == TOO_DEEP_VERDICT
    assert destructive_command_guard("echo rm " + "(" * 60, "/tmp/w") == TOO_DEEP_VERDICT
    doubled = "; ".join(f'X{i}="$X{i - 1}$X{i - 1}"' for i in range(1, 61))
    command = f'X0=aaaaaaaa; {doubled}; rm -rf "$X60"'
    assert destructive_command_guard(command, "/tmp/w") == UNREAD_VALUES_VERDICT


@pytest.mark.parametrize("command, refused", [
    # the cd stays in the work dir, goes above or beside it, or names no literal directory
    ("cd . && rm -rf *", True), ("cd ./ && rm -rf *", True), ("cd sub/.. && rm -rf ./*", True),
    ("cd /tmp/w && rm -rf *", True), ("cd ../w && rm -rf *", True), ("cd .. && rm -rf *", True),
    ("cd /tmp && rm -rf *", True), ("cd /. && rm -rf *", True), ("cd / && rm -rf *", True),
    ("cd // && rm -rf *", True), ("cd /tmp/w2 && rm -rf ../w", True),
    ("cd ../w2 && rm -rf ../w *", True), ("cd $PWD && rm -rf *", True),
    ('cd "$BUILD" && rm -rf *', True), ("cd ~ && rm -rf *", True), ("cd && rm -rf *", True),
    ("cd - && rm -rf *", True), ("cd -P sub && rm -rf *", True), ("pushd -n sub && rm -rf *", True),
    ("pushd +1 && rm -rf *", True), ('cd "" && rm -rf *', True), ("cd sub extra && rm -rf *", True),
    # the cd may not run, may fail, or runs in a child
    ("true || cd /tmp; rm -rf *", True), ("true || cd sub && rm -rf *", True),
    ("cd sub; rm -rf *", True), ("pushd sub; rm -rf ./*", True), ("cd sub || rm -rf *", True),
    ("! cd sub && rm -rf *", True), ("false | cd sub && rm -rf *", True),
    ("true |& cd sub && rm -rf *", True), ("(cd sub) && rm -rf *", True),
    ("cd sub & rm -rf *", True), ("sudo cd sub && rm -rf *", True),
    ("bash -c 'cd sub' && rm -rf *", True), ("true ||\ncd sub && rm -rf *", True),
    ("echo \\; cd sub && rm -rf *", True),
    ("echo x >& cd sub && rm -rf *", True), ("echo `date` cd sub && rm -rf *", True),
    ("until cd sub; do rm -rf *; done", True),
    # another command between the cd and the delete, or the delete after the cd's list
    ("cd sub && cd .. && rm -rf *", True), ("cd / && cd tmp && rm -rf *", True),
    ("pushd sub && popd && rm -rf *", True), ("cd sub && env -C .. rm -rf *", True),
    ("cd sub && rm -rf *; rm -rf *", True), ("cd sub && ls; rm -rf *", True),
    # a cd named before the one that enters: a function of that name, a cd the shell evaluates
    ("cd() { :; }; cd sub && rm -rf *", True), ("eval 'cd ..'; cd w && rm -rf *", True),
    # entered: a literal directory away from the work dir, right before the delete
    ("cd sub && rm -rf *", False), ("cd ./sub/ && rm -rf ./*", False),
    ('cd "sub" && rm -rf "$PWD"', False), ("pushd sub && rm -rf ./*", False),
    # FX05c: a ``cd`` after a new line starts a command like any other, and proves the delete
    ("ls\ncd sub && rm -rf *", False), ("mkdir -p sub && cd sub && rm -rf *", False),
    # FX05c: only ``&&`` proves a ``cd``; the scanner does not tell the branches of an ``if``
    ("if cd sub; then rm -rf \"$PWD\"; fi", True),
    ("ls; cd sub && rm -rf *", False), ("(cd sub && rm -rf *)", False),
    ("{ cd sub && rm -rf *; }", False), ("cd sub && rm -rf ../w", False),
    ("cd /tmp/w/build && rm -rf *", False), ("cd /var/tmp/x && rm -rf *", False),
    ("cd sub && rm -rf * && cd .. && ls", False),
])
def test_delete_from_inside_is_exempt_only_once_a_cd_left_the_work_dir(
    command: str, refused: bool,
) -> None:
    """Review 2026-11-11 S4-2: any ``cd`` or ``pushd`` before a delete named from inside the
    work dir (``rm -rf *``) exempted it, so ``cd . && rm -rf *``, ``true || cd /tmp; rm -rf *``
    and ``cd build; rm -rf *`` with no ``build`` emptied the work dir.  The delete is exempt
    only as a plain ``rm`` right after ``cd DIR &&`` (or ``cd DIR; then``), where the ``cd``
    starts a command of the shell that runs the ``rm`` and DIR is a literal path that is
    neither the work dir, nor above it, nor beside it (where ``../<name>`` is the work dir)."""
    verdict = destructive_command_guard(command, "/tmp/w")
    assert (verdict == DESTRUCTIVE_VERDICT) is refused, command


def test_delete_from_inside_follows_the_links_of_the_entered_directory(tmp_path: Path) -> None:
    """Audit 2026-11-12 B5/A5: the directory a ``cd`` enters was read from the command's text
    alone, so ``cd alias && rm -rf *`` through a link back to the work dir emptied it.  It is
    also taken as the filesystem names it; bash says where each ``cd`` lands (no delete runs)."""
    bash = find_bash()
    if bash is None:
        pytest.skip("Bash is the filesystem oracle")
    work = tmp_path / "w"
    (work / "sub").mkdir(parents=True)
    (tmp_path / "other" / "deep").mkdir(parents=True)
    (work / "alias").symlink_to(work, target_is_directory=True)
    (work / "up").symlink_to(tmp_path, target_is_directory=True)
    (work / "sub" / "back").symlink_to("..", target_is_directory=True)
    (work / "away").symlink_to(tmp_path / "other", target_is_directory=True)
    (work / "far").symlink_to(tmp_path / "other" / "deep", target_is_directory=True)
    (tmp_path / "other" / "x").symlink_to(work, target_is_directory=True)
    (tmp_path / "other" / "y").symlink_to(work, target_is_directory=True)
    (work / "y").mkdir()
    for entered, operand, refused in (
        ("alias", "*", True), ("up", "*", True), ("sub/back", "./*", True), ("up/w", "*", True),
        # no ``w/x``: bash falls back to the path as the filesystem takes it, ``other/x``
        ("far/../x", "*", True),
        # ``..`` of the directory the link names holds the work dir
        ("away", "../w", True),
        ("sub", "*", False), ("alias/sub", "*", False), ("far", "*", False),
        # ``w/y`` is a directory: bash enters it, and never looks at ``other/y``
        ("far/../y", "*", False),
    ):
        deleted = "../w" if operand == "../w" else "."
        pwd = "pwd -W" if sys.platform == "win32" else "pwd -P"
        landed_text = subprocess.run(
            [bash, "-c", f"cd {entered} && cd -P {deleted} && {pwd}"],
            cwd=work, check=True, capture_output=True, text=True,
        ).stdout.strip()
        landed = Path(landed_text).resolve()
        assert work.resolve().is_relative_to(landed) is refused, (entered, landed)
        verdict = destructive_command_guard(f"cd {entered} && rm -rf {operand}", str(work))
        assert (verdict == DESTRUCTIVE_VERDICT) is refused, entered
    if sys.platform != "win32":
        # A backslash is a file-name character on POSIX, not a separator.
        odd = tmp_path / "w\\z"
        odd.mkdir()
        (odd / "alias").symlink_to(odd, target_is_directory=True)
        verdict = destructive_command_guard("cd alias && rm -rf *", str(odd))
        assert verdict == DESTRUCTIVE_VERDICT


def test_a_cd_that_cdpath_may_redirect_exempts_no_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-21 S12/F1: bash looks a ``cd`` operand that neither ``/`` nor a ``.`` or
    ``..`` starts up in ``CDPATH`` first, so ``CDPATH=..; cd w && rm -rf *`` run in a work dir
    named ``w`` stays there and emptied it, exempt as a delete in ``w/w``.  Such a ``cd``
    proves nothing once the command names ``CDPATH`` or its shell inherits one; bash says
    where each ``cd`` lands (no delete runs)."""
    bash = find_bash()
    if bash is None:
        pytest.skip("Bash is the filesystem oracle")
    work = tmp_path / "w"
    (work / "sub").mkdir(parents=True)
    keep = work / "keep.txt"
    keep.write_text("keep")
    pwd = "pwd -W" if sys.platform == "win32" else "pwd -P"
    for entered, inherited, refused in (
        ("CDPATH=..; cd w", "", True), ("export CDPATH=..; pushd w", "", True),
        ("cd w", "..", True), ("pushd w", "elsewhere:..", True),
        # an operand ``CDPATH`` is not searched for, and one searched with none set
        ("CDPATH=..; cd ./sub", "", False), ("cd ../w/sub", "..", False),
        ("cd sub", "", False),
    ):
        monkeypatch.setenv("CDPATH", inherited)
        landed_text = subprocess.run(
            [bash, "-c", f"{entered} && {pwd}"],
            cwd=work, check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[-1]
        assert (Path(landed_text).resolve() == work.resolve()) is refused, (entered, landed_text)
        verdict = destructive_command_guard(f"{entered} && rm -rf *", str(work))
        assert (verdict == DESTRUCTIVE_VERDICT) is refused, (entered, inherited)
    # The Bash tool's shell inherits this process's ``CDPATH``; another environment's tools
    # pass the one their shell inherits.
    monkeypatch.setenv("CDPATH", "..")
    tools = UsefulTools(work_dir=str(work))
    assert tools.Bash("cd w && rm -rf *", "wipe") == DESTRUCTIVE_VERDICT
    assert keep.read_text() == "keep"
    (work / "sub" / "junk.txt").write_text("junk")
    assert tools.Bash("cd ./sub && rm -rf * && echo emptied", "allowed").strip() == "emptied"
    assert not (work / "sub" / "junk.txt").exists() and keep.read_text() == "keep"
    assert destructive_command_guard("cd w && rm -rf *", str(work), cdpath="".strip) is None
    monkeypatch.delenv("CDPATH")
    assert refuse_or_lift("cd w && rm -rf *", str(work), 30, cdpath="..".strip) == (
        DESTRUCTIVE_VERDICT, 30)


def test_guards_stay_linear_on_long_commands() -> None:
    """Repeated wrappers and assignments must not make the patterns backtrack exponentially.

    A 20 kB command of repeated ``sudo -n`` (or ``X=1``) words ran for over a
    second with nested wrapper repetitions; the single-wrapper prefix keeps it
    in milliseconds.  A 40 kB command of separators (``:;:;...``; audit
    2026-11-05 A09-4) ran for seconds, since from every separator the
    directory prefix and an assignment's value scanned the rest of the text
    for a ``/`` or a blank; they stop at the next separator.  A wrapper followed
    by thousands of ``rm`` (``kill``, ``pkill``, ``apt``...) words (audit
    2026-11-08 S7/F2) ran for seconds, quadratic in the command length: every
    word after the wrapper was a candidate command and each candidate rescanned
    the whole suffix for a protected operand; the wrapper's arguments now end at
    the first occurrence of the command the pattern matches (review 2026-11-08 R7
    S7/F2: whatever other guarded names its options hold, ``env -u kill -u pkill
    …``), so each suffix is scanned once per guarded command.  The shell's options
    after its ``-c`` cluster (audit 2026-11-08 S8) end at the first, possibly
    quoted, command name the same way, and the cluster's ``c`` is its first
    ``c``, so a 48 kB cluster is read once rather than once per split.  The
    reserved words a command may follow (audit 2026-11-10 C13-F3) repeat
    without a rescan either.  Nor do line continuations (audit 2026-11-21 S12/F4),
    which the guards remove before a second and a third reading, nor the quotes they
    remove for three more (review 2026-11-21 of S12): a quote nothing closes is
    scanned for once.  A long value costs its length in every word that expands it, and
    values built from one another double (audit 2026-11-22): the scan reads a megabyte of
    them, then the command is one it does not read.  The time is the guards' own, as CPU
    time tells it (:func:`guards_stay_linear`).
    """
    long_commands = [
        "sudo echo " + "rm " * 8000 + "ok",
        "sudo echo " + "kill " * 6000 + "ok",
        "sudo echo " + "pkill " * 5000 + "ok",
        "sudo echo " + "/bin/rm " * 4000 + "ok",
        "sudo echo " + "rm " * 8000 + "ok; echo /",
        "env " + "-u kill -u pkill -u rm " * 2000 + "ok",
        "env " + "-u kill " * 4000 + "rm -rf ./x",
        "bash " + "-c rm " * 5000 + "ok",
        "bash -c " + "'rm " * 8000 + "ok",
        "bash -c -e " + "x 'rm " * 4000 + "ok",
        "bash " + "-co posix " * 4000 + "ok",
        "bash -" + "c" * 48_000 + "1 x",
        "bash -c -e " + "echo hello " * 4400,
        "sudo echo " + "apt " * 6000 + "ok",
        "sudo echo " + "npm " * 6000 + "ok",
        "sudo -n " * 2499 + "true #xx",
        "sudo -n " + "X=1 " * 4996 + " true #x",
        "kill " + "-9 " * 5000 + "1234",
        "pkill " + "-f " * 3000 + "myserver",
        "rm -rf " + "./a " * 4000 + "./b",
        "npm " + "--flag " * 3000 + "test",
        ":;" * 20_000,
        ";A=" * 13_333,
        ";/" * 20_000,
        "|" * 40_000,
        "(" * 40_000,
        "{ " * 20_000,
        "{" * 40_000,
        "! " * 20_000,
        "then " * 8_000,
        "; if ! " * 5_000,
        "{ do rm " * 5_000,
        "; function f { coproc " * 4_000,
        "rm -rf " + "*x " * 13_000,
        "cd a; " * 6_000 + "rm -rf *x",
        "cd a && " * 6_000 + "rm -rf *",
        "cd a && rm -rf *; " * 3_000,
        "; cd " + "a/" * 10_000 + "$x && rm -rf *",
        "kill -9 " + '"-1"x ' * 6_000,
        "rm -rf " + "a \\\n" * 13_000 + "b",
        "# c \\\n" * 8_000 + "rm -rf x",
        "cd a && \\\n" * 5_000 + "rm -rf *",
        ";#\\\n" * 10_000,
        "rm -rf " + '"a b" ' * 6_600 + "x",
        'X="a b" ' * 5_000 + "true",
        "rm '" + ";rm '" * 8_000,
        "rm " + '"\\' * 20_000,
        "rm " + "$'" * 20_000,
        "rm " + "'\"" * 20_000,
        "rm -rf " + "a\\ " * 13_000,
        'A="' * 13_000,
        "X=" + "a" * 20_000 + "; echo" + ' "$X"' * 4_000,
        "X=" + "a" * 20_000 + "; " + "Y=$X;" * 4_000,
        "X0=aaaaaaaa; " + "; ".join(f'X{i}="$X{i - 1}$X{i - 1}"' for i in range(1, 61)),
        "a;" * 20_000 + "rm -rf /",
    ]
    for command in long_commands:
        assert guards_stay_linear(command, 1), command[:40]
    assert destructive_command_guard(long_commands[-1], "/work") == DESTRUCTIVE_VERDICT
    # A long argument list before the protected operand is still read through.
    assert destructive_command_guard("sudo rm -rf " + "./a " * 8000 + "/", "/work")
    assert lift_install_timeout("sudo pip " + "--flag " * 8000 + "install x", 30) == 900


def test_install_timeout_lift() -> None:
    """Installs and builds get at least 900 s; other commands keep their timeout."""
    for command in ["pip install requests", "python3 -m pip install -e .", "uv sync",
                    "uv pip install x", "sudo apt-get install -y gcc", "npm install",
                    "pnpm add lodash", "cargo build --release", "cd src && make -j8",
                    "cmake --build .", "go build ./...", "npm ci", "npm run build",
                    "echo prepare\nmake", "sudo -n apt-get install gcc",
                    "pip --cache-dir /tmp/pip install requests", "npm --prefix app install",
                    "sudo -u root -- apt install -y git",
                    # audit 2026-11-21 S12/F4: across a line continuation
                    "pip \\\ninstall sample", "sudo apt-get \\\n  install -y gcc", "ma\\\nke"]:
        assert lift_install_timeout(command, 30) == INSTALL_TIMEOUT_SECONDS, command
        assert lift_install_timeout(command, 1800) == 1800, command
    for command in ["uv run pytest", "npm get registry", "grep -R make .", "go test ./...",
                    "pip list", "echo make", "uv run pytest tests/test_build.py",
                    "npm test # not a build", "go run build.go", "pip \\\nlist"]:
        assert lift_install_timeout(command, 30) == 30, command


def test_refuse_or_lift_combines_both_guards(tmp_path: Path) -> None:
    """``refuse_or_lift`` returns the verdict and the lifted deadline in one call."""
    wd = str(tmp_path)
    assert refuse_or_lift(f"rm -rf {tmp_path.as_posix()}", wd, 30) == (DESTRUCTIVE_VERDICT, 30)
    assert refuse_or_lift("pip install requests", wd, 30) == (None, INSTALL_TIMEOUT_SECONDS)
    assert refuse_or_lift("echo hi", wd, 30) == (None, 30)


def test_bash_refuses_destructive_commands_without_running_them(tmp_path: Path) -> None:
    """``Bash`` returns the verdict and touches nothing; allowed deletes still run."""
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    (tmp_path / "build").mkdir()
    tools = UsefulTools(work_dir=str(tmp_path))
    named = tmp_path.as_posix()  # as_posix: bash would eat the backslashes of a Windows path
    assert tools.Bash(f"rm -rf {named}", "wipe") == DESTRUCTIVE_VERDICT
    assert tools.Bash("kill -9 -1", "kill all") == DESTRUCTIVE_VERDICT
    assert tools.Bash(f"rm -rf {named}", "wipe in background", background=True) == (
        DESTRUCTIVE_VERDICT)
    # Review 2026-11-08 R7 S7/F2: ``env -u kill -u pkill rm -rf <workdir>`` deletes the
    # work dir through a wrapper whose option values are spelled like guarded commands.
    assert tools.Bash(f"env -u kill -u pkill rm -rf {named}", "wipe") == DESTRUCTIVE_VERDICT
    # Audit 2026-11-10 C13-F3: the work dir named from inside it, where the call starts.
    assert tools.Bash("rm -rf ./*", "wipe") == DESTRUCTIVE_VERDICT
    # Review 2026-11-11 S4-2: a cd that stays here, may not run or fails exempts no delete
    # (each of these stays inside tmp_path whatever the guard answers).
    for command in ("cd . && rm -rf *", "true || cd build; rm -rf *", "cd nosuch; rm -rf *"):
        assert tools.Bash(command, "wipe") == DESTRUCTIVE_VERDICT, command
    # Audit 2026-11-21 S12/F4: bash joins the lines of a continued command before it runs it.
    for command in (f"rm -rf \\\n{named}", "rm -rf \\\n  ./*", "cd . && rm \\\n-rf *"):
        assert tools.Bash(command, "wipe") == DESTRUCTIVE_VERDICT, command
    # Review 2026-11-21 of S12: bash removes the quoting before it runs the command.
    for command in (f'X="a b" rm -rf {named}', f'rm -rf "x # y" {named}',
                    'rm -rf "$PWD"/*', 'rm -rf "no (1).txt" *'):
        assert tools.Bash(command, "wipe") == DESTRUCTIVE_VERDICT, command
    assert keep.read_text() == "keep"
    (tmp_path / "build" / "old (1) # x").write_text("junk")
    out = tools.Bash('X="a b" rm -rf "build/old (1) # x" && echo gone', "allowed")
    assert out.strip() == "gone" and not (tmp_path / "build" / "old (1) # x").exists()
    # The delete right after a cd into a directory of the work dir runs, there.
    (tmp_path / "build" / "junk.txt").write_text("junk")
    assert tools.Bash("cd build && rm -rf * && echo emptied", "allowed").strip() == "emptied"
    assert not (tmp_path / "build" / "junk.txt").exists() and keep.read_text() == "keep"
    (tmp_path / "build" / "junk.txt").write_text("junk")
    assert tools.Bash("rm -rf \\\n  build/junk.txt && echo gone", "allowed").strip() == "gone"
    assert not (tmp_path / "build" / "junk.txt").exists() and keep.read_text() == "keep"
    assert tools.Bash(f"rm -rf {named}/build && echo gone", "allowed").strip() == "gone"
    assert not (tmp_path / "build").exists()


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_bash_lifts_install_timeout(tmp_path: Path) -> None:
    """A build asked to run with a 1 s timeout is not killed after 1 s."""
    (tmp_path / "Makefile").write_text("all:\n\tsleep 2; echo built\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    started = time.monotonic()
    out = tools.Bash("make", "slow build", timeout_seconds=1)
    assert time.monotonic() - started >= 2
    assert "built" in out and "timeout" not in out.lower()
    # A non-build command with the same timeout is still killed at the deadline,
    # and the model sees what it printed before the kill.
    out = tools.Bash("echo phase-1; sleep 30", "slow probe", timeout_seconds=1)
    assert out.startswith("Error: Command execution timeout after 1s. Output before the timeout:")
    assert "phase-1" in out
    assert tools.Bash("sleep 30", "silent probe", timeout_seconds=1) == (
        "Error: Command execution timeout")


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_run_commands_parallel_guards_each_command(tmp_path: Path) -> None:
    """A destructive command in a batch is refused; its siblings still run."""
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    (tmp_path / "Makefile").write_text("all:\n\tsleep 2; echo built\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    report = tools.run_commands_parallel(
        json.dumps([f"rm -rf {tmp_path}", "echo sibling", "make"]), timeout_seconds=1)
    assert DESTRUCTIVE_VERDICT in report
    assert "sibling" in report
    assert "built" in report
    assert keep.read_text() == "keep"


def test_coding_sea_shares_the_guards() -> None:
    """The HarnessTax harness uses the shared patterns, not private copies."""
    from kiss.agents.sorcar import shell_guards

    assert coding_sea.INSTALL_COMMANDS is shell_guards.INSTALL_COMMANDS
    assert coding_sea.INSTALL_TIMEOUT_SECONDS == shell_guards.INSTALL_TIMEOUT_SECONDS
    assert shell_guards.destructive_pattern("/testbed/").search("rm -rf /testbed")
    # FX05c: by the command's text alone; the harness's container is not this host's to ask
    guard = shell_guards.destructive_pattern("/testbed")
    assert guard.search("cd sub && rm -rf ..") and guard.search("cd a/../.. && rm -rf *")
    assert guard.search("cd a/../b && rm -rf *") is None
    assert shell_guards.destructive_pattern("").search("rm -rf /")
    assert shell_guards.destructive_pattern("").search("rm -rf build") is None
    assert shell_guards.destructive_pattern("").search('d=$(pwd); rm -rf "$d"') is None


@pytest.fixture
def cleanup() -> Iterator[set[str]]:
    """Snapshot this process's containers; force-remove the new ones afterwards."""
    client = docker.from_env()
    before = image_container_ids(client)
    try:
        yield before
    finally:
        remove_new_image_containers(client, before)


@pytest.mark.slow
@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")
def test_docker_bash_applies_the_guards(
    cleanup: set[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Docker ``Bash`` and ``run_commands_parallel`` refuse destructive commands
    against the container work dir and keep a build alive past a short timeout."""
    mgr = DockerManager(IMAGE, workdir="/work")
    mgr.open()
    try:
        assert mgr.Bash("mkdir -p /work && echo keep > /work/keep.txt", "seed").strip() == ""
        assert mgr.Bash("rm -rf /work", "wipe") == DESTRUCTIVE_VERDICT
        assert mgr.Bash("kill -9 -1", "kill all") == DESTRUCTIVE_VERDICT
        assert mgr.Bash("cat /work/keep.txt", "check").strip() == "keep"
        report = mgr.run_commands_parallel(
            '["rm -rf /work/", "echo sibling", "cat /work/keep.txt"]', timeout_seconds=30)
        assert DESTRUCTIVE_VERDICT in report and "sibling" in report and "keep" in report
        # ``make`` is a build: the 1 s timeout is lifted, so the 2 s sleep completes.
        # Non-streaming Docker paths keep the output printed before the kill.
        out = mgr.Bash("echo phase-1; sleep 2; echo built", "slow probe", timeout_seconds=1)
        assert "timed out" in out and "phase-1" in out and "built" not in out, out
        report = mgr.run_commands_parallel('["echo phase-2; sleep 5"]', timeout_seconds=1)
        assert "phase-2" in report, report
        # The slim image has no ``make``; a stand-in build tool of that name
        # (``_CMD`` accepts a directory prefix) shows the lift end to end.
        mgr.Bash("printf '#!/bin/sh\\nsleep 2; echo built\\n' > /work/make; chmod +x /work/make",
                 "fake make")
        out = mgr.Bash("/work/make -j2", "slow build", timeout_seconds=1)
        assert "built" in out and "timed out" not in out, out
        # Round 2 of audit 2026-11-12 (B5): where the ``cd`` of a delete lands was asked of
        # this host, which holds no ``/work``: a link of the container back to its work dir
        # went unseen and the delete emptied it.  The container says where its paths lead.
        mgr.Bash("mkdir -p /work/sub /other/deep && ln -s /work /work/alias && ln -s / /work/up"
                 " && ln -s /other /work/away && ln -s /work /other/x", "links")
        for entered, operand in (("alias", "*"), ("up/work", "./*"), ("away/x", "*"),
                                 ("away", "../work")):
            command = f"cd {entered} && rm -rf {operand}"
            assert destructive_command_guard(command, "/work") is None  # as this host reads it
            assert mgr.Bash(command, "wipe") == DESTRUCTIVE_VERDICT, command
            assert DESTRUCTIVE_VERDICT in mgr.run_commands_parallel(json.dumps([command]))
        assert mgr.Bash("cat /work/keep.txt", "check").strip() == "keep"
        # A delete entered into a directory of the work dir, or one a link leads out to, runs.
        mgr.Bash("touch /work/sub/junk /other/deep/junk", "junk")
        for entered in ("sub", "alias/sub", "away/deep"):
            assert mgr.Bash(f"cd {entered} && rm -rf * && echo emptied", "ok").strip() == "emptied"
        assert mgr.Bash("ls /work/sub /other/deep; cat /work/keep.txt", "check").split() == [
            "/other/deep:", "/work/sub:", "keep"]
        # Review 2026-11-21 of S12/F1: the ``CDPATH`` bash looks a bare ``cd`` up in is the
        # container shell's.  This host's was asked: with ``/`` there and none here,
        # ``cd work && rm -rf *`` stayed in ``/work``, passed as a delete in ``/work/work``
        # and emptied it; with one here and none there, a delete in ``/work/sub`` was refused.
        mgr.exec_env = {"CDPATH": "/"}
        assert mgr.Bash("cd work && pwd -P", "where bash lands").split()[-1] == "/work"
        assert mgr.Bash("cd work && rm -rf *", "wipe") == DESTRUCTIVE_VERDICT
        assert DESTRUCTIVE_VERDICT in mgr.run_commands_parallel('["cd work && rm -rf ./*"]')
        mgr.exec_env = {}
        monkeypatch.setenv("CDPATH", "/")
        mgr.Bash("touch /work/sub/junk", "junk")
        assert mgr.Bash("cd sub && rm -rf * && echo emptied", "ok").strip() == "emptied"
        report = mgr.run_commands_parallel('["cd sub && rm -rf ./* && echo emptied"]')
        assert report.startswith("1 commands: 1 succeeded") and "emptied" in report, report
        monkeypatch.delenv("CDPATH")
        assert mgr.Bash("ls /work/sub; cat /work/keep.txt", "check").split() == ["keep"]
        # An answer nobody got (the task is stopped) runs no delete on a guess: the
        # command is reported as a stopped one is.
        mgr.Bash("touch /work/sub/kept", "seed")
        mgr.stop_event = threading.Event()
        mgr.stop_event.set()
        unanswered = mgr.Bash("cd sub && rm -rf *", "stopped")
        mgr.stop_event = None
        assert unanswered == f"{STOPPED_MESSAGE}\n[exit code: -1]"
        # Nor does a container that does not say what its ``CDPATH`` is: the ``printf`` of
        # that answer alone fails there, then hangs until the task is stopped.
        answers = '() { [ "$1" != %s ] || ACTION; builtin printf "$@"; }'
        mgr.exec_env = {"CDPATH": "/", "BASH_FUNC_printf%%": answers.replace("ACTION", "exit 7")}
        assert mgr.Bash("cd sub && rm -rf *", "unanswered") == (
            "Error: the command did not run: "
            "the environment did not say what its CDPATH is (status 7).")
        mgr.exec_env["BASH_FUNC_printf%%"] = answers.replace("ACTION", "sleep 30")
        mgr.stop_event = threading.Event()
        stop = threading.Timer(1.0, mgr.stop_event.set)
        stop.start()
        unanswered = mgr.Bash("cd sub && rm -rf *", "stopped")
        stop.join()
        mgr.stop_event, mgr.exec_env = None, {}
        assert unanswered == f"{STOPPED_MESSAGE}\n[exit code: -1]"
        assert mgr.Bash("ls /work/sub", "check").split() == ["kept"]
        # An oracle that timed out in a directory whose name reads like that message
        # is no Stop.
        mgr.Bash(f"mkdir -p '/work/{STOPPED_MESSAGE}/sub'", "seed")
        mgr.workdir = f"/work/{STOPPED_MESSAGE}"
        timed_out = mgr.Bash("cd sub && rm -f kept", "timeout", timeout_seconds=0)
        mgr.workdir = "/work"
        assert timed_out.startswith("Error: the command did not run:"), timed_out
        assert "timed out" in timed_out and "[exit code:" not in timed_out, timed_out
    finally:
        mgr.close()
    assert not os.path.exists(mgr.host_shared_path or "/nonexistent-kiss-shared")
