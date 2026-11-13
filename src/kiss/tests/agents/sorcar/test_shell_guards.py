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
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

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
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.processes import find_bash
from kiss.tests.agents.sorcar.docker_test_containers import (
    IMAGE,
    image_container_ids,
    is_docker_available,
    remove_new_image_containers,
)

BLOCKED = [
    "kill -9 -1",
    "kill -1",
    "echo cleanup; kill -TERM -1",
    "pkill -f .",
    "pkill -9 -f '.*'",
    "sudo rm -rf /",
    "rm -rf {wd}",
    "rm -rf {wd}/",
    "rm -rf '{wd}/*'",
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


def test_destructive_guard_protects_root_and_work_dir(tmp_path: Path) -> None:
    """Root and the work dir (any spelling) are refused; everything else runs."""
    wd = str(tmp_path)
    for command in BLOCKED:
        assert destructive_command_guard(command.format(wd=wd), wd) == DESTRUCTIVE_VERDICT, command
    for command in ALLOWED:
        assert destructive_command_guard(command.format(wd=wd), wd) is None, command
    # The trailing-slash spelling and the resolved path of a symlinked work dir
    # are protected too, and ``None`` protects the root only.
    link = tmp_path.parent / (tmp_path.name + "-link")
    link.symlink_to(tmp_path)
    assert destructive_command_guard(f"rm -rf {tmp_path}", str(link)) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {link}", str(link)) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {tmp_path}", f"{tmp_path}/") == DESTRUCTIVE_VERDICT
    assert destructive_command_guard("rm -rf /", None) == DESTRUCTIVE_VERDICT
    assert destructive_command_guard(f"rm -rf {tmp_path}", None) is None
    assert destructive_command_guard("rm -rf *", None) is None
    assert destructive_command_guard(f"rm -rf ../{tmp_path.name}", str(link)) == DESTRUCTIVE_VERDICT


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
    ("bash -c -e -u kill -u pkill 'rm -rf /tmp/w'", True),
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
    ("ls\ncd sub && rm -rf *", True), ("echo \\; cd sub && rm -rf *", True),
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
    ("if cd sub; then rm -rf \"$PWD\"; fi", False), ("mkdir -p sub && cd sub && rm -rf *", False),
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
    without a rescan either.
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
        "a;" * 20_000 + "rm -rf /",
    ]
    for command in long_commands:
        started = time.monotonic()
        destructive_command_guard(command, "/work")
        lift_install_timeout(command, 30)
        assert time.monotonic() - started < 1, command[:40]
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
                    "sudo -u root -- apt install -y git"]:
        assert lift_install_timeout(command, 30) == INSTALL_TIMEOUT_SECONDS, command
        assert lift_install_timeout(command, 1800) == 1800, command
    for command in ["uv run pytest", "npm get registry", "grep -R make .", "go test ./...",
                    "pip list", "echo make", "uv run pytest tests/test_build.py",
                    "npm test # not a build", "go run build.go"]:
        assert lift_install_timeout(command, 30) == 30, command


def test_refuse_or_lift_combines_both_guards(tmp_path: Path) -> None:
    """``refuse_or_lift`` returns the verdict and the lifted deadline in one call."""
    wd = str(tmp_path)
    assert refuse_or_lift(f"rm -rf {wd}", wd, 30) == (DESTRUCTIVE_VERDICT, 30)
    assert refuse_or_lift("pip install requests", wd, 30) == (None, INSTALL_TIMEOUT_SECONDS)
    assert refuse_or_lift("echo hi", wd, 30) == (None, 30)


def test_bash_refuses_destructive_commands_without_running_them(tmp_path: Path) -> None:
    """``Bash`` returns the verdict and touches nothing; allowed deletes still run."""
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    (tmp_path / "build").mkdir()
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Bash(f"rm -rf {tmp_path}", "wipe") == DESTRUCTIVE_VERDICT
    assert tools.Bash("kill -9 -1", "kill all") == DESTRUCTIVE_VERDICT
    assert tools.Bash(f"rm -rf {tmp_path}", "wipe in background", background=True) == (
        DESTRUCTIVE_VERDICT)
    # Review 2026-11-08 R7 S7/F2: ``env -u kill -u pkill rm -rf <workdir>`` deletes the
    # work dir through a wrapper whose option values are spelled like guarded commands.
    assert tools.Bash(f"env -u kill -u pkill rm -rf {tmp_path}", "wipe") == DESTRUCTIVE_VERDICT
    # Audit 2026-11-10 C13-F3: the work dir named from inside it, where the call starts.
    assert tools.Bash("rm -rf ./*", "wipe") == DESTRUCTIVE_VERDICT
    # Review 2026-11-11 S4-2: a cd that stays here, may not run or fails exempts no delete
    # (each of these stays inside tmp_path whatever the guard answers).
    for command in ("cd . && rm -rf *", "true || cd build; rm -rf *", "cd nosuch; rm -rf *"):
        assert tools.Bash(command, "wipe") == DESTRUCTIVE_VERDICT, command
    assert keep.read_text() == "keep"
    # The delete right after a cd into a directory of the work dir runs, there.
    (tmp_path / "build" / "junk.txt").write_text("junk")
    assert tools.Bash("cd build && rm -rf * && echo emptied", "allowed").strip() == "emptied"
    assert not (tmp_path / "build" / "junk.txt").exists() and keep.read_text() == "keep"
    # as_posix: bash would eat the backslashes of a Windows path.
    build = (tmp_path / "build").as_posix()
    assert tools.Bash(f"rm -rf {build} && echo gone", "allowed").strip() == "gone"
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
def test_docker_bash_applies_the_guards(cleanup: set[str]) -> None:
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
        # An answer nobody got (the task is stopped) runs no delete on a guess: the
        # command is reported as a stopped one is.
        mgr.Bash("touch /work/sub/kept", "seed")
        mgr.stop_event = threading.Event()
        mgr.stop_event.set()
        unanswered = mgr.run_bash("cd sub && rm -rf *", "stopped")
        mgr.stop_event = None
        assert unanswered.code == "-1" and unanswered.text.startswith(STOPPED_MESSAGE)
        assert mgr.Bash("ls /work/sub", "check").split() == ["kept"]
        # An oracle that timed out in a directory whose name reads like that message
        # is no Stop.
        mgr.Bash(f"mkdir -p '/work/{STOPPED_MESSAGE}/sub'", "seed")
        mgr.workdir = f"/work/{STOPPED_MESSAGE}"
        timed_out = mgr.run_bash("cd sub && rm -f kept", "timeout", timeout_seconds=0)
        mgr.workdir = "/work"
        assert timed_out.code == "error" and "timed out" in timed_out.text, timed_out
    finally:
        mgr.close()
    assert not os.path.exists(mgr.host_shared_path or "/nonexistent-kiss-shared")
