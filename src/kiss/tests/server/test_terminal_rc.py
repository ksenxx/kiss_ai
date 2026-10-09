# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The terminal tab's command history outlives the tab.

:mod:`kiss.server.terminal_rc` starts bash with ``--rcfile`` and zsh
with ``ZDOTDIR`` pointing at start-up files that run the user's own
dotfiles and then write every command to the history file as it is
entered.  The unit tests check the command and environment the module
builds and the files it writes; the end-to-end tests run real shells
on real ptys through :class:`TerminalService` with a scratch ``HOME``
and show that a second tab, opened while the first tab's shell is
still running, recalls the first tab's commands with Up, that the
user's own rc settings (a ``PROMPT_COMMAND`` of theirs, as a string or
an array; a zsh ``HISTFILE`` of their own) survive, and that the tab's
helper variables and functions are gone by the first prompt.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.core.processes import find_bash
from kiss.server import terminal_rc, terminal_tab
from kiss.server.terminal_rc import BASH_RC, ZSH_ZSHRC, rc_dir, with_persistent_history
from kiss.server.terminal_tab import TerminalService

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="terminal tabs need a pty",
)

_UP = "\x1b[A"
_ANSI = re.compile(r"\x1b\][^\x07]*\x07|\x1b\[[0-9;?]*[A-Za-z]|\x1b[=>]")
# Output that ends with a shell prompt: the line editor is waiting for input.
_PROMPT = re.compile(r"[$%#] $")


def _plain(text: str) -> str:
    """Strip escape sequences and carriage returns from terminal output."""
    return _ANSI.sub("", text).replace("\r", "")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch ``HOME`` (dotfiles, history) and ``KISS_HOME`` (rc files)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss-home"))
    monkeypatch.delenv("ZDOTDIR", raising=False)
    monkeypatch.delenv("HISTFILE", raising=False)
    monkeypatch.delenv("PROMPT_COMMAND", raising=False)
    return home


# ----------------------------------------------------------------------
# The command and environment
# ----------------------------------------------------------------------


def test_bash_gets_the_rc_file_and_keeps_its_other_arguments(home: Path) -> None:
    argv, env = with_persistent_history(["/bin/bash"])
    rc = rc_dir() / "bashrc"
    assert argv == ["/bin/bash", "--rcfile", str(rc)]
    assert env == {}
    assert rc.read_text(encoding="utf-8") == BASH_RC
    assert rc_dir() == Path(os.environ["KISS_HOME"]) / "terminal"


def test_bash_login_flag_becomes_the_login_variable(home: Path) -> None:
    # A login shell ignores --rcfile, so the rc file stands in for -l.
    argv, env = with_persistent_history(["/opt/homebrew/bin/bash", "-l"])
    assert argv == ["/opt/homebrew/bin/bash", "--rcfile", str(rc_dir() / "bashrc")]
    assert env == {"KISS_TERMINAL_LOGIN": "1"}


def test_zsh_gets_a_zdotdir_of_shims(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv, env = with_persistent_history(["/bin/zsh", "-l"])
    zdotdir = rc_dir() / "zdotdir"
    assert argv == ["/bin/zsh", "-l"]
    assert env == {"ZDOTDIR": str(zdotdir)}
    assert sorted(p.name for p in zdotdir.iterdir()) == [
        ".zlogin", ".zprofile", ".zshenv", ".zshrc",
    ]
    assert (zdotdir / ".zshrc").read_text(encoding="utf-8") == ZSH_ZSHRC
    # A ZDOTDIR of the user's own is passed along for the shims to use.
    monkeypatch.setenv("ZDOTDIR", str(home / "cfg"))
    _, env = with_persistent_history(["/bin/zsh"])
    assert env == {"ZDOTDIR": str(zdotdir), "KISS_USER_ZDOTDIR": str(home / "cfg")}


def test_other_shells_are_started_as_they_are(home: Path) -> None:
    for argv in (["/bin/sh"], ["/usr/bin/fish", "-l"], ["/tmp/test-shell", "-l"]):
        assert with_persistent_history(argv) == (argv, {})
    assert not rc_dir().exists()


def test_a_stale_rc_file_is_rewritten_and_a_current_one_left_alone(home: Path) -> None:
    rc = rc_dir() / "bashrc"
    rc.parent.mkdir(parents=True)
    rc.write_text("# an older version\n", encoding="utf-8")
    with_persistent_history(["/bin/bash"])
    assert rc.read_text(encoding="utf-8") == BASH_RC
    before = rc.stat().st_mtime_ns
    time.sleep(0.01)
    with_persistent_history(["/bin/bash"])
    assert rc.stat().st_mtime_ns == before
    assert not (rc.parent / "bashrc.tmp").exists()


def test_unwritable_rc_dir_leaves_the_shell_as_it_is(
    home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    blocker = home / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("KISS_HOME", str(blocker))
    with caplog.at_level("WARNING", logger=terminal_rc.__name__):
        assert with_persistent_history(["/bin/bash", "-l"]) == (["/bin/bash", "-l"], {})
        assert with_persistent_history(["/bin/zsh"]) == (["/bin/zsh"], {})
    assert "start-up files not written" in caplog.text


# ----------------------------------------------------------------------
# Real shells on real ptys
# ----------------------------------------------------------------------


class ConnPrinter:
    """Collects the events the service broadcasts."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(dict(event))

    def output(self, tab_id: str) -> str:
        with self.lock:
            return _plain(
                "".join(
                    e["data"] for e in self.events
                    if e["type"] == "terminalData" and e["tab_id"] == tab_id
                ),
            )

    def wait_for(self, predicate: Any, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        with self.lock:
            seen = [(e["type"], e.get("tab_id"), e.get("data", "")) for e in self.events]
        raise AssertionError(f"timed out waiting for terminal events; got {seen!r}")

    def ran(self, tab_id: str, start: int, marker: str) -> bool:
        """Whether a command typed when the output was *start* long has run.

        The shell echoes the typed line before running it, so *marker*
        may show up in the echo first; the command is done only once the
        prompt is back after the marker.
        """
        new = self.output(tab_id)[start:]
        return marker in new and _PROMPT.search(new) is not None


class Tabs:
    """Two terminal tabs on one service, with a scratch work dir."""

    def __init__(self, work_dir: Path) -> None:
        self.printer = ConnPrinter()
        self.svc = TerminalService(self.printer)
        self.work_dir = str(work_dir)

    def open(self, tab_id: str) -> None:
        """Open *tab_id* and wait for its first prompt (the line editor is up)."""
        self.svc.open(tab_id, "conn", self.work_dir, 200, 24)
        self.printer.wait_for(lambda: _PROMPT.search(self.printer.output(tab_id)))

    def run(self, tab_id: str, line: str, marker: str) -> str:
        """Type *line* + Enter; wait until the command has run and printed *marker*."""
        start = len(self.printer.output(tab_id))
        self.svc.input(tab_id, "conn", line + "\n")
        self.printer.wait_for(lambda: self.printer.ran(tab_id, start, marker))
        return self.printer.output(tab_id)

    def close(self) -> None:
        self.svc.shutdown()


def _diag_line(shell: str) -> str:
    """A command printing the state the tests check, as one ``DIAG`` line."""
    if shell == "zsh":
        funcs = (
            "$(typeset -f __kiss_terminal_prepare __kiss_terminal_capture "
            "__kiss_terminal_restore >/dev/null 2>&1 && echo kept || echo gone)"
        )
        login = "$([[ -o login ]] && echo y || echo n)"
    else:
        funcs = "gone"
        login = "$(shopt -q login_shell && echo y || echo n)"
    # ``exported`` counts ZDOTDIR in the environment: 1 when exported.
    # The line ends with ``diag-42``, which the echo of the command
    # itself does not contain: waiting for it waits for the output.
    return (
        "printf 'DIAG login=%s zdotdir=<%s> exported=%s kissvars=%s histfile=<%s> "
        "funcs=%s diag-%s\\n' "
        f'"{login}" "${{ZDOTDIR:-}}" "$(env | grep -c "^ZDOTDIR=")" '
        '"$(env | grep -c "^KISS_\\(TERMINAL\\|USER_ZDOTDIR\\)")" '
        f'"${{HISTFILE:-}}" "{funcs}" "$((40+2))"'
    )


def _bash() -> str:
    bash = find_bash()
    assert bash, "these tests need bash"
    return bash


def _recall_across_tabs(tabs: Tabs, history_file: Path) -> None:
    """Tab B, opened while tab A runs, recalls A's commands with Up."""
    tabs.open("A")
    tabs.run("A", "echo first-$((20+2))", "first-22")
    tabs.run("A", "echo second-$((40+2))", "second-42")
    # Written as soon as entered, not at the shell's exit.
    assert "echo second-$((40+2))" in history_file.read_text(encoding="utf-8")
    tabs.open("B")
    # Up Up recalls A's older command (the marker is in its output
    # only, so one hit means it ran once).
    out = tabs.run("B", _UP + _UP, "first-22")
    assert out.count("first-22") == 1
    # B's list is now A's two commands plus its own: Up Up is A's newest.
    out = tabs.run("B", _UP + _UP, "second-42")
    assert out.count("second-42") == 1


def test_bash_tab_recalls_the_commands_of_a_tab_still_running(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bash = _bash()
    monkeypatch.setenv("SHELL", bash)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [bash])
    # The user's rc file is read first, and its PROMPT_COMMAND keeps
    # running before every prompt.
    (home / ".bashrc").write_text(
        "RC_MARK=user-bashrc\nPROMPT_COMMAND='echo prompt-hook'\nHISTSIZE=200\n",
        encoding="utf-8",
    )
    tabs = Tabs(home)
    try:
        _recall_across_tabs(tabs, home / ".bash_history")
        out = tabs.run("B", 'echo "mark=$RC_MARK size=$HISTSIZE"', "mark=user-bashrc")
        assert "size=200" in out
        assert out.count("prompt-hook") >= 4
        out = tabs.run("B", 'declare -p PROMPT_COMMAND; shopt histappend', "histappend")
        # bash 4.4 and later print the newline as an escape, older ones as is.
        assert re.search(
            r"declare -- PROMPT_COMMAND=(\$'echo prompt-hook\\nhistory -a'"
            r'|"echo prompt-hook\nhistory -a")',
            out,
        )
        assert re.search(r"histappend\s+on", out)
        out = tabs.run("B", _diag_line("bash"), "diag-42")
        assert re.search(
            r"DIAG login=n zdotdir=<> exported=0 kissvars=0 "
            r"histfile=<[^>]*/\.bash_history> funcs=gone",
            out,
        )
    finally:
        tabs.close()


def test_bash_tabs_opened_together_on_an_empty_history_keep_each_others_commands(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two tabs that both started before any command was entered (no
    history file yet) write every command once, in the order entered,
    and a third tab recalls them all.  bash before 4.0 takes another
    path for such tabs, which must not rewrite the file."""
    bash = _bash()
    monkeypatch.setenv("SHELL", bash)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [bash])
    tabs = Tabs(home)
    try:
        tabs.open("A")
        tabs.open("B")
        tabs.run("A", "echo a-$((1+0))", "a-1")
        tabs.run("B", "echo b-$((1+0))", "b-1")
        tabs.run("A", "echo a-$((2+0))", "a-2")
        tabs.run("B", "echo b-$((2+0))", "b-2")
        assert (home / ".bash_history").read_text(encoding="utf-8") == (
            "echo a-$((1+0))\necho b-$((1+0))\necho a-$((2+0))\necho b-$((2+0))\n"
        )
        tabs.open("C")
        out = tabs.run("C", _UP + _UP + _UP + _UP, "a-1")
        assert out.count("a-1") == 1
    finally:
        tabs.close()


def test_bash_array_prompt_command_gets_the_hook_appended(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bash = _bash()
    version = subprocess.run(
        [bash, "-c", 'echo "${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if tuple(int(n) for n in version.split(".")) < (5, 1):
        pytest.skip(f"bash {version} runs only the first element of a PROMPT_COMMAND array")
    monkeypatch.setenv("SHELL", bash)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [bash])
    (home / ".bashrc").write_text(
        "PROMPT_COMMAND=('echo hook-one' 'echo hook-two')\n", encoding="utf-8",
    )
    tabs = Tabs(home)
    try:
        tabs.open("A")
        out = tabs.run("A", "declare -p PROMPT_COMMAND", "declare -a")
        # The hook is a third element; the user's two keep running.
        assert '[2]="history -a"' in out
        assert out.count("hook-two") >= 2
        tabs.run("A", "echo arr-$((50+5))", "arr-55")
        assert "echo arr-$((50+5))" in (home / ".bash_history").read_text(encoding="utf-8")
    finally:
        tabs.close()


def test_bash_login_shell_reads_the_login_files_instead(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bash = _bash()
    monkeypatch.setenv("SHELL", bash)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [bash, "-l"])
    (home / ".bash_profile").write_text(
        "PROFILE_MARK=user-profile\n. ~/.bashrc\n", encoding="utf-8",
    )
    (home / ".bashrc").write_text("RC_MARK=user-bashrc\n", encoding="utf-8")
    (home / ".profile").write_text("PROFILE_MARK=wrong-file\n", encoding="utf-8")
    tabs = Tabs(home)
    try:
        _recall_across_tabs(tabs, home / ".bash_history")
        out = tabs.run(
            "B", 'echo "p=$PROFILE_MARK r=$RC_MARK login=${KISS_TERMINAL_LOGIN:-unset}"',
            "p=user-profile",
        )
        assert "p=user-profile r=user-bashrc login=unset" in out
    finally:
        tabs.close()


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
def test_zsh_tab_recalls_the_commands_of_a_tab_still_running(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh
    monkeypatch.setenv("SHELL", zsh)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [zsh])
    # Without the system files (macOS's /etc/zshrc configures history)
    # no file configures a history file: the usual one, written at once.
    (home / ".zshenv").write_text(
        "ENV_MARK=user-zshenv\nunsetopt globalrcs\n", encoding="utf-8",
    )
    (home / ".zshrc").write_text("RC_MARK=user-zshrc\n", encoding="utf-8")
    tabs = Tabs(home)
    try:
        _recall_across_tabs(tabs, home / ".zsh_history")
        out = tabs.run(
            "B", 'echo "e=$ENV_MARK r=$RC_MARK save=$SAVEHIST size=$HISTSIZE"',
            "e=user-zshenv",
        )
        assert "e=user-zshenv r=user-zshrc save=10000 size=10000" in out
        out = tabs.run("B", _diag_line("zsh"), "diag-42")
        assert re.search(
            r"DIAG login=n zdotdir=<> exported=0 kissvars=0 "
            r"histfile=<[^>]*/\.zsh_history> funcs=gone",
            out,
        )
        out = tabs.run("B", "[[ -o incappendhistory ]] && echo inc-$((1+1))", "inc-2")
    finally:
        tabs.close()


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
def test_zsh_login_shell_with_its_own_zdotdir_and_history_file(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh
    monkeypatch.setenv("SHELL", zsh)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [zsh, "-l"])
    cfg = home / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("ZDOTDIR", str(cfg))
    # The user's files are read from their ZDOTDIR, in zsh's order,
    # and the history settings of their own are kept.
    (cfg / ".zshenv").write_text('ORDER="$ORDER env"\n', encoding="utf-8")
    (cfg / ".zprofile").write_text('ORDER="$ORDER profile"\n', encoding="utf-8")
    (cfg / ".zshrc").write_text(
        'ORDER="$ORDER rc"\nHISTFILE=$ZDOTDIR/own_history\nSAVEHIST=300\nHISTSIZE=400\n'
        "setopt sharehistory\n",
        encoding="utf-8",
    )
    (cfg / ".zlogin").write_text('ORDER="$ORDER login"\n', encoding="utf-8")
    tabs = Tabs(home)
    try:
        _recall_across_tabs(tabs, cfg / "own_history")
        out = tabs.run("B", 'echo "order=$ORDER save=$SAVEHIST size=$HISTSIZE"', "order=")
        assert "order= env profile rc login save=300 size=400" in out
        out = tabs.run("B", _diag_line("zsh"), "diag-42")
        assert re.search(
            rf"DIAG login=y zdotdir=<{re.escape(str(cfg))}> exported=1 kissvars=0 "
            rf"histfile=<{re.escape(str(cfg))}/own_history> funcs=gone",
            out,
        )
        # sharehistory was the user's choice; incappendhistory is not added to it.
        out = tabs.run(
            "B", "[[ -o sharehistory ]] && [[ ! -o incappendhistory ]] && echo opts-$((2+2))",
            "opts-4",
        )
    finally:
        tabs.close()


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
def test_zsh_dotfiles_keep_their_scope_and_an_unexported_zdotdir(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh
    monkeypatch.setenv("SHELL", zsh)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [zsh])
    cfg = home / "cfg"
    cfg.mkdir()
    (home / "bin").mkdir()
    # A ZDOTDIR set (not exported) by ~/.zshenv moves the other files;
    # a `typeset` in them declares a shell-wide variable, as it does
    # when zsh reads the files itself; `nounset` is fine.
    (home / ".zshenv").write_text('ZDOTDIR="$HOME/cfg"\n', encoding="utf-8")
    (cfg / ".zshrc").write_text(
        'typeset -U path\npath=("$HOME/bin" $path)\nsetopt nounset\nRC_MARK=cfg-zshrc\n',
        encoding="utf-8",
    )
    (home / ".zshrc").write_text("RC_MARK=wrong-file\n", encoding="utf-8")
    tabs = Tabs(home)
    try:
        _recall_across_tabs(tabs, cfg / ".zsh_history")
        out = tabs.run("B", 'echo "r=$RC_MARK p=${path[1]}"', "r=")
        assert f"r=cfg-zshrc p={home}/bin" in out
        out = tabs.run("B", _diag_line("zsh"), "diag-42")
        assert re.search(
            rf"DIAG login=n zdotdir=<{re.escape(str(cfg))}> exported=0 kissvars=0 "
            rf"histfile=<{re.escape(str(cfg))}/\.zsh_history> funcs=gone",
            out,
        )
        out = tabs.run("B", "[[ -o incappendhistory ]] && echo inc-$((1+1))", "inc-2")
    finally:
        tabs.close()


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
def test_zsh_login_shell_whose_zshrc_turns_rcs_off_is_still_cleaned_up(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zsh = shutil.which("zsh")
    assert zsh
    monkeypatch.setenv("SHELL", zsh)
    monkeypatch.setattr(terminal_tab, "default_shell", lambda: [zsh, "-l"])
    cfg = home / "cfg"
    cfg.mkdir()
    # ~/.zprofile moves ZDOTDIR, so .zshrc comes from the new place;
    # that .zshrc turns rcs off, so zsh reads no .zlogin at all.
    (home / ".zprofile").write_text('ZDOTDIR="$HOME/cfg"\n', encoding="utf-8")
    (cfg / ".zshrc").write_text("RC_MARK=cfg-zshrc\nunsetopt rcs\n", encoding="utf-8")
    (home / ".zshrc").write_text("RC_MARK=wrong-file\n", encoding="utf-8")
    (home / ".zlogin").write_text("LOGIN_MARK=ran\n", encoding="utf-8")
    (cfg / ".zlogin").write_text("LOGIN_MARK=ran\n", encoding="utf-8")
    tabs = Tabs(home)
    try:
        # zsh itself reads and saves no history file with rcs off, but
        # the commands still land in it as they are entered.
        tabs.open("A")
        tabs.run("A", "echo first-$((20+2))", "first-22")
        assert "echo first-$((20+2))" in (cfg / ".zsh_history").read_text(encoding="utf-8")
        out = tabs.run("A", 'echo "r=$RC_MARK l=${LOGIN_MARK:-none}"', "r=")
        assert "r=cfg-zshrc l=none" in out
        out = tabs.run("A", _diag_line("zsh"), "diag-42")
        assert re.search(
            rf"DIAG login=y zdotdir=<{re.escape(str(cfg))}> exported=0 kissvars=0 "
            rf"histfile=<{re.escape(str(cfg))}/\.zsh_history> funcs=gone",
            out,
        )
    finally:
        tabs.close()
