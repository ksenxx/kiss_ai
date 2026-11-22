# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S2, F1: the shell guards stay linear on newline-heavy commands.

``_CMD`` let the blanks after a separator run across newlines, so on a
command of blank lines every newline started a match attempt that
swallowed the rest of the text and backtracked through it: a 20 kB
no-op took seconds.  The blanks after a separator now stop at the next
newline, which is a separator of its own, so the set of commands
matched is unchanged.

A backquote starts a command too, and a word, an assignment's value and
a directory prefix ran across it: each backquote of a run rescanned the
rest of the text.  They stop at a backquote as at the other separators.
"""

from __future__ import annotations

import time

from kiss.agents.sorcar.shell_guards import (
    DESTRUCTIVE_VERDICT,
    INSTALL_TIMEOUT_SECONDS,
    destructive_command_guard,
    lift_install_timeout,
)


def _guard_seconds(command: str) -> float:
    """The CPU time the two guards take on *command*: wall time also counts the wait for a core."""
    started = time.thread_time()
    destructive_command_guard(command, "/work")
    lift_install_timeout(command, 30)
    return time.thread_time() - started


def guards_stay_linear(command: str, bound: float) -> bool:
    """Whether the guards read *command* in time linear in its length.

    Within *bound* seconds of CPU time; or, on a machine too slow or too loaded for that
    (full run 2026-11-22: a parallel run of the suite took commands an idle machine reads
    in 0.3 s and 0.6 s past 0.5 s and 1 s, on Linux, macOS and Windows), twice the text
    in less than three times the time: a quadratic guard takes four times as long.
    """
    seconds = _guard_seconds(command)
    return seconds < bound or _guard_seconds(command + command) < 3 * seconds


def test_guards_stay_linear_on_newline_heavy_commands() -> None:
    """A 20 kB command of blank lines is guarded in well under half a second.

    Of the guards' own time, as CPU time tells it (:func:`guards_stay_linear`).
    """
    for command in (
        " \n" * 10_000,
        "\n" * 20_000,
        "; \n" * 6_667,
        " \n" * 9_999 + "true",
        ("a\n" + " " * 100 + "\n") * 196,
        "printf '%s' '" + "`" * 20_000 + "'",
        "`/" * 10_000,
        "`A=" * 6_666,
        "`rm " * 5_000,
        "`sudo " * 3_333,
        "`bash -c " * 2_222,
        "`pip " * 4_000,
    ):
        assert destructive_command_guard(command, "/work") is None
        assert lift_install_timeout(command, 30) == 30
        assert guards_stay_linear(command, 0.5), repr(command[:20])


def test_commands_after_newlines_and_blanks_are_still_matched() -> None:
    """The commands the guards matched across blank lines and newlines still match."""
    for command in (
        ";\n  kill -1",
        "\n\n\t rm -rf /",
        "echo a; \n pkill -f .",
        "true &&\n\n  sudo -n rm -rf /work",
        "X=1 \\\n rm -rf '/work'",
        "echo a\r\nkill -9 -1",
        "echo `kill -1`",
        "X=`date` rm -rf /work",
        "sudo -u `whoami` rm -rf /",
        "echo `date`; `rm -rf /work`",
    ):
        assert destructive_command_guard(command, "/work") == DESTRUCTIVE_VERDICT, command
    for command in (
        "echo a;\n   pip install requests",
        "\n\n\tsudo apt-get install -y gcc",
        "cd src &&\n make -j8",
        "echo prepare\r\nnpm ci",
    ):
        assert lift_install_timeout(command, 30) == INSTALL_TIMEOUT_SECONDS, command
    # FX05c: a quoted mention is an argument of ``echo``, no command; a pid after ``-1`` exempts it.
    assert destructive_command_guard("echo 'a;\n kill -1'", "/work") is None
    assert destructive_command_guard("kill -1 1234\n", "/work") is None
    assert destructive_command_guard("kill -1 `cat server.pid`", "/work") is None
    assert lift_install_timeout("echo install\n", 30) == 30
    assert lift_install_timeout("pip install `cat requirements.txt`", 30) == INSTALL_TIMEOUT_SECONDS
