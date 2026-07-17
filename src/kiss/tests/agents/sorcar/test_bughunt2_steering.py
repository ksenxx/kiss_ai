# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 2: injection re-scanning in custom commands + stream leak in steering.

Bug 1 (``custom_commands.expand_command``): the three substitution
passes ran sequentially over the *result* of the previous pass, so
content INJECTED by an earlier pass was re-scanned by the later ones:

* ``_inject_files`` ran first and ``_inject_shell`` then scanned the
  injected file contents — a data file that happened to contain
  ``!`cmd``` got its command **executed** (arbitrary command execution
  from a file the user merely referenced with ``@{path}``).
* The argument-placeholder pass ran last over everything, so literal
  ``$1`` / ``$ARGUMENTS`` text inside injected file contents or shell
  output was replaced with the user's arguments, and
  ``has_placeholder`` was computed on the injected text — so a
  placeholder-free template whose injected file contained ``$1 `` also
  wrongly suppressed the append-args fallback.

This is the same bug class fixed by bughunt 7 for argument *values*,
now for file / shell injections: replacement text must never be
re-scanned.

Bug 2 (``cli_steering.AnchoredRepl.__enter__``): ``sys.stdout`` and
``sys.stderr`` were swapped to :class:`_StdoutProxy` *before*
``box.start()``; when ``start()`` raises (``termios.error`` on a
non-tty stdin, ``io.UnsupportedOperation``, …) the exception
propagated out of ``__enter__`` so ``__exit__`` never ran and the
process was left with its global ``sys.stdout`` / ``sys.stderr``
permanently pointing at proxies of a dead box.  The same leak existed
in ``SteeringSession.run``'s owned-box path (``box.start()`` before
the ``try``/``finally``).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from kiss.agents.sorcar.custom_commands import CustomCommand, expand_command


def _cmd(template: str) -> CustomCommand:
    """Return a minimal in-memory custom command with *template*."""
    return CustomCommand(
        name="c",
        description="",
        argument_hint="",
        template=template,
        source="user",
        path="/tmp/c.md",
    )


# ---------------------------------------------------------------------------
# Bug 1: injected content must never be re-scanned by later passes
# ---------------------------------------------------------------------------


def test_file_content_shell_marker_not_executed() -> None:
    """``!`cmd``` inside an @{file}'s CONTENTS must not be executed."""
    with tempfile.TemporaryDirectory() as wd:
        marker = Path(wd) / "pwned.txt"
        notes = Path(wd) / "notes.md"
        notes.write_text(
            "reminder: run !`touch pwned.txt` later\n", encoding="utf-8"
        )
        out = expand_command(_cmd("See @{notes.md} end."), "", wd)
        assert not marker.exists(), (
            "shell command inside injected file contents was executed"
        )
        assert "!`touch pwned.txt`" in out, (
            "file contents must be injected verbatim"
        )


def test_file_content_placeholders_not_expanded() -> None:
    """``$1``/``$ARGUMENTS`` inside file contents stay verbatim; args append."""
    with tempfile.TemporaryDirectory() as wd:
        notes = Path(wd) / "notes.md"
        notes.write_text("price is $1 per unit, see $ARGUMENTS", encoding="utf-8")
        out = expand_command(_cmd("Notes: @{notes.md}"), "foo bar", wd)
        # File contents verbatim — never substituted with the args.
        assert "price is $1 per unit, see $ARGUMENTS" in out
        # The template itself has no placeholder, so the args must
        # still be appended after two newlines.
        assert out.endswith("\n\nfoo bar")


def test_shell_output_placeholders_not_expanded() -> None:
    """``$1`` inside shell command OUTPUT stays verbatim; template's $1 expands."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(
            _cmd("Out: !`printf 'cost $1 ok'` arg=$1"), "X", wd
        )
        assert "cost $1 ok" in out, "shell output must be injected verbatim"
        assert "arg=X" in out, "template's own $1 must still expand"


def test_file_injection_and_placeholders_still_work_together() -> None:
    """Sanity: normal file + shell + placeholder expansion is unchanged."""
    with tempfile.TemporaryDirectory() as wd:
        (Path(wd) / "f.txt").write_text("FILE", encoding="utf-8")
        out = expand_command(
            _cmd("a=@{f.txt} b=!`echo SHELL` c=$1 d=$ARGUMENTS"),
            "one two",
            wd,
        )
        assert out == "a=FILE b=SHELL c=one d=one two"


# ---------------------------------------------------------------------------
# Bug 2: AnchoredRepl.__enter__ must not leak proxied streams on failure
# ---------------------------------------------------------------------------

_ENTER_LEAK_SCRIPT = r"""
import sys
from kiss.ui.cli.cli_steering import AnchoredRepl

orig_out, orig_err = sys.stdout, sys.stderr
repl = AnchoredRepl()
raised = False
try:
    repl.__enter__()
except BaseException:
    raised = True
restored = sys.stdout is orig_out and sys.stderr is orig_err
print(f"RAISED={raised} RESTORED={restored}", file=orig_err)
"""


def test_anchored_repl_enter_failure_restores_streams() -> None:
    """A failing ``box.start()`` must restore sys.stdout/sys.stderr.

    Runs a fresh interpreter with stdin redirected from ``/dev/null``
    (not a tty), so ``termios.tcgetattr`` inside ``_InputBox.start``
    raises.  ``__enter__`` must then restore the original streams
    before propagating — otherwise the process is stuck writing all
    further output through a proxy of a dead box.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _ENTER_LEAK_SCRIPT],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "RAISED=True" in proc.stderr, proc.stderr
    assert "RESTORED=True" in proc.stderr, (
        "sys.stdout/sys.stderr leaked as _StdoutProxy after failed "
        f"__enter__: {proc.stderr!r}"
    )
