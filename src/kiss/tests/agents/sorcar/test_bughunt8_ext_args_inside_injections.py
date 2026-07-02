# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 8 (SORCAR-EXT): arguments silently dropped by expand_command.

``expand_command`` documents: "If the template uses no
``$ARGUMENTS``/``$N`` placeholder and arguments were given, they are
appended after two newlines ... so they are never silently dropped."

The single-pass ``_INJECT_RE`` substitution matches ``!`command``` and
``@{path}`` constructs *before* the argument placeholders, so a ``$1``
or ``$ARGUMENTS`` that appears only *inside* those constructs is never
substituted with the user's arguments.  But ``has_placeholder`` was
computed on the raw template, where those inner ``$1``/``$ARGUMENTS``
occurrences still matched — so the append-args fallback was suppressed
and the user's arguments vanished from the expanded prompt entirely,
violating the documented never-silently-dropped guarantee.
"""

from __future__ import annotations

import tempfile

from kiss.agents.sorcar.custom_commands import CustomCommand, expand_command


def _cmd(template: str) -> CustomCommand:
    return CustomCommand(
        name="t",
        description="",
        argument_hint="",
        template=template,
        source="user",
        path="x",
    )


def test_positional_only_inside_shell_injection_appends_args() -> None:
    """$1 inside !`...` is not a template placeholder; args must be appended."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("Out: !`echo start $1 end`"), "world", wd)
    # The shell alternative wins the single-pass match, so $1 reaches
    # /bin/sh (where it expands to nothing) and is never replaced with
    # the user's argument.  The argument must therefore be appended.
    assert "world" in out, f"argument silently dropped: {out!r}"
    assert out == "Out: start end\n\nworld"


def test_arguments_only_inside_shell_injection_appends_args() -> None:
    """$ARGUMENTS inside !`...` must not suppress the append fallback."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("Out: !`echo got $ARGUMENTS.`"), "vital", wd)
    assert "vital" in out, f"argument silently dropped: {out!r}"
    assert out == "Out: got .\n\nvital"


def test_positional_only_inside_file_injection_appends_args() -> None:
    """$1 inside @{...} is part of the path, not a placeholder."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("Data: @{notes-$1.md}"), "topic", wd)
    assert "topic" in out, f"argument silently dropped: {out!r}"
    assert out == "Data: [could not read file: notes-$1.md]\n\ntopic"


def test_real_placeholder_outside_injection_still_counts() -> None:
    """A genuine $1 outside injections keeps the no-append behaviour."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("Run $1 via !`echo x`"), "job", wd)
    assert out == "Run job via x"
