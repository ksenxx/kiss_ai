# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 7: argument re-injection in custom-command template expansion.

A user-supplied argument whose *value* contains the literal text
``$ARGUMENTS`` (or ``$1`` … ``$9``) must be inserted verbatim.  The old
implementation ran the positional-placeholder pass and the
``$ARGUMENTS`` pass as two separate ``re.sub`` calls, so placeholder
text introduced into the template by the *first* pass (from the user's
argument value) was re-expanded by the *second* pass.
"""

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


def test_positional_value_containing_arguments_is_not_reexpanded() -> None:
    """A $1 value holding the literal ``$ARGUMENTS`` stays verbatim."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("Run $1 now"), "'$ARGUMENTS literal'", wd)
    assert out == "Run $ARGUMENTS literal now"


def test_arguments_value_containing_positional_is_not_reexpanded() -> None:
    """An $ARGUMENTS value holding the literal ``$1`` stays verbatim."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("Say $ARGUMENTS"), "$1 wins", wd)
    assert out == "Say $1 wins"


def test_positional_value_containing_other_positional() -> None:
    """A $1 value holding the literal ``$2`` must not pick up arg two."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("A $1 B $2"), "'$2' second", wd)
    assert out == "A $2 B second"


def test_normal_expansion_still_works() -> None:
    """Plain positional + $ARGUMENTS expansion is unchanged."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("a=$1 b=$2 all=$ARGUMENTS"), "one two", wd)
    assert out == "a=one b=two all=one two"


def test_missing_positional_is_empty_and_args_appended_when_no_placeholder() -> None:
    """Out-of-range positionals are empty; no-placeholder templates append args."""
    with tempfile.TemporaryDirectory() as wd:
        out = expand_command(_cmd("only $3 here"), "one", wd)
        assert out == "only  here"
        out2 = expand_command(_cmd("static text"), "extra args", wd)
        assert out2 == "static text\n\nextra args"
