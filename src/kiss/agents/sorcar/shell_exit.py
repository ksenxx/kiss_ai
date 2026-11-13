# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The exit of a shell command, handed from the tool that ran it to the coding gate as data.

A shell tool's result is the command's own output inside the tool's
status, and a reader of the text cannot tell the two apart: a passing
command whose output opened with ``Error (exit code 7):`` read as a
failed one, and a fan-out command that printed the section of a later
one gave that one its exit.  So the tool that ran the command says how
it ended (:func:`hand_over`) and the gate's result hook asks for it
(:func:`take`).  Both run on the thread that makes the tool call, which
is whose slot this is.  A result nobody handed anything over for was
written by the framework (the call was refused, the tool raised or was
interrupted, or made the command as an ``Edit``) or opens with the
tool's own lines (a job's report): its opening is not a command's output.

The words are the gate ledger's: an exit status (``"0"``, ``"7"``,
``"-9"``), ``"timeout"`` (killed at the deadline), ``"background"`` (left
running, as the job named), ``"refused"`` and ``"error"`` (the tool did
not run the command), and for a command of a fan-out also ``"killed by
signal N"``.
"""

from __future__ import annotations

import threading
from typing import NamedTuple


class Ran(NamedTuple):
    """A ``Bash`` result inside the tools: the text and how the command ended.

    Attributes:
        text: The result the model reads.
        code: The ledger's word for the command (see the module).
        job_id: The background job a ``"background"`` *code* names.
    """

    text: str
    code: str
    job_id: str = ""


def unrun(text: str) -> Ran:
    """Return the result of a command its tool did not run.

    Args:
        text: The tool's own text, never a command's output: a guard's
            ``REFUSED: ...`` or the tool's ``Error: ...``.
    """
    return Ran(text, "refused" if text.startswith("REFUSED") else "error")


def exit_word(code: int | None) -> str:
    """Return the ledger's word for the exit status *code* (``None``: killed at the deadline)."""
    return "timeout" if code is None else str(code)


_slot = threading.local()


def hand_over(asked: object, *codes: str, job_id: str = "") -> None:
    """Say how the commands of the tool call made with *asked* ended, for :func:`take`.

    Args:
        asked: The call's ``command`` (``Bash``) or ``commands``
            (``run_commands_parallel``) argument, as the tool got it.
        *codes: The ledger's word for each command, in order; none when
            the tool ran none (its arguments were invalid).
        job_id: The job a ``Bash`` command was started in or left running as.
    """
    _slot.handed = (asked, codes, job_id)


def take(asked: object = None) -> tuple[tuple[str, ...], str] | None:
    """Return what was handed over for the tool call made with *asked*, and empty the slot.

    Args:
        asked: The call's ``command`` or ``commands`` argument; ``None``
            only empties the slot (before a call, or for any other tool).

    Returns:
        ``(codes, job_id)`` as :func:`hand_over` got them; ``None`` when
        nothing was handed over (the tool did not return) or what was
        belongs to another call, one whose result no hook read.
    """
    handed, _slot.handed = getattr(_slot, "handed", None), None
    if handed is None or asked is None or handed[0] != asked:
        return None
    return handed[1], handed[2]
