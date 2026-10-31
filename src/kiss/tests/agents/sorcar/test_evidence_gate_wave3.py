# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 1, R4 finding 2: a mutating package or build command is not a check.

The first fix let ``uv run pytest``, ``npm test`` and ``make test`` count
as the check after an edit, but it read the word right after the tool as
its subcommand and the word after ``make`` as its only target: global
options (``npm --prefix web run build``, ``uv --project . sync``), a Cargo
toolchain (``cargo +stable fmt``), a later Make target (``make test
build``) and ``--fix`` all passed for checks, and a finish right after
them was let through with the edit unchecked.
"""

from __future__ import annotations

import pytest

from kiss.agents.sorcar import evidence_gate as eg

WRITES = (
    "npm --prefix src/kiss/webapp run build",
    "uv --project . sync",
    "cargo +stable fmt",
    "make test build",
    "npm run lint -- --fix",
    "uv run ruff check --fix .",
    "git -C repo checkout main",
    "pip --quiet install x",
    "make -k test install",
    "make V=1",
)
CHECKS = (
    "uv --project . run pytest -q",
    "npm --prefix src/kiss/webapp test",
    "cargo +stable test",
    "make test V=1",
    "make -k check",
    "make test check",
    "make test 2>&1 | tail -5",
    "make check && echo ok",
    "git -C repo diff",
    "pytest --fixtures",
)


@pytest.mark.parametrize("command", WRITES)
def test_a_mutating_command_after_an_edit_leaves_it_unchecked(command: str) -> None:
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {})
    gate.observe("Bash", {"command": command})
    verdict = gate.verdict({"success": True, "summary_in_html": "Done."})
    assert verdict is not None and eg.UNCHECKED_REASON in verdict, command


@pytest.mark.parametrize("command", CHECKS)
def test_a_check_after_an_edit_lets_the_finish_through(command: str) -> None:
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {})
    gate.observe("Bash", {"command": command})
    assert gate.verdict({"success": True, "summary_in_html": "Done."}) is None, command
