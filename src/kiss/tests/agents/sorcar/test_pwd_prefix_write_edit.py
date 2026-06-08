# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests pinning the ``PWD/`` prefix contract for Write/Edit.

The ``Read`` tool rewrites a literal ``PWD/`` prefix to the agent's
``work_dir`` (see ``test_read_tool_robustness.py``).  ``Write`` and
``Edit`` must honour the *same* contract: a model that does
``Read("PWD/x")`` and then ``Edit("PWD/x", ...)`` would otherwise hit
"File not found" on the very path ``Read`` just accepted, and
``Write("PWD/x", ...)`` would silently create a literal ``PWD/`` folder
relative to the process CWD instead of inside ``work_dir`` (which is a
different directory entirely when the agent runs inside a git worktree).

Each test exercises a real filesystem fixture (no mocks).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools


@pytest.fixture
def work_and_cwd():
    """Yield (work_dir, cwd) as two *distinct* real directories.

    The process CWD is deliberately set to a separate directory so that
    ``PWD/`` expansion (which must target ``work_dir``) is observably
    different from naive CWD-relative resolution.
    """
    work_dir = Path(tempfile.mkdtemp()).resolve()
    other_cwd = Path(tempfile.mkdtemp()).resolve()
    prev = Path.cwd()
    os.chdir(other_cwd)
    yield work_dir, other_cwd
    os.chdir(prev)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(other_cwd, ignore_errors=True)


def test_write_pwd_prefix_lands_in_work_dir(work_and_cwd) -> None:
    """``Write("PWD/x")`` must create ``<work_dir>/x`` (not ``<cwd>/PWD/x``)."""
    work_dir, other_cwd = work_and_cwd
    tools = UsefulTools(work_dir=str(work_dir))

    out = tools.Write("PWD/notes.md", "hello")

    assert "Successfully wrote" in out
    assert (work_dir / "notes.md").read_text() == "hello"
    # The literal ``PWD`` directory must NOT have been created under CWD.
    assert not (other_cwd / "PWD").exists()


def test_edit_pwd_prefix_finds_file_read_accepted(work_and_cwd) -> None:
    """``Edit("PWD/x")`` must edit the same file ``Read("PWD/x")`` accepts."""
    work_dir, _ = work_and_cwd
    (work_dir / "doc.txt").write_text("alpha beta")
    tools = UsefulTools(work_dir=str(work_dir))

    # Read accepts the PWD-prefixed path ...
    assert tools.Read("PWD/doc.txt") == "alpha beta"

    # ... and Edit on the very same path must succeed, not 404.
    out = tools.Edit("PWD/doc.txt", "beta", "gamma")

    assert "Successfully replaced" in out
    assert (work_dir / "doc.txt").read_text() == "alpha gamma"


def test_write_then_edit_roundtrip_via_pwd(work_and_cwd) -> None:
    """A full Write -> Edit roundtrip through the ``PWD/`` prefix works."""
    work_dir, _ = work_and_cwd
    tools = UsefulTools(work_dir=str(work_dir))

    tools.Write("PWD/sub/data.txt", "v1")
    assert (work_dir / "sub" / "data.txt").read_text() == "v1"

    out = tools.Edit("PWD/sub/data.txt", "v1", "v2")
    assert "Successfully replaced" in out
    assert (work_dir / "sub" / "data.txt").read_text() == "v2"


def test_pwd_prefix_uses_cwd_when_no_work_dir(work_and_cwd) -> None:
    """With no ``work_dir`` set, ``PWD/`` falls back to the process CWD."""
    _, other_cwd = work_and_cwd
    tools = UsefulTools()

    tools.Write("PWD/from_cwd.md", "x")

    assert (other_cwd / "from_cwd.md").read_text() == "x"
