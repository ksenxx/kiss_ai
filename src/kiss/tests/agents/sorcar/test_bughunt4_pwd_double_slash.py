# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt4: ``PWD//path`` must not escape the working directory.

``_expand_pwd_prefix("PWD//etc/passwd", base)`` computed
``os.path.join(base, "/etc/passwd")`` — and ``os.path.join`` discards
*base* when the second component is absolute, so the expansion escaped
the agent's working directory entirely instead of resolving inside it.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.useful_tools import UsefulTools, _expand_pwd_prefix


def test_expand_pwd_double_slash_stays_in_work_dir(tmp_path: Path) -> None:
    """``PWD//sub/f.txt`` must resolve under work_dir, not to ``/sub/f.txt``."""
    out = _expand_pwd_prefix("PWD//sub/f.txt", str(tmp_path))
    assert Path(out) == tmp_path / "sub" / "f.txt"


def test_expand_pwd_many_slashes(tmp_path: Path) -> None:
    """Any number of extra slashes after PWD/ must stay inside work_dir."""
    out = _expand_pwd_prefix("PWD///x.txt", str(tmp_path))
    assert Path(out) == tmp_path / "x.txt"


def test_expand_pwd_trailing_slashes_only(tmp_path: Path) -> None:
    """``PWD//`` (slashes only, no suffix) must resolve to work_dir itself."""
    out = _expand_pwd_prefix("PWD//", str(tmp_path))
    assert Path(out) == tmp_path


def test_write_then_read_pwd_double_slash(tmp_path: Path) -> None:
    """End-to-end: Write/Read with a PWD// path lands inside work_dir."""
    tools = UsefulTools(work_dir=str(tmp_path))
    msg = tools.Write("PWD//x.txt", "hello")
    assert "Successfully" in msg
    assert (tmp_path / "x.txt").read_text() == "hello"
    assert tools.Read("PWD//x.txt") == "hello"


def test_plain_pwd_paths_unchanged(tmp_path: Path) -> None:
    """Regression guard: normal PWD expansion behaviour is preserved."""
    assert _expand_pwd_prefix("PWD", str(tmp_path)) == str(tmp_path)
    assert Path(_expand_pwd_prefix("PWD/a.txt", str(tmp_path))) == tmp_path / "a.txt"
    assert _expand_pwd_prefix("/abs/p.txt", str(tmp_path)) == "/abs/p.txt"
    assert _expand_pwd_prefix("PWDFOO/x", str(tmp_path)) == "PWDFOO/x"
