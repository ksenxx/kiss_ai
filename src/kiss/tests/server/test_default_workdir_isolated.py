# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The test suite must never point a default server at this repository.

Since the diff/merge review was removed, ``_run_task_inner``
auto-commits a dirty working tree when a non-worktree task ends.  A
``VSCodeServer`` whose ``work_dir`` was never overridden used to fall
back to ``os.getcwd()`` — the developer repository when pytest runs
from the repo root — so any test driving ``_run_task`` on such a
server could commit the developer's in-progress work (this actually
happened once during the removal itself).  The autouse
``_isolated_default_workdir`` fixture in ``src/kiss/tests/conftest.py``
guards against that; this test pins the guard.
"""

from pathlib import Path

from kiss.server.server import VSCodeServer


def _repo_root() -> Path:
    """Return the root of the repository containing this test file."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / ".git").exists():
            return parent
    raise AssertionError("test file is not inside a git repository")


def test_default_server_workdir_is_not_the_developer_repo() -> None:
    """A defaulted ``VSCodeServer.work_dir`` must not be this repo."""
    server = VSCodeServer()
    work_dir = Path(server.work_dir).resolve()
    repo = _repo_root()
    assert work_dir != repo
    assert repo not in (work_dir, *work_dir.parents), (
        f"default work_dir {work_dir} points inside the developer repo "
        f"{repo}; the _isolated_default_workdir conftest guard is broken"
    )
    assert work_dir.exists()
