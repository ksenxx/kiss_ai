# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test for the 2026-10-26 sweep of ``kiss.core.config``.

``get_jobs_root`` (and an in-file ``_artifact_root`` helper, since inlined
into it) used to
take an optional ``base_dir`` that no caller in the repository passed; the
artifact root is fixed for the process (see
``test_artifact_dir_and_step_limit``).  The parameter is gone; this test
pins the only remaining behaviour: the jobs root is
``<checkout>/.kiss.artifacts/jobs`` and the per-process artifact
directory lives directly under it.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kiss.core.config import get_artifact_dir, get_jobs_root

_CHECKOUT = Path(__file__).resolve().parents[4]

# A process that waits for the instant named by its argument, then draws
# the artifact directory name every such process draws: same second, same seed.
_SAME_DRAW_WORKER = (
    "import random, sys, time\n"
    "from kiss.core.config import get_artifact_dir\n"
    "time.sleep(max(0.0, float(sys.argv[1]) - time.time()))\n"
    "random.seed(20261111)\n"
    "print(get_artifact_dir())\n"
)


def test_jobs_root_is_the_checkout_artifact_jobs_dir() -> None:
    """``get_jobs_root()`` names ``<checkout>/.kiss.artifacts/jobs``."""
    jobs_root = get_jobs_root()
    assert jobs_root == _CHECKOUT / ".kiss.artifacts" / "jobs"
    assert jobs_root.is_absolute()
    assert Path(get_artifact_dir()).parent == jobs_root


def test_processes_that_draw_the_same_name_get_an_artifact_dir_each() -> None:
    """Processes starting in one second with one random number do not share a directory.

    The directory was made with ``exist_ok=True``, so a name another
    process had just taken was handed out again and the two saved
    their trajectories over each other.  It is made exclusively now:
    a taken name is drawn again, in the same ``job_<time>_<number>``
    format.
    """
    for lead in (3, 6, 12):  # seconds for every worker to be up; a late one voids the round
        start = int(time.time()) + lead + 0.2
        workers = [
            subprocess.Popen(
                [sys.executable, "-c", _SAME_DRAW_WORKER, str(start)],
                stdout=subprocess.PIPE, text=True,
            )
            for _ in range(4)
        ]
        printed = [worker.communicate(timeout=120)[0].strip() for worker in workers]
        dirs = [Path(line) for line in printed if line]
        try:
            assert len(dirs) == len(workers), "a worker printed no directory"
            assert all(d.is_dir() and d.parent == get_jobs_root() for d in dirs)
            assert all(re.fullmatch(r"job_\d{4}(?:_\d{2}){5}_\d{1,7}", d.name) for d in dirs)
            same_second = len({d.name.rsplit("_", 1)[0] for d in dirs}) == 1
            if same_second:
                assert len(set(dirs)) == len(dirs), f"shared: {sorted(d.name for d in dirs)}"
                return
        finally:
            for d in set(dirs):
                d.rmdir()
    pytest.fail("the workers never drew their names within one second")
