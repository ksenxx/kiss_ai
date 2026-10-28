# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test for the 2026-10-26 sweep of ``kiss.core.config``.

``get_jobs_root`` (and the in-file ``_artifact_root`` it calls) used to
take an optional ``base_dir`` that no caller in the repository passed; the
artifact root is fixed for the process (see
``test_artifact_dir_and_step_limit``).  The parameter is gone; this test
pins the only remaining behaviour: the jobs root is
``<checkout>/.kiss.artifacts/jobs`` and the per-process artifact
directory lives directly under it.
"""

from __future__ import annotations

from pathlib import Path

from kiss.core.config import get_artifact_dir, get_jobs_root

_CHECKOUT = Path(__file__).resolve().parents[4]


def test_jobs_root_is_the_checkout_artifact_jobs_dir() -> None:
    """``get_jobs_root()`` names ``<checkout>/.kiss.artifacts/jobs``."""
    jobs_root = get_jobs_root()
    assert jobs_root == _CHECKOUT / ".kiss.artifacts" / "jobs"
    assert jobs_root.is_absolute()
    assert Path(get_artifact_dir()).parent == jobs_root
