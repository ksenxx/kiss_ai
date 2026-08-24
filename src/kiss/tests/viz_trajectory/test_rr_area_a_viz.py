# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the area-A fixes in the trajectory viz server.

Covers:

* A-R5 — the ``/api/jobs/<job>/trajectories`` route resolves the job
  directory once and hands it to ``load_job_trajectories``; the new
  optional ``job_dir`` parameter is honored and the two-argument call
  keeps resolving by name (backward compatibility).
* A-RC1 — ``_add_job_dirs`` must not crash when a recorded job
  directory is deleted by another process before its mtime is compared,
  and a same-named surviving candidate must replace the deleted
  recording instead of leaving a dead path mapped.
  The exact between-``glob``-and-``stat`` interleaving inside a single
  call cannot be reproduced without test doubles, but the identical
  code path — ``existing.stat()`` on a path that no longer exists — is
  reached deterministically here by deleting a directory recorded in
  ``found`` before the next root is scanned, exactly what a concurrent
  worktree cleanup does between two roots of the discovery walk.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

import yaml

import kiss.viz_trajectory.server as viz_server
from kiss.viz_trajectory.server import (
    _add_job_dirs,
    find_job_dir,
    load_job_trajectories,
)


def _write_trajectory(job_dir: Path, name: str, start: int) -> None:
    """Write a minimal trajectory YAML the parser accepts.

    Args:
        job_dir: Job directory receiving the ``trajectories`` subdir.
        name: Agent name recorded in the trajectory.
        start: Run start timestamp used for ordering.
    """
    trajectories = job_dir / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "id": 1,
        "run_start_timestamp": start,
        "run_end_timestamp": start + 10,
        "model": "test-model",
        "command": "do things",
        "step_count": 2,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with (trajectories / f"trajectory_{start}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)


class AddJobDirsPrunedDirTest(unittest.TestCase):
    """A-RC1: a job dir pruned mid-discovery must not kill the walk."""

    def setUp(self) -> None:
        self.tmp = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deleted_recorded_dir_does_not_crash_the_scan(self) -> None:
        """A pruned recorded copy is replaced by the surviving duplicate."""
        root_a = self.tmp / "a" / "jobs"
        root_b = self.tmp / "b" / "jobs"
        (root_a / "job_2024_01_01_00_00_00_1").mkdir(parents=True)
        (root_b / "job_2024_01_01_00_00_00_1").mkdir(parents=True)
        (root_b / "job_2024_01_01_00_00_00_2").mkdir(parents=True)

        found: dict[str, Path] = {}
        _add_job_dirs(root_a, found)
        # Another process (worktree cleanup) prunes the recorded copy
        # before the walk reaches the second root.
        shutil.rmtree(root_a / "job_2024_01_01_00_00_00_1")

        _add_job_dirs(root_b, found)  # must not raise FileNotFoundError

        # The scan carried on: the job unique to root_b was recorded.
        self.assertIn("job_2024_01_01_00_00_00_2", found)
        # And the duplicate name now maps to the copy that still exists,
        # not the deleted path recorded from the first root.
        self.assertEqual(
            found["job_2024_01_01_00_00_00_1"],
            root_b / "job_2024_01_01_00_00_00_1",
        )

# The remaining branch of the split stat — the *candidate* ``job_dir``
# vanishing so its own ``stat()`` raises and the recorded copy is kept —
# is reachable only when the deletion lands between ``job_dir.is_dir()``
# and ``job_dir.stat()`` inside one loop iteration.  ``is_dir()`` swallows
# every ``OSError`` (a pruned or permission-blocked path just returns
# False and the loop ``continue``s earlier), so the branch cannot be
# reached deterministically without instrumenting or doubling the code
# under test; it is documented here instead.

    def test_scan_still_prefers_newer_copy_when_both_exist(self) -> None:
        """The mtime comparison itself is unchanged for surviving dirs."""
        root_a = self.tmp / "a" / "jobs"
        root_b = self.tmp / "b" / "jobs"
        old = root_a / "job_2024_01_01_00_00_00_1"
        new = root_b / "job_2024_01_01_00_00_00_1"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        import os

        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))

        found: dict[str, Path] = {}
        _add_job_dirs(root_a, found)
        _add_job_dirs(root_b, found)

        self.assertEqual(found["job_2024_01_01_00_00_00_1"], new)


class LoadJobTrajectoriesResolvedOnceTest(unittest.TestCase):
    """A-R5: the pre-resolved job_dir is used instead of a second walk."""

    def setUp(self) -> None:
        self.tmp = Path(mkdtemp())
        self.jobs = self.tmp / "jobs"
        self.jobs.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_explicit_job_dir_is_used_not_rediscovered(self) -> None:
        """A job dir invisible to discovery still loads when passed in."""
        hidden = self.tmp / "elsewhere" / "job_2024_03_03_00_00_00_9"
        _write_trajectory(hidden, "hidden-agent", 100)
        # Discovery from the artifact dir cannot see it...
        self.assertIsNone(find_job_dir(self.jobs, hidden.name))
        # ...but the pre-resolved path loads without re-resolution.
        loaded = load_job_trajectories(self.jobs, hidden.name, job_dir=hidden)
        self.assertEqual([t["name"] for t in loaded], ["hidden-agent"])

    def test_two_argument_call_still_resolves_by_name(self) -> None:
        """Backward compatibility: omitting job_dir resolves as before."""
        job = self.jobs / "job_2024_03_03_00_00_00_1"
        _write_trajectory(job, "agent-one", 5)
        loaded = load_job_trajectories(self.jobs, job.name)
        self.assertEqual([t["name"] for t in loaded], ["agent-one"])

    def test_unknown_job_without_job_dir_returns_empty(self) -> None:
        """A name that resolves nowhere yields an empty list, not an error."""
        self.assertEqual(load_job_trajectories(self.jobs, "job_none_0"), [])

    def test_route_returns_trajectories_end_to_end(self) -> None:
        """The Flask route serves trajectories after the single resolution."""
        job = self.jobs / "job_2024_03_03_00_00_00_2"
        _write_trajectory(job, "route-agent", 7)
        old_dir = viz_server.ARTIFACT_DIR
        viz_server.ARTIFACT_DIR = self.jobs
        try:
            client = viz_server.app.test_client()
            response = client.get(f"/api/jobs/{job.name}/trajectories")
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual([t["name"] for t in body], ["route-agent"])

            missing = client.get("/api/jobs/job_missing_0/trajectories")
            self.assertEqual(missing.status_code, 404)
        finally:
            viz_server.ARTIFACT_DIR = old_dir


if __name__ == "__main__":
    unittest.main()
