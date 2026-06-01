"""Integration tests for trajectory-viewer job discovery.

Verify that :func:`kiss.viz_trajectory.server.list_jobs` aggregates jobs from
every ``.kiss.artifacts/jobs`` root beneath the project - the main directory,
git worktrees, nested artifact directories - so the viewer loads all jobs, and
that :func:`load_job_trajectories` resolves jobs stored in those extra roots.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import yaml

from kiss.viz_trajectory.server import (
    find_job_dir,
    list_jobs,
    load_job_trajectories,
)


def _write_trajectory(jobs_root: Path, job: str, name: str) -> None:
    """Create a minimal trajectory YAML under ``<jobs_root>/<job>/trajectories``."""
    traj_dir = jobs_root / job / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "id": 1,
        "run_start_timestamp": 100,
        "run_end_timestamp": 200,
        "model": "test-model",
        "command": "do something",
        "step_count": 3,
        "max_steps": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    (traj_dir / "trajectory_0.yaml").write_text(yaml.safe_dump(data))


class TestJobDiscovery(TestCase):
    """Test aggregation of jobs across multiple artifact roots."""

    def setUp(self) -> None:
        self.project = Path(tempfile.mkdtemp(prefix="kiss_viz_test_"))
        self.main_jobs = self.project / ".kiss.artifacts" / "jobs"
        self.main_jobs.mkdir(parents=True)

    def test_main_jobs_listed(self) -> None:
        """Jobs in the primary artifact root are listed."""
        _write_trajectory(self.main_jobs, "job_2024_01_01_00_00_00_1", "Main")
        jobs = list_jobs(self.main_jobs)
        self.assertEqual([j["name"] for j in jobs], ["job_2024_01_01_00_00_00_1"])
        self.assertEqual(jobs[0]["trajectory_count"], 1)

    def test_worktree_jobs_aggregated(self) -> None:
        """Jobs stored inside a git worktree's artifact root are also listed."""
        _write_trajectory(self.main_jobs, "job_2024_01_01_00_00_00_1", "Main")
        wt_jobs = (
            self.project / ".kiss-worktrees" / "wt1" / ".kiss.artifacts" / "jobs"
        )
        wt_jobs.mkdir(parents=True)
        _write_trajectory(wt_jobs, "job_2024_02_02_00_00_00_2", "Worktree")

        names = {j["name"] for j in list_jobs(self.main_jobs)}
        self.assertEqual(
            names, {"job_2024_01_01_00_00_00_1", "job_2024_02_02_00_00_00_2"}
        )

    def test_nested_and_sibling_jobs_aggregated(self) -> None:
        """Jobs in nested and sibling artifact roots are listed too."""
        _write_trajectory(self.main_jobs, "job_2024_01_01_00_00_00_1", "Main")
        nested = self.main_jobs / ".kiss.artifacts" / "jobs"
        nested.mkdir(parents=True)
        _write_trajectory(nested, "job_2024_03_03_00_00_00_3", "Nested")
        sibling = self.project / "src" / ".kiss.artifacts" / "jobs"
        sibling.mkdir(parents=True)
        _write_trajectory(sibling, "job_2024_04_04_00_00_00_4", "Sibling")

        names = {j["name"] for j in list_jobs(self.main_jobs)}
        self.assertEqual(
            names,
            {
                "job_2024_01_01_00_00_00_1",
                "job_2024_03_03_00_00_00_3",
                "job_2024_04_04_00_00_00_4",
            },
        )

    def test_sorted_newest_first(self) -> None:
        """Aggregated jobs are sorted newest-first by their timestamp."""
        _write_trajectory(self.main_jobs, "job_2024_01_01_00_00_00_1", "Old")
        wt_jobs = (
            self.project / ".kiss-worktrees" / "wt1" / ".kiss.artifacts" / "jobs"
        )
        wt_jobs.mkdir(parents=True)
        _write_trajectory(wt_jobs, "job_2024_12_31_00_00_00_2", "New")

        names = [j["name"] for j in list_jobs(self.main_jobs)]
        self.assertEqual(
            names, ["job_2024_12_31_00_00_00_2", "job_2024_01_01_00_00_00_1"]
        )

    def test_load_trajectories_from_worktree_job(self) -> None:
        """Trajectories load for a job that lives only in a worktree root."""
        wt_jobs = (
            self.project / ".kiss-worktrees" / "wt1" / ".kiss.artifacts" / "jobs"
        )
        wt_jobs.mkdir(parents=True)
        _write_trajectory(wt_jobs, "job_2024_02_02_00_00_00_2", "Worktree")

        trajectories = load_job_trajectories(self.main_jobs, "job_2024_02_02_00_00_00_2")
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["name"], "Worktree")

    def test_find_missing_job_returns_none(self) -> None:
        """A job name absent from every root resolves to ``None``."""
        self.assertIsNone(find_job_dir(self.main_jobs, "job_does_not_exist_0"))

    def test_non_standard_root_only_scans_directly(self) -> None:
        """A jobs dir not under ``.kiss.artifacts`` is scanned directly only."""
        plain = Path(tempfile.mkdtemp(prefix="kiss_viz_plain_")) / "jobs"
        plain.mkdir(parents=True)
        _write_trajectory(plain, "job_2024_05_05_00_00_00_5", "Plain")
        # A sibling artifact root must NOT be aggregated for a non-standard dir.
        names = {j["name"] for j in list_jobs(plain)}
        self.assertEqual(names, {"job_2024_05_05_00_00_00_5"})
