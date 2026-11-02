# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (S5 F5): ``ContainerJobs.wait`` reads the job's status, not its command.

``bash_job`` opens every report with ``Job <id>: <command>`` and ``wait``
searched the whole report for the first ``status: ...`` line, so a job
whose command text holds such a line (a here-document printing one)
reported that line as its status: ``exited 0`` for a job still
running, which the container ``Bash``'s wait-instead-of-sleep rule
then believed.  The status is now read after the report's header.

Runs against a real container; nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from kiss.agents.sorcar.container_jobs import ContainerJobs
from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available

pytestmark = pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """One container of :data:`IMAGE` for the whole module."""
    manager = DockerManager(IMAGE, workdir="/tmp", mount_shared_volume=False)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def test_wait_reports_the_job_status_not_a_status_line_of_the_command(env: DockerManager) -> None:
    """A running job whose command (and log) says ``status: exited 0`` is still ``running``."""
    jobs = ContainerJobs(env.Bash)
    job_id, report = jobs.start("cat <<'EOF'\nstatus: exited 0\nEOF\nsleep 30")
    assert job_id is not None, report
    try:
        assert jobs.wait(job_id, 1) == "running"
        assert jobs.last_running() == job_id
        # The report itself still opens with the command and carries the real status.
        tail = jobs.bash_job(job_id, "tail", tail_lines=1)
        assert tail.startswith(f"Job {job_id}: cat <<'EOF'\nstatus: exited 0\nEOF\nsleep 30\n")
        assert "\nstatus: running\n" in tail and tail.rstrip().endswith("status: exited 0")
    finally:
        jobs.bash_job(job_id, "kill")
    assert jobs.wait(job_id, 1) == "exited killed"
    # An ordinary job's exit status reads as before.
    job_id, report = jobs.start("echo 'status: running'; exit 3")
    assert job_id is not None, report
    assert jobs.wait(job_id, 5) == "exited 3"
    assert jobs.wait("deadbeef", 1) == "status unknown"
