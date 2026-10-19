# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cron (2026-10-18 concurrency review, slice E): in_chat work dir and run_now snapshot.

E1-F1: an ``in_chat`` job without a ``work_dir`` ran in the per-run
scratch directory, which ``_execute_job`` removes when the run ends —
so the chat the user is supposed to continue in pointed at a deleted
directory.  Such a job now runs in its own stable directory
``$KISS_HOME/cron/chats/<job_id>``.

E1-F2: ``cron_job("run_now")`` loaded the job before taking the store
lock and ran that copy, so an edit committed while it waited for the
lock was not seen.  The load now happens under the lock.

Real job store under an isolated ``KISS_HOME``, real threads, real
shell commands; no mocks.  The daemon is unreachable on purpose (the
endpoint file does not exist), which is a recorded error, not a mock.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.cron_agent import cron_job, load_jobs
from kiss.tests.conftest import is_root, posix_only


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the job store and point the daemon endpoint at a missing file."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    return tmp_path


def _created(reply: str) -> dict:
    parsed = yaml.safe_load(reply)
    assert "created" in parsed, parsed
    return dict(parsed["created"])


def test_in_chat_job_without_work_dir_runs_in_a_stable_chat_directory(tmp_path: Path) -> None:
    """The chat's work directory exists after the run and is the same on the next run."""
    job = _created(
        cron_job(
            "create",
            name="morning",
            prompt="/good_morning now",
            schedule="0 8 * * *",
            deliver="none",
            in_chat=True,
        )
    )
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert not stored.get("work_dir")

    reply = yaml.safe_load(cron_job("run_now", job_id=job["id"]))
    assert reply["ran"]["last_status"] == "error"  # no daemon: the run is recorded as failed
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert "Cannot connect" in stored["last_summary"]

    chat_dir = tmp_path / "cron" / "chats" / job["id"]
    assert chat_dir.is_dir(), "the in_chat run's work directory must survive the run"
    assert cron_agent._job_work_dir(stored, tmp_path / "scratch") == chat_dir
    # The per-run scratch directories are still cleaned up.
    runs_dir = tmp_path / "cron" / "runs"
    assert not runs_dir.exists() or list(runs_dir.iterdir()) == []
    # A job with its own work_dir keeps it; a plain prompt job keeps the scratch.
    assert (
        cron_agent._job_work_dir(
            {**stored, "work_dir": str(tmp_path)},
            tmp_path / "scratch",
        )
        == tmp_path
    )
    assert (
        cron_agent._job_work_dir(
            {**stored, "in_chat": False},
            tmp_path / "scratch",
        )
        == tmp_path / "scratch"
    )


@posix_only("directory permission bits")
@pytest.mark.skipif(is_root(), reason="root ignores directory permission bits")
def test_in_chat_directory_that_cannot_be_created_is_a_recorded_run_error(
    tmp_path: Path,
) -> None:
    """A failed chat-directory mkdir is the run's error, and the scratch dir is removed."""
    job = _created(
        cron_job(
            "create",
            name="morning",
            prompt="/good_morning now",
            schedule="0 8 * * *",
            deliver="none",
            in_chat=True,
        )
    )
    chats_dir = tmp_path / "cron" / "chats"
    chats_dir.mkdir(parents=True)
    os.chmod(chats_dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        reply = yaml.safe_load(cron_job("run_now", job_id=job["id"]))
    finally:
        os.chmod(chats_dir, stat.S_IRWXU)
    assert reply["ran"]["last_status"] == "error", reply
    stored = next(j for j in load_jobs() if j["id"] == job["id"])
    assert stored["last_status"] == "error"
    assert "PermissionError" in stored["last_summary"], stored["last_summary"]
    assert not (chats_dir / job["id"]).exists()
    runs_dir = tmp_path / "cron" / "runs"
    assert not runs_dir.exists() or list(runs_dir.iterdir()) == []


def test_run_now_executes_the_job_as_stored_when_the_lock_is_released(tmp_path: Path) -> None:
    """An edit committed while ``run_now`` waits for the store lock is what runs."""
    job = _created(
        cron_job(
            "create",
            name="echo",
            command="echo OLD-OUTPUT",
            schedule="0 8 * * *",
            deliver="local",
        )
    )
    replies: list[str] = []

    def run_now() -> None:
        replies.append(cron_job("run_now", job_id=job["id"]))

    with cron_agent._jobs_lock(blocking=True):
        thread = threading.Thread(target=run_now, daemon=True)
        thread.start()
        # Let run_now reach the store lock (and, before the fix, load its
        # stale copy of the job) while this thread still holds it.
        time.sleep(1.0)
        assert thread.is_alive()
        jobs = load_jobs()
        for stored in jobs:
            if stored["id"] == job["id"]:
                stored["command"] = "echo NEW-OUTPUT"
        cron_agent.save_jobs(jobs)
    thread.join(timeout=60)
    assert not thread.is_alive()

    reply = yaml.safe_load(replies[0])
    assert reply["ran"]["last_status"] == "ok", reply
    output = (tmp_path / "cron" / "output" / f"{job['id']}.md").read_text()
    assert "NEW-OUTPUT" in output
    assert "OLD-OUTPUT" not in output


def test_run_now_of_a_job_removed_while_waiting_for_the_lock_does_not_run(
    tmp_path: Path,
) -> None:
    """A job removed under the lock is reported as missing, not run once more."""
    job = _created(
        cron_job(
            "create",
            name="gone",
            command="echo SHOULD-NOT-RUN",
            schedule="0 8 * * *",
            deliver="local",
        )
    )
    replies: list[str] = []

    def run_now() -> None:
        replies.append(cron_job("run_now", job_id=job["id"]))

    with cron_agent._jobs_lock(blocking=True):
        thread = threading.Thread(target=run_now, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert thread.is_alive()
        cron_agent.save_jobs([j for j in load_jobs() if j["id"] != job["id"]])
    thread.join(timeout=60)
    assert not thread.is_alive()

    reply = yaml.safe_load(replies[0])
    assert reply == {"error": f"no job with id {job['id']!r}"}, reply
    assert not (tmp_path / "cron" / "output" / f"{job['id']}.md").exists()
