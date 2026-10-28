"""A cron prompt job whose prompt is a CHANNEL slash command must run.

Sweep 2026-10-26, slice ``sorcar-cron-sea``, finding F1: ``_run_prompt_job``
passed the run's scratch directory as an *explicit* ``work_dir`` option on
every prompt job.  A slash-command SEA deriving from ``ChannelSea`` locks
``work_dir`` to ``$KISS_HOME/channel_work``, so a scheduled ``/slack …`` /
``/gmail …`` / any ``/<channel> …`` prompt was refused on every run with
"the script locks work_dir=…".  The scratch directory is already the
dispatcher's default (``make_run_agent_tool(scratch_dir)``), so the option is
only needed when the job names its own ``work_dir``.

End-to-end against the real SEA registry (a ``ChannelSea`` registered through
``$KISS_HOME/SEAS.md``) and the real dispatcher; only ``daemon_client.run`` is
captured so no daemon is needed.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent, daemon_client, sea_commands

CHANNEL_SEA = (
    "from kiss.agents.seas.base.base_sea import ChannelSea\n"
    "\n"
    "class MychanSea(ChannelSea):\n"
    "    def description(self) -> str:\n"
    "        return 'a channel'\n"
)


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate KISS_HOME and register a ``/mychan`` channel SEA through SEAS.md."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    seas = tmp_path / "seas" / "mychan"
    seas.mkdir(parents=True)
    (seas / "mychan_sea.py").write_text(CHANNEL_SEA)
    (home / "SEAS.md").write_text(str(tmp_path / "seas") + "\n")
    sea_commands._reset_for_tests()
    assert sea_commands.get_command("mychan") is not None
    yield home
    sea_commands._reset_for_tests()


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def capture_run(prompt: str, **kwargs: object) -> daemon_client.TaskResult:
        captured.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(text="done", success=True, cost=0.0, tokens=0, steps=0)

    monkeypatch.setattr(daemon_client, "run", capture_run)
    return captured


def test_channel_slash_prompt_job_runs_in_the_channel_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _home: Path,
) -> None:
    """A ``/mychan …`` prompt job runs the channel SEA in its locked work dir."""
    captured = _capture(monkeypatch)
    scratch = tmp_path / "run"
    scratch.mkdir()
    job = {"id": "abcd1234", "name": "chan", "prompt": "/mychan post the daily digest"}
    status, text = cron_agent._run_prompt_job(
        job, scratch, cron_agent._job_work_dir(job, scratch), cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert (status, text) == ("ok", "done")
    assert captured[0]["work_dir"] == str(_home / "channel_work")
    assert captured[0]["auto_classify"] is False


def test_plain_prompt_job_still_runs_in_the_scratch_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a job ``work_dir`` a plain prompt runs in the scratch directory (the default)."""
    captured = _capture(monkeypatch)
    scratch = tmp_path / "run"
    scratch.mkdir()
    job = {"id": "abcd1234", "name": "plain", "prompt": "summarise the day"}
    status, _ = cron_agent._run_prompt_job(
        job, scratch, cron_agent._job_work_dir(job, scratch), cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert status == "ok"
    assert captured[0]["work_dir"] == str(scratch)


def test_job_work_dir_is_still_passed_and_still_clashes_with_a_channel_sea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job that names a ``work_dir`` keeps it; a channel command then clashes (clearly)."""
    captured = _capture(monkeypatch)
    scratch = tmp_path / "run"
    scratch.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    job = {"id": "abcd1234", "name": "proj", "prompt": "lint it", "work_dir": str(project)}
    status, _ = cron_agent._run_prompt_job(
        job, scratch, cron_agent._job_work_dir(job, scratch), cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert status == "ok"
    assert captured[0]["work_dir"] == str(project)

    job["prompt"] = "/mychan post"
    status, text = cron_agent._run_prompt_job(
        job, scratch, cron_agent._job_work_dir(job, scratch), cron_agent.PROMPT_TIMEOUT_SECONDS,
    )
    assert status == "error"
    assert "locks work_dir=" in str(text)
    assert len(captured) == 1
