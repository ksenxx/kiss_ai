# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the Terminal-Bench harness (daemon home, launcher, scoring).

``SeaHarborAgent.run`` itself needs Harbor's Docker environment, a running
benchmark daemon and a paid model, so it is exercised by a real ``harbor run``
smoke trial rather than here; :func:`trial.run_trial` is covered on its
failure path (no daemon), which runs the same client code.  The agent class
test needs the ``harbor`` package and is skipped without it.
``main_container`` (Compose-label lookup of the ``main`` service) and
``skills_note`` need a running Harbor environment: they were verified with
smoke trials on ``nextjs-performance`` (a ``network_mode: service:`` task)
and ``cumulative-layout-shift`` (a task with a ``skills_dir``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kiss.agents.seas.coding import coding_sea
from kiss.benchmarks.terminal_bench import daemon, run, summarize, trial
from kiss.tests.conftest import IS_WINDOWS, posix_only


def write_trial(job_dir: Path, task: str, suffix: str, reward: float | None, cost: float | None,
                exception: str | None = None, minutes: float = 10.0) -> Path:
    """Write a Harbor-style ``result.json`` for one trial and return its directory."""
    start = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    trial = job_dir / f"{task}__{suffix}"
    trial.mkdir(parents=True)
    result = {
        "trial_name": trial.name,
        "task_name": f"terminal-bench/{task}",
        "agent_result": None if cost is None else {"cost_usd": cost, "metadata": {"tokens": 1000}},
        "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
        "exception_info": None if exception is None else {"exception_type": exception},
        "agent_execution": {
            "started_at": start.isoformat(),
            "finished_at": (start + timedelta(minutes=minutes)).isoformat(),
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    return trial


def test_summarize_scores_job_against_dataset(tmp_path: Path) -> None:
    job = tmp_path / "job"
    dataset = tmp_path / "dataset"
    for name in ("alpha", "beta", "gamma"):
        (dataset / name).mkdir(parents=True)
        (dataset / name / "task.toml").write_text("schema_version = '1.0'\n")
    (dataset / "not-a-task").mkdir()  # no task.toml: ignored
    write_trial(job, "alpha", "a1", 1.0, 2.5)
    write_trial(job, "alpha", "a2", 0.0, 1.5, exception="AgentTimeoutError")
    write_trial(job, "beta", "b1", None, None, exception="EnvironmentStartError", minutes=0)
    (job / "config.json").write_text("{}")  # not a trial directory
    (job / "no-result__x").mkdir()  # unfinished trial: no result.json
    (job / "weird__y").mkdir()
    (job / "weird__y" / "result.json").write_text(json.dumps({"other": 1}))  # not a trial

    summary = summarize.summarize([job], dataset, attempts=2)

    assert list(summary["tasks"]) == ["alpha", "beta", "gamma"]
    alpha = summary["tasks"]["alpha"]
    assert (alpha["attempts"], alpha["passed"], alpha["mean_reward"]) == (2, 1, 0.5)
    assert alpha["cost_usd"] == 4.0 and alpha["tokens"] == 2000
    assert alpha["errors"] == ["AgentTimeoutError"] and alpha["seconds"] == [600.0, 600.0]
    beta = summary["tasks"]["beta"]
    assert (beta["attempts"], beta["passed"]) == (1, 0)
    assert (beta["cost_usd"], beta["seconds"]) == (0.0, [0.0])
    gamma = summary["tasks"]["gamma"]
    assert (gamma["attempts"], gamma["passed"], gamma["mean_reward"]) == (0, 0, 0.0)
    overall = summary["overall"]
    assert overall["tasks"] == 3 and overall["tasks_with_trials"] == 2
    assert overall["attempts_run"] == 3 and overall["attempts_passed"] == 1
    assert overall["mean_reward"] == pytest.approx(1 / 6)
    assert overall["tasks_solved_at_least_once"] == 1 and overall["errors"] == 2
    assert overall["cost_usd"] == 4.0 and overall["tokens"] == 2000

    markdown = summarize.to_markdown(summary)
    assert "**16.7%** (1 of 6 attempts, 3 tasks x 2 attempts)" in markdown
    assert "| alpha | 1 | 2 | 4.00 | 10, 10 | AgentTimeoutError |" in markdown
    assert "| gamma | 0 | 0 | 0.00 | - |  |" in markdown


def test_summarize_without_dataset_and_zero_attempts(tmp_path: Path) -> None:
    job = tmp_path / "job"
    write_trial(job, "alpha", "a1", 1.0, 0.25)
    summary = summarize.summarize([job], None, attempts=0)
    assert list(summary["tasks"]) == ["alpha"]
    assert summary["tasks"]["alpha"]["mean_reward"] == 0.0
    assert summary["overall"]["mean_reward"] == 0.0
    summary = summarize.summarize([job], None, attempts=1)
    assert summary["overall"]["mean_reward"] == 1.0


def test_summarize_rerun_job_replaces_a_task(tmp_path: Path) -> None:
    job = tmp_path / "job"
    rerun = tmp_path / "rerun"
    write_trial(job, "alpha", "a1", 1.0, 1.0)
    write_trial(job, "beta", "b1", None, None, exception="ValueError")
    write_trial(job, "beta", "b2", None, None, exception="ValueError")
    write_trial(rerun, "beta", "b3", 1.0, 3.0)
    write_trial(rerun, "beta", "b4", 0.0, 2.0)
    summary = summarize.summarize([job, rerun], None, attempts=2)
    beta = summary["tasks"]["beta"]
    assert (beta["attempts"], beta["passed"], beta["cost_usd"], beta["errors"]) == (2, 1, 5.0, [])
    assert summary["overall"]["job"] == "job + rerun"
    assert summary["overall"]["attempts_passed"] == 2 and summary["overall"]["errors"] == 0


def test_summarize_cli_writes_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job = tmp_path / "job"
    write_trial(job, "alpha", "a1", 1.0, 0.25)
    assert summarize.main([str(job), "--attempts", "1"]) == 0
    assert json.loads((job / "summary.json").read_text())["overall"]["attempts_passed"] == 1
    assert (job / "summary.md").read_text().startswith("# job\n")
    assert "**100.0%**" in capsys.readouterr().out


def test_harbor_command_for_kiss_agent_and_oracle(tmp_path: Path) -> None:
    parser_args = ["--job-name", "j", "--jobs-dir", str(tmp_path), "-x", "gpu-task",
                   "-x", "other/full-name", "--budget-usd", "12.5", "--home", "/h"]
    with pytest.raises(SystemExit):
        run.main(["--jobs-dir", str(tmp_path)])  # --job-name is required
    # The KISS agent needs a running daemon: its endpoint file is missing here.
    assert run.main(parser_args) == 2

    cmd = run.harbor_command(run.build_parser().parse_args(parser_args))
    assert cmd[:4] == ["uv", "run", "harbor", "run"]
    assert cmd[cmd.index("-a") + 1] == run.KISS_AGENT
    assert cmd[cmd.index("-m") + 1] == "claude-opus-5-5"
    assert "budget_usd=12.5" in cmd and "home=/h" in cmd
    assert cmd[cmd.index("--cpus") + 1] == "limit" and cmd[cmd.index("--memory") + 1] == "limit"
    excluded = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-x"]
    assert excluded == ["terminal-bench/gpu-task", "other/full-name"]
    # the defaults: highest effort, the trial's model as reviewer and blind
    # verifier, local Docker
    assert "reasoning_effort=max" in cmd and "reviewer=same" in cmd
    assert "blind_verifier=same" in cmd
    assert "-e" not in cmd and "-i" not in cmd

    gpu = run.harbor_command(run.build_parser().parse_args([
        "--job-name", "g", "--env", "modal", "-i", "gpu-task", "-i", "org/t",
        "--effort", "", "--reviewer", "none", "--blind-verifier", "claude-haiku-5"]))
    assert gpu[gpu.index("-e") + 1] == "modal"
    # the Modal environment needs Harbor's ``modal`` extra at the installed version
    assert gpu[:5] == ["uv", "run", "--with", f"harbor[modal]=={run.harbor_version()}", "harbor"]
    assert run.harbor_version()[0].isdigit()
    assert [gpu[i + 1] for i, a in enumerate(gpu) if a == "-i"] == [
        "terminal-bench/gpu-task", "org/t"]
    assert "reviewer=none" in gpu and not any(a.startswith("reasoning_effort") for a in gpu)
    assert "blind_verifier=claude-haiku-5" in gpu

    oracle_args = run.build_parser().parse_args(["--job-name", "o", "--agent", "oracle"])
    oracle = run.harbor_command(oracle_args)
    assert "-m" not in oracle and "--ak" not in oracle
    assert oracle[oracle.index("-a") + 1] == "oracle"


@posix_only("CreateProcess resolves only uv.exe on PATH; the fake uv is a sh script")
def test_run_main_launches_oracle_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` runs the built command; a fake ``uv`` on PATH records the argv it received."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_file = tmp_path / "argv"
    (fake_bin / "uv").write_text(f"#!/bin/sh\necho \"$@\" > {argv_file}\nexit 7\n")
    (fake_bin / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    assert run.main(["--job-name", "o", "--agent", "oracle", "--jobs-dir", str(tmp_path)]) == 7
    argv = argv_file.read_text().split()
    assert argv[:3] == ["run", "harbor", "run"]
    assert "--job-name" in argv and argv[argv.index("-a") + 1] == "oracle"


def test_prepare_home_copies_keys_and_writes_config(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what Path.home() reads on Windows
    (tmp_path / ".kiss").mkdir()
    (tmp_path / ".kiss" / "api_keys.env").write_text("X=1\n")
    (tmp_path / ".kiss" / "MODEL_INFO.json").write_text("{}")
    home = tmp_path / "bench-home"
    work_dir = daemon.prepare_home(home)
    assert work_dir == home / "scratch" and (work_dir / ".git").is_dir()
    assert (home / "api_keys.env").read_text() == "X=1\n"
    assert (home / "MODEL_INFO.json").read_text() == "{}"
    assert not (home / "MY_MODELS.json").exists()
    config = json.loads((home / "config.json").read_text())
    assert config["use_web_browser"] is False and config["use_memory"] is False
    assert config["classify_tasks"] is False and config["work_dir"] == str(work_dir)
    # A second call keeps the existing copies and repository.
    (home / "api_keys.env").write_text("X=2\n")
    assert daemon.prepare_home(home) == work_dir
    assert (home / "api_keys.env").read_text() == "X=2\n"


def _parent_pid(pid: int) -> int:
    """Return the parent pid of *pid* on Windows.

    The venv's ``python.exe`` there is uv's trampoline, which runs the
    interpreter as a child, so the daemon's own pid is that child's.
    """
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').ParentProcessId"],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()
    return int(out)


def test_daemon_runs_no_cron_scheduler(tmp_path: Path) -> None:
    """The benchmark daemon neither adopts the user's cron jobs nor seeds the defaults.

    The user's daemon home (``~/.kiss``) holds a scheduled job and is named
    by the ``~/.kiss-web-home`` hand-over marker; a kiss-web daemon started
    from a sibling home would adopt that job and seed the default ones.
    The benchmark daemon must do neither (they would spend money mid-run)
    and must leave the marker alone, so the user's next daemon start does
    not adopt from the benchmark home.
    """
    import socket

    user_home = tmp_path / ".kiss"
    (user_home / "cron").mkdir(parents=True)
    job = {"id": "abcd1234", "name": "Run the tests", "prompt": "", "command": "pytest -q",
           "schedule": "every 2h", "deliver": "none", "model_name": "", "max_budget": "",
           "work_dir": "", "timeout": "", "enabled": True, "one_shot": False,
           "until_delivered": False, "in_chat": False, "created_at": 1.0,
           "next_run_at": 4e9, "failure_streak": 0}
    (user_home / "cron" / "jobs.json").write_text(json.dumps([job]))
    marker = tmp_path / ".kiss-web-home"
    marker.write_text(f"{user_home}\n")
    home = tmp_path / ".kiss-tb"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    env.pop("KISS_HOME", None)
    log = tmp_path / "daemon.log"
    with log.open("w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "kiss.benchmarks.terminal_bench.daemon",
             "--home", str(home), "--port", str(port)],
            env=env, stdout=subprocess.DEVNULL, stderr=log_file, text=True,
        )
    try:
        # The ready line is logged after the cron decision (the endpoint
        # file is published before it), so it is the milestone to wait for.
        deadline = time.monotonic() + 120
        while proc.poll() is None and time.monotonic() < deadline:
            if "daemon ready: cron scheduler off" in log.read_text():
                break
            time.sleep(0.2)
        assert proc.poll() is None and "daemon ready: cron scheduler off" in log.read_text(), (
            log.read_text())
        daemon_pid = json.loads((home / daemon.ENDPOINT_FILE).read_text())["pid"]
        assert daemon_pid == proc.pid or (IS_WINDOWS and _parent_pid(daemon_pid) == proc.pid), (
            f"endpoint pid {daemon_pid} is not the daemon started as pid {proc.pid}")
        assert not (home / "cron").exists()
        assert marker.read_text() == f"{user_home}\n"
        assert json.loads((user_home / "cron" / "jobs.json").read_text()) == [job]
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    assert "adopted" not in log.read_text() and "seeded" not in log.read_text()


def test_run_trial_records_daemon_failure(tmp_path: Path) -> None:
    """Without a daemon the client raises; the trial records the error and no spend."""
    config = {"container": "c0ffee", "workdir": "/app", "prompt": "do it",
              "model": "claude-opus-5-5", "max_budget": 1.0}
    metrics = trial.run_trial(tmp_path, config, tmp_path / "missing.json", threading.Event())
    assert metrics["model"] == "claude-opus-5-5" and metrics["container"] == "c0ffee"
    assert metrics["agent_success"] is False and metrics["cost_usd"] == 0.0
    assert metrics["error"] and metrics["seconds"] >= 0
    assert metrics["review_cost_usd"] == 0.0
    # finish reviews an earlier run logged in the same directory are not charged to
    # this run: only records appended after the run starts count
    # (review_usage(path, since) sums them, see test_coding_sea_evidence_gate)
    (tmp_path / "trajectory.jsonl").write_text(
        'not json\n{"event": "review", "cost_usd": 0.25, "tokens": 1200}\n'
        '{"event": "review", "cost_usd": 0.5, "tokens": 300}\n{"event": "llm_call"}\n')
    metrics = trial.run_trial(tmp_path, config, tmp_path / "missing.json", threading.Event())
    assert metrics["review_cost_usd"] == 0.0
    assert metrics["cost_usd"] == 0.0 and metrics["tokens"] == 0
    # no verified checkpoint was logged by this run: nothing to submit
    assert metrics["submitted"] == "none"
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["prompt"] == "do it" and written["max_budget"] == 1.0
    assert (tmp_path / "sea.py").is_file()
    # a verified checkpoint of a gone container: the restore is attempted and fails closed
    (tmp_path / "trajectory.jsonl").write_text(
        '{"event": "checkpoint", "name": "verified", "verified": true, "exit_code": 0}\n')
    metrics = trial.run_trial(tmp_path, config, tmp_path / "missing.json", threading.Event())
    assert metrics["submitted"] == "none"  # the checkpoint predates this run's offset
    config_path = tmp_path / "config.json"
    assert coding_sea.submit_best_checkpoint(config_path) == "kept"  # no verifier to run
    (tmp_path / "trajectory.jsonl").write_text(
        '{"event": "checkpoint", "name": "verified", "verified": true, "exit_code": 0}\n')
    config_path.write_text(json.dumps({**config, "container": "exec:http://127.0.0.1:9/",
                                       "trajectory": str(tmp_path / "trajectory.jsonl")}))
    assert coding_sea.submit_best_checkpoint(config_path) == "kept"  # unreachable bridge


def test_harbor_agent_declares_name_version_and_options() -> None:
    harbor_agent = pytest.importorskip("kiss.benchmarks.terminal_bench.harbor_agent")
    agent = harbor_agent.SeaHarborAgent(logs_dir=Path("/tmp/x"), model_name="m",
                                        budget_usd="3", home="/h")
    assert harbor_agent.SeaHarborAgent.name() == "kiss-sorcar"
    assert agent.version() == harbor_agent.AGENT_VERSION
    assert agent.options.budget_usd == 3.0 and agent.options.home == "/h"
    assert agent.capabilities.skills is True and agent.capabilities.mcp_servers is True
    from harbor.models.task.config import MCPServerConfig  # type: ignore[import-untyped]

    assert harbor_agent.mcp_note([]) == ""
    note = harbor_agent.mcp_note([
        MCPServerConfig(name="playwright", transport="sse", url="http://playwright-mcp:3080/sse"),
        MCPServerConfig(name="fs", transport="stdio", command="npx", args=["server-fs", "/app"]),
    ])
    assert "- playwright (transport sse): http://playwright-mcp:3080/sse" in note
    assert "- fs (transport stdio): npx server-fs /app" in note
    defaults = harbor_agent.SeaHarborAgent(logs_dir=Path("/tmp/x"), model_name="m")
    assert defaults.options.home == str(daemon.DEFAULT_HOME)
    assert defaults.options.finish_gate is True and defaults.options.budget_usd == 100.0
    assert defaults.options.reviewer == harbor_agent.SAME_MODEL
    assert defaults.options.blind_verifier == harbor_agent.SAME_MODEL
    assert defaults.options.reasoning_effort == ""
    gated_off = harbor_agent.SeaHarborAgent(logs_dir=Path("/tmp/x"), model_name="m",
                                            finish_gate="false", reviewer="none",
                                            blind_verifier="none", reasoning_effort="max")
    assert gated_off.options.finish_gate is False
    assert gated_off.options.reviewer == "none" and gated_off.options.reasoning_effort == "max"
    assert gated_off.options.blind_verifier == "none"


def test_daemon_module_help_runs() -> None:
    out = subprocess.run([sys.executable, "-m", "kiss.benchmarks.terminal_bench.daemon", "--help"],
                         capture_output=True, text=True, check=True)
    assert "--home" in out.stdout and "--port" in out.stdout
