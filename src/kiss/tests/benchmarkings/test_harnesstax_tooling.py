# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the HarnessTax result tooling on a synthetic results tree.

The tests build a small ``results/<phase>`` tree (SWE-bench Lite trial
directories and Harbor-style Terminal-Bench trial directories) plus a daemon
ledger database with the real schema, then run the ``aggregate``, ``audit``,
``analyze`` and ``report`` command-line tools as subprocesses with
``HARNESSTAX_RESULTS_ROOT`` / ``HARNESSTAX_HOME`` pointing at the tree.

Not covered here: ``harbor_agent`` and the container attach need a live
daemon, a model and a container; they were exercised by hand on a real trial.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE = "synthetic"
MODEL = "gpt-5.6-luna"
PROMPT_TIMEOUT = "prompt of the timed-out trial"
PROMPT_FALLBACK = "prompt of the contaminated trial"
DEADLINE_TS = 1_800_000_000.0


def trailer(step: int, tokens: int, cost: float) -> str:
    """The usage trailer KISS appends to every tool result."""
    return (
        f"Steps: {step}/10000, Context: 1,000/500,000 tokens, Total tokens: {tokens:,}, "
        f"Budget: ${cost:.4f}/$15.00, "
    )


def write_trajectory(path: Path, turns: int, base_ts: float, tool_result: str = "ok") -> None:
    """A trajectory with *turns* model calls, one tool call each.

    As in a real run, call N's ``new_messages`` carry the tool result of call
    N-1 with its usage trailer (spend through call N-1); call 1 has none.
    """
    with path.open("w") as fh:
        for turn in range(1, turns + 1):
            ts = base_ts + turn
            previous = [
                {
                    "role": "tool",
                    "content": tool_result
                    + trailer(turn - 1, 1000 * (turn - 1), 0.01 * (turn - 1)),
                }
            ]
            fh.write(
                json.dumps(
                    {
                        "event": "llm_call",
                        "turn": turn,
                        "ts": ts,
                        "new_messages": previous
                        if turn > 1
                        else [{"role": "user", "content": "task"}],
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "event": "tool",
                        "turn": turn,
                        "ts": ts + 0.5,
                        "tool": "bash",
                        "args": {"command": "true"},
                        "result": tool_result,
                    }
                )
                + "\n"
            )


def swe_trial(
    root: Path,
    task: str,
    rep: int,
    resolved: bool | None,
    cost: float,
    error: str = "",
    agent_success: bool = True,
) -> Path:
    """One SWE-bench Lite trial directory."""
    directory = root / PHASE / "swebench-lite" / f"{task}__{MODEL}__r{rep}"
    directory.mkdir(parents=True)
    write_trajectory(directory / "trajectory.jsonl", 4, 1_700_000_000.0)
    (directory / "config.json").write_text(json.dumps({"model": MODEL, "prompt": f"fix {task}"}))
    (directory / "result.json").write_text(
        json.dumps(
            {
                "benchmark": "swebench-lite",
                "task": task,
                "rep": rep,
                "phase": PHASE,
                "model": MODEL,
                "agent_success": agent_success,
                "cost_usd": cost,
                "tokens": 4000,
                "steps": 4,
                "turns": 4,
                "seconds": 12.0,
                "task_id": f"swe-{task}-{rep}",
                "final_text": "done",
                "error": error,
                "patch_chars": 10,
                "resolved": resolved,
            }
        )
    )
    return directory


def tb_trial(
    root: Path,
    task: str,
    reward: float | None,
    cost: float,
    exception: str = "",
    prompt: str = "",
    agent_result: bool = True,
    suffix: str = "abc",
) -> Path:
    """One Harbor-layout Terminal-Bench trial directory."""
    directory = root / PHASE / "tb2" / f"{PHASE}-{MODEL}" / f"{task}__{suffix}"
    (directory / "agent").mkdir(parents=True)
    (directory / "agent" / "config.json").write_text(
        json.dumps({"model": MODEL, "prompt": prompt or f"do {task}"})
    )
    os.utime(directory / "agent" / "config.json", (DEADLINE_TS - 100, DEADLINE_TS - 100))
    write_trajectory(directory / "agent" / "trajectory.jsonl", 6, DEADLINE_TS - 90)
    if agent_result:
        (directory / "agent" / "result.json").write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "cost_usd": cost,
                    "tokens": 6000,
                    "steps": 6,
                    "turns": 6,
                    "seconds": 30.0,
                    "task_id": "",
                    "error": "",
                    "agent_success": True,
                    "final_text": "done",
                }
            )
        )
    metadata = {"model": MODEL, "turns": 6, "tokens": 6000, "seconds": 30.0, "error": ""}
    (directory / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "task_id": {"path": task},
                "config": {"agent": {"model_name": MODEL}},
                "agent_result": {"cost_usd": cost, "metadata": metadata} if agent_result else None,
                "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
                "exception_info": {"exception_type": exception} if exception else None,
                "agent_execution": {
                    "started_at": "2027-01-15T08:00:00Z",
                    "finished_at": "2027-01-15T08:00:00Z",
                },
            }
        )
    )
    return directory


def make_ledger(home: Path) -> None:
    """A daemon database with the real ``task_history``/``events`` schema."""
    con = sqlite3.connect(home / "sorcar.db")
    con.execute(
        "CREATE TABLE task_history (id TEXT PRIMARY KEY, model TEXT, task TEXT, timestamp REAL, "
        "cost REAL, tokens INTEGER, steps INTEGER, end_ts REAL)"
    )
    con.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, seq INTEGER, "
        "event_json TEXT, timestamp REAL)"
    )
    submitted = DEADLINE_TS - 100
    con.execute(
        "INSERT INTO task_history VALUES (?,?,?,?,?,?,?,?)",
        ("task-timeout", MODEL, PROMPT_TIMEOUT, submitted, 0.9, 9000, 9, DEADLINE_TS + 50),
    )
    con.execute(
        "INSERT INTO task_history VALUES (?,?,?,?,?,?,?,?)",
        ("task-fallback", MODEL, PROMPT_FALLBACK, submitted, 0.3, 3000, 3, DEADLINE_TS),
    )
    # Usage events: three before the deadline, one after (must be ignored with a cutoff).
    for seq, (offset, cost, tokens, model) in enumerate(
        [
            (-80, 0.1, 1000, MODEL),
            (-60, 0.2, 2000, MODEL),
            (-40, 0.3, 3000, MODEL),
            (40, 0.9, 9000, MODEL),
        ]
    ):
        con.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?,?,?,?)",
            (
                "task-timeout",
                seq,
                json.dumps(
                    {
                        "type": "usage_info",
                        "cost": f"${cost}",
                        "total_tokens": tokens,
                        "total_steps": seq + 1,
                        "model": model,
                        "ts": int((DEADLINE_TS + offset) * 1000),
                    }
                ),
                DEADLINE_TS + offset,
            ),
        )
    for seq, model in enumerate([MODEL, "claude-opus-4-8", MODEL]):
        con.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?,?,?,?)",
            (
                "task-fallback",
                seq,
                json.dumps(
                    {
                        "type": "usage_info",
                        "cost": f"${0.1 * (seq + 1):.1f}",
                        "total_tokens": 1000 * (seq + 1),
                        "total_steps": seq + 1,
                        "model": model,
                        "ts": int((DEADLINE_TS - 50 + seq) * 1000),
                    }
                ),
                DEADLINE_TS - 50 + seq,
            ),
        )
    con.commit()
    con.close()


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """Synthetic results root + daemon home, with the environment to point tools at them."""
    root = tmp_path / "results"
    home = tmp_path / "home"
    home.mkdir()
    make_ledger(home)
    # SWE-bench Lite: task A solved once out of two attempts, task B failed once,
    # one ungraded attempt (skipped), one attempt with a daemon error (incomplete).
    swe_trial(root, "astropy__astropy-7746", 1, True, 0.10)
    swe_trial(root, "astropy__astropy-7746", 2, False, 0.30, agent_success=False)
    swe_trial(root, "django__django-11019", 1, False, 0.20)
    swe_trial(root, "django__django-11019", 2, None, 0.20)
    swe_trial(root, "django__django-11019", 3, None, 0.00, error="TimeoutError: client")
    # Terminal-Bench: one clean solve, one Harbor agent timeout whose spend must
    # be read from the ledger up to the deadline, one fallback-contaminated
    # attempt, one cancelled attempt (incomplete).
    tb_trial(root, "regex-log", 1.0, 0.05)
    tb_trial(
        root,
        "train-fasttext",
        0.0,
        0.0,
        exception="AgentTimeoutError",
        prompt=PROMPT_TIMEOUT,
        agent_result=False,
    )
    tb_trial(root, "fix-git", 1.0, 0.30, prompt=PROMPT_FALLBACK)
    tb_trial(root, "sam-cell-seg", None, 0.0, exception="CancelledError", agent_result=False)
    return {"root": root, "home": home}


def run_tool(
    tree: dict[str, Path], module: str, *args: str, **env_overrides: str
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m benchmarkings.harnesstax.<module>`` against the synthetic tree."""
    env = dict(
        os.environ,
        HARNESSTAX_RESULTS_ROOT=str(tree["root"]),
        HARNESSTAX_HOME=str(tree["home"]),
        PYTHONPATH=str(REPO_ROOT),
        **env_overrides,
    )
    return subprocess.run(
        [sys.executable, "-m", f"benchmarkings.harnesstax.{module}", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_aggregate_uses_task_means_ledger_and_skips_incomplete(tree: dict[str, Path]) -> None:
    """Success is the mean of per-task means; timeout spend is the ledger usage up to the deadline.

    Bench records carry the deadline-bounded ledger usage, not the runaway trajectory.
    """
    proc = run_tool(tree, "aggregate", "--phase", PHASE)
    assert proc.returncode == 0, proc.stderr
    assert "skipping incomplete trial (CancelledError)" in proc.stderr
    assert "unequal attempts per task" in proc.stderr
    summary = json.loads((tree["root"] / PHASE / "summary.json").read_text())
    swe = next(r for r in summary["summary"] if r["benchmark"] == "swebench-lite")
    # task A: (1 + 0) / 2 = 0.5, task B: 0 (ungraded attempts skipped) -> 0.25
    assert swe["trials"] == 3 and swe["tasks"] == 2
    assert swe["success_rate"] == pytest.approx(0.25)
    assert swe["cost_per_rollout"] == pytest.approx(((0.10 + 0.30) / 2 + 0.20) / 2)
    assert swe["cost_per_solve"] == pytest.approx(swe["cost_per_rollout"] / 0.25)
    assert 0.0 <= swe["success_ci"][0] <= swe["success_rate"] <= swe["success_ci"][1] <= 1.0
    tb = next(r for r in summary["summary"] if r["benchmark"] == "tb2")
    assert tb["trials"] == 3  # cancelled trial excluded
    timeout_rec = next(r for r in summary["records"] if r["task"] == "train-fasttext")
    assert (
        timeout_rec["cost"] == pytest.approx(0.3)
        and timeout_rec["turns"] == 3
        and timeout_rec["tokens"] == 3000
    )
    assert timeout_rec["success"] is False and "AgentTimeoutError" in timeout_rec["error"]


def test_aggregate_falls_back_to_trajectory_without_ledger(tree: dict[str, Path]) -> None:
    """Without a ledger the timeout spend comes from the trajectory trailers before the deadline."""
    (tree["home"] / "sorcar.db").unlink()
    proc = run_tool(tree, "aggregate", "--phase", PHASE)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tree["root"] / PHASE / "summary.json").read_text())
    rec = next(r for r in summary["records"] if r["task"] == "train-fasttext")
    # trajectory: 6 calls at DEADLINE-89 .. DEADLINE-84, all before the deadline; the last
    # trailer (in call 6) reports the spend through call 5
    assert rec["turns"] == 6 and rec["cost"] == pytest.approx(0.05) and rec["tokens"] == 5000


def test_audit_flags_and_quarantines(tree: dict[str, Path]) -> None:
    """The audit lists incomplete and fallback-served trials and moves them out of the tree."""
    proc = run_tool(tree, "audit", "--phase", PHASE)
    assert proc.returncode == 0, proc.stderr
    assert "3 flagged trial(s)" in proc.stdout
    assert (
        "contaminated" in proc.stdout
        and "claude-opus-4-8" in proc.stdout
        and "fix-git" in proc.stdout
    )
    assert "CancelledError" in proc.stdout and "TimeoutError: client" in proc.stdout
    proc = run_tool(tree, "audit", "--phase", PHASE, "--quarantine")
    assert proc.returncode == 0, proc.stderr
    quarantine = tree["root"] / PHASE / "_quarantine"
    assert (quarantine / "tb2" / "fix-git__abc").is_dir()
    assert (quarantine / "tb2" / "sam-cell-seg__abc").is_dir()
    assert (quarantine / "swebench-lite" / f"django__django-11019__{MODEL}__r3").is_dir()
    proc = run_tool(tree, "aggregate", "--phase", PHASE)
    assert "skipping" not in proc.stderr
    proc = run_tool(tree, "audit", "--phase", PHASE, "--rerun-plan")
    assert proc.returncode == 0, proc.stderr
    assert (
        f"tb2_runner --phase {PHASE} --models {MODEL} --tasks fix-git --reps 1 --concurrency 1 "
        "--job-suffix=-rerun1-fix-git"
    ) in proc.stdout
    assert "--tasks sam-cell-seg --reps 1" in proc.stdout
    # once a rerun job holds the attempt it drops out of the plan; a second generation gets a
    # new suffix
    (tree["root"] / PHASE / "tb2" / f"{PHASE}-{MODEL}-rerun1-fix-git" / "fix-git__new").mkdir(
        parents=True
    )
    proc = run_tool(tree, "audit", "--phase", PHASE, "--rerun-plan")
    assert "fix-git" not in proc.stdout.split("flagged trial(s)")[1]
    swe_trial(tree["root"], "django__django-11019", 3, None, 0.0, error="TimeoutError: again")
    proc = run_tool(tree, "audit", "--phase", PHASE, "--quarantine")
    assert proc.returncode == 0, proc.stderr
    # a second contaminated fix-git attempt: one already re-run, so one more is planned
    # (generation 2)
    tb_trial(tree["root"], "fix-git", 1.0, 0.30, prompt=PROMPT_FALLBACK, suffix="def")
    proc = run_tool(tree, "audit", "--phase", PHASE, "--quarantine", "--rerun-plan")
    assert "--tasks fix-git --reps 1 --concurrency 1 --job-suffix=-rerun2-fix-git" in proc.stdout


def test_analyze_and_report(tree: dict[str, Path]) -> None:
    """The trajectory analysis and the HTML report run end to end on the tree."""
    run_tool(tree, "audit", "--phase", PHASE, "--quarantine")
    proc = run_tool(tree, "aggregate", "--phase", PHASE)
    assert proc.returncode == 0, proc.stderr
    proc = run_tool(
        tree,
        "analyze",
        "--phase",
        PHASE,
        "--failures-only",
        "--benchmark",
        "swebench-lite",
        "--rep",
        "2",
    )
    assert proc.returncode == 0, proc.stderr
    assert "gave up" in proc.stdout and "== per model ==" in proc.stdout
    proc = run_tool(tree, "analyze", "--phase", PHASE, "--model", MODEL)
    assert proc.returncode == 0, proc.stderr
    assert "infra/timeout" in proc.stdout and "solved" in proc.stdout
    out = tree["root"] / "report.html"
    proc = run_tool(tree, "report", "--phase", PHASE, "--out", str(out))
    assert (
        proc.returncode != 0
        and "incomplete results" in proc.stderr
        and "claude-fable-5" in proc.stderr
    )
    proc = run_tool(tree, "report", "--phase", PHASE, "--out", str(out), "--allow-incomplete")
    assert proc.returncode == 0, proc.stderr
    html = out.read_text()
    assert html.count("<svg") == 8 and "GPT-5.6 Luna · KISS Sorcar" in html
    assert "Incomplete run" in html and "gpt-5.6-luna 87 slot(s) off" in html
    assert "n/a</text>" not in html  # the blog has per-task Pi data for luna on both benchmarks
    assert "1/2 · 0/3</text>" in html  # astropy: KISS solved one of two attempts, Pi none


RECORDED_SUMMARY = (
    REPO_ROOT / "benchmarkings" / "harnesstax" / "results" / "baseline" / "summary.json"
)


@pytest.mark.skipif(not RECORDED_SUMMARY.is_file(), reason="recorded baseline results not present")
def test_report_prose_on_the_recorded_study() -> None:
    """The full discussion renders from the recorded study and every claim check holds."""
    from benchmarkings.harnesstax import report

    html = report.render("baseline")
    assert (
        "Every per-model success rate" in html
        or "fall inside the blog's confidence interval" in html
    )
    assert "Incomplete run" not in html
    assert html.count("<svg") == 8


def test_trajectory_metrics_and_turn_count(tmp_path: Path) -> None:
    """Turns, cost and tokens are read from trailers, optionally only up to a cutoff time."""
    from benchmarkings.harnesstax import trials

    write_trajectory(tmp_path / "trajectory.jsonl", 5, 100.0)
    (tmp_path / "trajectory.jsonl").open("a").write("not json\n")
    # call N carries the trailer of call N-1, so the last trailer reports the spend through call 4
    assert trials.trajectory_metrics(tmp_path) == {"turns": 5, "cost_usd": 0.04, "tokens": 4000}
    assert trials.trajectory_metrics(tmp_path, before_ts=102.6) == {
        "turns": 2,
        "cost_usd": 0.01,
        "tokens": 1000,
    }
    assert trials.count_turns(tmp_path) == 5
    assert trials.trajectory_metrics(tmp_path / "missing") == {
        "turns": 0,
        "cost_usd": 0.0,
        "tokens": 0,
    }


def test_tb2_runner_lifts_harbor_agent_deadline(tmp_path: Path) -> None:
    """A new job lifts Harbor's agent deadline; an existing job directory is resumed as is."""
    from benchmarkings.harnesstax import tb2_runner

    tasks = ["regex-log", "train-fasttext"]
    command = tb2_runner.harbor_command(tmp_path, "job", MODEL, tasks, 3, 6)
    multiplier = command[command.index("--agent-timeout-multiplier") + 1]
    assert float(multiplier) == tb2_runner.AGENT_TIMEOUT_MULTIPLIER >= 1000
    assert command[command.index("-m") + 1] == MODEL and command[command.index("-k") + 1] == "3"
    assert [command[i + 1] for i, a in enumerate(command) if a == "-i"] == tasks
    (tmp_path / "job").mkdir()
    resume = tb2_runner.harbor_command(tmp_path, "job", MODEL, ["regex-log"], 3, 6)
    assert resume[-3:] == ["resume", "-p", str(tmp_path / "job")] and "-i" not in resume


def test_tb2_runner_job_finished(tmp_path: Path) -> None:
    """Harbor writes the job summary at start; only ``finished_at`` marks completion."""
    from benchmarkings.harnesstax import tb2_runner

    summary = tmp_path / "result.json"
    assert tb2_runner.job_finished(summary) is False
    summary.write_text(json.dumps({"started_at": "x", "finished_at": None}))
    assert tb2_runner.job_finished(summary) is False
    summary.write_text(json.dumps({"started_at": "x", "finished_at": "2027-01-01T00:00:00Z"}))
    assert tb2_runner.job_finished(summary) is True
    summary.write_text("{broken")
    assert tb2_runner.job_finished(summary) is False


def test_daemon_pinned_catalog_drops_fallbacks(tmp_path: Path) -> None:
    """The private catalog copy has no ``fallback`` routes in either catalog shape."""
    from benchmarkings.harnesstax import daemon

    source = tmp_path / "MODEL_INFO.json"
    target = tmp_path / "pinned.json"
    source.write_text(
        json.dumps(
            {
                "a": {"fallback": "b", "input_price_per_1M": 1},
                "b": {"input_price_per_1M": 2},
                "_doc": "comment",
            }
        )
    )
    daemon.write_pinned_catalog(source, target)
    pinned = json.loads(target.read_text())
    assert pinned == {
        "a": {"input_price_per_1M": 1},
        "b": {"input_price_per_1M": 2},
        "_doc": "comment",
    }
    # an existing pinned catalog keeps its prices; only fallbacks are stripped
    source.write_text(json.dumps({"a": {"fallback": "b", "input_price_per_1M": 9}}))
    daemon.write_pinned_catalog(source, target)
    assert json.loads(target.read_text())["a"] == {"input_price_per_1M": 1}
    target.unlink()
    source.write_text(json.dumps([{"name": "a", "fallback": "b"}, {"name": "b"}]))
    daemon.write_pinned_catalog(source, target)
    assert json.loads(target.read_text()) == [{"name": "a"}, {"name": "b"}]


def test_hooks_log_every_call_and_answer_interactive_tools(tmp_path: Path) -> None:
    """Every LLM call is counted (no cap); the tool hook logs calls and answers human-only tools."""
    import docker

    from benchmarkings.harnesstax import sea_core

    # The hook ends the trial when its container is gone, so the test needs a
    # live one; without a Docker daemon the liveness check is skipped.
    container_name = "c"
    live = None
    try:
        client = docker.from_env()
        client.ping()
        live = client.containers.run("python:3.11-slim", "sleep infinity", detach=True)
        container_name = live.id
    except Exception:
        pass
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "container": container_name,
                "workdir": "/app",
                "prompt": "p",
                "model": MODEL,
                "trajectory": str(tmp_path / "trajectory.jsonl"),
            }
        )
    )
    harness = sea_core.ContainerHarness(str(config))
    try:
        for expected in range(1, 202):
            assert harness.on_llm_call([{"role": "user", "content": "x"}]) == [
                {"role": "user", "content": "x"}
            ]
            assert harness.turns == expected
        # no wall-clock limit: tool results carry no time note
        tool_result = {"role": "tool", "content": "out"}
        assert harness.on_llm_call([tool_result]) == [{"role": "tool", "content": "out"}]
    finally:
        if live is not None:
            live.remove(force=True)
    if live is not None:
        # the container is gone: the next model call ends the trial
        from kiss.core.kiss_error import BudgetExceededError

        with pytest.raises(BudgetExceededError):
            harness.on_llm_call([])
    assert harness.on_tool_call("Bash", {"command": "ls"}) == "OK"
    assert harness.on_tool_call("ask_user_question", {"question": "?"}) != "OK"
    assert harness.on_tool_call("talk", {"text": "hi", "language": "en"}) != "OK"
    assert harness.on_tool_call("run_agent", {"agent": "slack", "task": "x"}) != "OK"
    assert harness.docker_image() == f"container:{container_name}"
    assert harness.if_append_basic_tools()
    assert not harness.use_memory() and not harness.use_web_tools()
    assert "/app" in harness.system_prompt() and "wall-clock" not in harness.system_prompt()
    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events].count("llm_call") == 202 + (live is not None)
    tool_events = [e for e in events if e["event"] == "tool_call"]
    assert [e["blocked"] for e in tool_events] == [False, True, True, True]



def test_audit_agent_error_and_deadline_aware_contamination(tree: dict[str, Path]) -> None:
    """A non-timeout SEA error is incomplete; fallback calls after Harbor's deadline do not count.

    Contamination is judged only on calls made before the deadline.
    """
    broken = tb_trial(tree["root"], "nginx-request-logging", 0.0, 0.0)
    agent = json.loads((broken / "agent" / "result.json").read_text())
    agent["error"] = "ConnectionError: daemon died"
    (broken / "agent" / "result.json").write_text(json.dumps(agent))
    timed_out = tb_trial(tree["root"], "pypi-server", 0.0, 0.0, exception="AgentTimeoutError")
    agent = json.loads((timed_out / "agent" / "result.json").read_text())
    agent["error"] = "TimeoutError: Task did not finish within 855 seconds"
    (timed_out / "agent" / "result.json").write_text(json.dumps(agent))
    con = sqlite3.connect(tree["home"] / "sorcar.db")
    con.execute(
        "INSERT INTO task_history VALUES (?,?,?,?,?,?,?,?)",
        ("task-late", MODEL, "do pypi-server", DEADLINE_TS - 100, 0.2, 2000, 2, DEADLINE_TS + 50),
    )
    for seq, (offset, model) in enumerate([(-30, MODEL), (30, "claude-opus-4-8")]):
        con.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?,?,?,?)",
            (
                "task-late",
                seq,
                json.dumps(
                    {
                        "type": "usage_info",
                        "cost": "$0.1",
                        "total_tokens": 1000,
                        "total_steps": seq + 1,
                        "model": model,
                        "ts": int((DEADLINE_TS + offset) * 1000),
                    }
                ),
                DEADLINE_TS + offset,
            ),
        )
    con.commit()
    con.close()
    proc = run_tool(tree, "audit", "--phase", PHASE)
    assert proc.returncode == 0, proc.stderr
    assert "ConnectionError: daemon died" in proc.stdout
    assert "nginx-request-logging" in proc.stdout
    # timed out, and the fallback call came after the deadline
    assert "pypi-server" not in proc.stdout
    assert "4 flagged trial(s)" in proc.stdout


def test_append_to_last_tool_result_handles_every_message_shape() -> None:
    """The note lands in tool results of every provider shape.

    It is never appended to prompts or assistant turns.
    """
    from benchmarkings.harnesstax import sea_core

    anthropic: dict[str, Any] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": "out"}],
    }
    anthropic_list: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t",
                "content": [{"type": "text", "text": "out"}],
            }
        ],
    }
    responses: dict[str, Any] = {"type": "function_call_output", "call_id": "c", "output": "out"}
    chat: dict[str, Any] = {"role": "tool", "tool_call_id": "c", "content": "out"}
    for message in (anthropic, anthropic_list, responses, chat):
        assert sea_core.append_to_last_tool_result(message, "[note]")
    assert anthropic["content"][0]["content"] == "out\n\n[note]"
    assert anthropic_list["content"][0]["content"][-1]["text"] == "[note]"
    assert responses["output"] == "out\n\n[note]"
    assert chat["content"] == "out\n\n[note]"
    prompt: dict[str, Any] = {"role": "user", "content": "task"}
    assistant: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    for other in (prompt, assistant, "not a dict"):
        assert not sea_core.append_to_last_tool_result(other, "[note]")
    assert prompt == {"role": "user", "content": "task"}
    assert assistant["content"] == [{"type": "text", "text": "hi"}]


def test_changed_definitions_and_test_paths() -> None:
    """The edit locator names the enclosing definitions of a change.

    The test-path heuristic covers common layouts; Python, Go receiver methods
    and exported JS functions are recognised.
    """
    from benchmarkings.harnesstax import test_context

    changed = test_context.changed_definitions
    old = (
        "class Field:\n    def check(self):\n        return 1\n\n"
        "    def other(self):\n        pass\n"
    )
    new = (
        "class Field:\n    def check(self):\n        if True:\n            return 2\n\n"
        "    def other(self):\n        pass\n"
    )
    assert changed(old, new) == ["check", "Field"]
    # a deleted line points at the definition that lost it; a new top-level function names itself
    assert changed(new, old) == ["check", "Field"]
    assert changed("x = 1\n", "x = 1\ndef helper():\n    return 3\n") == ["helper"]
    # generic names are dropped, a change outside any definition yields nothing
    assert changed("def main():\n    a\n", "def main():\n    b\n") == []
    assert changed("a = 1\n", "a = 2\n") == []
    # Go receiver methods and exported JS functions are recognised too
    go_old = "func (s *Server) Start() {\n\treturn\n}\n"
    assert changed(go_old, go_old.replace("return", "run()")) == ["Start"]
    js_old = "export async function load() {\n  a\n}\n"
    assert changed(js_old, js_old.replace("  a\n", "  b\n")) == ["load"]
    for path in (
        "tests/test_x.py",
        "pkg/x_test.go",
        "src/a.spec.ts",
        "spec/a_spec.rb",
        "src/__tests__/a.js",
        "FooTests.cs",
        "testing/util.py",
    ):
        assert test_context.is_test_path(path), path
    for path in ("django/db/models/fields.py", "src/contest.py", "latest/run.py"):
        assert not test_context.is_test_path(path), path
    assert test_context.find_referencing_tests("c", "/w", [], "/w/a.py") == []


def test_edit_tool_results_list_referencing_tests(tmp_path: Path) -> None:
    """Editing an existing source file appends the tests that mention the changed definitions."""
    import docker

    from benchmarkings.harnesstax import sea_core

    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        pytest.skip("Docker is not available")
    live = client.containers.run("python:3.11-slim", "sleep infinity", detach=True)
    try:
        setup = (
            "mkdir -p /repo/pkg /repo/tests && cd /repo && "
            "printf 'class Field:\\n    def check(self):\\n        return 1\\n' > pkg/fields.py && "
            "printf 'from pkg.fields import Field\\n\\ndef test_check():\\n"
            "    assert Field().check() == 1\\n' > tests/test_fields.py && "
            "printf 'def test_other():\\n    Field\\n    Field\\n    Field\\n' "
            "> tests/test_other.py && "
            "printf 'x = 1\\n' > tests/test_unrelated.py && printf 'Field\\n' > pkg/notes.txt"
        )
        assert live.exec_run(["sh", "-c", setup]).exit_code == 0
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "container": live.id,
                    "workdir": "/repo",
                    "prompt": "p",
                    "model": MODEL,
                    "trajectory": str(tmp_path / "trajectory.jsonl"),
                    "test_context": True,
                }
            )
        )
        harness = sea_core.ContainerHarness(str(config))
        assert "tests that reference the changed definitions" in harness.system_prompt()
        # an Edit of a source file: snapshot, then the change lands in the container
        edit = {"file_path": "pkg/fields.py", "old_string": "1", "new_string": "2"}
        assert harness.on_tool_call("Edit", edit) == "OK"
        live.exec_run(["sh", "-c", "sed -i 's/return 1/return 2/' /repo/pkg/fields.py"])
        # edits of test files, missing files and non-string paths are ignored
        harness.on_tool_call("Write", {"file_path": "/repo/tests/test_new.py", "content": "x"})
        harness.on_tool_call("Write", {"file_path": "/repo/pkg/new_module.py", "content": "x"})
        harness.on_tool_call("Edit", {"file_path": 3})
        message = {"type": "function_call_output", "call_id": "c", "output": "Edited"}
        harness.on_llm_call([message])
        note = message["output"]
        assert "code you changed in pkg/fields.py (check, Field" in note
        assert "tests/test_other.py (3)" in note and "tests/test_fields.py (3)" in note
        assert "test_unrelated" not in note and "notes.txt" not in note
        # mentions both names: ranks first
        assert note.index("test_fields.py") < note.index("test_other.py")
        assert note.endswith("before you finish.]")  # nothing follows: no wall-clock note
        events = [
            json.loads(line)
            for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        ]
        assert [e["tests"] for e in events if e["event"] == "test_context"] == [
            [["tests/test_fields.py", 3], ["tests/test_other.py", 3]]
        ]
        # the same tests are not listed twice; an edit that changed nothing adds nothing
        fields = "/repo/pkg/fields.py"
        harness.on_tool_call("Edit", {"file_path": fields, "old_string": "2", "new_string": "3"})
        live.exec_run(["sh", "-c", "sed -i 's/return 2/return 3/' /repo/pkg/fields.py"])
        harness.on_tool_call("Edit", {"file_path": fields, "old_string": "q", "new_string": "r"})
        again = {"type": "function_call_output", "call_id": "c", "output": "Edited"}
        harness.on_llm_call([again])
        assert "Existing tests" not in again["output"]
        # a file deleted between snapshot and model call is skipped
        harness.on_tool_call("Edit", {"file_path": fields, "old_string": "3", "new_string": "4"})
        live.exec_run(["rm", "/repo/pkg/fields.py"])
        gone = {"type": "function_call_output", "call_id": "c", "output": "Edited"}
        harness.on_llm_call([gone])
        assert "Existing tests" not in gone["output"]
        # the feature is off unless the trial config enables it
        config.write_text(
            json.dumps(
                {
                    "container": live.id,
                    "workdir": "/repo",
                    "prompt": "p",
                    "model": MODEL,
                    "trajectory": str(tmp_path / "t2.jsonl"),
                }
            )
        )
        off = sea_core.ContainerHarness(str(config))
        off.on_tool_call("Edit", edit)
        assert off.pending_edits == []
    finally:
        live.remove(force=True)


def test_verification_pass_runs_fresh_context_after_first_run(tmp_path: Path) -> None:
    """A trial runs the task, then a fresh-context verification pass on the same container.

    Needs a running benchmark daemon (``HARNESSTAX_TEST_SOCK``) and Docker;
    skipped otherwise, since the pass is only observable end to end.
    """
    import docker

    from benchmarkings.harnesstax import trials

    sock = os.environ.get("HARNESSTAX_TEST_SOCK", "")
    if not sock or not Path(sock).exists():
        pytest.skip("set HARNESSTAX_TEST_SOCK to a benchmark daemon socket")
    client = docker.from_env()
    live = client.containers.run(
        "python:3.11-slim", "sleep infinity", detach=True, working_dir="/app"
    )
    model = os.environ.get("HARNESSTAX_TEST_MODEL", "gpt-5.6-luna")
    assert live.id is not None
    container_id = live.id
    try:
        live.exec_run(["mkdir", "-p", "/app"])
        os.environ["HARNESSTAX_VERIFY_PASS"] = "1"
        try:
            metrics = trials.run_sea_trial(
                tmp_path,
                container_id,
                "/app",
                "Create the file /app/hello.txt containing exactly the line `hello` "
                "and nothing else.",
                model,
                Path(sock),
            )
        finally:
            del os.environ["HARNESSTAX_VERIFY_PASS"]
        assert live.exec_run(["cat", "/app/hello.txt"]).output.decode().strip() == "hello"
        second = metrics["verify_pass"]
        assert second is not None and second["error"] == "" and second["agent_success"]
        assert metrics["agent_success"] and metrics["cost_usd"] >= second["cost_usd"] > 0
        events = [
            json.loads(line)
            for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
        ]
        prompts = [
            json.dumps(e["new_messages"])
            for e in events
            if e.get("event") == "llm_call" and e["turn"] == 1
        ]
        assert len(prompts) == 2
        assert "Do not trust that report" in prompts[1] and "Do not trust" not in prompts[0]
        assert metrics["turns"] == sum(1 for e in events if e.get("event") == "llm_call")
        # the pass is off by default
        again = trials.run_sea_trial(
            tmp_path,
            container_id,
            "/app",
            "Print the content of /app/hello.txt.",
            model,
            Path(sock),
        )
        assert again["verify_pass"] is None
    finally:
        live.remove(force=True)
    assert trials.plain_text("<p>Did <b>x</b></p>\n<ul><li>y</li></ul>") == "Did x y"
