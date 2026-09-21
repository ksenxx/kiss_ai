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

Not covered here: the turn cap in ``sea_core.on_llm_call`` and the client
timeout in ``harbor_agent`` need a live daemon, a model and a container; they
were exercised by hand with ``HARNESSTAX_MAX_TURNS=3`` on a real trial.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

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


def test_aggregate_counts_spend_only_through_the_turn_cap(tree: dict[str, Path]) -> None:
    """An attempt that ran past the cap is billed through its cap-th call.

    The ledger is preferred; the trajectory trailers are the fallback.
    """
    # Every synthetic trial has 4 calls; with a cap of 3 each is over the cap.  The SWE
    # trials are not in the ledger (trajectory fallback: call 4's trailer = spend through
    # call 3); the timed-out TB trial is (ledger: first 3 usage events).
    proc = run_tool(tree, "aggregate", "--phase", PHASE, HARNESSTAX_MAX_TURNS="3")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tree["root"] / PHASE / "summary.json").read_text())
    swe = next(r for r in summary["records"] if r["benchmark"] == "swebench-lite")
    assert swe["turns"] == 3 and swe["cost"] == pytest.approx(0.03) and swe["tokens"] == 3000
    timed_out = next(r for r in summary["records"] if r["task"] == "train-fasttext")
    assert timed_out["turns"] == 3 and timed_out["cost"] == pytest.approx(0.3)
    # the same trials are flagged as overruns: tool calls ran after the cap
    proc = run_tool(tree, "audit", "--phase", PHASE, HARNESSTAX_MAX_TURNS="3")
    assert proc.returncode == 0, proc.stderr
    assert (
        proc.stdout.count("overrun") >= 5
        and "1 tool call(s) executed after the 3-turn cap" in proc.stdout
    )
    # with the real cap nothing is an overrun
    proc = run_tool(tree, "audit", "--phase", PHASE)
    assert "overrun" not in proc.stdout


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
    # a cap of 2 counts two calls and the spend through call 2 (read from call 3's trailer)
    assert trials.trajectory_metrics(tmp_path, max_turns=2) == {
        "turns": 2,
        "cost_usd": 0.02,
        "tokens": 2000,
    }
    assert trials.trajectory_metrics(tmp_path, max_turns=9) == {
        "turns": 5,
        "cost_usd": 0.04,
        "tokens": 4000,
    }
    assert trials.count_turns(tmp_path) == 5
    assert trials.trajectory_metrics(tmp_path / "missing") == {
        "turns": 0,
        "cost_usd": 0.0,
        "tokens": 0,
    }


def test_harbor_agent_timeout_from_task_toml(tmp_path: Path) -> None:
    """The client timeout is the task's agent limit minus the margin, with a floor and a default."""
    from benchmarkings.harnesstax import harbor_agent

    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "task.toml").write_text("[agent]\ntimeout_sec = 1200.0\n")
    env = task_dir / "environment"
    assert harbor_agent.agent_timeout_seconds(env) == 1200 - harbor_agent.DEADLINE_MARGIN_SECONDS
    (task_dir / "task.toml").write_text("[agent]\ntimeout_sec = 50\n")
    assert harbor_agent.agent_timeout_seconds(env) == 60
    (task_dir / "task.toml").write_text("[verifier]\ntimeout_sec = 50\n")
    assert harbor_agent.agent_timeout_seconds(env) == harbor_agent.DEFAULT_TIMEOUT_SECONDS
    (task_dir / "task.toml").write_text("not = [toml\n")
    assert harbor_agent.agent_timeout_seconds(env) == harbor_agent.DEFAULT_TIMEOUT_SECONDS
    assert (
        harbor_agent.agent_timeout_seconds(tmp_path / "nope" / "environment")
        == harbor_agent.DEFAULT_TIMEOUT_SECONDS
    )


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


def test_turn_cap_ends_the_run_before_the_extra_call(tmp_path: Path) -> None:
    """Calls 1..N pass through and are logged; call N+1 raises the terminal error promptly."""
    import threading

    from benchmarkings.harnesstax import sea_core
    from kiss.core.kiss_error import BudgetExceededError

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "container": "c",
                "prompt": "p",
                "model": MODEL,
                "max_turns": 3,
                "trajectory": str(tmp_path / "trajectory.jsonl"),
            }
        )
    )
    harness = sea_core.ContainerHarness(str(config))
    for expected in (1, 2, 3):
        assert harness.on_llm_call([{"role": "user", "content": "x"}]) == [
            {"role": "user", "content": "x"}
        ]
        assert harness.turns == expected
    outcome: list[BaseException | None] = []
    worker = threading.Thread(target=lambda: outcome.append(_call_capped(harness)))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "the capped call deadlocked"
    assert isinstance(outcome[0], BudgetExceededError)
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert events == ["llm_call", "llm_call", "llm_call", "turn_cap"]
    assert harness.turns == 3


def _call_capped(harness: object) -> BaseException | None:
    """Invoke the hook once more and return the exception it raised (or None)."""
    try:
        harness.on_llm_call([])  # type: ignore[attr-defined]
    except BaseException as exc:  # noqa: BLE001 - the test inspects the type
        return exc
    return None


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
    assert "ConnectionError: daemon died" in proc.stdout and "nginx-request-logging" in proc.stdout
    assert (
        "pypi-server" not in proc.stdout
    )  # timed out, and the fallback call came after the deadline
    assert "4 flagged trial(s)" in proc.stdout
