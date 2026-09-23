# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of :mod:`benchmarkings.harnesstax.skillopt_bridge`.

A synthetic ``results/<phase>`` tree (Harbor-style Terminal-Bench trials and
SWE-bench Lite trials, graded and ungraded) is turned into a SkillOpt eval set
through the ``build`` CLI, and the eval set is loaded back with SkillOpt's own
loader so the env class, the imported trajectories and the selection tasks
are checked the way the optimizer sees them.

Not covered here: ``SweBenchEnv.rollouts`` starts an official SWE-bench
evaluation image (one to three GB each), runs a model against it and grades
with the ``swebench`` harness; it is exercised by the real optimization run
(see ``benchmarkings/harnesstax/results/skillopt``).  ``SweBenchEnv.grade``
is covered up to the harness call: with no patches it never runs the harness,
and with an interpreter lacking ``swebench`` the harness fails and every
instance is reported as errored.  Its network-flake regrade loop reads the
harness's ``test_output.txt`` files, so it too needs a real harness run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarkings.harnesstax import skillopt_bridge, swebench_runner
from benchmarkings.harnesstax.skillopt_bridge import SweBenchEnv, compact_trajectory, main
from kiss.agents.seas.skillopt_sea import load_evals, make_env

TRAILER = (
    "\n\nSteps: 3/10000, Context: 6,500/400,000 tokens, Total tokens: 12,642, "
    "Budget: $0.1012/$50.00, "
)


def _events(prompt: str) -> str:
    """A HarnessTax trajectory: prompt, a tool call and result, a blocked call, a dead container."""
    events: list[dict[str, Any]] = [
        {
            "event": "llm_call",
            "turn": 1,
            "new_messages": [
                {"role": "system", "content": "hidden"},
                {"role": "user", "content": prompt},
            ],
        },
        {
            "event": "tool_call",
            "turn": 1,
            "tool": "Bash",
            "args": {"command": "ls /app"},
            "blocked": False,
        },
        {
            "event": "llm_call",
            "turn": 2,
            "new_messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "a.py\nb.py" + TRAILER,
                        },
                        {"type": "text", "text": "  "},
                    ],
                }
            ],
        },
        {
            "event": "tool_call",
            "turn": 2,
            "tool": "ask_user_question",
            "args": {"question": "ok?"},
            "blocked": True,
        },
        {
            "event": "llm_call",
            "turn": 3,
            "new_messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "No human."
                                    + TRAILER
                                    + "\nNote appended after the trailer.",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {"event": "container_gone", "turn": 3},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\nnot json\n"


def _tb2_trial(
    root: Path, task: str, suffix: str, model: str | None, reward: float | None, **meta: Any
) -> None:
    trial = root / "tb2" / "nocap-x" / f"{task}__{suffix}"
    (trial / "agent").mkdir(parents=True)
    result: dict[str, Any] = {"task_name": task, "trial_name": f"{task}__{suffix}"}
    if model:
        result["agent_result"] = {
            "metadata": {
                "model": model,
                "cost_usd": 1.5,
                "tokens": 1000,
                "steps": 7,
                "final_text": "<p>Did <b>it</b></p>",
                **meta,
            }
        }
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward}}
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "agent" / "config.json").write_text(
        json.dumps({"prompt": f"solve {task}", "model": model})
    )
    (trial / "agent" / "trajectory.jsonl").write_text(_events(f"solve {task}"))


def _swe_trial(
    root: Path, task: str, model: str, rep: int, resolved: bool | None, **extra: Any
) -> None:
    trial = root / "swebench-lite" / f"{task}__{model}__r{rep}"
    trial.mkdir(parents=True)
    result = {
        "task": task,
        "rep": rep,
        "model": model,
        "resolved": resolved,
        "cost_usd": 0.3,
        "tokens": 700,
        "steps": 5,
        "final_text": "<p>fixed</p>",
        "error": "",
        "patch_chars": 12,
        **extra,
    }
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "config.json").write_text(json.dumps({"prompt": f"fix {task}", "model": model}))
    (trial / "trajectory.jsonl").write_text(_events(f"fix {task}"))


@pytest.fixture
def phase(tmp_path: Path) -> Path:
    root = tmp_path / "nocap"
    _tb2_trial(root, "alpha", "aaa1111", "claude-fable-5", 1.0)
    _tb2_trial(root, "alpha", "aaa2222", "claude-fable-5", 0.0, error="ran out of budget")
    _tb2_trial(root, "beta", "bbb1111", "claude-fable-5", 1.0)
    _tb2_trial(root, "gamma", "ccc1111", None, None)  # crashed before the agent ran: skipped
    _tb2_trial(root, "delta", "ddd1111", "gpt-5.6-sol", 0.0)  # another model: not imported
    _swe_trial(root, "django__django-11019", "claude-fable-5", 1, True)
    _swe_trial(root, "django__django-11019", "claude-fable-5", 2, False, eval_error=True)
    _swe_trial(root, "sympy__sympy-20590", "claude-fable-5", 1, None)  # ungraded: skipped
    _swe_trial(root, "sympy__sympy-20590", "claude-fable-5", 2, False, patch_chars=0)
    return root


def test_compact_trajectory_strips_trailers_and_marks_blocked_calls(phase: Path) -> None:
    """The transcript keeps prompt, calls and results, drops the system message and bad lines."""
    path = phase / "tb2" / "nocap-x" / "alpha__aaa1111" / "agent" / "trajectory.jsonl"
    transcript = compact_trajectory(path, 44)
    assert transcript == [
        {"role": "user", "text": "solve alpha"},
        {"role": "assistant", "text": '[call Bash] {"command": "ls /app"}'},
        {"role": "tool", "text": "a.py\nb.py"},
        {"role": "assistant", "text": "[call ask_user_question] (answered without r"},
        {"role": "tool", "text": "No human.\nNote appended after the trailer."},
        {"role": "tool", "text": "[the trial container disappeared]"},
    ]
    assert compact_trajectory(phase / "missing.jsonl", 44) == []
    # Only the exact trailer goes; prose mentioning "Steps:" and nested trailers' neighbours stay.
    assert skillopt_bridge._TRAILER.sub("", "Steps: do this" + TRAILER + "then that" + TRAILER) == (
        "Steps: do thisthen that"
    )


def test_build_writes_the_eval_set_skillopt_loads(
    phase: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Failed trials train, successes are sampled, selection tasks get the benchmark prompt."""
    out = tmp_path / "out"
    assert (
        main(
            [
                "build",
                "--phase-dir",
                str(phase),
                "--out",
                str(out),
                "--select",
                "django__django-11019,sympy__sympy-20590",
                "--swebench-python",
                sys.executable,
                "--max-successes",
                "1",
                "--message-chars",
                "60",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "train": 4,
        "failed": 3,
        "select": 2,
        "evals": str(out / "evals.json"),
        "rollouts": str(out / "rollouts.json"),
    }
    rollouts = {r["task_id"]: r for r in json.loads((out / "rollouts.json").read_text())}
    failed_ids = {
        "tb2/alpha/claude-fable-5/aaa2222",
        "swe/django__django-11019/claude-fable-5/r2",
        "swe/sympy__sympy-20590/claude-fable-5/r2",
    }
    assert failed_ids <= set(rollouts) and len(rollouts) == 4
    assert set(rollouts) - failed_ids <= {
        "tb2/alpha/claude-fable-5/aaa1111",
        "tb2/beta/claude-fable-5/bbb1111",
        "swe/django__django-11019/claude-fable-5/r1",
    }
    assert sum(r["passed"] for r in rollouts.values()) == 1
    failed_tb2 = rollouts["tb2/alpha/claude-fable-5/aaa2222"]
    assert failed_tb2["verdict"] == "verifier reward 0; error: ran out of budget"
    assert failed_tb2["success"] is False and failed_tb2["cost"] == 1.5 and failed_tb2["steps"] == 7
    assert failed_tb2["trajectory"][0] == {"role": "user", "text": "solve alpha"}
    assert failed_tb2["trajectory"][-1] == {"role": "result", "text": "ran out of budget"}
    django_r2 = rollouts["swe/django__django-11019/claude-fable-5/r2"]
    assert django_r2["verdict"] == "not resolved (patch did not apply or was empty)"
    sympy_r2 = rollouts["swe/sympy__sympy-20590/claude-fable-5/r2"]
    assert sympy_r2["verdict"] == "not resolved (empty patch)"
    assert sympy_r2["trajectory"][-1] == {"role": "result", "text": "fixed"}
    assert sympy_r2["success"] is True

    raw = json.loads((out / "evals.json").read_text())
    assert raw["env"]["class"] == "benchmarkings.harnesstax.skillopt_bridge:SweBenchEnv"
    evals = load_evals(out / "evals.json")
    assert [t.id for t in evals.tasks if t.split == "select"] == [
        "django__django-11019",
        "sympy__sympy-20590",
    ]
    select = evals.tasks[-1]
    problem = swebench_runner.load_instances(["sympy__sympy-20590"])[0]["problem_statement"]
    assert select.prompt == swebench_runner.PROMPT.format(problem_statement=problem)
    train = [t for t in evals.tasks if t.split == "train"]
    assert len(train) == 4 and {t.id for t in train} == set(rollouts)
    assert (
        next(t for t in train if t.id == "tb2/alpha/claude-fable-5/aaa2222").prompt == "solve alpha"
    )
    assert [r.task_id for r in evals.train_rollouts] == [t.id for t in train]
    env = make_env(evals.env, evals.rollout)
    assert isinstance(env, SweBenchEnv) and env.swebench_python == sys.executable
    assert env.workdir == "/testbed" and env.test_context == ""


def test_build_and_select_errors(phase: Path, tmp_path: Path) -> None:
    """Unknown selection instances and models without trials are errors."""
    with pytest.raises(ValueError, match="not among the sampled"):
        skillopt_bridge.select_tasks(["nope__nope-1"])
    with pytest.raises(ValueError, match="no finished trials"):
        skillopt_bridge.build(
            phase, tmp_path / "o", ["claude-haiku-4-5"], ["django__django-11019"], sys.executable
        )
    with pytest.raises(ValueError, match="at least one"):
        skillopt_bridge.build(phase, tmp_path / "o", ["gpt-5.6-sol"], [], sys.executable)
    assert not (tmp_path / "o").exists()
    # Two models' trials of the same instance keep distinct ids.
    summary = skillopt_bridge.build(
        phase,
        tmp_path / "o",
        ["gpt-5.6-sol", "claude-fable-5"],
        ["django__django-11019"],
        sys.executable,
        max_successes=0,
    )
    assert summary["train"] == 4 and summary["failed"] == 4
    ids = [r.task_id for r in load_evals(tmp_path / "o" / "evals.json").train_rollouts]
    assert "tb2/delta/gpt-5.6-sol/ddd1111" in ids and len(set(ids)) == 4


def test_grade_without_patches_or_without_the_harness(tmp_path: Path) -> None:
    """No patch means nothing to grade; a failed harness run marks every instance errored."""
    env = SweBenchEnv(sys.executable)
    assert env.grade(tmp_path, [("a", ""), ("b", "  \n")]) == (set(), set())
    assert not (tmp_path / "_eval").exists()
    # ``python -m swebench.harness.run_evaluation`` fails without the package installed
    # in the interpreter's environment, so no report is written.
    resolved, errored = env.grade(
        tmp_path, [("django__django-11019", "diff --git a/x b/x\n"), ("z", "")]
    )
    assert (resolved, errored) == (set(), {"django__django-11019"})
    lines = (tmp_path / "_eval" / "predictions.jsonl").read_text().splitlines()
    predictions = [json.loads(line) for line in lines]
    assert predictions == [
        {
            "instance_id": "django__django-11019",
            "model_name_or_path": "skillopt",
            "model_patch": "diff --git a/x b/x\n",
        }
    ]
    assert skillopt_bridge.unattended_verdict("Bash", {}) == "OK"
    assert skillopt_bridge.unattended_verdict("ask_user_question", {}) != "OK"
