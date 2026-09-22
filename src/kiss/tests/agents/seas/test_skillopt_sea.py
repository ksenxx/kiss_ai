# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the SkillOpt agent (:mod:`kiss.agents.seas.skillopt_sea`).

The optimizer's rollouts run in-process with :class:`SorcarAgent`, so the
whole loop is exercised against the scripted local model server: the
"model" runs the eval commands through the real Bash tool, the analysts
and ranker answer with scripted JSON, and the test checks the gate, the
state file, the candidate SEA and the proposal on disk.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas import sh_sea, skillopt_sea
from kiss.agents.seas.skillopt_sea import (
    EvalTask,
    OptimizeConfig,
    Patch,
    SeaTarget,
    SkillTarget,
    apply_patch,
    edit_budget,
    format_report,
    load_evals,
    load_target,
    parse_patches,
    run_optimization,
    run_rollout,
    verify,
)
from kiss.agents.sorcar import sea_commands
from kiss.server.agent_file import apply_agent_overrides
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
    tool_call_body,
)

_SH_SEA = Path(sh_sea.__file__).resolve()
_SKILLOPT_SEA = Path(skillopt_sea.__file__).resolve()
_EVALS = _SKILLOPT_SEA.parent / "evals" / "sh_sea_evals.json"


def _rollout(command: str, output: str) -> list[bytes]:
    """Script one ``/sh`` rollout: a Bash call, then finish with *output*."""
    return [
        tool_call_body("Bash", {"command": command, "description": "run"}, 500),
        finish_body(f"<pre>{output}</pre>", 600),
    ]


def _patches_body(*patches: dict[str, Any], prose: str = "") -> bytes:
    """Script an analyst/ranker reply carrying *patches* (optionally wrapped in prose)."""
    return text_body(prose + json.dumps({"patches": list(patches)}))


def _copy_sh_sea(tmp_path: Path) -> Path:
    dest = tmp_path / "sh_sea.py"
    shutil.copy(_SH_SEA, dest)
    return dest


def _evals(tmp_path: Path, tasks: list[dict[str, Any]], **extra: Any) -> Path:
    path = tmp_path / "evals.json"
    path.write_text(json.dumps({"tasks": tasks, **extra}), encoding="utf-8")
    return path


def _config(
    tmp_path: Path, target: Path, evals: Path, url: str, **overrides: Any
) -> OptimizeConfig:
    values: dict[str, Any] = {
        "target": target,
        "evals": evals,
        "out_dir": tmp_path / "out",
        "model": MODEL,
        "model_config": {"base_url": url, "api_key": "local"},
        "max_workers": 1,
        "max_steps": 5,
    }
    values.update(overrides)
    return OptimizeConfig(**values)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def test_sea_target_reads_splices_and_validates_the_prompt_constant(tmp_path: Path) -> None:
    """The trainable text of ``sh_sea.py`` is SYSTEM_PROMPT; candidates keep every other line."""
    target = load_target(_copy_sh_sea(tmp_path))
    assert isinstance(target, SeaTarget)
    assert target.kind == "sea"
    assert target.text() == sh_sea.system_prompt()
    tricky = 'Line one with a """triple""" and a back\\slash.\nLine two ends with a quote "'
    assert target.validate(tricky) == ""
    candidate = target.materialize(tricky, tmp_path / "cand")
    assert candidate.path.name == "sh_candidate.py"
    assert candidate.text() == tricky
    # Every getter but the prompt is untouched: the candidate is loaded like a SEA.
    kwargs = candidate.rollout_kwargs()
    assert kwargs["base_system_prompt"] == tricky
    assert kwargs["tool_profile"] == "bash"
    assert kwargs["is_parallel"] is False
    assert kwargs["web_tools"] is False
    assert kwargs["use_memory"] is False
    assert "tools" not in kwargs
    original_lines = _SH_SEA.read_text(encoding="utf-8").splitlines()
    candidate_lines = candidate.path.read_text(encoding="utf-8").splitlines()
    start = original_lines.index("SYSTEM_PROMPT = (")
    end = original_lines.index(")", start)
    assert candidate_lines[: start + 1] == original_lines[: start + 1]
    assert candidate_lines[-(len(original_lines) - end) :] == original_lines[end:]
    proposal = target.write_proposal("single line prompt")
    assert proposal == target.path.with_name("sh_sea.py.proposed")
    assert "'single line prompt'" in proposal.read_text(encoding="utf-8")
    adopted = tmp_path / "adopted" / "sh_sea.py"
    adopted.parent.mkdir()
    shutil.copy(proposal, adopted)
    assert SeaTarget(adopted).text() == "single line prompt"
    assert target.text() == sh_sea.system_prompt()


def test_sea_target_accepts_inline_literals_and_rejects_other_shapes(tmp_path: Path) -> None:
    """``return "..."`` works too; a computed prompt or a missing getter is an error."""
    annotated = tmp_path / "annotated_sea.py"
    annotated.write_text(
        'PROMPT: str = "typed"\n\n\ndef system_prompt():\n    return PROMPT\n', encoding="utf-8"
    )
    assert SeaTarget(annotated).text() == "typed"
    assert SeaTarget(annotated).materialize("t2", tmp_path / "t2").text() == "t2"
    inline = tmp_path / "inline_sea.py"
    inline.write_text('def system_prompt():\n    return "inline prompt"\n', encoding="utf-8")
    target = SeaTarget(inline)
    assert target.text() == "inline prompt"
    assert target.validate("changed") == ""
    assert target.materialize("changed", tmp_path / "c").text() == "changed"

    computed = tmp_path / "computed_sea.py"
    computed.write_text('def system_prompt():\n    return "a" + "b"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="string literal"):
        SeaTarget(computed).text()
    unbound = tmp_path / "unbound_sea.py"
    unbound.write_text("def system_prompt():\n    return MISSING\n", encoding="utf-8")
    with pytest.raises(ValueError, match="module-level assignment"):
        SeaTarget(unbound).text()
    no_return = tmp_path / "noreturn_sea.py"
    no_return.write_text("def system_prompt():\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="return statement"):
        SeaTarget(no_return).text()
    without = tmp_path / "without_sea.py"
    without.write_text("def tools():\n    return []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no top-level system_prompt"):
        SeaTarget(without).text()
    empty = tmp_path / "empty_sea.py"
    empty.write_text('def system_prompt():\n    return ""\n', encoding="utf-8")
    with pytest.raises(ValueError, match="returned nothing"):
        SeaTarget(empty).rollout_kwargs()
    with pytest.raises(ValueError, match="not found"):
        SeaTarget(tmp_path / "nope_sea.py")
    with pytest.raises(ValueError, match="unsupported target"):
        load_target(tmp_path / "x.txt")


def test_sea_target_maps_every_getter_and_a_tools_file(tmp_path: Path) -> None:
    """Optional getters (tools list or path, hooks, model, ...) all reach the rollout kwargs."""
    tools_file = tmp_path / "extra_tools.py"
    tools_file.write_text(
        'def greet(name: str) -> str:\n    """Greet.\n\n    Args:\n        name: Who.\n\n'
        '    Returns:\n        Text.\n    """\n    return "hi " + name\n\n\n'
        "def tools():\n    return [greet]\n",
        encoding="utf-8",
    )
    sea = tmp_path / "full_sea.py"
    sea.write_text(
        "PROMPT = 'p'\n\n\ndef system_prompt():\n    return PROMPT\n\n\n"
        "def tool_profile():\n    return 'shell'\n\n\ndef model():\n    return 'm'\n\n\n"
        "def model_config():\n    return {'k': 1}\n\n\ndef append_to_system_prompt():\n"
        "    return 'suffix'\n\n\ndef if_append_basic_tools():\n    return False\n\n\n"
        "def docker_image():\n    return 'img'\n\n\ndef use_web_tools():\n    return True\n\n\n"
        "def llm_call_hook():\n    return lambda m: m\n\n\ndef tool_call_hook():\n"
        "    return lambda n, a: 'OK'\n\n\n"
        "def prompt():\n    return 'fixed prompt'\n\n\ndef append_to_prompt():\n"
        "    return ' suffix'\n\n\nfrom pathlib import Path\n\n\n"
        f"def tools():\n    return Path({str(tools_file)!r})\n",
        encoding="utf-8",
    )
    kwargs = SeaTarget(sea).rollout_kwargs()
    assert kwargs["base_system_prompt"] == "p"
    assert kwargs["tool_profile"] == "shell"
    assert kwargs["model_name"] == "m"
    assert kwargs["model_config"] == {"k": 1}
    assert kwargs["system_prompt"] == "suffix"
    assert kwargs["append_basic_tools"] is False
    assert kwargs["docker_image"] == "img"
    assert kwargs["web_tools"] is True
    assert callable(kwargs["llm_call_hook"]) and callable(kwargs["tool_call_hook"])
    assert [t.__name__ for t in kwargs["tools"]] == ["greet"]
    assert kwargs["prompt"] == "fixed prompt" and kwargs["append_to_prompt"] == " suffix"
    inline_tools = tmp_path / "list_sea.py"
    inline_tools.write_text(
        "def helper():\n    return 1\n\n\ndef system_prompt():\n    return 'q'\n\n\n"
        "def tools():\n    return [helper]\n",
        encoding="utf-8",
    )
    assert [t.__name__ for t in SeaTarget(inline_tools).rollout_kwargs()["tools"]] == ["helper"]


def test_sea_fingerprint_detects_code_changes_outside_the_prompt(tmp_path: Path) -> None:
    """The confinement gate compares ASTs with the prompt blanked."""
    base = (
        "X = 'a'\n\n\ndef system_prompt():\n    return X\n\n\n"
        "def use_worktree():\n    return False\n"
    )
    same_code = base.replace("'a'", "'totally different prompt'")
    changed = base.replace("return False", "return True")
    assert skillopt_sea._fingerprint(base) == skillopt_sea._fingerprint(same_code)
    assert skillopt_sea._fingerprint(base) != skillopt_sea._fingerprint(changed)
    assert ast.literal_eval(skillopt_sea._string_literal('ends with "')) == 'ends with "'
    assert ast.literal_eval(skillopt_sea._string_literal('a\n"""\\\n')) == 'a\n"""\\\n'


def test_skill_target_uses_the_whole_file_as_text(tmp_path: Path) -> None:
    """A SKILL.md is its own text and is injected as an appended system prompt."""
    skill = tmp_path / "SKILL.md"
    skill.write_text("---\nname: demo\n---\n# Demo\nDo things.\n", encoding="utf-8")
    target = load_target(skill)
    assert isinstance(target, SkillTarget)
    assert target.kind == "skill"
    assert target.text() == skill.read_text(encoding="utf-8")
    assert target.rollout_kwargs() == {"system_prompt": target.text()}
    assert target.validate("anything") == ""
    candidate = target.materialize("# New\n", tmp_path / "cand")
    assert candidate.path == tmp_path / "cand" / "SKILL.md"
    assert candidate.text() == "# New\n"
    assert target.write_proposal("# P\n").read_text(encoding="utf-8") == "# P\n"
    assert target.proposal_path() == tmp_path / "SKILL.md.proposed"


# --------------------------------------------------------------------------
# Eval set, verification, patches
# --------------------------------------------------------------------------


def test_load_evals_normalizes_and_validates(tmp_path: Path) -> None:
    """String ``expect`` becomes a list, ids default, duplicates and bad splits are errors."""
    tasks, defaults = load_evals(
        _evals(
            tmp_path,
            [
                {"prompt": "echo a", "expect": "a"},
                {"id": "b", "prompt": "echo b", "split": "select"},
            ],
            rollout={"max_steps": 3},
        )
    )
    assert [t.id for t in tasks] == ["task1", "b"]
    assert tasks[0].expect == ["a"] and tasks[1].split == "select"
    assert defaults == {"max_steps": 3}
    bare = tmp_path / "bare.json"
    bare.write_text('[{"id": "x", "prompt": "p"}]', encoding="utf-8")
    assert load_evals(bare)[0][0].id == "x" and load_evals(bare)[1] == {}
    with pytest.raises(ValueError, match="unique"):
        load_evals(_evals(tmp_path, [{"id": "a", "prompt": "p"}, {"id": "a", "prompt": "q"}]))
    with pytest.raises(ValueError, match="split"):
        load_evals(_evals(tmp_path, [{"id": "a", "prompt": "p", "split": "test"}]))
    real_tasks, real_defaults = load_evals(_EVALS)
    assert len(real_tasks) == 10 and real_defaults["web_tools"] is False


def test_verify_grades_expect_regex_check_and_success(tmp_path: Path) -> None:
    """Each verifier kind fails on its own condition; HTML is stripped first."""
    task = EvalTask(
        id="t", prompt="p", expect=["a &amp; b"], expect_regex=r"4\s*\n\s*5", check="test -f made"
    )
    assert verify(task, "<pre>a &amp;amp; b\n4\n5</pre>", True, tmp_path) == (
        False,
        "check exited 1: ",
    )
    (tmp_path / "made").touch()
    assert verify(task, "<pre>a &amp;amp; b\n4\n5</pre>", False, tmp_path) == (True, "pass")
    assert verify(task, "<pre>a &amp;amp; b 4 5</pre>", True, tmp_path)[1].startswith("regex")
    assert verify(task, "nothing", True, tmp_path) == (
        False,
        "missing expected text: ['a &amp; b']",
    )
    bare = EvalTask(id="t", prompt="p")
    assert verify(bare, "", True, tmp_path) == (True, "pass")
    assert verify(bare, "", False, tmp_path) == (False, "rollout did not finish successfully")


def test_patches_parse_apply_and_budget() -> None:
    """Patch JSON is found inside prose, bad ops are dropped, anchors must exist."""
    reply = (
        'Sure. {"note": 1} then {"patches": [{"op": "add", "anchor": "", "text": "x"}, '
        '{"op": "nope"}, {"op": "delete", "anchor": "b"}]} trailing'
    )
    patches = parse_patches(reply)
    assert [(p.op, p.anchor, p.text) for p in patches] == [("add", "", "x"), ("delete", "b", "")]
    assert parse_patches("no json here") == []
    assert parse_patches('{"patches": "not a list"} {broken') == []
    text = "a b c"
    assert apply_patch(text, Patch("add", "", "\nd\n")) == "a b c\nd\n"
    assert apply_patch(text, Patch("add", "b", "+")) == "a b+ c"
    assert apply_patch(text, Patch("replace", "b", "B")) == "a B c"
    assert apply_patch(text, Patch("delete", " b", "")) == "a c"
    assert apply_patch(text, Patch("add", "zzz", "+")) is None
    assert apply_patch(text, Patch("replace", "zzz", "+")) is None
    assert [edit_budget(i, 5, 4, 2) for i in range(5)] == [4, 4, 3, 2, 2]
    assert edit_budget(0, 1, 4, 2) == 4
    assert edit_budget(3, 5, 2, 2) == 2


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_one_round_accepts_a_candidate_that_passes_more_selection_tasks(tmp_path: Path) -> None:
    """Rollouts run the real Bash tool; the gate accepts 1.0 > 0.5 and writes the proposal.

    Request order (max_workers=1): best on a, best on b, failure analyst
    (b failed), success analyst (a passed), ranker, candidate on a and b.
    """
    sea = _copy_sh_sea(tmp_path)
    evals = _evals(
        tmp_path,
        [
            {"id": "a", "prompt": "echo alpha", "expect": ["alpha"]},
            {
                "id": "b",
                "prompt": "echo beta",
                "expect": ["beta"],
                "check": "test -f marker",
                "setup": "touch marker",
            },
        ],
    )
    patch = {"op": "add", "anchor": "", "text": "Return the exact output.", "rationale": "r"}
    script = [
        *_rollout("echo alpha", "alpha"),
        *_rollout("echo beta", "wrong"),
        _patches_body(patch, prose="Analysis first.\n"),
        _patches_body(),
        _patches_body(patch, {"op": "replace", "anchor": "missing anchor", "text": "x"}),
        *_rollout("echo alpha", "alpha"),
        *_rollout("echo beta", "beta"),
    ]
    with serve(script) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, evals, url))
    assert len(requests) == 11
    tool_names = [{t["function"]["name"] for t in r["tools"]} for r in requests if r.get("tools")]
    assert tool_names == [{"Bash", "finish"}] * 8
    bash_result = str(requests[1]["messages"][-1]["content"])
    assert "alpha" in bash_result
    analyst_prompt = str(requests[4]["messages"][-1]["content"])
    assert "FAILED" in analyst_prompt and "missing expected text: ['beta']" in analyst_prompt
    assert "echo beta" in analyst_prompt
    assert "PASSED" in str(requests[5]["messages"][-1]["content"])
    ranker_prompt = str(requests[6]["messages"][-1]["content"])
    assert "at most 4 patches" in ranker_prompt and "Return the exact output." in ranker_prompt
    assert "- b: 'echo beta' -> missing expected text: ['beta']" in ranker_prompt
    candidate_system = str(requests[7]["messages"][0]["content"])
    assert candidate_system.startswith(sh_sea.system_prompt() + "\nReturn the exact output.")
    assert "# Restricted tool profile: bash" in candidate_system

    assert report["improved"] is True
    assert report["best_score"] == 1.0 and report["rounds_done"] == 1
    round1 = report["rounds"][0]
    assert round1["best_score"] == 0.5 and round1["candidate_score"] == 1.0
    assert round1["accepted"] is True and round1["edit_budget"] == 4
    assert round1["best_results"] == {"a": True, "b": False}
    assert round1["candidate_results"] == {"a": True, "b": True}
    assert [p["op"] for p in round1["patches"]] == ["add", "replace"]
    assert report["total_cost"] > 0
    proposal = Path(report["proposal"])
    assert proposal == tmp_path / "sh_sea.py.proposed"
    assert SeaTarget(proposal.with_suffix("")).text() == sh_sea.system_prompt()
    ns: dict[str, Any] = {}
    exec(compile(proposal.read_text(encoding="utf-8"), str(proposal), "exec"), ns)  # noqa: S102
    assert ns["system_prompt"]() == sh_sea.system_prompt() + "\nReturn the exact output.\n"
    assert ns["tool_profile"]() == "bash"
    assert sea.read_text(encoding="utf-8") == _SH_SEA.read_text(encoding="utf-8")

    out = tmp_path / "out"
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert state["best_text"].endswith("Return the exact output.\n") and state["rounds_done"] == 1
    assert state["rejected"] == []
    rollouts = json.loads(
        (out / "round_01" / "train" / "rollouts.json").read_text(encoding="utf-8")
    )
    assert [r["passed"] for r in rollouts] == [True, False]
    trajectory = rollouts[1]["trajectory"]
    roles = [m["role"] for m in trajectory]
    assert roles[:2] == ["user", "assistant"] and roles[-1] == "result"
    assert "system" not in roles and roles.count("user") == 1
    calls = [m["text"] for m in trajectory if m["role"] == "assistant"]
    assert calls[0].startswith('[call Bash] {"command": "echo beta"')
    assert calls[1].startswith('[call finish] {"success": true')
    assert any("echo beta" in m["text"] for m in rollouts[1]["trajectory"])
    assert (out / "round_01" / "candidate" / "sh_candidate.py").exists()
    assert (out / "round_01" / "candidate.txt").read_text(encoding="utf-8") == state["best_text"]
    assert (out / "round_01" / "train" / "b" / "marker").exists()
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "skipped patch with missing anchor" in log and "accepted: 1.000 > 0.500" in log
    text = format_report(report)
    assert "proposal written to" in text and "+Return the exact output." in text
    assert skillopt_sea.status(str(out)).splitlines()[3:] == text.splitlines()[3:]


def test_second_round_resumes_and_rejects_a_tie_or_regression(tmp_path: Path) -> None:
    """Resuming continues from the accepted text; a candidate scoring <= best is buffered."""
    sea = _copy_sh_sea(tmp_path)
    evals = _evals(
        tmp_path,
        [
            {"id": "a", "prompt": "echo alpha", "expect": ["alpha"], "split": "train"},
            {"id": "b", "prompt": "echo beta", "expect": ["beta"]},
            {"id": "c", "prompt": "echo gamma", "expect": ["gamma"], "split": "select"},
        ],
    )
    first = {"op": "add", "anchor": "", "text": "First rule.", "rationale": ""}
    # Round 1: train = a, b; select = b, c (c is rolled out separately for the best).
    script = [
        *_rollout("echo alpha", "alpha"),
        *_rollout("echo beta", "wrong"),
        *_rollout("echo gamma", "gamma"),
        _patches_body(first),
        _patches_body(),
        _patches_body(first),
        *_rollout("echo beta", "beta"),
        *_rollout("echo gamma", "gamma"),
    ]
    with serve(script) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, evals, url))
    assert len(requests) == 13
    assert report["rounds"][0]["accepted"] is True
    assert report["rounds"][0]["best_results"] == {"b": False, "c": True}
    assert (tmp_path / "out" / "round_01" / "select_best" / "rollouts.json").exists()

    second = {"op": "replace", "anchor": "First rule.", "text": "Second rule.", "rationale": ""}
    # Round 2 (resumed): everything passes for the best; the candidate loses c -> rejected.
    script = [
        *_rollout("echo alpha", "alpha"),
        *_rollout("echo beta", "beta"),
        *_rollout("echo gamma", "gamma"),
        _patches_body(second),
        _patches_body(second),
        *_rollout("echo beta", "beta"),
        *_rollout("echo gamma", "nope"),
    ]
    with serve(script) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, evals, url))
    assert len(requests) == 12
    assert report["rounds_done"] == 2 and len(report["rounds"]) == 2
    round2 = report["rounds"][1]
    assert round2["round"] == 2 and round2["accepted"] is False
    assert round2["best_score"] == 1.0 and round2["candidate_score"] == 0.5
    assert round2["reason"] == "rejected: 0.500 <= 1.000"
    assert "First rule." in str(requests[6]["messages"][-1]["content"])
    assert "Second rule." in str(requests[8]["messages"][0]["content"])
    state = json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))
    assert state["best_text"].endswith("First rule.\n")
    assert [r["round"] for r in state["rejected"]] == [2]
    assert state["rejected"][0]["patches"][0]["text"] == "Second rule."
    assert "First rule." in (tmp_path / "sh_sea.py.proposed").read_text(encoding="utf-8")

    # Round 3: the ranker sees the rejected buffer; patches that change nothing end the round.
    script = [
        *_rollout("echo alpha", "alpha"),
        *_rollout("echo beta", "beta"),
        *_rollout("echo gamma", "gamma"),
        _patches_body(second),
        _patches_body({"op": "delete", "anchor": "absent", "text": ""}),
    ]
    with serve(script) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, evals, url))
    assert len(requests) == 8
    assert "Second rule." in str(requests[7]["messages"][-1]["content"])
    assert report["rounds"][2]["reason"] == "ranked patches did not change the text"

    # Round 4: no patches at all ends the round before ranking; ``fresh`` restarts from scratch.
    script = [
        *_rollout("echo alpha", "alpha"),
        *_rollout("echo beta", "beta"),
        *_rollout("echo gamma", "gamma"),
        _patches_body(),
    ]
    with serve(script) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, evals, url, fresh=True))
    assert len(requests) == 7
    assert report["rounds_done"] == 1 and report["improved"] is False
    assert report["rounds"][0]["reason"] == "analysts proposed no patches"
    assert "no candidate beat the original" in format_report(report)
    assert not (tmp_path / "sh_sea.py.proposed").exists()
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["log.txt", "round_01", "state.json"]

    # Round 5: a different eval set in the same out_dir is another run, not a resume.
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps({"tasks": [{"id": "z", "prompt": "echo zeta", "expect": ["zeta"]}]}),
        encoding="utf-8",
    )
    script = [*_rollout("echo zeta", "zeta"), _patches_body()]
    with serve(script) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, other, url))
    assert len(requests) == 3 and report["rounds_done"] == 1
    assert "starting over" in (out / "log.txt").read_text(encoding="utf-8")


def test_cost_cap_crashed_rollouts_and_epoch_batches(tmp_path: Path) -> None:
    """``max_cost`` stops before a round; a rollout exception is a failed task, not a crash."""
    sea = _copy_sh_sea(tmp_path)
    evals = _evals(tmp_path, [{"id": "a", "prompt": "echo alpha", "expect": ["alpha"]}])
    with serve([finish_body("<p>x</p>")]) as (url, requests):
        report = run_optimization(_config(tmp_path, sea, evals, url, max_cost=0.0))
    assert requests == [] and report["rounds_done"] == 0 and report["rounds"] == []
    assert "stopping: spent" in (tmp_path / "out" / "log.txt").read_text(encoding="utf-8")

    # An unknown tool_profile makes SorcarAgent.run raise: a failed rollout, not a crash.
    broken = tmp_path / "broken_sea.py"
    broken.write_text(
        "def system_prompt():\n    return 'p'\n\n\ndef tool_profile():\n    return 'bogus'\n",
        encoding="utf-8",
    )
    three = _evals(tmp_path, [{"id": t, "prompt": f"echo {t}", "expect": [t]} for t in "abc"])
    with serve([_patches_body()]) as (url, requests):
        report = run_optimization(
            _config(tmp_path, broken, three, url, out_dir=tmp_path / "out2", epochs=2, batch_size=2)
        )
    # Two epochs over three tasks in batches of two: [a, b], [c], [a, b], [c].
    assert report["rounds_done"] == 4
    assert [r["train_tasks"] for r in report["rounds"]] == [["a", "b"], ["c"], ["a", "b"], ["c"]]
    assert [r["edit_budget"] for r in report["rounds"]] == [4, 4, 3, 2]
    assert all(r["best_score"] == 0.0 for r in report["rounds"])
    rollouts = json.loads((tmp_path / "out2" / "round_02" / "train" / "rollouts.json").read_text())
    assert rollouts[0]["passed"] is False and "bogus" in rollouts[0]["error"]
    assert [len(r["messages"]) for r in requests] == [1] * 4
    # Resuming continues the epoch cycle where it stopped.
    with serve([_patches_body()]) as (url, requests):
        report = run_optimization(
            _config(tmp_path, broken, three, url, out_dir=tmp_path / "out2", epochs=1, batch_size=2)
        )
    assert [r["train_tasks"] for r in report["rounds"][4:]] == [["a", "b"], ["c"]]


def test_run_rollout_applies_eval_defaults_and_prompt_getters(tmp_path: Path) -> None:
    """Eval-set ``rollout`` defaults reach ``SorcarAgent.run``; prompt getters shape the prompt."""
    fixed = tmp_path / "fixed_sea.py"
    fixed.write_text(
        "def system_prompt():\n    return 'p'\n\n\ndef prompt():\n    return 'fixed prompt'\n\n\n"
        "def append_to_prompt():\n    return ' suffix'\n\n\n"
        "def tool_profile():\n    return 'bash'\n",
        encoding="utf-8",
    )
    task = EvalTask(id="t", prompt="ignored task prompt", expect=["done"])
    with serve([finish_body("<p>done</p>")]) as (url, requests):
        cfg = _config(tmp_path, fixed, tmp_path / "unused.json", url)
        rollout = run_rollout(
            SeaTarget(fixed),
            task,
            cfg,
            tmp_path / "w1",
            {"append_basic_tools": False, "max_steps": 7},
        )
    assert rollout.passed and rollout.steps == 1
    assert {t["function"]["name"] for t in requests[0]["tools"]} == {"finish"}
    user = [m for m in requests[0]["messages"] if m["role"] == "user"]
    assert "fixed prompt suffix" in str(user[0]["content"])
    assert "ignored task prompt" not in str(user[0]["content"])
    with serve(_rollout("echo hi", "hi")) as (url, requests):
        cfg = _config(tmp_path, fixed, tmp_path / "unused.json", url)
        rollout = run_rollout(SeaTarget(fixed), task, cfg, tmp_path / "w2", {"max_steps": 7})
    assert "Steps: 1/7" in str(requests[1]["messages"][-1]["content"])
    assert rollout.passed is False and rollout.verdict == "missing expected text: ['done']"


def test_missing_split_is_an_error(tmp_path: Path) -> None:
    """An eval set with no train task (or no selection task) cannot be optimized."""
    sea = _copy_sh_sea(tmp_path)
    evals = _evals(tmp_path, [{"id": "a", "prompt": "p", "split": "select"}])
    with pytest.raises(ValueError, match="at least one train task"):
        run_optimization(_config(tmp_path, sea, evals, "http://127.0.0.1:9/v1"))


# --------------------------------------------------------------------------
# SEA surface and CLI
# --------------------------------------------------------------------------


def test_sea_getters_and_tools_follow_the_contract(tmp_path: Path) -> None:
    """The SkillOpt SEA is registered as ``/skillopt`` and exposes optimize/status."""
    assert skillopt_sea.system_prompt().startswith("You are SkillOpt")
    assert [t.__name__ for t in skillopt_sea.tools()] == ["optimize", "status"]
    assert skillopt_sea.tool_profile() == "shell"
    assert skillopt_sea.use_worktree() is False
    assert skillopt_sea.auto_commit() is False
    assert skillopt_sea.classify_tasks() is False
    assert skillopt_sea.is_parallel() is False
    assert skillopt_sea.use_web_tools() is False
    assert skillopt_sea.use_memory() is False
    assert sea_commands.get_command("skillopt") == _SKILLOPT_SEA
    cmd: dict[str, Any] = {"agentPath": str(_SKILLOPT_SEA)}
    assert "autoCommit" in apply_agent_overrides(cmd)
    assert cmd["autoCommit"] is False and cmd["useWorktree"] is False
    assert cmd["toolProfile"] == "shell" and cmd["toolsFile"] == str(_SKILLOPT_SEA)
    assert cmd["systemPrompt"] == skillopt_sea.SYSTEM_PROMPT
    assert skillopt_sea.status(str(tmp_path / "none")) == f"no state.json under {tmp_path / 'none'}"
    # The tool wrapper with a zero cost cap runs no round and needs no model.
    sea = _copy_sh_sea(tmp_path)
    evals = _evals(tmp_path, [{"id": "a", "prompt": "echo a", "expect": ["a"]}])
    text = skillopt_sea.optimize(str(sea), str(evals), max_cost=0.0, model="unused-model")
    assert "rounds done: 0" in text
    assert (tmp_path / ".skillopt" / "sh_sea" / "state.json").exists()
    text = skillopt_sea.optimize(str(sea), str(evals), out_dir=str(tmp_path / "o"), max_cost=0.0)
    assert (tmp_path / "o" / "state.json").exists()


def test_cli_runs_the_optimizer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m kiss.agents.seas.skillopt_sea`` drives the same loop from the terminal."""
    sea = _copy_sh_sea(tmp_path)
    evals = _evals(tmp_path, [{"id": "a", "prompt": "echo alpha", "expect": ["alpha"]}])
    script = [*_rollout("echo alpha", "alpha"), _patches_body()]
    with serve(script) as (url, requests):
        code = skillopt_sea.main(
            [
                "--target",
                str(sea),
                "--evals",
                str(evals),
                "--out-dir",
                str(tmp_path / "cli"),
                "--model",
                MODEL,
                "--model-config",
                json.dumps({"base_url": url, "api_key": "local"}),
                "--max-workers",
                "1",
                "--max-steps",
                "4",
                "--fresh",
            ]
        )
    assert code == 0 and len(requests) == 3
    out = capsys.readouterr().out
    assert "rounds done: 1" in out and "analysts proposed no patches" in out
