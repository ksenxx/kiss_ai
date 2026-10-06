# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the SkillOpt agent (:mod:`kiss.agents.seas.skillopt.skillopt_sea`).

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

from kiss.agents.seas.sh import sh_sea
from kiss.agents.seas.skillopt import skillopt_sea
from kiss.agents.seas.skillopt.skillopt_sea import (
    ConstantTarget,
    Env,
    EvalTask,
    OptimizeConfig,
    Patch,
    Rollout,
    SeaTarget,
    SkillTarget,
    Target,
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
from kiss.agents.sorcar.agent_file import apply_agent_overrides
from kiss.agents.sorcar.sea_settings import SeaError, resolve_settings
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
    tool_call_body,
)
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon

_SH_SEA = Path(sh_sea.__file__).resolve()
_SKILLOPT_SEA = Path(skillopt_sea.__file__).resolve()
_EVALS = Path(sh_sea.__file__).resolve().parent / "evals" / "sh_sea_evals.json"


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
    assert target.text() == sh_sea.ShSea().system_prompt("")
    tricky = 'Line one with a """triple""" and a back\\slash.\nLine two ends with a quote "'
    assert target.validate(tricky) == ""
    candidate = target.materialize(tricky, tmp_path / "cand")
    assert candidate.path.name == "sh_candidate.py"
    assert candidate.text() == tricky
    # Everything but the prompt is untouched: the candidate is loaded like a
    # SEA, so its ``settings()`` (``worker`` kind, Bash profile) reach the
    # rollout exactly as the daemon would apply them.
    kwargs = candidate.rollout_kwargs()
    assert kwargs["system_prompt_hook"]("DEFAULT PROMPT") == tricky
    assert kwargs["tool_profile"] == "bash"
    assert kwargs["is_parallel"] is False
    assert kwargs["web_tools"] is False
    assert kwargs["use_memory"] is False
    assert "tools_hook" not in kwargs and "prompt" not in kwargs
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
    assert target.text() == sh_sea.ShSea().system_prompt("")


def test_sea_target_accepts_inline_literals_and_rejects_other_shapes(tmp_path: Path) -> None:
    """``return "..."`` works too; a computed prompt or a missing method is an error."""
    annotated = tmp_path / "annotated_sea.py"
    annotated.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

PROMPT: str = "typed"


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return PROMPT
""", encoding="utf-8"
    )
    assert SeaTarget(annotated).text() == "typed"
    assert SeaTarget(annotated).materialize("t2", tmp_path / "t2").text() == "t2"
    inline = tmp_path / "inline_sea.py"
    inline.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return "inline prompt"
""", encoding="utf-8")
    target = SeaTarget(inline)
    assert target.text() == "inline prompt"
    assert target.validate("changed") == ""
    assert target.materialize("changed", tmp_path / "c").text() == "changed"

    computed = tmp_path / "computed_sea.py"
    computed.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return "a" + "b"
""", encoding="utf-8")
    with pytest.raises(ValueError, match="string literal"):
        SeaTarget(computed).text()
    unbound = tmp_path / "unbound_sea.py"
    unbound.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return MISSING
""", encoding="utf-8")
    with pytest.raises(ValueError, match="module-level assignment"):
        SeaTarget(unbound).text()
    no_return = tmp_path / "noreturn_sea.py"
    no_return.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        pass
""", encoding="utf-8")
    with pytest.raises(ValueError, match="return statement"):
        SeaTarget(no_return).text()
    without = tmp_path / "without_sea.py"
    without.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def tools(self, tools):
        return tools + []
""", encoding="utf-8")
    with pytest.raises(ValueError, match=r"no system_prompt\(\) method"):
        SeaTarget(without).text()
    empty = tmp_path / "empty_sea.py"
    empty.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return ""
""", encoding="utf-8")
    with pytest.raises(ValueError, match="returned nothing"):
        SeaTarget(empty).rollout_kwargs()
    with pytest.raises(ValueError, match="not found"):
        SeaTarget(tmp_path / "nope_sea.py")
    with pytest.raises(ValueError, match="unsupported target"):
        load_target(tmp_path / "x.txt")


def test_sea_target_maps_every_setting_and_getter(tmp_path: Path) -> None:
    """Every ``settings()`` key and method (tools, hooks, ...) reaches the rollout kwargs."""
    sea = tmp_path / "full_sea.py"
    sea.write_text(
        '''
from kiss.agents.seas.base.base_sea import BaseSea

PROMPT = 'p'


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return PROMPT

    def prompt(self, task):
        return 'fixed prompt suffix'

    def settings(self, settings):
        return settings | {
            'tool_profile': 'none', 'model': 'm', 'model_config': {'k': 1},
            'docker_image': 'img', 'use_web_tools': True,
        }

    def llm_call_hook(self, new_messages):
        return new_messages + ['seen']

    def tool_call_hook(self, name, args):
        return 'refused' if name == 'Bash' else 'OK'

    def tools(self, tools):
        return tools + [greet]


def greet(name: str) -> str:
    """Greet.

    Args:
        name: Who.

    Returns:
        Text.
    """
    return "hi " + name
''',
        encoding="utf-8",
    )
    kwargs = SeaTarget(sea).rollout_kwargs()
    assert kwargs["system_prompt_hook"]("DEFAULT PROMPT") == "p"
    assert kwargs["tool_profile"] == "none"
    assert kwargs["model_name"] == "m"
    assert kwargs["model_config"] == {"k": 1}
    # The ``none`` profile with ``tools()`` fixes the whole tool set: no basic tools.
    assert kwargs["append_basic_tools"] is False
    assert kwargs["docker_image"] == "img"
    assert kwargs["web_tools"] is True
    assert kwargs["llm_call_hook"](["m"]) == ["m", "seen"]
    assert kwargs["tool_call_hook"]("Bash", {}) == "refused"
    assert kwargs["tool_call_hook"]("Read", {}) == "OK"
    assert [t.__name__ for t in kwargs["tools_hook"]([greet_stub])] == ["greet_stub", "greet"]
    assert kwargs["prompt"]("any task") == "fixed prompt suffix"
    assert "add_to_prompt" not in kwargs and "tools" not in kwargs
    added_tools = tmp_path / "list_sea.py"
    added_tools.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

def helper():
    return 1


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 'q'

    def tools(self, tools):
        return tools + [helper]
""",
        encoding="utf-8",
    )
    added = SeaTarget(added_tools).rollout_kwargs()
    # ``tools()`` extends the basic toolset: no profile is forced
    # and the eval set's ``append_basic_tools`` default stands.
    assert [t.__name__ for t in added["tools_hook"]([])] == ["helper"]
    assert "append_basic_tools" not in added and "tool_profile" not in added
    # Removed legacy getters are ordinary module functions: they configure nothing.
    legacy = tmp_path / "legacy_sea.py"
    legacy.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

def helper():
    return 1


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 'q'


def tools():
    return [helper]


def model():
    return 'm'


def tool_profile():
    return 'shell'


def append_to_system_prompt():
    return 'suffix'
""",
        encoding="utf-8",
    )
    only = SeaTarget(legacy).rollout_kwargs()
    assert list(only) == ["system_prompt_hook"] and only["system_prompt_hook"]("x") == "q"


def greet_stub(name: str) -> str:
    """Stand-in built-in tool handed to a SEA's ``tools()``."""
    return name


def test_sea_target_rejects_invalid_tool_getters(tmp_path: Path) -> None:
    """Same contract as the daemon: ``tools()`` returns a list of callables, never a path."""
    for name, body in (
        ("path", "        return '/some/tools.py'"),
        ("ints", "        return tools + [1]"),
    ):
        bad = tmp_path / f"bad_{name}_sea.py"
        bad.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
            "class Sea(BaseSea):\n"
            "    def system_prompt(self, system_prompt):\n        return 'q'\n\n"
            f"    def tools(self, tools):\n{body}\n",
            encoding="utf-8",
        )
        # The fold runs when the rollout asks for its tools, in the daemon's words.
        tools_hook = SeaTarget(bad).rollout_kwargs()["tools_hook"]
        with pytest.raises(SeaError, match=r"tools\(\) .*must return a list of tool callables"):
            tools_hook([])
    malformed = tmp_path / "malformed_sea.py"
    malformed.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 'q'

    def settings(self, settings):
        return settings | {'tool_profile': 7}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be str, got int"):
        SeaTarget(malformed).rollout_kwargs()


def test_sea_fingerprint_detects_code_changes_outside_the_prompt(tmp_path: Path) -> None:
    """The confinement gate compares ASTs with the prompt blanked."""
    base = (
        """
from kiss.agents.seas.base.base_sea import BaseSea

X = 'a'


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return X

    def settings(self, settings):
        return settings | {'use_worktree': False}
"""
    )
    same_code = base.replace("'a'", "'totally different prompt'")
    changed = base.replace("False", "True")
    find = skillopt_sea._prompt_constant
    assert skillopt_sea._fingerprint(base, find) == skillopt_sea._fingerprint(same_code, find)
    assert skillopt_sea._fingerprint(base, find) != skillopt_sea._fingerprint(changed, find)
    assert ast.literal_eval(skillopt_sea._string_literal('ends with "')) == 'ends with "'
    assert ast.literal_eval(skillopt_sea._string_literal('a\n"""\\\n')) == 'a\n"""\\\n'


_TEMPLATE_MODULE = '''\
"""A harness that formats its prompt inside a class."""

SYSTEM_PROMPT = """\\
Work in {workdir}.
Be careful.{test_context}
"""
LIMIT = 3


class Harness:
    def system_prompt(self):
        return SYSTEM_PROMPT.format(workdir="/app", test_context="")
'''


def test_constant_target_reads_splices_and_keeps_the_template_fields(tmp_path: Path) -> None:
    """A named constant is the text; candidates must keep its code and its format fields."""
    module = tmp_path / "harness_core.py"
    module.write_text(_TEMPLATE_MODULE, encoding="utf-8")
    target = load_target(module, "SYSTEM_PROMPT")
    assert isinstance(target, ConstantTarget) and target.kind == "constant"
    assert target.text() == "Work in {workdir}.\nBe careful.{test_context}\n"
    assert target.rollout_kwargs() == {"base_system_prompt": target.text()}
    new = "Work in {workdir}.{test_context}\nVerify with the real checker.\n"
    assert target.validate(new) == ""
    candidate = target.materialize(new, tmp_path / "cand")
    assert candidate.path.name == "harness_core_candidate.py"
    assert isinstance(candidate, ConstantTarget) and candidate.text() == new
    ns: dict[str, Any] = {}
    exec(compile(candidate.path.read_text(encoding="utf-8"), "c", "exec"), ns)  # noqa: S102
    assert ns["Harness"]().system_prompt() == "Work in /app.\nVerify with the real checker.\n"
    assert ns["LIMIT"] == 3
    assert "replacement fields" in target.validate("Work in {workdir}.\n")
    assert "replacement fields" in target.validate(new + " {extra}")
    assert "not a valid format template" in target.validate(new + " stray }")
    assert "not a valid format template" in target.validate(
        new.replace("{workdir}", "{workdir:bad}")
    )
    assert "not a valid format template" in target.validate(new.replace("{workdir}", "{workdir!x}"))
    assert target.validate('"""' + new) == ""
    assert "does not compile" in ConstantTarget(module, "LIMIT").validate("x")
    with pytest.raises(ValueError, match="no module-level assignment"):
        ConstantTarget(module, "MISSING").text()
    with pytest.raises(ValueError, match="identifier"):
        ConstantTarget(module, "not-a-name")
    assert target.proposal_path() == tmp_path / "harness_core.py.proposed"
    assert target.write_proposal(new).read_text(encoding="utf-8") == candidate.path.read_text(
        encoding="utf-8"
    )
    assert isinstance(load_target(module), SeaTarget)


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
    evals = load_evals(
        _evals(
            tmp_path,
            [
                {"prompt": "echo a", "expect": "a"},
                {"id": "b", "prompt": "echo b", "split": "select"},
            ],
            rollout={"max_steps": 3},
        )
    )
    tasks = evals.tasks
    assert [t.id for t in tasks] == ["task1", "b"]
    assert tasks[0].expect == ["a"] and tasks[1].split == "select"
    assert evals.rollout == {"max_steps": 3}
    assert evals.env is None and evals.train_rollouts == []
    bare = tmp_path / "bare.json"
    bare.write_text('[{"id": "x", "prompt": "p"}]', encoding="utf-8")
    assert load_evals(bare).tasks[0].id == "x" and load_evals(bare).rollout == {}
    with pytest.raises(ValueError, match="unique"):
        load_evals(_evals(tmp_path, [{"id": "a", "prompt": "p"}, {"id": "a", "prompt": "q"}]))
    with pytest.raises(ValueError, match="split"):
        load_evals(_evals(tmp_path, [{"id": "a", "prompt": "p", "split": "test"}]))
    with pytest.raises(ValueError, match="module:Class"):
        load_evals(_evals(tmp_path, [{"id": "a", "prompt": "p"}], env={"class": "nocolon"}))
    real = load_evals(_EVALS)
    assert len(real.tasks) == 10 and real.rollout["web_tools"] is False


def test_load_evals_imports_train_rollouts_and_env(tmp_path: Path) -> None:
    """``train_rollouts`` is read relative to the eval set; unknown task ids are an error."""
    rollouts = [
        {
            "task_id": "a",
            "passed": False,
            "verdict": "wrong",
            "result": "r",
            "success": True,
            "trajectory": [{"role": "user", "text": "p"}],
            "cost": 0.1,
            "tokens": 5,
            "steps": 2,
        }
    ]
    (tmp_path / "rollouts.json").write_text(json.dumps(rollouts), encoding="utf-8")
    evals = load_evals(
        _evals(
            tmp_path,
            [
                {"id": "a", "prompt": "p", "split": "train"},
                {"id": "b", "prompt": "q", "split": "select"},
            ],
            train_rollouts="rollouts.json",
            env={
                "class": "kiss.agents.seas.skillopt.skillopt_sea:InProcessEnv",
                "defaults": {"max_steps": 2},
            },
        )
    )
    assert [r.task_id for r in evals.train_rollouts] == ["a"] and evals.train_rollouts[
        0
    ].error == ""
    env = skillopt_sea.make_env(evals.env, evals.rollout)
    assert isinstance(env, skillopt_sea.InProcessEnv) and env.defaults == {"max_steps": 2}
    assert isinstance(skillopt_sea.make_env(None, {"x": 1}), skillopt_sea.InProcessEnv)
    with pytest.raises(ValueError, match="not an Env"):
        skillopt_sea.make_env(
            {"class": "kiss.agents.seas.skillopt.skillopt_sea:EvalTask", "id": "a", "prompt": "p"},
            {},
        )
    with pytest.raises(ValueError, match="split 'train'"):
        load_evals(_evals(tmp_path, [{"id": "b", "prompt": "q"}], train_rollouts="rollouts.json"))
    with pytest.raises(ValueError, match="split 'train'"):
        load_evals(_evals(tmp_path, [{"id": "a", "prompt": "p"}], train_rollouts="rollouts.json"))


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
    sh = sh_sea.ShSea()
    assert candidate_system.startswith(
        sh.system_prompt("").rstrip("\n") + "\nReturn the exact output."
    )
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
    assert SeaTarget(proposal.with_suffix("")).text() == sh.system_prompt("")
    ns: dict[str, Any] = {}
    exec(compile(proposal.read_text(encoding="utf-8"), str(proposal), "exec"), ns)  # noqa: S102
    proposed = ns["ShSea"]()
    assert (
        proposed.system_prompt("DEFAULT PROMPT")
        == sh.system_prompt("").rstrip("\n") + "\nReturn the exact output.\n"
    )
    # Every other line of the SEA is kept: its ``settings()`` still pin the Bash profile.
    assert proposed.settings({}) == sh.settings({}) == {
        "kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"],
    }
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


class MarkerEnv(Env):
    """A benchmark stand-in: a task passes when the text carries *marker* (or the task is ``easy``).

    Every call is appended to *log* as ``[work_root name, task ids]`` so the
    test can check which tasks were rolled out and which were imported.
    """

    def __init__(self, log: str, marker: str) -> None:
        self.log = Path(log)
        self.marker = marker

    def rollouts(
        self, target: Target, tasks: list[EvalTask], cfg: OptimizeConfig, work_root: Path
    ) -> list[Rollout]:
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps([work_root.name, [t.id for t in tasks]]) + "\n")
        text = target.text()
        rollouts = []
        for task in tasks:
            passed = self.marker in text or task.id == "easy"
            verdict = "pass" if passed else f"text lacks {self.marker!r}"
            rollouts.append(
                Rollout(
                    task.id,
                    passed,
                    verdict,
                    "r",
                    True,
                    [{"role": "result", "text": "r"}],
                    0.01,
                    10,
                    1,
                )
            )
        return rollouts


def test_imported_trajectories_and_a_custom_env_drive_a_constant_target(tmp_path: Path) -> None:
    """Imported trajectories are the training rollouts of every round; the env gates the candidates.

    Model calls (no rollout touches the model): failure analyst and success
    analyst over the imported trajectories, then the ranker, in both rounds.
    """
    module = tmp_path / "harness_core.py"
    module.write_text(_TEMPLATE_MODULE, encoding="utf-8")
    long_trajectory = [{"role": "assistant", "text": "[call Bash] " + "x" * 1500}] * 20
    rollouts = [
        {
            "task_id": "t1",
            "passed": False,
            "verdict": "tests failed",
            "result": "",
            "success": True,
            "trajectory": long_trajectory,
            "cost": 0.5,
            "tokens": 100,
            "steps": 20,
        },
        {
            "task_id": "t2",
            "passed": True,
            "verdict": "resolved",
            "result": "",
            "success": True,
            "trajectory": [{"role": "result", "text": "done"}],
            "cost": 0.2,
            "tokens": 50,
            "steps": 5,
        },
    ]
    (tmp_path / "rollouts.json").write_text(json.dumps(rollouts), encoding="utf-8")
    log = tmp_path / "env_calls.jsonl"
    evals = _evals(
        tmp_path,
        [
            {"id": "t1", "prompt": "fix the bug", "split": "train"},
            {"id": "t2", "prompt": "add a flag", "split": "train"},
            {"id": "easy", "prompt": "select 1", "split": "select"},
            {"id": "hard", "prompt": "select 2", "split": "select"},
        ],
        train_rollouts="rollouts.json",
        env={"class": f"{__name__}:MarkerEnv", "log": str(log), "marker": "Verify"},
    )
    patch1 = {
        "op": "add",
        "anchor": "Be careful.",
        "text": " Verify with the checker.",
        "rationale": "",
    }
    patch2 = {"op": "add", "anchor": "", "text": "Also keep notes.", "rationale": ""}
    script = [
        _patches_body(patch1),
        _patches_body(),
        _patches_body(patch1),
        _patches_body(patch2),
        _patches_body(),
        _patches_body(patch2),
    ]
    with serve(script) as (url, requests):
        report = run_optimization(
            _config(
                tmp_path,
                module,
                evals,
                url,
                constant="SYSTEM_PROMPT",
                epochs=2,
                trajectory_total_chars=1000,
            )
        )
    assert len(requests) == 6
    failure_prompt = str(requests[0]["messages"][-1]["content"])
    assert "## Task t1" in failure_prompt and "tests failed" in failure_prompt
    assert "characters omitted" in failure_prompt and len(failure_prompt) < 6000
    assert "## Task t2" in str(requests[1]["messages"][-1]["content"])
    assert "- hard: 'select 2' -> text lacks 'Verify'" in str(
        requests[2]["messages"][-1]["content"]
    )
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls == [
        ["select_best", ["easy", "hard"]],
        ["select_candidate", ["easy", "hard"]],
        ["select_best", ["easy", "hard"]],
        ["select_candidate", ["easy", "hard"]],
    ]
    # Round 2's failure analyst sees the imported failure with the accepted text.
    round2_failure = str(requests[3]["messages"][-1]["content"])
    assert "## Task t1" in round2_failure and "Verify with the checker." in round2_failure
    rounds = report["rounds"]
    assert rounds[0]["accepted"] is True and rounds[0]["best_score"] == 0.5
    assert rounds[0]["candidate_score"] == 1.0
    assert rounds[1]["accepted"] is False and rounds[1]["reason"] == "rejected: 1.000 <= 1.000"
    assert report["kind"] == "constant" and report["best_score"] == 1.0
    proposal = Path(report["proposal"])
    assert proposal == tmp_path / "harness_core.py.proposed"
    ns: dict[str, Any] = {}
    exec(compile(proposal.read_text(encoding="utf-8"), str(proposal), "exec"), ns)  # noqa: S102
    assert (
        ns["Harness"]().system_prompt() == "Work in /app.\nBe careful. Verify with the checker.\n"
    )
    assert module.read_text(encoding="utf-8") == _TEMPLATE_MODULE
    log_text = (tmp_path / "out" / "log.txt").read_text(encoding="utf-8")
    assert log_text.count("using 2 imported trajectories for the training batch") == 2
    assert "rolling out the current best on the training batch" not in log_text
    assert not (tmp_path / "out" / "round_01" / "train").exists()
    assert not (tmp_path / "out" / "round_02" / "train").exists()


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
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 'p'

    def settings(self, settings):
        return settings | {'tool_profile': 'bogus'}
""",
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


def test_run_rollout_applies_eval_defaults_and_prompt_settings(tmp_path: Path) -> None:
    """Eval-set ``rollout`` defaults reach ``SorcarAgent.run``; prompt settings shape the prompt."""
    fixed = tmp_path / "fixed_sea.py"
    fixed.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 'p'

    def prompt(self, task):
        return 'fixed prompt suffix'

    def settings(self, settings):
        return settings | {'tool_profile': 'bash'}
""",
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


def test_run_rollout_sub_task_inherits_the_prompt_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``run_agent`` sub-task of a rollout gets the eval set's ``prompt_suffix``.

    ``run_rollout`` appends the defaults' ``prompt_suffix`` to the prompt
    (after the SEA's ``prompt(task)`` shaped it) and hands it to
    ``SorcarAgent.run(prompt_suffix=...)``, so the dispatch inherits it
    like a daemon-run task's ``appendToPrompt``; the SEA's own suffix
    belongs to the prompt body and is not inherited.  The dispatch is
    read off a local daemon stand-in reached through ``KISS_SORCAR_LOCAL``.
    """
    sea = tmp_path / "suffix_sea.py"
    sea.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 'p'

    def prompt(self, task):
        return task + '\\n\\nROLLOUT-SUFFIX'
""",
        encoding="utf-8",
    )
    daemon = RecordingDaemon(text="child ok", tokens=1, steps=1, chat_id="c")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    task = EvalTask(id="t", prompt="parent task", expect=["done"])
    bodies = [
        tool_call_body("run_agent", {"task": "child task", "agent": "", "timeout": "30"}, 500),
        finish_body("<p>done</p>", 600),
    ]
    try:
        with serve(bodies) as (url, requests):
            cfg = _config(tmp_path, sea, tmp_path / "unused.json", url)
            rollout = run_rollout(
                SeaTarget(sea), task, cfg, tmp_path / "w1",
                {"max_steps": 7, "prompt_suffix": "\n\nEVAL-SUFFIX"},
            )
    finally:
        daemon.close()
    assert rollout.passed, rollout
    user = [m for m in requests[0]["messages"] if m["role"] == "user"]
    assert str(user[0]["content"]).rstrip().endswith("ROLLOUT-SUFFIX\n\nEVAL-SUFFIX")
    (call,) = daemon.run_commands
    assert call["prompt"] == "child task"
    assert call["appendToPrompt"] == "\n\nEVAL-SUFFIX"
    assert "child ok" in str(requests[1]["messages"][-1]["content"])


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
    skillopt = skillopt_sea.SkilloptSea()
    assert skillopt.system_prompt("DEFAULT PROMPT").startswith("You are SkillOpt")
    assert [t.__name__ for t in skillopt.tools([])] == ["optimize", "status"]
    assert skillopt.tools([greet_stub])[0] is greet_stub
    declared = {"kind": "worker", "tool_profile": "shell"}
    assert skillopt.settings({}) == declared
    assert skillopt.settings({"model": "m", "kind": "session"}) == {"model": "m", **declared}
    # ``worker`` turns worktree, auto-commit, classifier, fan-out, browser
    # and memory off; ``system_prompt`` is a method, not a settings key.
    effective = {
        "kind": "worker",
        "tool_profile": "shell",
        "use_worktree": False,
        "auto_commit": False,
        "auto_classify": False,
        "allow_fan_out": False,
        "use_web_tools": False,
        "use_memory": False,
    }
    assert resolve_settings(declared) == effective
    assert skillopt.system_prompt("") == skillopt_sea.SYSTEM_PROMPT
    # The class is the contract: the module keeps no getter of the old shape.
    assert_no_removed_getters(skillopt_sea)
    assert sea_commands.get_command("skillopt") == _SKILLOPT_SEA
    assert sea_commands.slash_command_task("/skillopt x.py evals.json") == (
        "x.py evals.json", _SKILLOPT_SEA
    )
    assert sea_commands.sea_settings(_SKILLOPT_SEA) == effective
    cmd: dict[str, Any] = {"agentPath": str(_SKILLOPT_SEA), "prompt": "x.py evals.json"}
    assert apply_agent_overrides(cmd) == {
        "systemPromptHook", "toolsHook", "toolProfile", "useWorktree", "autoCommit",
        "classifyTasks", "isParallel", "useWebTools", "useMemory",
    }
    assert cmd["autoCommit"] is False and cmd["useWorktree"] is False
    assert cmd["toolProfile"] == "shell" and cmd["prompt"] == "x.py evals.json"
    assert "appendBasicTools" not in cmd and "systemPrompt" not in cmd and "tools" not in cmd
    assert cmd["systemPromptHook"]("DEFAULT PROMPT") == skillopt_sea.SYSTEM_PROMPT
    assert [t.__name__ for t in cmd["toolsHook"]([greet_stub])] == [
        "greet_stub", "optimize", "status",
    ]
    assert cmd["_runConfig"]["sea"] == "skillopt" and cmd["_runConfig"]["kind"] == "worker"
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
    """``python -m kiss.agents.seas.skillopt.skillopt_sea`` drives the loop from the terminal."""
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


def test_sea_target_rollout_composes_the_base_class_and_runs_prompt_only_on_the_real_task(
    tmp_path: Path,
) -> None:
    """A rollout evaluates the whole inheritance chain; ``prompt(task)`` sees no placeholder."""
    base = tmp_path / "base_sea.py"
    base.write_text(
        '''
from kiss.agents.seas.base.base_sea import BaseSea

def helper(x: str) -> str:
    """Help.

    Args:
        x: X.

    Returns:
        X.
    """
    return x


class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'tool_profile': 'none'}

    def tools(self, tools):
        return tools + [helper]

    def system_prompt(self, system_prompt):
        return system_prompt + "\\n\\n" + 'BASE RULES'
''',
        encoding="utf-8",
    )
    target = tmp_path / "strict_sea.py"
    target.write_text(
        "import json\n\nfrom kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        "PROMPT = 'strict'\n\n\n"
        f"class Sea(sea_class({str(base)!r})):\n"
        "    def system_prompt(self, system_prompt):\n        return PROMPT\n\n"
        "    def prompt(self, task):\n        return json.loads(task)['text']\n",
        encoding="utf-8",
    )
    kwargs = SeaTarget(target).rollout_kwargs()
    # The trainable constant replaces the prompt; settings and tools are the base's.
    assert kwargs["system_prompt_hook"]("DEFAULT PROMPT") == "strict"
    assert kwargs["tool_profile"] == "none" and kwargs["append_basic_tools"] is False
    assert [t.__name__ for t in kwargs["tools_hook"]([])] == ["helper"]
    assert kwargs["prompt"]('{"text": "real task"}') == "real task"
    # A candidate copy in a scratch directory still finds the absolute base.
    assert SeaTarget(target).validate("tuned") == ""
    # A base next to the file is found from there, but not from a candidate
    # copy in a scratch directory: rejected at validation.
    relative = tmp_path / "relative_sea.py"
    relative.write_text(
        """
from pathlib import Path

from kiss.agents.sorcar.sea_commands import sea_class

PROMPT = 'rel'


class Sea(sea_class('base_sea.py', relative_to=Path(__file__).parent)):
    def system_prompt(self, system_prompt):
        return PROMPT
""",
        encoding="utf-8",
    )
    assert SeaTarget(relative).rollout_kwargs()["system_prompt_hook"]("x") == "rel"
    assert SeaTarget(relative).text() == "rel"
    assert "not an existing Python" in SeaTarget(relative).validate("tuned")
    # Every class of the chain contributes, base first, without ``super()``:
    # the base's text stays in the prompt.
    composed = tmp_path / "composed_sea.py"
    composed.write_text(
        f"from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Sea(sea_class({str(base)!r})):\n"
        "    def system_prompt(self, system_prompt):\n"
        "        return system_prompt + '\\n\\nSTRICT'\n",
        encoding="utf-8",
    )
    composed_kwargs = SeaTarget(composed).rollout_kwargs()
    assert composed_kwargs["system_prompt_hook"]("DEFAULT") == "DEFAULT\n\nBASE RULES\n\nSTRICT"
    with pytest.raises(ValueError, match="string literal"):
        SeaTarget(composed).text()  # not a trainable constant
