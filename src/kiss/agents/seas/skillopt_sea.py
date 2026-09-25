# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SkillOpt agent — optimizes the prompt text of a skill or of a SEA against an eval set.

Implements the outer loop of SkillOpt (arXiv 2605.23904) for two kinds of
*target*:

* :class:`SkillTarget` — a ``SKILL.md`` file.  The whole file is the trainable
  text; a rollout runs the default Sorcar agent with the text appended to its
  system prompt.
* :class:`SeaTarget` — a Sorcar Extension Agent (``*_sea.py``).  The trainable
  text is the string constant returned by the file's ``system_prompt()``
  getter (directly or through a module-level constant).  A candidate is a
  copy of the file with that constant replaced; every other getter is kept,
  so a rollout runs the candidate exactly as ``/name`` would.  A candidate is
  rejected before any rollout when it changes anything but that constant.
* :class:`ConstantTarget` — any Python module plus the name of a module-level
  string constant (``--constant SYSTEM_PROMPT``).  The constant's value is the
  trainable text; it may be a ``str.format`` template, and a candidate must
  keep exactly the original's replacement fields.  This is how a SEA whose
  prompt is built inside a class or a function (for example the HarnessTax
  container SEA, ``benchmarkings/harnesstax/sea_core.py``) is optimized.

One optimization round: roll the current best out on the training batch,
let a failure analyst and a success analyst propose patches over
minibatches of trajectories, let one ranker call merge and rank the patches
(and the previously rejected ones) down to the round's edit budget, apply
them, roll the candidate out on the selection tasks, and accept it only if
its pass rate is strictly higher than the best's.  Accepted text is written
next to the target as ``<target>.proposed``; the target itself is never
modified.

Three ways to run it::

    /skillopt optimize src/kiss/agents/seas/sh_sea.py with
        src/kiss/agents/seas/evals/sh_sea_evals.json for one epoch

    run_agent(agent="skillopt", task="...")           # from another agent

    uv run python -m kiss.agents.seas.skillopt_sea \\
        --target src/kiss/agents/seas/sh_sea.py \\
        --evals src/kiss/agents/seas/evals/sh_sea_evals.json \\
        --out-dir tmp/skillopt/sh_sea --model claude-fable-5-1

Eval set JSON::

    {"rollout": {"web_tools": false, "max_steps": 30},
     "tasks": [{"id": "echo", "prompt": "echo hi", "expect": ["hi"]},
               {"id": "two", "prompt": "seq 1 2", "expect_regex": "1\\\\s*\\\\n\\\\s*2",
                "setup": "touch f", "check": "test -f f", "split": "select"}]}

A task passes when every ``expect`` substring appears in the rollout's
result (HTML tags stripped), ``expect_regex`` matches it, and ``check`` (a
shell command run in the rollout's scratch directory with the result in
``$RESULT``) exits 0.  ``setup`` and ``check`` run under the same shell as
the rollout's ``Bash`` tool (``sh`` on POSIX, Git bash on Windows), so
``touch f`` / ``test -f f`` work everywhere.  A task without any of the
three passes when the rollout finished successfully.  ``split`` is
``train``, ``select`` or absent (both).  Rollouts run in-process with
:class:`SorcarAgent`, so the loop needs no daemon and its spend is folded
into the calling task.

Two optional eval-set entries plug a benchmark in:

* ``"env": {"class": "pkg.module:Class", ...}`` names an :class:`Env`
  subclass (built with the remaining entries as keyword arguments) that
  replaces the in-process rollouts and their grading, e.g. a class that
  runs each task in its own Docker container and grades with the
  benchmark's official harness.
* ``"train_rollouts": "rollouts.json"`` (relative to the eval set) imports
  trajectories of the target's current text — a ``rollouts.json`` as the
  optimizer itself writes — so the analysts can mine an existing
  experiment's failures instead of re-rolling the training batch.  A task
  with an imported trajectory is never rolled out (the env may not even be
  able to run it): the trajectory stands in for it in every round, so give
  such tasks ``split: train`` and keep the selection tasks runnable.

Candidate SEAs are written to scratch directories, so a SEA target must be
self-contained: one that imports sibling modules or reads files next to
``__file__`` fails the pre-rollout gate ("cannot be loaded as a SEA").
"""

from __future__ import annotations

import argparse
import ast
import difflib
import html
import importlib
import json
import logging
import math
import os
import re
import shutil
import string
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.useful_tools import _popen_kwargs
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import flatten_content_to_text
from kiss.core.utils import substitute_prompt_args

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


class Target(ABC):
    """A file carrying a trainable text, and how to roll a candidate text out."""

    kind = ""

    def __init__(self, path: Path) -> None:
        """Bind the target to *path* (resolved; must exist)."""
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError(f"target file not found: {self.path}")

    @abstractmethod
    def text(self) -> str:
        """Return the trainable text currently stored in the file."""

    @abstractmethod
    def render(self, text: str) -> str:
        """Return the full file content that carries *text*."""

    @abstractmethod
    def candidate_name(self) -> str:
        """Return the file name a candidate copy is written under."""

    @abstractmethod
    def rollout_kwargs(self) -> dict[str, Any]:
        """Return the ``SorcarAgent.run`` keyword arguments that make a rollout use this file."""

    def validate(self, text: str) -> str:
        """Return why *text* cannot be a candidate, or ``""`` when it can."""
        return ""

    def materialize(self, text: str, dest_dir: Path) -> Target:
        """Write a candidate copy carrying *text* into *dest_dir* and return it as a target."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / self.candidate_name()
        dest.write_text(self.render(text), encoding="utf-8")
        return type(self)(dest)

    def proposal_path(self) -> Path:
        """Return where accepted text is written (``<target>.proposed``)."""
        return self.path.with_name(self.path.name + ".proposed")

    def write_proposal(self, text: str) -> Path:
        """Write the full file carrying *text* to :meth:`proposal_path` and return it."""
        path = self.proposal_path()
        path.write_text(self.render(text), encoding="utf-8")
        return path


class SkillTarget(Target):
    """A ``SKILL.md`` whose whole content is the trainable text."""

    kind = "skill"

    def text(self) -> str:
        """Return the file content."""
        return self.path.read_text(encoding="utf-8")

    def render(self, text: str) -> str:
        """The file is the text."""
        return text

    def candidate_name(self) -> str:
        """Candidates keep the skill's file name."""
        return self.path.name

    def rollout_kwargs(self) -> dict[str, Any]:
        """Append the skill text to the default agent's system prompt."""
        return {"system_prompt": self.text()}


def _assigned_value(node: ast.stmt, name: str) -> ast.expr | None:
    """Return the value *node* assigns to *name* (``X = ...`` or ``X: str = ...``) or ``None``."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target: ast.expr = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target = node.target
    else:
        return None
    return node.value if isinstance(target, ast.Name) and target.id == name else None


def _prompt_constant(tree: ast.Module) -> ast.Constant:
    """Return the string-constant node that ``system_prompt()`` returns in *tree*."""
    func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "system_prompt"),
        None,
    )
    if func is None:
        raise ValueError("no top-level system_prompt() function")
    last = func.body[-1] if func.body else None
    if not isinstance(last, ast.Return) or last.value is None:
        raise ValueError("system_prompt() must end with a return statement")
    value: ast.expr = last.value
    if isinstance(value, ast.Name):
        assigned = [_assigned_value(n, value.id) for n in tree.body]
        values = [v for v in assigned if v is not None]
        if not values:
            raise ValueError(
                f"system_prompt() returns {value.id}, which is not a module-level assignment"
            )
        value = values[-1]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value
    raise ValueError(
        "system_prompt() must return a string literal or a module-level string constant"
    )


def _string_literal(text: str) -> str:
    """Return a Python literal evaluating to *text* (triple-quoted when multi-line)."""
    body = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    literal = '"""\\\n' + body + '"""' if "\n" in text else repr(text)
    if ast.literal_eval(literal) != text:
        literal = repr(text)
    return literal


def _splice(source: str, node: ast.Constant, text: str) -> str:
    """Return *source* with the literal at *node* replaced by a literal for *text*."""
    lines = source.splitlines(keepends=True)
    start = sum(len(line.encode()) for line in lines[: node.lineno - 1]) + node.col_offset
    end_lineno = node.end_lineno or node.lineno
    end = sum(len(line.encode()) for line in lines[: end_lineno - 1]) + (node.end_col_offset or 0)
    raw = source.encode()
    return (raw[:start] + _string_literal(text).encode() + raw[end:]).decode()


def _constant_node(tree: ast.Module, name: str) -> ast.Constant:
    """Return the string-literal node of the last module-level assignment to *name* in *tree*."""
    values = [v for v in (_assigned_value(n, name) for n in tree.body) if v is not None]
    if not values:
        raise ValueError(f"no module-level assignment to {name}")
    value = values[-1]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value
    raise ValueError(f"{name} must be assigned a string literal")


def _fingerprint(source: str, find: Callable[[ast.Module], ast.Constant]) -> str:
    """Return the AST dump of *source* with the constant that *find* locates blanked."""
    tree = ast.parse(source)
    find(tree).value = ""
    return ast.dump(tree)


def _format_fields(text: str) -> set[str]:
    """Return the ``str.format`` field names of *text* (``ValueError`` on unbalanced braces)."""
    return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}


def _execute_sea(path: Path) -> dict[str, Any]:
    """Execute the SEA file at *path* and return its namespace."""
    from kiss.server.tools_file import execute_python_file

    return execute_python_file(str(path), ValueError, "SEA")


class SeaTarget(Target):
    """A SEA whose trainable text is the constant returned by ``system_prompt()``."""

    kind = "sea"

    def _source(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def text(self) -> str:
        """Return the prompt constant's value."""
        return str(_prompt_constant(ast.parse(self._source())).value)

    def render(self, text: str) -> str:
        """Return the file with the prompt constant replaced by *text*."""
        source = self._source()
        return _splice(source, _prompt_constant(ast.parse(source)), text)

    def candidate_name(self) -> str:
        """``foo_sea.py`` -> ``foo_candidate.py`` (never a ``*_sea.py`` name)."""
        stem = self.path.stem.removesuffix("_sea")
        return f"{stem}_candidate.py"

    def validate(self, text: str) -> str:
        """Reject text whose candidate does not compile, changes code, or is not returned as is."""
        try:
            candidate = self.render(text)
            compile(candidate, str(self.path), "exec")
            if _fingerprint(candidate, _prompt_constant) != _fingerprint(
                self._source(), _prompt_constant
            ):
                return "candidate changes code outside the system_prompt() constant"
        except (SyntaxError, ValueError) as exc:
            return f"candidate does not compile: {exc}"
        try:
            with tempfile.TemporaryDirectory(prefix="skillopt-") as tmp:
                path = Path(tmp) / self.candidate_name()
                path.write_text(candidate, encoding="utf-8")
                returned = _execute_sea(path)["system_prompt"]()
        except Exception as exc:  # noqa: BLE001 - any failure is a gate reason
            return f"candidate cannot be loaded as a SEA: {exc}"
        if returned != text:
            return "candidate's system_prompt() does not return the proposed text"
        return ""

    def rollout_kwargs(self) -> dict[str, Any]:
        """Map the file's getters onto ``SorcarAgent.run`` arguments."""
        ns = _execute_sea(self.path)
        kwargs: dict[str, Any] = {
            "base_system_prompt": str(_call_getter(ns, "system_prompt") or "")
        }
        if not kwargs["base_system_prompt"]:
            raise ValueError(f"{self.path.name}: system_prompt() returned nothing")
        for getter, key in (
            ("prompt", "prompt"),
            ("append_to_prompt", "append_to_prompt"),
            ("tool_profile", "tool_profile"),
            ("is_parallel", "is_parallel"),
            ("use_web_tools", "web_tools"),
            ("use_memory", "use_memory"),
            ("append_to_system_prompt", "system_prompt"),
            ("if_append_basic_tools", "append_basic_tools"),
            ("docker_image", "docker_image"),
            ("model", "model_name"),
            ("model_config", "model_config"),
            ("llm_call_hook", "llm_call_hook"),
            ("tool_call_hook", "tool_call_hook"),
        ):
            value = _call_getter(ns, getter)
            if value is not None:
                kwargs[key] = value
        tools = _call_getter(ns, "tools")
        if isinstance(tools, list):
            kwargs["tools"] = tools
        elif isinstance(tools, (str, os.PathLike)) and os.fspath(tools):
            from kiss.server.tools_file import load_tools_file

            kwargs["tools"] = load_tools_file(os.fspath(tools))
        return kwargs


def _call_getter(ns: dict[str, Any], name: str) -> Any:
    """Call the SEA getter *name* in namespace *ns*; ``None`` when it is not defined."""
    getter = ns.get(name)
    return getter() if callable(getter) else None


class ConstantTarget(Target):
    """A Python module whose trainable text is the module-level string constant *constant*.

    The constant may be a ``str.format`` template (``"... {workdir} ..."``); a
    candidate must keep exactly the original's replacement fields so the
    module's ``.format(...)`` call keeps working.  Rolled out in-process, the
    text is used as the agent's system prompt; an eval set whose ``env`` runs
    the module's own harness (a container benchmark, say) formats it itself.
    """

    kind = "constant"

    def __init__(self, path: Path, constant: str) -> None:
        """Bind the target to the constant *constant* of the module at *path*."""
        super().__init__(path)
        if not constant.isidentifier():
            raise ValueError(f"constant name must be an identifier, not {constant!r}")
        self.constant = constant

    def _source(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def _find(self, tree: ast.Module) -> ast.Constant:
        return _constant_node(tree, self.constant)

    def text(self) -> str:
        """Return the constant's value."""
        return str(self._find(ast.parse(self._source())).value)

    def render(self, text: str) -> str:
        """Return the module with the constant replaced by *text*."""
        source = self._source()
        return _splice(source, self._find(ast.parse(source)), text)

    def candidate_name(self) -> str:
        """``foo.py`` -> ``foo_candidate.py``."""
        return f"{self.path.stem.removesuffix('_sea')}_candidate.py"

    def materialize(self, text: str, dest_dir: Path) -> Target:
        """Write a candidate module carrying *text* into *dest_dir* and return it as a target."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / self.candidate_name()
        dest.write_text(self.render(text), encoding="utf-8")
        return ConstantTarget(dest, self.constant)

    def validate(self, text: str) -> str:
        """Reject text whose module does not compile, changes code, or breaks the template."""
        try:
            candidate = self.render(text)
            compile(candidate, str(self.path), "exec")
            if _fingerprint(candidate, self._find) != _fingerprint(self._source(), self._find):
                return f"candidate changes code outside the {self.constant} constant"
        except (SyntaxError, ValueError) as exc:
            return f"candidate does not compile: {exc}"
        try:
            fields = _format_fields(text)
        except ValueError as exc:
            return f"candidate is not a valid format template: {exc}"
        original = _format_fields(self.text())
        if fields != original:
            return (
                f"candidate changes the template's replacement fields: "
                f"{sorted(fields)} != {sorted(original)}"
            )
        try:
            text.format(**dict.fromkeys(fields, ""))
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            return f"candidate is not a valid format template: {exc}"
        return ""

    def rollout_kwargs(self) -> dict[str, Any]:
        """Use the constant as the agent's system prompt."""
        return {"base_system_prompt": self.text()}


def load_target(path: str | Path, constant: str = "") -> Target:
    """Return the target for *path*: a :class:`ConstantTarget` when *constant* names a
    module-level string, else the :class:`SeaTarget` (``.py``) or :class:`SkillTarget` (``.md``)."""
    p = Path(path)
    if constant:
        return ConstantTarget(p, constant)
    if p.suffix == ".py":
        return SeaTarget(p)
    if p.suffix == ".md":
        return SkillTarget(p)
    raise ValueError(f"unsupported target {p}: expected a *.py SEA or a SKILL.md")


# --------------------------------------------------------------------------
# Eval set and verification
# --------------------------------------------------------------------------


@dataclass
class EvalTask:
    """One eval task: a prompt for the agent and how to grade its result."""

    id: str
    prompt: str
    expect: list[str] = field(default_factory=list)
    expect_regex: str = ""
    check: str = ""
    setup: str = ""
    split: str = ""


@dataclass
class EvalSet:
    """A loaded eval set.

    ``rollout`` holds the ``SorcarAgent.run`` defaults of in-process rollouts;
    ``env`` is ``{"class": "module:Class", ...kwargs}`` naming an :class:`Env`
    that replaces them (``None`` for in-process rollouts); ``train_rollouts``
    are imported trajectories of the target's original text; a task with
    one is never rolled out, the trajectory is its training rollout.
    """

    tasks: list[EvalTask]
    rollout: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] | None = None
    train_rollouts: list[Rollout] = field(default_factory=list)


def load_evals(path: Path) -> EvalSet:
    """Load the eval set at *path* (a ``{"tasks": [...], ...}`` object or a bare task list)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {"tasks": data}
    raw_tasks = data["tasks"]
    tasks: list[EvalTask] = []
    for i, item in enumerate(raw_tasks):
        expect = item.get("expect", [])
        if isinstance(expect, str):
            expect = [expect]
        tasks.append(
            EvalTask(
                id=str(item.get("id") or f"task{i + 1}"),
                prompt=str(item["prompt"]),
                expect=[str(e) for e in expect],
                expect_regex=str(item.get("expect_regex", "")),
                check=str(item.get("check", "")),
                setup=str(item.get("setup", "")),
                split=str(item.get("split", "")),
            )
        )
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("eval task ids must be unique")
    for task in tasks:
        if task.split not in ("", "train", "select"):
            raise ValueError(f"task {task.id}: split must be train, select or absent")
    env = data.get("env")
    if env is not None and not (isinstance(env, dict) and ":" in str(env.get("class", ""))):
        raise ValueError('env must be an object with a "class": "module:Class" entry')
    imported: list[Rollout] = []
    if data.get("train_rollouts"):
        rollouts_path = Path(path).parent / str(data["train_rollouts"])
        imported = load_rollouts(rollouts_path)
        train_ids = {t.id for t in tasks if t.split == "train"}
        unknown = sorted({r.task_id for r in imported} - train_ids)
        if unknown:
            # An imported trajectory never counts as a selection rollout: it
            # describes the original text, not the candidate under test.
            raise ValueError(
                f"train_rollouts must name tasks with split 'train'; not so: {unknown[:5]}"
            )
    return EvalSet(tasks, dict(data.get("rollout", {})), env, imported)


def load_rollouts(path: Path) -> list[Rollout]:
    """Load a ``rollouts.json`` (a list of :class:`Rollout` dicts, as the optimizer writes them)."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Rollout(**item) for item in items]


def plain_text(result: str) -> str:
    """Strip HTML tags and entities from a rollout result."""
    return html.unescape(re.sub(r"<[^>]+>", "", result))


def verify(task: EvalTask, result: str, success: bool, work_dir: Path) -> tuple[bool, str]:
    """Grade *result* for *task*; return ``(passed, verdict)``."""
    text = plain_text(result)
    missing = [e for e in task.expect if e not in text]
    if missing:
        return False, f"missing expected text: {missing!r}"
    if task.expect_regex and not re.search(task.expect_regex, text):
        return False, f"regex {task.expect_regex!r} did not match"
    if task.check:
        env = {**os.environ, "RESULT": text, "SUCCESS": "1" if success else "0"}
        proc = subprocess.run(
            **_popen_kwargs(task.check),
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return (
                False,
                f"check exited {proc.returncode}: {(proc.stdout + proc.stderr).strip()[:300]}",
            )
    if not (task.expect or task.expect_regex or task.check) and not success:
        return False, "rollout did not finish successfully"
    return True, "pass"


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------


@dataclass
class Rollout:
    """The outcome of running one candidate on one task."""

    task_id: str
    passed: bool
    verdict: str
    result: str
    success: bool
    trajectory: list[dict[str, str]]
    cost: float
    tokens: int
    steps: int
    error: str = ""


@dataclass
class OptimizeConfig:
    """Everything one optimization run needs."""

    target: Path
    evals: Path
    out_dir: Path
    model: str
    optimizer_model: str = ""
    model_config: dict[str, Any] | None = None
    epochs: int = 1
    batch_size: int = 40
    edit_budget: int = 4
    edit_floor: int = 2
    minibatch: int = 8
    max_workers: int = 4
    rollout_budget: float = 1.0
    max_steps: int = 30
    max_cost: float = 20.0
    fresh: bool = False
    trajectory_chars: int = 3000
    trajectory_total_chars: int = 20000
    constant: str = ""


def _parse_result(raw: str) -> tuple[bool, str]:
    """Split an agent's YAML result into ``(success, summary)``; raw text when not YAML."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict) and "summary" in data:
        return bool(data.get("success", True)), str(data.get("summary") or "")
    return True, raw


def _message_summary(message: Any, limit: int) -> dict[str, str]:
    """Reduce one provider message to ``{"role", "text"}`` for the analysts."""
    if not isinstance(message, dict):
        return {"role": "?", "text": str(message)[:limit]}
    text = flatten_content_to_text(message.get("content"))
    for call in message.get("tool_calls") or []:
        fn = call.get("function", call) if isinstance(call, dict) else {}
        text += f"\n[call {fn.get('name', '?')}] {fn.get('arguments', '')}"
    return {"role": str(message.get("role", "?")), "text": text[:limit]}


class _TrajectoryRecorder:
    """Rollout hooks that keep a compact transcript: prompts, tool calls and tool results.

    ``KISSAgent`` hands ``llm_call_hook`` only the messages added since the
    previous call (the prompt, then tool results), never the assistant turn,
    so the assistant's decisions are captured through ``tool_call_hook``.
    The target's own hooks, if any, still run after the recording.
    """

    def __init__(self, limit: int, inner_llm_hook: Any, inner_tool_hook: Any) -> None:
        self.trajectory: list[dict[str, str]] = []
        self.limit = limit
        self.inner_llm_hook = inner_llm_hook
        self.inner_tool_hook = inner_tool_hook

    def __call__(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Record *messages* (the ``llm_call_hook``), then defer to the target's hook."""
        for message in messages:
            if not (isinstance(message, dict) and message.get("role") == "system"):
                self.trajectory.append(_message_summary(message, self.limit))
        return self.inner_llm_hook(messages) if self.inner_llm_hook is not None else messages

    def tool_call(self, name: str, args: dict[str, Any]) -> str:
        """Record a tool call (the ``tool_call_hook``), then defer to the target's hook."""
        rendered = json.dumps(args, ensure_ascii=False, default=str)
        self.trajectory.append(
            {"role": "assistant", "text": f"[call {name}] {rendered}"[: self.limit]}
        )
        return self.inner_tool_hook(name, args) if self.inner_tool_hook is not None else "OK"


def run_rollout(
    target: Target,
    task: EvalTask,
    cfg: OptimizeConfig,
    work_dir: Path,
    defaults: dict[str, Any] | None = None,
) -> Rollout:
    """Run *task* against *target* in-process inside *work_dir* and grade it.

    *defaults* is the eval set's ``rollout`` object (``SorcarAgent.run``
    keyword arguments plus ``max_steps``); the target's own getters override
    it.  A target defining ``prompt()`` replaces the task's prompt with its
    fixed one (such a SEA varies only through ``setup``); ``append_to_prompt()``
    is appended to the task's prompt.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _live_agent_usage

    work_dir.mkdir(parents=True, exist_ok=True)
    if task.setup:
        subprocess.run(
            **_popen_kwargs(task.setup),
            cwd=work_dir,
            check=True,
            capture_output=True,
            timeout=120,
        )
    kwargs: dict[str, Any] = {"web_tools": False, "is_parallel": False, "use_memory": False}
    kwargs.update(defaults or {})
    kwargs.update(target.rollout_kwargs())
    prompt = str(kwargs.pop("prompt", "") or task.prompt) + str(kwargs.pop("append_to_prompt", ""))
    record = _TrajectoryRecorder(
        cfg.trajectory_chars, kwargs.pop("llm_call_hook", None), kwargs.pop("tool_call_hook", None)
    )
    trajectory = record.trajectory
    agent = SorcarAgent(f"skillopt rollout {task.id}")
    error = ""
    success, summary = False, ""
    try:
        raw = agent.run(
            model_name=kwargs.pop("model_name", cfg.model),
            prompt_template=prompt,
            work_dir=str(work_dir),
            max_steps=int(kwargs.pop("max_steps", cfg.max_steps)),
            max_budget=cfg.rollout_budget,
            model_config=kwargs.pop("model_config", cfg.model_config),
            verbose=False,
            llm_call_hook=record,
            tool_call_hook=record.tool_call,
            **kwargs,
        )
        success, summary = _parse_result(raw)
    except Exception as exc:  # noqa: BLE001 - a crashed rollout is a failed rollout
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("rollout %s crashed: %s", task.id, error)
    cost, tokens, steps = _live_agent_usage(agent)
    trajectory.append({"role": "result", "text": (error or summary)[: cfg.trajectory_chars]})
    if error:
        passed, verdict = False, error
    else:
        passed, verdict = verify(task, summary, success, work_dir)
    return Rollout(
        task.id, passed, verdict, summary, success, trajectory, cost, tokens, steps, error
    )


class Env(ABC):
    """Where a candidate is rolled out on eval tasks and how the rollouts are graded."""

    @abstractmethod
    def rollouts(
        self, target: Target, tasks: list[EvalTask], cfg: OptimizeConfig, work_root: Path
    ) -> list[Rollout]:
        """Roll *target* out on *tasks*; return one :class:`Rollout` per task, in task order.

        Per-task artifacts (scratch directories, logs) go under *work_root*.
        """


class InProcessEnv(Env):
    """Rollouts with :class:`SorcarAgent` in this process, graded by the tasks' own rules."""

    def __init__(self, defaults: dict[str, Any] | None = None) -> None:
        """*defaults* is the eval set's ``rollout`` object."""
        self.defaults = defaults or {}

    def rollouts(
        self, target: Target, tasks: list[EvalTask], cfg: OptimizeConfig, work_root: Path
    ) -> list[Rollout]:
        """Run :func:`run_rollout` for every task, ``cfg.max_workers`` at a time."""
        with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
            return list(
                pool.map(
                    run_rollout,
                    repeat(target),
                    tasks,
                    repeat(cfg),
                    [work_root / t.id for t in tasks],
                    repeat(self.defaults),
                )
            )


def make_env(spec: dict[str, Any] | None, defaults: dict[str, Any]) -> Env:
    """Build the eval set's :class:`Env`: ``spec["class"]`` (``module:Class``) with the
    remaining entries as keyword arguments, or :class:`InProcessEnv` when *spec* is ``None``."""
    if not spec:
        return InProcessEnv(defaults)
    kwargs = dict(spec)
    module_name, _, class_name = str(kwargs.pop("class")).partition(":")
    env = getattr(importlib.import_module(module_name), class_name)(**kwargs)
    if not isinstance(env, Env):
        raise ValueError(f"{module_name}:{class_name} is not an Env")
    return env


# --------------------------------------------------------------------------
# Patches and analysts
# --------------------------------------------------------------------------


@dataclass
class Patch:
    """One edit: ``add`` after *anchor* (at the end if empty), ``replace`` it, or ``delete`` it."""

    op: str
    anchor: str
    text: str
    rationale: str = ""


def parse_patches(reply: str) -> list[Patch]:
    """Extract the ``{"patches": [...]}`` object from a model reply; ``[]`` when absent."""
    start = reply.find("{")
    while start != -1:
        try:
            data, _ = json.JSONDecoder().raw_decode(reply[start:])
        except json.JSONDecodeError:
            start = reply.find("{", start + 1)
            continue
        if isinstance(data, dict) and isinstance(data.get("patches"), list):
            patches = []
            for item in data["patches"]:
                if not isinstance(item, dict) or item.get("op") not in ("add", "replace", "delete"):
                    continue
                patches.append(
                    Patch(
                        op=str(item["op"]),
                        anchor=str(item.get("anchor", "")),
                        text=str(item.get("text", "")),
                        rationale=str(item.get("rationale", "")),
                    )
                )
            return patches
        start = reply.find("{", start + 1)
    return []


def apply_patch(text: str, patch: Patch) -> str | None:
    """Return *text* with *patch* applied, or ``None`` when its anchor is missing."""
    if patch.op == "add":
        if not patch.anchor:
            return text.rstrip("\n") + "\n" + patch.text.strip("\n") + "\n"
        if patch.anchor not in text:
            return None
        return text.replace(patch.anchor, patch.anchor + patch.text, 1)
    if patch.anchor not in text:
        return None
    return text.replace(patch.anchor, patch.text if patch.op == "replace" else "", 1)


def edit_budget(round_index: int, total_rounds: int, initial: int, floor: int) -> int:
    """Cosine-decay the per-round edit budget from *initial* to *floor*."""
    if total_rounds <= 1 or initial <= floor:
        return initial
    progress = round_index / (total_rounds - 1)
    return math.floor(floor + (initial - floor) * 0.5 * (1 + math.cos(math.pi * progress)) + 0.5)


PATCH_FORMAT = """\
Reply with one JSON object and nothing else:
{"patches": [{"op": "add" | "replace" | "delete",
              "anchor": "<exact substring of the current text; empty + op=add appends at the end>",
              "text": "<new text; empty for delete>",
              "rationale": "<one sentence>"}]}
Anchors must be copied verbatim from the current text. Propose at most {max_patches}
patches, each a self-contained instruction the agent can follow. Do not restate what the
text already says. Prefer short, general rules over task-specific ones. Do not mention
specific eval tasks by their exact command, and do not hard-code expected answers."""

FAILURE_ANALYST_PROMPT = """\
You improve the instructions of an AI agent. Below is the agent's current instruction
text and trajectories of tasks the agent FAILED. Find what the instructions lack or say
wrong that let these failures happen, and propose patches that would make the agent pass
tasks like these without hurting other tasks.

# Current instruction text
<<<
{text}
>>>

# Failed trajectories
{trajectories}

{format}"""

SUCCESS_ANALYST_PROMPT = """\
You improve the instructions of an AI agent. Below is the agent's current instruction
text and trajectories of tasks the agent PASSED. Distill what made them work into
guidance that is missing from the text, and remove or tighten redundant or confusing
wording. Propose nothing when the text is already right for these tasks.

# Current instruction text
<<<
{text}
>>>

# Passed trajectories
{trajectories}

{format}"""

RANKER_PROMPT = """\
You are merging patches proposed by several analysts for an AI agent's instruction text.
Select and merge at most {budget} patches that are most likely to raise the agent's pass
rate on tasks like those in the eval set. Drop duplicates, merge overlapping patches into
one, drop patches that conflict with each other, and drop patches equivalent to ones that
were rejected before (they did not improve the pass rate when tried). Keep anchors exact.

# Current instruction text
<<<
{text}
>>>

# Current pass rate on the selection tasks: {score}

# Eval tasks and the current text's verdict on each
{verdicts}

# Proposed patches
{patches}

# Previously rejected patches
{rejected}

{format}"""


def _clip(text: str, limit: int) -> str:
    """Return *text* when it fits *limit*, else its head and tail around an omission mark."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n  [... {len(text) - limit} characters omitted ...]\n{text[-tail:]}"


def _format_trajectories(
    rollouts: list[Rollout], tasks: dict[str, EvalTask], limit: int = 20000
) -> str:
    """Render *rollouts* for an analyst, each trajectory clipped to *limit* characters."""
    parts = []
    for r in rollouts:
        task = tasks[r.task_id]
        steps = _clip("\n".join(f"  [{m['role']}] {m['text']}" for m in r.trajectory), limit)
        parts.append(
            f"## Task {r.task_id}\nprompt: {task.prompt}\nverdict: {r.verdict}\n"
            f"steps={r.steps} cost=${r.cost:.4f}\n{steps}"
        )
    return "\n\n".join(parts)


def _format_patches(patches: list[Patch]) -> str:
    return json.dumps([asdict(p) for p in patches], indent=1) if patches else "(none)"


def _distinct(patches: list[Patch]) -> list[Patch]:
    """Keep the first patch for every ``(op, anchor)`` pair, preserving order."""
    seen: set[tuple[str, str]] = set()
    unique: list[Patch] = []
    for patch in patches:
        if (patch.op, patch.anchor) not in seen:
            seen.add((patch.op, patch.anchor))
            unique.append(patch)
    return unique


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------


def _calling_agent() -> Any:
    """Return the Sorcar agent whose task thread is running this code, or ``None``."""
    try:
        from kiss.server.agent_state import current_agent
    except Exception:  # noqa: BLE001 - not running inside the daemon
        return None
    return current_agent()


def _attribute(parent: Any, cost: float, tokens: int, steps: int) -> None:
    """Fold a rollout's or analyst's spend into *parent* (no-op without a parent)."""
    if parent is None or cost <= 0 and tokens <= 0:
        return
    from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

    _attribute_sub_usage(parent, cost, tokens, steps)


_RESUME_KEYS = ("target", "evals", "task_ids", "original_text")
"""State fields that must match for ``state.json`` to be resumed."""


class Optimizer:
    """Runs SkillOpt rounds for one target and eval set, persisting state in ``out_dir``."""

    def __init__(self, cfg: OptimizeConfig, parent: Any = None) -> None:
        """Load the target, the eval set and any earlier state under ``cfg.out_dir``."""
        self.cfg = cfg
        self.parent = parent
        self.target = load_target(cfg.target, cfg.constant)
        evals = load_evals(cfg.evals)
        self.tasks = evals.tasks
        self.env = make_env(evals.env, evals.rollout)
        self.imported = {r.task_id: r for r in evals.train_rollouts}
        self.tasks_by_id = {t.id: t for t in self.tasks}
        self.train = [t for t in self.tasks if t.split != "select"]
        self.select = [t for t in self.tasks if t.split != "train"]
        if not self.train or not self.select:
            raise ValueError("the eval set needs at least one train task and one selection task")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = cfg.out_dir / "state.json"
        original = self.target.text()
        self.state: dict[str, Any] = {
            "target": str(self.target.path),
            "kind": self.target.kind,
            "evals": str(cfg.evals.resolve()),
            "task_ids": [t.id for t in self.tasks],
            "original_text": original,
            "best_text": original,
            "best_score": None,
            "rounds_done": 0,
            "rejected": [],
            "history": [],
            "total_cost": 0.0,
        }
        if cfg.fresh:
            self._discard_earlier_run()
        elif self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            same_run = all(saved.get(key) == self.state[key] for key in _RESUME_KEYS)
            if same_run:
                self.state = saved
            else:
                self._log("state.json belongs to another target or eval set; starting over")

    def _discard_earlier_run(self) -> None:
        """Remove an earlier run's state, log, round artifacts and proposal from disk."""
        for path in self.cfg.out_dir.iterdir():
            if path.name in ("state.json", "log.txt"):
                path.unlink()
            elif path.is_dir() and path.name.startswith("round_"):
                shutil.rmtree(path)
        self.target.proposal_path().unlink(missing_ok=True)

    # -- helpers ---------------------------------------------------------

    def _log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        logger.info("skillopt: %s", message)
        with (self.cfg.out_dir / "log.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def _spend(self, cost: float, tokens: int, steps: int) -> None:
        self.state["total_cost"] = float(self.state["total_cost"]) + cost
        _attribute(self.parent, cost, tokens, steps)

    def _rollouts(self, target: Target, tasks: list[EvalTask], work_root: Path) -> list[Rollout]:
        """Roll *target* out on *tasks* through the env; results in task order."""
        work_root.mkdir(parents=True, exist_ok=True)
        results = self.env.rollouts(target, tasks, self.cfg, work_root)
        for r in results:
            self._spend(r.cost, r.tokens, r.steps)
            self._log(
                f"  {r.task_id}: {'PASS' if r.passed else 'FAIL'} ({r.verdict}) "
                f"steps={r.steps} ${r.cost:.4f}"
            )
        (work_root / "rollouts.json").write_text(
            json.dumps([asdict(r) for r in results], indent=1),
            encoding="utf-8",
        )
        return results

    def _ask(self, prompt: str, name: str) -> str:
        """One non-agentic call to the optimizer model; ``""`` when the call fails or is refused."""
        agent = KISSAgent(f"skillopt {name}")
        try:
            return agent.run(
                model_name=self.cfg.optimizer_model or self.cfg.model,
                prompt_template=prompt,
                is_agentic=False,
                verbose=False,
                model_config=self.cfg.model_config,
            )
        except Exception as exc:  # noqa: BLE001 - a refused analyst must not end the run
            self._log(f"  {name} call failed: {type(exc).__name__}: {exc}")
            return ""
        finally:
            self._spend(agent.budget_used, agent.total_tokens_used, 1)

    def _analyze(self, rollouts: list[Rollout], template: str, name: str) -> list[Patch]:
        """Run one analyst over minibatches of *rollouts* and collect its patches."""
        patches: list[Patch] = []
        for batch in _chunks(rollouts, self.cfg.minibatch):
            prompt = substitute_prompt_args(
                template,
                {
                    "text": self.state["best_text"],
                    "trajectories": _format_trajectories(
                        batch, self.tasks_by_id, self.cfg.trajectory_total_chars
                    ),
                    "format": substitute_prompt_args(
                        PATCH_FORMAT, {"max_patches": str(self.cfg.edit_budget)}
                    ),
                },
            )
            patches.extend(parse_patches(self._ask(prompt, name)))
        self._log(f"  {name}: {len(patches)} patches from {len(rollouts)} trajectories")
        return patches

    def _rank(
        self, patches: list[Patch], budget: int, score: float, rollouts: list[Rollout]
    ) -> list[Patch]:
        """Merge and rank *patches* down to *budget* with one model call.

        *rollouts* (the current text's rollouts) are summarized as per-task
        verdicts: they ground the request in the actual tasks, without which
        claude-fable-5-1 refuses a bare "rank these instruction patches" prompt.
        """
        verdicts = "\n".join(
            f"- {r.task_id}: {self.tasks_by_id[r.task_id].prompt!r} -> {r.verdict}"
            for r in rollouts
        )
        rejected = [
            {"score_when_tried": r["score"], "patches": r["patches"]}
            for r in self.state["rejected"]
        ]
        prompt = substitute_prompt_args(
            RANKER_PROMPT,
            {
                "budget": str(budget),
                "text": self.state["best_text"],
                "score": f"{score:.3f}",
                "verdicts": verdicts or "(none)",
                "patches": _format_patches(patches),
                "rejected": json.dumps(rejected, indent=1) if rejected else "(none)",
                "format": substitute_prompt_args(PATCH_FORMAT, {"max_patches": str(budget)}),
            },
        )
        reply = self._ask(prompt, "ranker")
        if not reply:
            # The ranker call failed: fall back to the analysts' order, one patch per anchor.
            unique = _distinct(patches)[:budget]
            self._log(f"  ranker unavailable; applying the first {len(unique)} distinct patches")
            return unique
        return parse_patches(reply)[:budget]

    @staticmethod
    def _score(rollouts: list[Rollout]) -> float:
        return sum(r.passed for r in rollouts) / len(rollouts) if rollouts else 0.0

    # -- the loop ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run ``cfg.epochs`` epochs of rounds and return the final report dict."""
        rounds_per_epoch = math.ceil(len(self.train) / max(1, self.cfg.batch_size))
        total = self.cfg.epochs * rounds_per_epoch
        first = int(self.state["rounds_done"])
        self._log(
            f"target={self.target.path} kind={self.target.kind} tasks={len(self.tasks)} "
            f"train={len(self.train)} select={len(self.select)} rounds={total} "
            f"model={self.cfg.model} optimizer={self.cfg.optimizer_model or self.cfg.model}"
        )
        for k in range(first, first + total):
            if float(self.state["total_cost"]) >= self.cfg.max_cost:
                self._log(
                    f"stopping: spent ${self.state['total_cost']:.2f} "
                    f">= max_cost ${self.cfg.max_cost:.2f}"
                )
                break
            offset = (k % rounds_per_epoch) * self.cfg.batch_size
            batch = self.train[offset : offset + self.cfg.batch_size]
            budget = edit_budget(k - first, total, self.cfg.edit_budget, self.cfg.edit_floor)
            self._log(f"round {k + 1}/{first + total}: edit budget {budget}")
            self._round(k, budget, batch)
            self.state["rounds_done"] = k + 1
            self._save()
        self._save()
        return self.report()

    def _round(self, k: int, budget: int, batch: list[EvalTask]) -> None:
        round_dir = self.cfg.out_dir / f"round_{k + 1:02d}"
        summary: dict[str, Any] = {
            "round": k + 1,
            "edit_budget": budget,
            "train_tasks": [t.id for t in batch],
            "best_score": None,
            "candidate_score": None,
            "accepted": False,
            "reason": "",
            "patches": [],
            "cost_before": float(self.state["total_cost"]),
        }
        self.state["history"].append(summary)

        best = self.target.materialize(self.state["best_text"], round_dir / "best")
        # A task with an imported trajectory is offline: the env may not be able
        # to run it at all (a Terminal-Bench trial fed to an SWE-bench env), so
        # its imported trajectory stands in for its training rollout every round.
        train_rollouts = [self.imported[t.id] for t in batch if t.id in self.imported]
        if train_rollouts:
            self._log(f"using {len(train_rollouts)} imported trajectories for the training batch")
        todo = [t for t in batch if t.id not in self.imported]
        if todo:
            self._log("rolling out the current best on the training batch")
            train_rollouts += self._rollouts(best, todo, round_dir / "train")
        by_id = {r.task_id: r for r in train_rollouts}
        missing = [t for t in self.select if t.id not in by_id]
        if missing:
            self._log("rolling out the current best on the remaining selection tasks")
            for r in self._rollouts(best, missing, round_dir / "select_best"):
                by_id[r.task_id] = r
        best_rollouts = [by_id[t.id] for t in self.select]
        best_score = self._score(best_rollouts)
        summary["best_score"] = best_score
        summary["best_results"] = {r.task_id: r.passed for r in best_rollouts}
        self.state["best_score"] = best_score
        self._log(f"best pass rate on selection tasks: {best_score:.3f}")

        failed = [r for r in train_rollouts if not r.passed]
        passed = [r for r in train_rollouts if r.passed]
        patches: list[Patch] = []
        if failed:
            patches += self._analyze(failed, FAILURE_ANALYST_PROMPT, "failure analyst")
        if passed:
            patches += self._analyze(passed, SUCCESS_ANALYST_PROMPT, "success analyst")
        (round_dir / "proposed_patches.json").write_text(_format_patches(patches), encoding="utf-8")
        if not patches:
            summary["reason"] = "analysts proposed no patches"
            self._log(summary["reason"])
            return

        ranked = self._rank(patches, budget, best_score, best_rollouts)
        summary["patches"] = [asdict(p) for p in ranked]
        candidate_text = self.state["best_text"]
        for patch in ranked:
            applied = apply_patch(candidate_text, patch)
            if applied is None:
                self._log(f"  skipped patch with missing anchor: {patch.anchor[:60]!r}")
                continue
            candidate_text = applied
        if candidate_text == self.state["best_text"]:
            summary["reason"] = "ranked patches did not change the text"
            self._log(summary["reason"])
            return
        reason = self.target.validate(candidate_text)
        (round_dir / "candidate.txt").write_text(candidate_text, encoding="utf-8")
        if reason:
            summary["reason"] = f"rejected before rollout: {reason}"
            self.state["rejected"].append(
                {"round": k + 1, "score": None, "patches": summary["patches"]}
            )
            self._log(summary["reason"])
            return

        candidate = self.target.materialize(candidate_text, round_dir / "candidate")
        self._log("rolling out the candidate on the selection tasks")
        cand_rollouts = self._rollouts(candidate, self.select, round_dir / "select_candidate")
        cand_score = self._score(cand_rollouts)
        summary["candidate_score"] = cand_score
        summary["candidate_results"] = {r.task_id: r.passed for r in cand_rollouts}
        if cand_score > best_score:
            summary["accepted"] = True
            summary["reason"] = f"accepted: {cand_score:.3f} > {best_score:.3f}"
            self.state["best_text"] = candidate_text
            self.state["best_score"] = cand_score
            self.state["proposal"] = str(self.target.write_proposal(candidate_text))
        else:
            summary["reason"] = f"rejected: {cand_score:.3f} <= {best_score:.3f}"
            self.state["rejected"].append(
                {"round": k + 1, "score": cand_score, "patches": summary["patches"]}
            )
        self._log(summary["reason"])

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Return the run's report: scores per round, the diff, and the proposal path."""
        original = self.state["original_text"]
        best = self.state["best_text"]
        diff = _diff(original, best)
        return {
            "target": str(self.target.path),
            "kind": self.target.kind,
            "tasks": len(self.tasks),
            "rounds_done": self.state["rounds_done"],
            "best_score": self.state["best_score"],
            "improved": best != original,
            "proposal": self.state.get("proposal", ""),
            "total_cost": round(float(self.state["total_cost"]), 4),
            "rounds": self.state["history"],
            "diff": diff,
            "out_dir": str(self.cfg.out_dir),
        }


def _diff(original: str, best: str) -> str:
    """Unified diff of two texts (line-based, no trailing-newline artifacts)."""
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            best.splitlines(),
            "original",
            "best",
            n=2,
            lineterm="",
        )
    )


def _rate(value: Any) -> str:
    """Format a pass rate, or ``n/a`` when it was never measured."""
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def format_report(report: dict[str, Any]) -> str:
    """Render :meth:`Optimizer.report` as text for the model or the terminal."""
    lines = [
        f"target: {report['target']} ({report['kind']})",
        f"tasks: {report['tasks']}  rounds done: {report['rounds_done']}  "
        f"best pass rate: {_rate(report['best_score'])}  total cost: ${report['total_cost']:.4f}",
        f"state and per-round artifacts: {report['out_dir']}",
    ]
    for r in report["rounds"]:
        lines.append(
            f"round {r['round']}: edit budget {r['edit_budget']}, best {_rate(r['best_score'])}, "
            f"candidate {_rate(r['candidate_score'])}, {r['reason']}"
        )
        for p in r["patches"]:
            lines.append(f"    {p['op']} @ {p['anchor'][:50]!r}: {p['text'][:120]!r}")
    if report["improved"]:
        lines.append(f"proposal written to: {report['proposal']}")
        lines.append("diff original -> best:\n" + report["diff"])
    else:
        lines.append(
            "no candidate beat the original; the target is unchanged and no proposal was written"
        )
    return "\n".join(lines)


def run_optimization(cfg: OptimizeConfig, parent: Any = None) -> dict[str, Any]:
    """Run the optimizer for *cfg* and return its report dict."""
    return Optimizer(cfg, parent).run()


# --------------------------------------------------------------------------
# SEA getters and tools
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are SkillOpt, an optimizer for agent instruction texts. You can optimize three kinds
of target: a skill (a SKILL.md file), a Sorcar Extension Agent (a *_sea.py file whose
system_prompt() getter returns a string constant), or a named module-level string
constant of any Python file (pass `constant`, e.g. SYSTEM_PROMPT, for a SEA that formats
its prompt inside a class). The optimization loop is implemented in your tools; you
drive it and report the outcome.

Procedure:
1. Identify from the user's request: the target file (and constant name, if any), the
   eval set JSON, how many epochs (default 1), the rollout model (default: your own
   model) and where to keep artifacts (default: <work_dir>/tmp/skillopt/<target stem>).
   If the eval set path is missing, ask the user for it; do not write one yourself unless
   asked. An eval set may name its own rollout env (`"env"`) and imported trajectories
   (`"train_rollouts"`) that stand in for their tasks' training rollouts.
2. Call `optimize` once with those arguments. It runs every round (rollouts, analysts,
   ranking, gate) and returns a report. It can take many minutes; that is expected.
3. Report back: pass rate before and after, which patches were accepted or rejected and
   why, the diff, the proposal path, and total cost. Quote the report's numbers; do not
   invent results. Rollout artifacts (trajectories, patches, candidates) are under the
   out_dir if the user wants details.

Rules:
- Never edit the target file. Accepted text goes to <target>.proposed for the user to adopt.
- Do not run the eval tasks or the target yourself with Bash; `optimize` does the rollouts.
- Use `status` to summarize an earlier run's state without spending anything.
"""
"""The operating manual for the orchestrating model."""


def optimize(
    target: str,
    evals: str,
    out_dir: str = "",
    epochs: int = 1,
    model: str = "",
    optimizer_model: str = "",
    edit_budget: int = 4,
    max_workers: int = 4,
    rollout_budget: float = 1.0,
    max_cost: float = 20.0,
    fresh: bool = False,
    constant: str = "",
) -> str:
    """Optimize a skill or SEA prompt against an eval set and return the report.

    Args:
        target: Path of the SKILL.md or *_sea.py to optimize (never modified).
        evals: Path of the eval set JSON (see the module docstring for the format).
        constant: Name of a module-level string constant of *target* to optimize
            instead of its ``system_prompt()`` text (for a SEA that builds its
            prompt inside a class or function).
        out_dir: Directory for state, logs, trajectories and candidates
            (default: ``<target dir>/.skillopt/<target stem>``).
        epochs: Passes over the training tasks (one round per batch of 40 tasks).
        model: Model that runs the rollouts (default: the calling agent's model).
        optimizer_model: Model for the analysts and the ranker (default: *model*).
        edit_budget: Maximum patches applied in the first round (decays to 2).
        max_workers: Parallel rollouts.
        rollout_budget: USD cap per rollout.
        max_cost: USD cap for the whole run; rounds stop once reached.
        fresh: Ignore earlier state in *out_dir* and restart from the target's text.

    Returns:
        The textual report of every round, the diff and the proposal path.
    """
    parent = _calling_agent()
    target_path = Path(target).expanduser()
    if not model:
        model = str(getattr(parent, "model_name", "") or "")
    if not model:
        from kiss.core.models.model_info import get_default_model

        model = get_default_model()
    cfg = OptimizeConfig(
        target=target_path,
        evals=Path(evals).expanduser(),
        out_dir=Path(out_dir).expanduser()
        if out_dir
        else target_path.parent / ".skillopt" / target_path.stem,
        model=model,
        optimizer_model=optimizer_model,
        epochs=max(1, epochs),
        edit_budget=max(1, edit_budget),
        max_workers=max(1, max_workers),
        rollout_budget=rollout_budget,
        max_cost=max_cost,
        fresh=fresh,
        constant=constant,
    )
    return format_report(run_optimization(cfg, parent))


def status(out_dir: str) -> str:
    """Summarize the state of an earlier optimization run without running anything.

    Args:
        out_dir: The ``out_dir`` an earlier ``optimize`` call used.

    Returns:
        The same textual report ``optimize`` returns, built from ``state.json``.
    """
    state_path = Path(out_dir).expanduser() / "state.json"
    if not state_path.exists():
        return f"no state.json under {out_dir}"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    diff = _diff(state["original_text"], state["best_text"])
    return format_report(
        {
            "target": state["target"],
            "kind": state["kind"],
            "tasks": "?",
            "rounds_done": state["rounds_done"],
            "best_score": state["best_score"],
            "improved": state["best_text"] != state["original_text"],
            "proposal": state.get("proposal", ""),
            "total_cost": float(state["total_cost"]),
            "rounds": state["history"],
            "diff": diff,
            "out_dir": str(Path(out_dir).expanduser()),
        }
    )


def system_prompt() -> str:
    """Replace the default system prompt with the SkillOpt manual."""
    return SYSTEM_PROMPT


def tools() -> list[Any]:
    """Expose the optimizer to the orchestrating model."""
    return [optimize, status]


def tool_profile() -> str:
    """Bash and Read for inspecting the target, eval set and proposal; no editing tools."""
    return "shell"


def use_worktree() -> bool:
    """Nothing is committed; run directly in the work dir."""
    return False


def auto_commit() -> bool:
    """Never commit: the proposal is for the user to adopt."""
    return False


def classify_tasks() -> bool:
    """The task is always the same kind; skip classification."""
    return False


def is_parallel() -> bool:
    """Rollouts run in the tool's own thread pool."""
    return False


def use_web_tools() -> bool:
    """No browsing."""
    return False


def use_memory() -> bool:
    """No persistent memory."""
    return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the optimizer from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description="SkillOpt for skills and SEAs")
    parser.add_argument("--target", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--optimizer-model", default="")
    parser.add_argument("--model-config", default="", help="JSON dict passed to the model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--edit-budget", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rollout-budget", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-cost", type=float, default=20.0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--constant", default="", help="module-level string constant of --target to optimize"
    )
    parser.add_argument("--trajectory-total-chars", type=int, default=20000)
    args = parser.parse_args(argv)
    cfg = OptimizeConfig(
        target=Path(args.target),
        evals=Path(args.evals),
        out_dir=Path(args.out_dir),
        model=args.model,
        optimizer_model=args.optimizer_model,
        model_config=json.loads(args.model_config) if args.model_config else None,
        epochs=args.epochs,
        edit_budget=args.edit_budget,
        max_workers=args.max_workers,
        rollout_budget=args.rollout_budget,
        max_steps=args.max_steps,
        max_cost=args.max_cost,
        fresh=args.fresh,
        trajectory_total_chars=args.trajectory_total_chars,
        constant=args.constant,
    )
    print(format_report(run_optimization(cfg)))
    return 0


if __name__ == "__main__":
    # ``python -m`` loads this file as ``__main__``; an env module importing
    # ``kiss.agents.seas.skillopt_sea`` would otherwise get a second copy of
    # ``Env`` and fail the ``isinstance`` check in ``make_env``.
    from kiss.agents.seas.skillopt_sea import main as _main

    raise SystemExit(_main())
