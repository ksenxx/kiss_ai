# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20 (B7): the SEA command, lint, docs, apply and skills fixes.

End-to-end through the public entry points:

- ``run_picked_hook(..., sea=<loaded>)`` runs the hook of the SEA it was
  handed, without a second registry lookup that may say "not a model"
  meanwhile (F1);
- an ``async def`` SEA method is a ``SeaError`` and nothing else: the
  coroutine is closed, so no ``RuntimeWarning: coroutine ... was never
  awaited`` follows it, whichever method was made async (F6);
- ``apply_run`` compares the evaluated prompt with the task the run
  was evaluated on, not with the command's ``prompt`` field (F7);
- ``sea_docs.render`` / ``sea docs`` render each table once per run
  through the shared memo (F8);
- ``lint_all(..., fix=True)`` is what ``sea lint --fix`` runs (F4);
- a skill's resource listing keeps its order and skips, without walking
  ``node_modules`` / ``.git`` / ``__pycache__`` (F5).
"""

from __future__ import annotations

import argparse
import gc
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar import sea_commands, sea_docs, sea_lint
from kiss.agents.sorcar.sea_apply import apply_run
from kiss.agents.sorcar.sea_commands import check_sea, evaluate_sea, load_sea, run_picked_hook
from kiss.agents.sorcar.sea_settings import SeaError
from kiss.agents.sorcar.skills import Skill, load_skill_content
from kiss.core.config import kiss_home


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the registry state before and after each test, and the test ``SEAS.md``."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _write_sea(folder: Path, name: str, source: str) -> Path:
    """Write ``<folder>/<name>/<name>_sea.py`` with *source* and return it."""
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


_UNREGISTERED_HOOK_SEA = """\
from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "writes a marker when picked, but registers as no model"

    def on_picked_as_model(self, work_dir):
        Path(work_dir, "picked.txt").write_text("picked")
        return "marker written"
"""


def test_picked_hook_runs_the_sea_it_was_handed(tmp_path: Path) -> None:
    """A loaded SEA's hook runs although the registry maps the name to no model SEA."""
    path = _write_sea(tmp_path / "seas", "unlisted", _UNREGISTERED_HOOK_SEA)
    assert sea_commands.model_sea("unlisted") is None
    run_picked_hook("unlisted", str(tmp_path), sea=load_sea(path))
    assert (tmp_path / "picked.txt").read_text() == "picked"
    # Without a loaded SEA the name is resolved through the registry: no entry, no hook.
    (tmp_path / "picked.txt").unlink()
    run_picked_hook("unlisted", str(tmp_path))
    assert not (tmp_path / "picked.txt").exists()


_ASYNC_SEA = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    async def {method}(self, *args):
        return None
"""


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("settings", "a dict"),
        ("prompt", "a string"),
        ("system_prompt", "a string"),
        ("tools", "a list of tool callables (not a file path)"),
        (
            "tool_call_hook",
            "a Verdict (ALLOW, or refuse(text), from kiss.agents.seas.base.base_sea)",
        ),
        ("llm_call_hook", "a list"),
    ],
)
def test_async_method_is_one_sea_error_without_a_never_awaited_warning(
    tmp_path: Path,
    method: str,
    expected: str,
) -> None:
    """Every checker closes the coroutine an ``async def`` method returns."""
    path = _write_sea(tmp_path / "seas", "coro", _ASYNC_SEA.format(method=method))
    message = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_sea(path, require_description=False)
        except SeaError as exc:
            message = str(exc)
        gc.collect()
    assert message == f"{method}() of SEA {str(path)!r} must return {expected}, got coroutine"
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_apply_run_compares_the_prompt_with_the_evaluated_task() -> None:
    """``SeaRun.task`` is the reference: a command without ``prompt`` gets none written."""
    run = evaluate_sea([BaseSea()], "do it", "t1")
    assert run.task == "do it" and run.prompt == "do it"
    cmd: dict[str, object] = {}
    assert apply_run(cmd, [BaseSea()], run) == set()
    assert "prompt" not in cmd
    rewritten = evaluate_sea([BaseSea()], "task {task_id}", "t1")
    cmd = {"prompt": "task t1"}  # the client already filled the id in; the run did too
    assert apply_run(cmd, [BaseSea()], rewritten) == {"prompt"}
    assert cmd["prompt"] == "task t1"


_PAGE = "intro\n<!-- sea-docs: precedence -->\nold\n<!-- /sea-docs -->\nend\n"


def test_render_shares_one_table_memo_across_pages(tmp_path: Path) -> None:
    """A table rendered for one page is reused, not regenerated, for the next."""
    tables: dict[str, str] = {}
    first = sea_docs.render(_PAGE, tables)
    assert set(tables) == {"precedence"} and tables["precedence"] in first
    tables["precedence"] = "MEMO"
    assert (
        sea_docs.render(_PAGE, tables)
        == "intro\n<!-- sea-docs: precedence -->\nMEMO\n<!-- /sea-docs -->\nend\n"
    )
    assert sea_docs.render(_PAGE) == first  # a page on its own renders afresh
    pages = [tmp_path / "a.md", tmp_path / "b.md"]
    for page in pages:
        page.write_text(_PAGE, encoding="utf-8")
    args = argparse.Namespace(files=[str(p) for p in pages], check=True)
    assert sea_docs.run(args) == 1
    args.check = False
    assert sea_docs.run(args) == 0
    assert [p.read_text(encoding="utf-8") for p in pages] == [first, first]
    args.check = True
    assert sea_docs.run(args) == 0


_RENAMED_KEY_SEA = """\
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "uses a former settings key"

    def settings(self, settings):
        return settings | {"classify_tasks": False}
"""


def test_lint_all_fix_rewrites_then_lints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sea lint --fix`` (``run``) and ``lint_all(fix=True)`` rewrite the renamed key first."""
    path = _write_sea(tmp_path / "seas", "renamed", _RENAMED_KEY_SEA)
    assert [f.code for f in sea_lint.lint_all([path])] == ["renamed-key"]
    code = sea_lint.run(argparse.Namespace(paths=[str(path)], fix=True, registered=False))
    out = capsys.readouterr().out
    assert f"fixed {path}:9:" in out and "renamed-key" not in out
    assert '"auto_classify": False' in path.read_text(encoding="utf-8")
    assert code == 0 and out.endswith("sea lint: 0 finding(s)\n")
    assert sea_lint.lint_all([path], fix=True) == [] and capsys.readouterr().out == ""


def test_skill_resources_keep_their_order_and_skip_bookkeeping_trees(tmp_path: Path) -> None:
    """Files in path order, subdirectories where their name sorts; skipped trees never listed."""
    skill = tmp_path / "skill"
    for rel in (
        "SKILL.md",
        "z.txt",
        "a-x.txt",
        "a/b.txt",
        "a/sub/SKILL.md",
        "a/sub/c.txt",
        "README.md",
        "node_modules/pkg/index.js",
        "lib/__pycache__/x.pyc",
        "lib/x.py",
        ".git/HEAD",
    ):
        (skill / rel).parent.mkdir(parents=True, exist_ok=True)
        (skill / rel).write_text("x", encoding="utf-8")
    (skill / "lib" / ".git").write_text("gitdir: elsewhere", encoding="utf-8")  # a worktree's
    (skill / "link").symlink_to(skill / "a")  # a symlinked directory is not entered
    (skill / "SKILL.md").write_text("---\nname: skill\ndescription: d\n---\nbody\n")
    content = load_skill_content(Skill("skill", "d", str(skill / "SKILL.md"), "user"))
    listed = [line[6:-7] for line in content.splitlines() if line.startswith("<file>")]
    assert listed == ["README.md", "a/b.txt", "a/sub/c.txt", "a-x.txt", "lib/x.py", "z.txt"]
    assert "listing truncated" not in content
