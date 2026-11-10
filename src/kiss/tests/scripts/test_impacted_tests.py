# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``kiss.scripts.impacted_tests`` on a synthetic source tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.scripts import impacted_tests as script


def _tree(tmp_path: Path) -> Path:
    """A ``src/`` with a chain core -> mid -> top, tests at every level, and noise."""
    src = tmp_path / "src"
    for rel, text in {
        "kiss/__init__.py": "",
        "kiss/core/__init__.py": "",
        "kiss/core/core.py": "X = 1\n",
        "kiss/mid.py": "from kiss.core.core import X\n",
        "kiss/top.py": "import kiss.mid\n",
        "kiss/other.py": "import json\n",
        "kiss/tests/__init__.py": "",
        "kiss/tests/test_core.py": "from kiss.core import core\n",
        "kiss/tests/test_mid.py": "from kiss.mid import X\n",
        "kiss/tests/test_top.py": "import kiss.top\n",
        "kiss/tests/test_other.py": "import kiss.other\n",
        "kiss/tests/helper.py": "import kiss.core.core\n",
        "kiss/tests/broken.py": "def (\n",
        "kiss/__pycache__/junk.py": "import kiss.core.core\n",
    }.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return src


def test_tests_within_the_depth_bound_are_selected(tmp_path: Path) -> None:
    src = _tree(tmp_path)
    core = src / "kiss" / "core" / "core.py"
    names = lambda paths: [p.name for p in paths]  # noqa: E731
    assert names(script.impacted_tests([core], src, depth=1)) == ["test_core.py"]
    assert names(script.impacted_tests([core], src, depth=2)) == ["test_core.py", "test_mid.py"]
    assert names(script.impacted_tests([core], src, depth=0)) == [
        "test_core.py", "test_mid.py", "test_top.py",
    ]
    # A changed test is impacted by itself; helpers and non-.py files are not tests.
    assert names(script.impacted_tests([src / "kiss" / "tests" / "test_other.py"], src)) == [
        "test_other.py",
    ]
    assert script.impacted_tests([src / "kiss" / "tests" / "helper.py"], src, depth=1) == []
    assert script.impacted_tests([tmp_path / "README.md", src / "kiss" / "nope.py"], src) == []
    # A relative path resolves against the tree's parent (the repo root).
    assert names(script.impacted_tests([Path("src/kiss/mid.py")], src, depth=1)) == ["test_mid.py"]


def test_relative_imports_and_deleted_modules_reach_their_tests(tmp_path: Path) -> None:
    """``from ._helper import x`` resolves to the package; a deleted module keeps its importers."""
    src = _tree(tmp_path)
    for rel, text in {
        "kiss/pkg/__init__.py": "from ._helper import H\n",
        "kiss/pkg/_helper.py": "H = 1\n",
        "kiss/pkg/sub/__init__.py": "",
        "kiss/pkg/sub/deep.py": "from .. import H\nfrom .._helper import H as H2\n",
        "kiss/tests/test_pkg.py": "import kiss.pkg\n",
        "kiss/tests/test_deep.py": "import kiss.pkg.sub.deep\n",
        "kiss/tests/test_gone.py": "from kiss.gone import thing\n",
    }.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    helper = src / "kiss" / "pkg" / "_helper.py"
    assert script.imported_modules(src / "kiss" / "pkg" / "sub" / "deep.py", src) == {
        "kiss.pkg", "kiss.pkg.H", "kiss.pkg._helper", "kiss.pkg._helper.H",
    }
    names = lambda paths: [p.name for p in paths]  # noqa: E731
    # No test imports _helper directly; its importers (pkg, deep) are one hop away.
    assert script.impacted_tests([helper], src, depth=1) == []
    assert names(script.impacted_tests([helper], src, depth=2)) == ["test_deep.py", "test_pkg.py"]
    # kiss/gone.py no longer exists; the test importing it is still selected.
    assert names(script.impacted_tests([src / "kiss" / "gone.py"], src)) == ["test_gone.py"]


PREFIX = "uv run pytest -q -p no:cacheprovider --timeout=600 "


def _paths(command: str) -> list[str]:
    assert command.startswith(PREFIX), command
    return command.removeprefix(PREFIX).split()


def test_split_commands_balance_by_size_and_never_exceed_the_tests(tmp_path: Path) -> None:
    src = _tree(tmp_path)
    tests = script.impacted_tests([src / "kiss" / "core" / "core.py"], src, depth=0)
    commands = script.split_commands(tests, 2, root=tmp_path)
    assert len(commands) == 2
    listed = sorted(p for c in commands for p in _paths(c))
    assert listed == sorted(t.relative_to(tmp_path).as_posix() for t in tests)
    assert len(script.split_commands(tests, 10, root=tmp_path)) == 3
    assert script.split_commands([], 4, root=tmp_path) == []
    # --timeout 0 leaves the flag out; another value replaces the default.
    assert script.split_commands(tests, 1, root=tmp_path, timeout=0) == [
        "uv run pytest -q -p no:cacheprovider " + " ".join(listed),
    ]
    assert script.split_commands(tests, 1, root=tmp_path, timeout=5)[0].startswith(
        "uv run pytest -q -p no:cacheprovider --timeout=5 src/",
    )


def test_split_commands_balance_by_recorded_durations(tmp_path: Path) -> None:
    """A recorded slow file gets a split of its own; unrecorded files weigh by size."""
    src = _tree(tmp_path)
    tests = script.impacted_tests([src / "kiss" / "core" / "core.py"], src, depth=0)
    # test_top.py is made the largest file yet records as the quickest;
    # test_core.py records as slow.
    (src / "kiss" / "tests" / "test_top.py").write_text("import kiss.top\n" + "# padding\n" * 200)
    by_size = script.split_commands(tests, 2, root=tmp_path)
    assert _paths(by_size[0]) == ["src/kiss/tests/test_top.py"]
    durations = {"src/kiss/tests/test_core.py": 90.0, "src/kiss/tests/test_top.py": 1.0}
    by_time = script.split_commands(tests, 2, root=tmp_path, durations=durations)
    assert _paths(by_time[0]) == ["src/kiss/tests/test_core.py"]
    # test_mid.py has no record: it weighs its size at the median recorded rate
    # (far under 90 s) and shares the split of the quick file.
    assert _paths(by_time[1]) == ["src/kiss/tests/test_mid.py", "src/kiss/tests/test_top.py"]


def test_owned_tests_are_the_changed_tests_and_those_named_for_changed_modules(
    tmp_path: Path, capsys,
) -> None:
    src = _tree(tmp_path)
    tests = src / "kiss" / "tests"
    (tests / "test_core_more.py").write_text("import kiss.core.core\n")
    (tests / "test_corex.py").write_text("import kiss.core.core\n")  # not named for core
    (src / "kiss" / "pkg" / "__init__.py").parent.mkdir()
    (src / "kiss" / "pkg" / "__init__.py").write_text("")
    (tests / "test_pkg.py").write_text("import kiss.pkg\n")
    names = lambda paths: [p.name for p in paths]  # noqa: E731
    # No import hops: mid.py imports core, yet a change of core selects only test_core*.
    assert names(script.owned_tests([src / "kiss" / "core" / "core.py"], src)) == [
        "test_core.py", "test_core_more.py",
    ]
    # A changed test selects itself; a helper, a deleted test and a non-.py file nothing.
    assert names(script.owned_tests([tests / "test_other.py", tests / "helper.py"], src)) == [
        "test_other.py",
    ]
    assert script.owned_tests([tests / "test_gone.py", tmp_path / "README.md"], src) == []
    # A package's __init__ is named by the package; a module with no test is
    # reported on stderr; a relative path resolves against the repo root.
    assert names(script.owned_tests([src / "kiss" / "pkg" / "__init__.py"], src)) == [
        "test_pkg.py",
    ]
    assert script.owned_tests([Path("src/kiss/other.py"), Path("src/kiss/top.py")], src) == [
        tests / "test_other.py", tests / "test_top.py",
    ]
    assert script.owned_tests([src / "kiss" / "mid.py"], src) == [tests / "test_mid.py"]
    (tests / "test_mid.py").unlink()
    assert script.owned_tests([src / "kiss" / "mid.py"], src) == []
    assert "no test module named for" in capsys.readouterr().err


def test_main_prints_paths_or_the_splits_json(capsys) -> None:
    """On the real repository the script's own test names itself."""
    me = Path(__file__)
    assert script.main([str(me)]) == 0
    assert capsys.readouterr().out.strip() == str(me.relative_to(script.PROJECT_ROOT))
    assert script.main([str(me), "--splits", "4"]) == 0
    commands = json.loads(capsys.readouterr().out)
    mine = me.relative_to(script.PROJECT_ROOT).as_posix()
    assert commands == [f"{PREFIX}{mine}"]
    assert script.main([str(me), "--splits", "4", "--timeout", "0"]) == 0
    assert json.loads(capsys.readouterr().out) == [f"uv run pytest -q -p no:cacheprovider {mine}"]
    # --owned on the script itself: its own test, no hops; without paths it refuses.
    assert script.main(["--owned", "src/kiss/scripts/impacted_tests.py"]) == 0
    assert capsys.readouterr().out.strip() == str(me.relative_to(script.PROJECT_ROOT))
    with pytest.raises(SystemExit) as refused:
        script.main(["--owned"])
    assert refused.value.code == 2
    assert "--owned needs the files you changed" in capsys.readouterr().err
