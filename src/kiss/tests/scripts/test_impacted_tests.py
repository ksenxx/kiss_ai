# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``kiss.scripts.impacted_tests`` on a synthetic source tree."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_split_commands_balance_by_size_and_never_exceed_the_tests(tmp_path: Path) -> None:
    src = _tree(tmp_path)
    tests = script.impacted_tests([src / "kiss" / "core" / "core.py"], src, depth=0)
    commands = script.split_commands(tests, 2, root=tmp_path)
    assert len(commands) == 2
    prefix = "uv run pytest -q -p no:cacheprovider src/kiss/tests/"
    assert all(c.startswith(prefix) for c in commands)
    listed = sorted(p for c in commands for p in c.split()[6:])
    assert listed == sorted(t.relative_to(tmp_path).as_posix() for t in tests)
    assert len(script.split_commands(tests, 10, root=tmp_path)) == 3
    assert script.split_commands([], 4, root=tmp_path) == []


def test_main_prints_paths_or_the_splits_json(capsys) -> None:
    """On the real repository the script's own test names itself."""
    me = Path(__file__)
    assert script.main([str(me)]) == 0
    assert capsys.readouterr().out.strip() == str(me.relative_to(script.PROJECT_ROOT))
    assert script.main([str(me), "--splits", "4"]) == 0
    commands = json.loads(capsys.readouterr().out)
    mine = me.relative_to(script.PROJECT_ROOT).as_posix()
    assert commands == [f"uv run pytest -q -p no:cacheprovider {mine}"]
