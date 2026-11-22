# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``kiss.scripts.impacted_tests`` on a synthetic source tree."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.core.processes import find_bash
from kiss.scripts import impacted_tests as script
from kiss.tests.conftest import posix_only


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
        "kiss/tests/test_rel.py": "from ..pkg.sub.deep import H2\n",
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
    assert names(script.impacted_tests([helper], src, depth=2)) == [
        "test_deep.py", "test_pkg.py", "test_rel.py",
    ]
    # Importing a submodule runs the __init__.py of every package above it first,
    # whether the import names it absolutely (test_deep) or relatively (test_rel).
    sub_init = src / "kiss" / "pkg" / "sub" / "__init__.py"
    for depth in (1, script.DEFAULT_DEPTH, 0):
        selected = script.impacted_tests([sub_init], src, depth)
        assert names(selected) == ["test_deep.py", "test_rel.py"], depth
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


def test_split_commands_quote_their_paths_for_the_shell(tmp_path: Path) -> None:
    """A path with a space or a shell metacharacter stays one word of its command.

    bash, which runs the commands, is asked for the words it reads; nothing is executed.
    """
    bash = find_bash()
    if bash is None:
        pytest.skip("no bash to read the command")
    odd = tmp_path / "my tests; echo $HOME" / "test_it's.py"
    odd.parent.mkdir()
    for test in (odd, tmp_path / "test_plain.py"):
        test.write_text("")
    [command] = script.split_commands([odd, tmp_path / "test_plain.py"], 1, root=tmp_path)
    read = subprocess.run(
        [bash, "-c", f'words() {{ printf "%s\\0" "$@"; }}; words {command}'],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert read.stdout.split("\0")[:-1] == [
        *PREFIX.split(), "my tests; echo $HOME/test_it's.py", "test_plain.py",
    ], read.stderr


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
    mine = me.relative_to(script.PROJECT_ROOT).as_posix()
    assert script.main([str(me)]) == 0
    assert capsys.readouterr().out.strip() == mine
    assert script.main([str(me), "--splits", "4"]) == 0
    commands = json.loads(capsys.readouterr().out)
    assert commands == [f"{PREFIX}{mine}"]
    assert script.main([str(me), "--splits", "4", "--timeout", "0"]) == 0
    assert json.loads(capsys.readouterr().out) == [f"uv run pytest -q -p no:cacheprovider {mine}"]
    # --owned on the script itself: its own test, no hops; without paths it refuses.
    assert script.main(["--owned", "src/kiss/scripts/impacted_tests.py"]) == 0
    assert capsys.readouterr().out.strip() == mine
    with pytest.raises(SystemExit) as refused:
        script.main(["--owned"])
    assert refused.value.code == 2
    assert "--owned needs the files you changed" in capsys.readouterr().err


def test_a_changed_conftest_selects_every_test_below_it(tmp_path: Path) -> None:
    """pytest loads a ``conftest.py`` for the tests under its directory without an import."""
    src = _tree(tmp_path)
    tests = src / "kiss" / "tests"
    for rel in ("conftest.py", "sub/conftest.py", "sub/test_below.py", "sub/deep/test_deeper.py",
                "side/test_beside.py"):
        (tests / rel).parent.mkdir(parents=True, exist_ok=True)
        (tests / rel).write_text("")
    below = ["test_below.py", "test_deeper.py"]
    hit = script.impacted_tests([tests / "sub" / "conftest.py"], src, depth=1)
    assert sorted(p.name for p in hit) == below
    # The suite's conftest and the repository's own (above src) reach every test.
    everything = sorted([*below, "test_beside.py", "test_core.py", "test_mid.py", "test_other.py",
                         "test_top.py"])
    for conftest in (tests / "conftest.py", Path("conftest.py")):
        assert sorted(p.name for p in script.impacted_tests([conftest], src)) == everything
    # A conftest that imports a changed module stands for the tests below it, which
    # are one hop further than it is: core -> mid -> sub/conftest.py -> sub's tests.
    (tests / "sub" / "conftest.py").write_text("import kiss.mid\n")
    core, mid = src / "kiss" / "core" / "core.py", src / "kiss" / "mid.py"
    names = lambda paths: sorted(p.name for p in paths)  # noqa: E731
    assert names(script.impacted_tests([mid], src, depth=1)) == ["test_mid.py"]
    assert names(script.impacted_tests([mid], src)) == [*below, "test_mid.py", "test_top.py"]
    assert names(script.impacted_tests([core], src)) == ["test_core.py", "test_mid.py"]
    assert names(script.impacted_tests([core], src, depth=0)) == [
        "test_below.py", "test_core.py", "test_deeper.py", "test_mid.py", "test_top.py",
    ]
    # Importing kiss.mid runs kiss/__init__.py first, for the conftest's tests as for it.
    package = script.impacted_tests([src / "kiss" / "__init__.py"], src)
    assert names(package) == [name for name in everything if name != "test_beside.py"]


def _fixture(signature: str, body: str, options: str = "") -> str:
    """Return the source of a fixture ``def signature`` that runs *body*."""
    return f"import pytest\n@pytest.fixture{options}\ndef {signature}:\n    {body}\n"


def _test(asks: str = "") -> str:
    """Return the source of a test that passes once it has the fixtures it *asks* for."""
    return f"def test_it({asks}):\n    pass\n"


_LAZY = _fixture("value()", "import kiss.helper")
_WRAPPED = _fixture("wrapped(value)", "return value")
_ANNOTATED = "def annotation():\n    import kiss.helper\ndef unused() -> annotation():\n    pass\n"
_CONFTEST_TREE = {
    "helper.py": "raise AssertionError('kiss.helper is broken')\n",
    "fixtures.py": _WRAPPED,
    "shared.py": _fixture("outer(inner)", "pass"),
    # Audit X1/F2: what a conftest imports as it loads, every test below it runs.
    "tests/eager/conftest.py": (
        "from kiss.helper import VALUE\n" + _fixture("value()", "return VALUE")
    ),
    "tests/eager/test_fails.py": _test("value"),
    # The six trees of the FX21b review.  A helper nothing names runs for no test, ...
    "tests/unused/conftest.py": "def unused():\n    import kiss.helper\n",
    "tests/unused/test_passes.py": _test(),
    # ... a fixture for the tests that ask for it, by its ``name=``, ...
    "tests/renamed/conftest.py": (
        _fixture("_optional()", "import kiss.helper", "(name='optional')")
    ),
    "tests/renamed/test_fails.py": _test("optional"),
    "tests/renamed/test_passes.py": _test(),
    # ... or through a fixture of a conftest below (``values`` is another word).
    "tests/child/conftest.py": _LAZY,
    "tests/child/test_passes.py": _test("values=()"),
    "tests/child/deep/conftest.py": _WRAPPED,
    "tests/child/deep/test_fails.py": _test("wrapped"),
    "tests/child/deep/test_passes.py": _test(),
    # An annotation is evaluated as the conftest loads, whatever function it is on (from
    # Python 3.14 only when something reads it, so there the selection is to spare), ...
    "tests/annotation/conftest.py": _ANNOTATED,
    f"tests/annotation/test_{'fails' if sys.version_info < (3, 14) else 'spare'}.py": _test(),
    # ... unless the module postpones its annotations: spared too, as something may read
    # them later (FX21c review: a header may be evaluated, it need not be).
    "tests/postponed/conftest.py": "from __future__ import annotations\n" + _ANNOTATED,
    "tests/postponed/test_spare.py": _test(),
    # A default is evaluated as the conftest loads, and may call a function by its name in a
    # string (FX21c review); the fixture's own ``name=`` above is no such call.
    "tests/literal/conftest.py": (
        "import sys\ndef load():\n    import kiss.helper\n"
        "def unused(x=getattr(sys.modules[__name__], 'load')()):\n    pass\n"
    ),
    "tests/literal/test_fails.py": _test(),
    # A test asks through a fixture it imports from a module that names the conftest's.
    "tests/imported/conftest.py": _LAZY,
    "tests/imported/test_fails.py": "from kiss.fixtures import wrapped\n" + _test("wrapped"),
    "tests/imported/test_passes.py": _test(),
    # pytest runs a hook, and an autouse fixture with all it asks for, for every test below.
    "tests/hook/conftest.py": "def pytest_runtest_setup(item):\n    import kiss.helper\n",
    "tests/hook/test_fails.py": _test(),
    "tests/auto/conftest.py": (
        _fixture("_auto(_check)", "pass", "(scope='function', autouse=True)")
        + _fixture("_check(_really)", "pass")
        + _fixture("_really()", "import kiss.helper")
    ),
    "tests/auto/test_fails.py": _test(),
    # A mark asks for a fixture, as does the string another fixture hands to pytest.  The
    # default written out, ``autouse=False``, runs it for no other test (FX21c review).
    "tests/asked/conftest.py": (
        _fixture("value()", "import kiss.helper", "(autouse=False)")
        + _fixture("twice(request)", "request.getfixturevalue('value')")
    ),
    "tests/asked/test_fails.py": "import pytest\n@pytest.mark.usefixtures('twice')\n" + _test(),
    "tests/asked/test_passes.py": _test(),
    # An import for type checkers never runs; the ``else`` of one always does.
    "tests/typed/conftest.py": (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import kiss.helper\n"
    ),
    "tests/typed/test_passes.py": _test(),
    "tests/typed/untyped/conftest.py": (
        "import typing\nif typing.TYPE_CHECKING:\n    pass\nelse:\n    import kiss.helper\n"
    ),
    "tests/typed/untyped/test_fails.py": _test(),
    # A fixture of the same name in another directory is another fixture.
    "tests/plain/conftest.py": _fixture("value()", "return 1"),
    "tests/plain/test_passes.py": _test("value"),
    # Spared, not ruled out: another decorator may run the function it is handed, and a
    # conftest that imports a module naming the fixture may hold a fixture that asks for it.
    "tests/decorated/conftest.py": (
        "import contextlib\n@contextlib.contextmanager\ndef unused():\n"
        "    import kiss.helper\n    yield\n"
    ),
    "tests/decorated/test_spare.py": _test(),
    "tests/registered/conftest.py": (
        "from kiss.shared import outer\n" + _fixture("inner()", "import kiss.helper")
    ),
    "tests/registered/test_fails.py": _test("outer"),
    "tests/registered/test_spare.py": _test(),
    # The repository's own conftest, above src (written apart): a fixture as any other.
    "tests/above/test_fails.py": _test("from_above"),
    "tests/above/test_passes.py": _test(),
}
"""Files under ``src/kiss`` in which ``kiss.helper`` cannot be imported: real pytest fails each
``test_fails.py``, passes the others, and the selector cannot rule out a ``test_spare.py``."""


def test_a_conftest_import_selects_the_tests_that_run_it(tmp_path: Path) -> None:
    """A conftest's import of a changed module selects the tests below it that run the import.

    Real pytest says which those are, on :data:`_CONFTEST_TREE`: the selector names every
    test a broken ``helper.py`` fails and, of those that pass, only the spared ones.  Before,
    it named none (the import graph ends at a conftest), then every test below the conftest.
    """
    src = tmp_path / "src"
    tests = src / "kiss" / "tests"
    for rel, text in _CONFTEST_TREE.items():
        path = src / "kiss" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        (path.parent / "__init__.py").touch()
    (tmp_path / "conftest.py").write_text(_fixture("from_above()", "import kiss.helper"))
    ran = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", "-o", "addopts=",
         "--continue-on-collection-errors", f"--rootdir={tmp_path}", f"--confcutdir={tmp_path}",
         str(tests)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False,
        env={**os.environ, "PYTHONPATH": str(src), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
             "PYTEST_ADDOPTS": "", "COLUMNS": "200"},
    )
    every = {p.relative_to(tests).as_posix() for p in tests.rglob("test_*.py")}
    fails = {test for test in every if test.endswith("test_fails.py")}
    spare = {test for test in every if test.endswith("test_spare.py")}
    # What fails names its module: a test, or the directory of a conftest that cannot load.
    error = r"^ERROR src/kiss/tests/(\S+?)(?:::\w+)? - AssertionError: kiss\.helper is broken$"
    nodes = re.findall(error, ran.stdout, re.MULTILINE)
    failed = {test for test in every for node in nodes if f"{test}/".startswith(f"{node}/")}
    assert failed == fails, ran.stdout + ran.stderr
    assert f"{len(every - fails)} passed, {len(nodes)} errors" in ran.stdout
    helper = [src / "kiss" / "helper.py"]
    for depth in (script.DEFAULT_DEPTH, 0):
        selected = script.impacted_tests(helper, src, depth)
        assert {p.relative_to(tests).as_posix() for p in selected} == fails | spare, depth
    # The conftest is one hop from the module, the tests it stands for are two.
    assert script.impacted_tests(helper, src, depth=1) == []


def _cli(env: dict[str, str], *command: str) -> subprocess.CompletedProcess[str]:
    """Run ``python *command`` with *env* added to the environment, for at most 300 s.

    Its output is decoded as file names are (``os.fsdecode``), so a listed path reads as the
    file's name does here.
    """
    return subprocess.run(
        [sys.executable, *command], env={**os.environ, **env}, capture_output=True, timeout=300,
        check=False, encoding=sys.getfilesystemencoding(), errors=sys.getfilesystemencodeerrors(),
    )


def test_git_failing_in_the_repository_is_an_error_not_an_empty_selection(tmp_path: Path) -> None:
    """No changed files could be listed: stderr says so, the exit is non-zero, no ``[]``."""
    index = tmp_path / "index"
    index.write_text("not an index")  # real git: "index file smaller than expected", exit 128
    env = {"GIT_INDEX_FILE": str(index)}
    done = _cli(env, "-m", "kiss.scripts.impacted_tests", "--splits", "2")
    assert (done.returncode, done.stdout) == (1, ""), done.stderr
    assert "cannot list the changed files: git diff --name-only" in done.stderr
    assert "HEAD failed" in done.stderr


@pytest.mark.slow
@posix_only("stalls git on a FIFO")
def test_git_stalling_is_an_error_too(tmp_path: Path) -> None:
    """Real git blocks opening ``GIT_TRACE`` while the FIFO has no reader; the bound is 30 s."""
    fifo = tmp_path / "trace"
    os.mkfifo(fifo)
    try:
        done = _cli({"GIT_TRACE": str(fifo)}, "-m", "kiss.scripts.impacted_tests", "--splits", "2")
    finally:
        os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))  # frees a git still blocked on it
    assert (done.returncode, done.stdout) == (1, ""), done.stderr
    assert "cannot list the changed files" in done.stderr and "timed out" in done.stderr


def test_outside_a_repository_nothing_changed(tmp_path: Path) -> None:
    """A copy of the script with no ``.git`` above it finds no changed files, as documented."""
    copy = tmp_path / "src" / "kiss" / "scripts" / "impacted_tests.py"
    copy.parent.mkdir(parents=True)
    shutil.copy(script.__file__, copy)
    done = _cli({}, str(copy), "--splits", "2")
    assert (done.returncode, done.stdout) == (0, "[]\n"), done.stderr


def _git(repo: Path, *command: str) -> None:
    """Run ``git *command`` in *repo* under no configuration but an identity, for at most 60 s."""
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *command], cwd=repo,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
        check=True, capture_output=True, timeout=60,
    )


def test_changes_of_a_project_below_the_repository_root_are_found_renames_included(
    tmp_path: Path,
) -> None:
    """Audit SC1: ``git diff --name-only`` names its paths from the top of the repository, a
    staged rename by its new name alone, and a name that is not ASCII in quotes.  The first lost
    every tracked change of a project kept below the root (``.git`` above it), the second the
    tests that still import the old name, the third the file so named.  Read as text, the
    listing also lost a file with a carriage return in its name: the CR became a line feed.
    And ``print`` could not write a test's name that is not UTF-8 to a UTF-8 terminal."""
    repo = tmp_path / "repo"
    kiss = repo / "project" / "src" / "kiss"
    for rel, text in {
        "scripts/impacted_tests.py": Path(script.__file__).read_text(encoding="utf-8"),
        "mod.py": "X = 1\n",
        "old.py": "Y = 1\n",
        "tests/test_mod.py": "import kiss.mod\n",
        "tests/test_old.py": "import kiss.old\n",
        "tests/test_new.py": "import kiss.new\n",
        "tests/test_untouched.py": "import json\n",
    }.items():
        (kiss / rel).parent.mkdir(parents=True, exist_ok=True)
        (kiss / rel).write_text(text, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "mv", "project/src/kiss/old.py", "project/src/kiss/new.py")
    (kiss / "mod.py").write_text("X = 2\n")
    (kiss / "tests" / "test_fresh.py").write_text("import json\n")  # untracked
    (kiss / "tests" / "test_naïve.py").write_text("import json\n")
    latin = b"test_caf\xe9.py"  # not UTF-8: read and listed all the same
    with contextlib.suppress(OSError, UnicodeError):  # a system that refuses the names has none
        (kiss / os.fsdecode(b"caf\xe9.py")).write_text("Z = 1\n")
        (kiss / "tests" / os.fsdecode(latin)).write_text("import json\n")
    # PYTHONIOENCODING=utf-8: the strict UTF-8 stdout of a terminal under LANG=en_US.UTF-8.
    done = _cli({"PYTHONIOENCODING": "utf-8"}, str(kiss / "scripts" / "impacted_tests.py"))
    assert done.returncode == 0, done.stderr
    listed = [f"src/kiss/tests/test_{name}.py" for name in ("fresh", "mod", "naïve", "new", "old")]
    if latin in os.listdir(os.fsencode(kiss / "tests")):
        listed.insert(0, f"src/kiss/tests/{os.fsdecode(latin)}")
    assert done.stdout.split() == listed
    odd = kiss / "tests" / "test_carriage\rreturn.py"
    with contextlib.suppress(OSError):  # as above: Windows has no such name
        odd.write_text("import json\n")
    done = _cli({}, str(kiss / "scripts" / "impacted_tests.py"), "--splits", "1")
    assert done.returncode == 0, done.stderr
    [command] = json.loads(done.stdout)
    assert ("src/kiss/tests/test_carriage\rreturn.py" in shlex.split(command)) is odd.exists()
