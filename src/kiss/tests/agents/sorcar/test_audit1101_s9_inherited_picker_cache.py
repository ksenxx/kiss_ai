# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Editing the base SEA of a model-picker SEA must update the picker verdict.

``_registers_as_model`` caches each SEA's ``register_as_model()``
verdict per file revision, and a SEA may inherit that method from a
custom base (``class Child(sea_class(parent))``).  The cache was keyed
on the child's revision alone, so an edit of the parent alone (the
child untouched) kept the old verdict for good: ``model_sea`` still
offered a child whose fresh load no longer registers, and the reverse
edit left a newly registering child out of the picker.  The verdict is
now stamped with every file of the class's inheritance chain, as each
was read before it ran (an edit while the child's load is in flight is
re-read, not cached under the new revision), and with the script every
``sea_class("name")`` resolved to (a parent registered or rebound
later is re-read).

Real files, the real loader and registry; nothing is mocked.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import threading
import time
import types
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands, sea_settings
from kiss.core.config import kiss_home
from kiss.tests.agents.sorcar.conftest import isolated_sea_registry
from kiss.tests.server.pause import Pause


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Isolate the registry state and the session-wide ``SEAS.md`` per test."""
    with isolated_sea_registry():
        yield


def _write_parent(parent: Path, registers: bool | str, cls: str = "Parent") -> None:
    """Write the base SEA class *cls* whose ``register_as_model`` returns the expression
    *registers*."""
    parent.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        f"class {cls}(BaseSea):\n"
        "    def register_as_model(self):\n"
        f"        return {registers}\n",
        encoding="utf-8",
    )


def test_parent_edit_flips_the_inherited_picker_verdict(tmp_path: Path) -> None:
    """Only the parent changes; the child's verdict follows it in both directions."""
    folder = tmp_path / "seas"
    parent = tmp_path / "parent_sea.py"
    name = "derivedpicker"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Child(sea_class({str(parent)!r})):\n"
        "    def description(self):\n"
        "        return 'Inherits the model registration of its base.'\n",
        encoding="utf-8",
    )
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")

    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child
    stamp = child.stat()

    _write_parent(parent, False)
    sea_commands.refresh_registry()
    assert sea_commands.load_sea(child).register_as_model() is False
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) is None, "the picker still offers the child"
    assert name not in sea_commands.model_seas()

    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child, "the picker lost the re-enabled child"
    assert sea_commands.model_seas()[name] == child

    # A parent mid-edit does not load: the child is no picker entry
    # while it is broken, and its repair (the child still untouched)
    # brings the child back without a daemon restart.
    parent.write_text("def register_as_model(:\n", encoding="utf-8")
    assert sea_commands.model_sea(name) is None
    assert sea_commands.model_sea(name) is None, "a broken load is cached until a file changes"
    _write_parent(parent, True)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"


@pytest.mark.parametrize("parent_state", ["broken", "missing"])
def test_a_parent_broken_at_the_first_load_is_re_read_once_repaired(
    tmp_path: Path, parent_state: str
) -> None:
    """The very first load fails in the parent; its repair alone must bring the child in."""
    folder = tmp_path / "seas"
    parent = tmp_path / "parent_sea.py"
    name = "firstbroken"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Child(sea_class({str(parent)!r})):\n"
        "    pass\n",
        encoding="utf-8",
    )
    if parent_state == "broken":
        parent.write_text("class Parent(:\n", encoding="utf-8")
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert name in sea_commands.list_commands()
    assert sea_commands.model_sea(name) is None
    assert name not in sea_commands.model_seas()
    stamp = child.stat()

    # Only the parent changes: it now loads and registers.
    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.load_sea(child).register_as_model() is True
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"
    assert sea_commands.model_seas()[name] == child


def test_a_parent_edited_while_the_child_loads_is_re_read(tmp_path: Path) -> None:
    """The child's module body is still running when the parent is edited: its verdict came
    from the old parent and must not be cached under the new parent's revision."""
    folder = tmp_path / "seas"
    parent = tmp_path / "parent_sea.py"
    ready, resume = tmp_path / "ready", tmp_path / "resume"
    name = "racedpicker"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Child(sea_class({str(parent)!r})):\n"
        "    pass\n\n\n"
        f"Path({str(ready)!r}).touch()\n"
        "for _ in range(1000):  # the test edits the parent before it lets the load go on\n"
        f"    if Path({str(resume)!r}).exists():\n"
        "        break\n"
        "    time.sleep(0.01)\n",
        encoding="utf-8",
    )
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    _write_parent(parent, False)
    sea_commands.refresh_registry()
    verdicts: list[Path | None] = []

    def query() -> None:
        verdicts.append(sea_commands.model_sea(name))

    worker = threading.Thread(target=query, name="picker-query")
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "the child never got to its old parent's class"
        _write_parent(parent, True)
    finally:
        resume.touch()
        worker.join(10)
    assert not worker.is_alive() and verdicts == [None]
    assert sea_commands.load_sea(child).register_as_model() is True
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child, "the old verdict is cached under the new parent"


@pytest.mark.parametrize("in_package", [False, True], ids=["module", "package"])
def test_a_parent_imported_the_ordinary_way_is_re_read_once_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_package: bool
) -> None:
    """``from parent import Parent`` instead of ``sea_class``: Python keeps the module once
    imported, so the re-executed child derived from the parent's old class while its verdict
    was stamped with the parent's new revision, and the picker never followed the edit.

    A parent in a package (``from package import parent``, ``Child(parent.Parent)``) is
    reached through the attribute ``import`` bound on the package, not through
    ``sys.modules``: the re-execution replaced the ``sys.modules`` entry alone, so the child
    kept deriving from the old class and every picker read repeated the stale verdict."""
    folder = tmp_path / "seas"
    module = f"parent_{uuid.uuid4().hex[:8]}"  # sys.modules is process-wide: a fresh name
    name = "importedpicker"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    if in_package:
        package = f"package_{uuid.uuid4().hex[:8]}"
        (tmp_path / package).mkdir()
        (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
        parent = tmp_path / package / f"{module}.py"
        child.write_text(
            f"from {package} import {module}\n\n\n"
            f"class Child({module}.Parent):\n"
            "    pass\n",
            encoding="utf-8",
        )
        monkeypatch.delitem(sys.modules, package, raising=False)
        monkeypatch.delitem(sys.modules, f"{package}.{module}", raising=False)
    else:
        parent = tmp_path / f"{module}.py"
        child.write_text(
            f"from {module} import Parent\n\n\n"
            "class Child(Parent):\n"
            "    pass\n",
            encoding="utf-8",
        )
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")

    # The parent is not there yet.  No file names an absent module (``from parent import
    # Parent``) or submodule (``from package import parent``), so that failure is read
    # again at every lookup: the parent created later, the child untouched, brings it in.
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    _write_parent(parent, True)
    importlib.invalidate_caches()  # what ``import`` asks of whoever creates a module
    assert sea_commands.model_sea(name) == child, "the parent created later was not read"

    _write_parent(parent, False)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    stamp = child.stat()

    # Only the parent changes, in both directions.
    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) == child, "the child still derives from the old parent"
    assert sea_commands.model_seas()[name] == child
    assert sea_commands.load_sea(child).register_as_model() is True

    _write_parent(parent, False)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None, "the picker still offers the child"
    assert sea_commands.load_sea(child).register_as_model() is False

    # A parent mid-edit: the child is no entry while it is broken, and
    # its repair alone brings the child back.
    parent.write_text("class Parent(:\n", encoding="utf-8")
    assert sea_commands.model_sea(name) is None
    _write_parent(parent, True)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"

    # An edit that keeps the size within the same second: the bytecode
    # ``import`` cached for the earlier revision passes Python's
    # whole-second mtime and size check, so a re-import ran it instead
    # of the current source and the new revision was stamped as read.
    second = (parent.stat().st_mtime_ns // 1_000_000_000) * 1_000_000_000
    _write_parent(parent, "1 == 1")
    os.utime(parent, ns=(second + 100_000_000, second + 100_000_000))
    assert sea_commands.model_sea(name) == child
    _write_parent(parent, "1 == 0")
    os.utime(parent, ns=(second + 200_000_000, second + 200_000_000))
    assert parent.read_text(encoding="utf-8").endswith("return 1 == 0\n")
    assert sea_commands.model_sea(name) is None, "the parent's old bytecode was re-run"
    assert sea_commands.model_sea(name) is None, "the stale verdict was cached"


@pytest.mark.parametrize(
    ("encoding", "prefix", "header"),
    [
        ("utf-8", b"\xef\xbb\xbf", "# a byte-order mark ahead, as Windows editors write it\n"),
        ("latin-1", b"", "# coding: latin-1\n# caf\xe9\n"),
    ],
    ids=["utf-8-bom", "latin-1-cookie"],
)
def test_an_imported_parent_is_re_read_in_the_source_encoding_import_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    prefix: bytes,
    header: str,
) -> None:
    """A parent with a UTF-8 byte-order mark or a ``coding`` cookie: ``import`` reads it as
    Python source (the mark skipped, the cookie honoured), but its re-execution ran the text
    decoded as UTF-8 (the mark is then ``U+FEFF``, a syntax error; latin-1 bytes do not
    decode), so the refresh failed where the import succeeded and the picker cached a false
    verdict for a child whose fresh load registers."""
    folder = tmp_path / "seas"
    module = f"parent_{uuid.uuid4().hex[:8]}"
    parent = tmp_path / f"{module}.py"
    name = "encodedpicker"
    child = folder / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        f"from {module} import Parent\n\n\nclass Child(Parent):\n    pass\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")

    def write_parent(registers: bool) -> None:
        """The plain parent, in *encoding* behind *prefix* and *header*."""
        _write_parent(parent, registers)
        parent.write_bytes(prefix + (header + parent.read_text(encoding="utf-8")).encode(encoding))

    write_parent(True)
    sea_commands.refresh_registry()
    assert sea_commands.load_sea(child).register_as_model() is True
    assert sea_commands.model_sea(name) == child, "the parent's re-execution rejected its source"
    assert sea_commands.model_sea(name) == child, "the false verdict was cached"
    assert sea_commands.model_seas()[name] == child

    # An edit of such a parent is followed too, in both directions.
    write_parent(False)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None, "the picker still offers the child"
    write_parent(True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child, "the picker lost the re-enabled child"


def test_an_imported_grandparent_edit_re_reads_the_unchanged_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Leaf(Middle)``, ``Middle(Grand)``, both imported the ordinary way, only ``Grand``
    edited: the grandparent's module was re-executed but not the unchanged parent's, whose
    class still derived from the old grandparent, so the re-executed leaf inherited the old
    verdict under the grandparent's new stamp.  Every module deriving from a re-executed one
    is re-executed too, root first, before the leaf runs again."""
    folder = tmp_path / "seas"
    tag = uuid.uuid4().hex[:8]
    grand, middle = tmp_path / f"grand_{tag}.py", tmp_path / f"middle_{tag}.py"
    name = "chainpicker"
    leaf = folder / name / f"{name}_sea.py"
    leaf.parent.mkdir(parents=True)
    middle.write_text(
        f"from grand_{tag} import Grand\n\n\nclass Middle(Grand):\n    pass\n", encoding="utf-8"
    )
    leaf.write_text(
        f"from middle_{tag} import Middle\n\n\nclass Leaf(Middle):\n    pass\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")

    _write_parent(grand, False, "Grand")
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    stamps = (middle.stat().st_mtime_ns, leaf.stat().st_mtime_ns)

    for registers in (True, False, True):
        _write_parent(grand, registers, "Grand")
        sea_commands.refresh_registry()
        expected = leaf if registers else None
        assert sea_commands.model_sea(name) == expected, "the leaf derives from the old parent"
        assert sea_commands.model_sea(name) == expected, "the stale verdict was cached"
        assert sys.modules[f"middle_{tag}"].Middle().register_as_model() is registers
    assert (middle.stat().st_mtime_ns, leaf.stat().st_mtime_ns) == stamps


def test_two_threads_re_reading_one_imported_parent_re_execute_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two children import the same edited parent and are queried at once: the parent's
    re-execution is serialised per file, so the second thread waits for it and then finds
    the re-executed module instead of running the parent a second time."""
    folder = tmp_path / "seas"
    module = f"parent_{uuid.uuid4().hex[:8]}"
    parent = tmp_path / f"{module}.py"
    ready, resume, runs = tmp_path / "ready", tmp_path / "resume", tmp_path / "runs"
    names = ("firsttwin", "secondtwin")
    for name in names:
        child = folder / name / f"{name}_sea.py"
        child.parent.mkdir(parents=True)
        child.write_text(
            f"from {module} import Parent\n\n\nclass Child(Parent):\n    pass\n", encoding="utf-8"
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")

    def write_parent(registers: bool, blocking: bool) -> None:
        """The parent counts its executions in ``runs``; *blocking* makes it wait for ``resume``."""
        wait = (
            f"Path({str(ready)!r}).touch()\n"
            "for _ in range(1000):\n"
            f"    if Path({str(resume)!r}).exists():\n"
            "        break\n"
            "    time.sleep(0.01)\n"
        )
        parent.write_text(
            "import time\n"
            "from pathlib import Path\n"
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
            f"with Path({str(runs)!r}).open('a') as f:\n"
            "    f.write('run\\n')\n"
            + (wait if blocking else "")
            + "\n\nclass Parent(BaseSea):\n"
            "    def register_as_model(self):\n"
            f"        return {registers}\n",
            encoding="utf-8",
        )

    write_parent(False, blocking=False)
    sea_commands.refresh_registry()
    assert not set(names) & set(sea_commands.model_seas())
    before = runs.read_text(encoding="utf-8").count("run")

    write_parent(True, blocking=True)
    sea_commands.refresh_registry()
    verdicts: dict[str, Path | None] = {}

    def query(name: str) -> None:
        verdicts[name] = sea_commands.model_sea(name)

    first = threading.Thread(target=query, args=(names[0],), name="picker-first")
    second = threading.Thread(target=query, args=(names[1],), name="picker-second")
    first.start()
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "the first thread never got into the parent's re-execution"
        second.start()
        key = sea_settings._module_name(parent)
        while sea_settings._waited_module(second.ident) != key and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sea_settings._waited_module(second.ident) == key, "the second thread did not wait"
        assert runs.read_text(encoding="utf-8").count("run") == before + 1
    finally:
        resume.touch()
        first.join(10)
        second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert verdicts == {name: folder / name / f"{name}_sea.py" for name in names}
    assert runs.read_text(encoding="utf-8").count("run") == before + 1, "re-executed twice"


def test_a_named_parent_registered_or_rebound_later_is_re_read(tmp_path: Path) -> None:
    """``sea_class("name")``: the child's verdict follows what the registry resolves the
    name to, a parent that was missing and a higher-precedence folder alike."""
    low, high = tmp_path / "low", tmp_path / "high"
    child = low / "namedchild" / "namedchild_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        'class Child(sea_class("namedparent")):\n'
        "    pass\n",
        encoding="utf-8",
    )
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{low}\n{high}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("namedchild") is None  # no such parent yet

    (low / "namedparent").mkdir()
    _write_parent(low / "namedparent" / "namedparent_sea.py", True)
    sea_commands.refresh_registry()
    assert sea_commands.load_sea(child).register_as_model() is True
    assert sea_commands.model_sea("namedchild") == child, "the parent registered later"

    # The bottom folder of ``SEAS.md`` wins: the name now resolves to
    # another file, the one it resolved to before untouched.
    (high / "namedparent").mkdir(parents=True)
    _write_parent(high / "namedparent" / "namedparent_sea.py", False)
    sea_commands.refresh_registry()
    assert sea_commands.get_command("namedparent") == high / "namedparent" / "namedparent_sea.py"
    assert sea_commands.load_sea(child).register_as_model() is False
    assert sea_commands.model_sea("namedchild") is None, "the rebound parent was not re-read"


def _imported_child(tmp_path: Path, name: str, body: str = "") -> tuple[str, Path, Path]:
    """Write ``seas/<name>/<name>_sea.py`` deriving from ``Parent`` of a fresh ordinary module;
    return the module name, the parent's file and the child's.  *body* follows the class."""
    module = f"parent_{uuid.uuid4().hex[:8]}"  # sys.modules is process-wide: a fresh name
    child = tmp_path / "seas" / name / f"{name}_sea.py"
    child.parent.mkdir(parents=True)
    child.write_text(
        "import time\n"
        "from pathlib import Path\n"
        f"from {module} import Parent\n\n\n"
        "class Child(Parent):\n"
        "    pass\n" + body,
        encoding="utf-8",
    )
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{tmp_path / 'seas'}\n", encoding="utf-8")
    return module, tmp_path / f"{module}.py", child


def _try_import(module: str) -> None:
    """Import *module*, on a thread of its own: a file its source needs and does not find is
    the test's subject, not the thread's."""
    with contextlib.suppress(FileNotFoundError):
        importlib.import_module(module)


def test_an_imported_parent_re_executed_runs_in_a_fresh_module_published_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's new source ran in the old module's own dict: a revision that raised
    half-way had already rebound the globals of every instance loaded before (another task's
    SEA changed behaviour under it), and a name the new source dropped survived, so a child
    importing a class no longer in the file kept loading and kept its picker entry.  The new
    source runs in a fresh module that replaces the old one only once it ran through."""
    name = "freshpicker"
    module, parent, child = _imported_child(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)

    def write_parent(model: str, registers: bool, tail: str = "") -> None:
        """A ``Parent`` whose ``settings`` lays the module global ``MODEL``; *tail* follows."""
        parent.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
            f"MODEL = {model!r}\n\n\n"
            "class Parent(BaseSea):\n"
            "    def register_as_model(self):\n"
            f"        return {registers}\n\n"
            "    def settings(self, settings):\n"
            "        return settings | {'model': MODEL}\n" + tail,
            encoding="utf-8",
        )

    write_parent("old-model", True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child
    loaded = sea_commands.load_sea(child)  # another task's SEA, loaded before the edits
    assert sea_commands.base_settings([loaded])["model"] == "old-model"

    # A revision that raises half-way: nothing of it is published.
    write_parent("half-written", False, "\n\nraise RuntimeError('mid-edit')\n")
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    assert sea_commands.base_settings([loaded])["model"] == "old-model", (
        "the failed re-execution rebound the globals of an instance loaded before"
    )
    assert sys.modules[module].MODEL == "old-model"

    # A revision without ``Parent``: the child's import fails like a fresh one would.
    parent.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "class Replacement(BaseSea):\n"
        "    def register_as_model(self):\n"
        "        return True\n",
        encoding="utf-8",
    )
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None, "a class dropped from the source survived"
    assert not hasattr(sys.modules[module], "Parent")
    with pytest.raises(sea_settings.SeaError, match="cannot import name 'Parent'"):
        sea_commands.load_sea(child)
    assert sea_commands.base_settings([loaded])["model"] == "old-model"

    write_parent("new-model", True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child
    assert sea_commands.base_settings([sea_commands.load_sea(child)])["model"] == "new-model"
    assert sea_commands.base_settings([loaded])["model"] == "old-model"


@pytest.mark.parametrize("fails", [False, True], ids=["valid-edit", "edit-that-raises"])
def test_an_imported_parent_re_executed_is_registered_under_its_name_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """The fresh module ran while ``sys.modules`` still named the old one, where ``@dataclass``
    under ``from __future__ import annotations`` looks a string annotation up: a ``ClassVar``
    the edit newly imports was taken for a field, the class raised and the child lost its
    picker entry until the daemon restarted (audit 2026-11-10, S9 F1).  ``sys.modules`` names
    the fresh module while it runs, behind the module's import lock: an ``import`` in another
    thread waits for the run and reads the finished module, or the old one's names when the
    run raised."""
    name = "classvarpicker"
    module, parent, child = _imported_child(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    _write_parent(parent, True)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child
    old_parent = sys.modules[module].Parent
    ready, resume = threading.Event(), threading.Event()
    handshake = types.ModuleType(f"handshake_{module}")  # what the parent's source imports
    vars(handshake).update(ready=ready, resume=resume)
    monkeypatch.setitem(sys.modules, handshake.__name__, handshake)
    parent.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, fields\n"
        "from typing import ClassVar\n"
        f"import {handshake.__name__} as test\n"
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "@dataclass\n"
        "class Metadata:\n"
        "    label: ClassVar[str] = 'fresh'\n"
        "    required: str\n\n\n"
        "FIELDS = [field.name for field in fields(Metadata)]\n\n\n"
        "class Parent(BaseSea):\n"
        "    def register_as_model(self):\n"
        "        return FIELDS == ['required']\n\n\n"
        "test.ready.set()\n"
        "test.resume.wait(10)  # the test imports the module from another thread meanwhile\n"
        + ("raise RuntimeError('mid-edit')\n" if fails else ""),
        encoding="utf-8",
    )
    verdicts: list[Path | None] = []
    imported: list[type] = []

    def query() -> None:
        verdicts.append(sea_commands.model_sea(name))

    def ordinary_import() -> None:
        imported.append(__import__(module, fromlist=["Parent"]).Parent)

    picker = threading.Thread(target=query, name="picker-query")
    importer = threading.Thread(target=ordinary_import, name="ordinary-import")
    picker.start()
    try:
        assert ready.wait(10), f"the edited parent did not get past its dataclass: {verdicts}"
        importer.start()
        importer.join(0.3)
        assert importer.is_alive(), "an import read the module while its source still ran"
    finally:
        resume.set()
        picker.join(10)
    importer.join(10)
    assert verdicts == [None if fails else child], "the valid edit cost the child its entry"
    assert sea_commands.model_sea(name) == verdicts[0], "the next lookup disagrees"
    assert imported == [sys.modules[module].Parent], "the waiting import read another module"
    assert (imported[0] is old_parent) is fails
    if not fails:  # the fields an ordinary import of the same source finds
        refreshed = sys.modules[module].FIELDS
        monkeypatch.delitem(sys.modules, module)
        assert importlib.import_module(module).FIELDS == refreshed == ["required"]


@pytest.mark.parametrize(
    ("usable", "own", "shape"),
    [(True, False, ""), (False, False, ""), (None, False, ""), (None, True, ""),
     (None, True, "failed"), (None, False, "attr"), (None, False, "package"),
     (None, False, "nested")],
    ids=["usable-half-imported", "not-yet-usable", "no-class-yet", "own-import-ended",
         "own-import-failed", "module-attribute", "submodule-attribute", "base-of-sea-class"],
)
def test_an_imported_parent_mid_import_in_a_cycle_with_this_thread_is_re_read_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, usable: bool | None, own: bool, shape: str
) -> None:
    """The re-execution takes the module's import lock.  Another thread that holds it, importing
    the module for the first time, and waits for a module this thread is importing would wait
    for ever: importlib reports the cycle.  The refresh failed on it, and the picker kept that
    verdict of an unchanged file for good (review R1 of the fix).  The parent is left as
    ``import`` leaves it, half imported, and what it answered is read again after that import:
    its ``register_as_model`` may need a name the rest of its source binds.  A parent that
    blocks above its class (*usable* ``None``) fails the child's own ``from parent import
    Parent``: that failure was cached under the child's revision alone (review R2).  With
    *own* the child itself, not an import around the lookup, imports the blocker: the parent's
    import has ended (the child waits for it) by the time the picker reads the failure.
    *shape* ``failed``: it ended in a failure of the parent's own (what it needs is not there
    yet), and the module the child read half-way has left ``sys.modules``: the child's failure
    was cached all the same, under the files this thread had imported, and outlived the cause.
    *shape* ``attr``: the child reads ``parent.Parent``, a native ``AttributeError``;
    ``package``: the parent is a submodule, and ``package.parent`` is what cannot be read yet;
    ``nested``: its base, loaded by ``sea_class``, does the failing import, two ``SeaError``
    down; all three were cached too (review r3 R1).  The same statement failing on a parent
    fully imported is the file's failure: one lookup loads the child, the next 100 do not."""
    name = "cyclepicker"
    module, parent, child = _imported_child(tmp_path, name)
    blocker = f"blocker_{module}"
    if shape == "package":  # the parent becomes a submodule of a fresh package
        (package := tmp_path / f"package_{module}").mkdir()
        (package / "__init__.py").touch()
        module, parent = f"{package.name}.{module}", package / parent.name
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_parent(parent, True if usable else "IMPORTED")
    source, blocking = parent.read_text(encoding="utf-8"), f"import {blocker}\nIMPORTED = True\n"
    order = [blocking, source] if usable is None else [source, blocking]  # None: above the class
    needed = tmp_path / "needed"  # what the parent of ``failed`` cannot be imported without
    if shape == "failed":
        order.append(f"open({str(needed)!r}).close()\n")
    parent.write_text("\n\n".join(order), encoding="utf-8")
    other = threading.Thread(target=_try_import, args=(module,))  # imports the parent
    handshake = types.ModuleType(f"handshake_{blocker}")  # what the blocker and the child import
    vars(handshake)["other"] = other
    monkeypatch.setitem(sys.modules, handshake.__name__, handshake)
    inside = f"from {module} import Parent" if own else f"INSIDE = sea_commands.model_sea({name!r})"
    runs, base = tmp_path / "runs", tmp_path / "base_sea.py"  # an ``x`` per load of the child
    script = f"with open({str(runs)!r}, 'a') as f: f.write('x')\n" + child.read_text("utf-8")
    if own:  # the child imports the blocker, which fails, and waits for the parent's import
        script = (
            f"import {handshake.__name__} as test\n"
            f"try:\n    import {blocker}\nfinally:\n    test.other.join(60)\n" + script
        )
    if shape in ("attr", "package"):  # ``import parent`` and ``class Child(parent.Parent)``
        script = script.replace(f"from {parent.stem} import Parent", f"import {module}")
        script = script.replace("(Parent)", f"({module}.Parent)")
    if shape == "nested":  # the failing import moves into a base that ``sea_class`` loads
        base.write_text(script, encoding="utf-8")
        sea_class = f"sea_commands.sea_class({str(base)!r})"
        script = f"from kiss.agents.sorcar import sea_commands\nclass Child({sea_class}): pass\n"
    child.write_text(script, encoding="utf-8")
    (tmp_path / f"{blocker}.py").write_text(  # imports the parent in a thread that waits for it
        "import sys\n"
        "from importlib._bootstrap import _ModuleLock\n"
        f"import {handshake.__name__} as test\n"
        "from kiss.agents.sorcar import sea_commands\n"
        "from kiss.tests.server.pause import Pause\n\n\n"
        "def _waits_for_this_module(frame_locals):\n"
        "    return frame_locals['self'].name == __name__\n\n\n"
        # Once.  This source runs again when its import failed (*own*): in the other thread,
        # which imports this module itself where CPython has the fix of gh-143650 (a thread
        # that waited for a failed import got the half-run module before), or in the child's
        # next load.
        "with Pause(_ModuleLock.has_deadlock.__code__, _waits_for_this_module) as held:\n"
        "    if test.other.ident is None:\n"
        "        test.other.start()\n"
        "        assert held.paused.wait(10), 'the other import never waited for this module'\n"
        f"        {inside}\n"
        f"        INITIALIZING = sys.modules[{module!r}].__spec__._initializing\n",
        encoding="utf-8",
    )
    sea_commands.refresh_registry()
    if own:
        assert sea_commands.model_sea(name) is None
        if shape == "failed":
            assert module not in sys.modules, "the other import did not fail"
            needed.touch()  # no file of the child's changes
        else:
            assert not vars(sys.modules[module].__spec__)["_initializing"], "the import still runs"
    else:
        cycle = importlib.import_module(blocker)
        other.join(10)
        assert cycle.INITIALIZING and sys.modules[module].IMPORTED, "the other import was cut"
        assert cycle.INSIDE == (child if usable else None)
    assert sea_commands.model_sea(name) == child, "the verdict of the half-imported parent stuck"
    assert sea_commands.model_sea(name) == child
    _write_parent(parent, True, "Renamed")  # ``Parent`` is gone for good: the file's failure
    runs.write_text("", encoding="utf-8")
    assert not any(sea_commands.model_sea(name) for _ in range(101))
    assert runs.read_text(encoding="utf-8") == "x", "a genuine failure was loaded again"


def test_an_imported_parent_another_thread_is_importing_is_left_to_that_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-execution waited for the module's import lock without a bound, in a C-level wait
    no Stop reaches: behind another thread's first import of the parent, with no cycle for
    importlib to report, a stopped task stayed in the refresh until that import ended (audit
    2026-11-10, B/S9-F2).  The lock is taken only when no other thread holds it, so there is
    no wait left to stop: the refresh returns while the import is still open, having touched
    neither that module nor its initializing mark, and puts back the module the same file
    was imported as under a second name, which it had re-executed by then.  The file keeps
    the revision no file has, so the next refresh re-executes both."""
    name = f"parent_{uuid.uuid4().hex[:8]}"  # sys.modules is process-wide: a fresh name
    (package := tmp_path / f"package_{name}").mkdir()
    (package / "__init__.py").touch()
    parent, qualified = package / f"{name}.py", f"{package.name}.{name}"
    ready, resume = threading.Event(), threading.Event()
    handshake = types.ModuleType(f"handshake_{name}")  # what the parent's source imports
    vars(handshake).update(ready=ready, resume=resume)
    monkeypatch.setitem(sys.modules, handshake.__name__, handshake)
    parent.write_text(
        f"import {handshake.__name__} as test\n"
        "if '.' in __name__:  # the import under the package's name is the one held open\n"
        "    test.ready.set()\n"
        "    assert test.resume.wait(30)\n"
        "VALUE = 1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(package))
    plain = importlib.import_module(name)  # the same file under a second name, imported whole
    with ThreadPoolExecutor(max_workers=2) as pool:
        importing = pool.submit(importlib.import_module, qualified)
        try:
            assert ready.wait(10), "the first import never got into the parent's source"
            half = sys.modules[qualified]
            refreshing = pool.submit(sea_settings.refresh_imported_module, parent)
            assert refreshing.result(10) is False, "a module mid-import was re-executed"
            assert not hasattr(half, "VALUE"), "the refresh returned only after the import"
            assert sys.modules[qualified] is half and vars(half.__spec__)["_initializing"]
            assert sys.modules[name] is plain, "the module that ran through was not put back"
            assert sea_settings.imported_revision(parent) == sea_settings._MID_IMPORT
            assert sea_settings._module_name(parent) not in sea_settings._EXECUTING_OWNER
        finally:
            resume.set()
        assert importing.result(10) is half and half.VALUE == 1, "the import was cut short"
    assert sea_settings.refresh_imported_module(parent), "the deferred refresh never ran"
    assert sys.modules[name] is not plain and sys.modules[qualified] is not half
    assert sys.modules[qualified].VALUE == sys.modules[name].VALUE == 1
    assert sea_settings.imported_revision(parent) == sea_settings.parsed_sea(parent).stamp


def test_an_imported_parent_edited_while_the_child_reloads_is_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's re-execution superseded the child's class, so the child runs again; the
    parent is edited while that second run is under way.  The verdict comes from the class
    the parent's module held when it ran, and was stamped with the parent's revision on disk
    after the load (the edited one), so the picker never followed the edit: the stamp is the
    revision the parent's module was re-executed from."""
    name = "reloadpicker"
    ready, resume, runs = tmp_path / "ready", tmp_path / "resume", tmp_path / "runs"
    module, parent, child = _imported_child(
        tmp_path, name,
        f"\n\nwith Path({str(runs)!r}).open('a') as f:\n"
        "    f.write('x')\n"
        f"if Path({str(runs)!r}).read_text().count('x') == 2:  # the reload\n"
        f"    Path({str(ready)!r}).touch()\n"
        "    for _ in range(1000):  # the test edits the parent before it lets the load go on\n"
        f"        if Path({str(resume)!r}).exists():\n"
        "            break\n"
        "        time.sleep(0.01)\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    _write_parent(parent, False)
    sea_commands.refresh_registry()
    verdicts: list[Path | None] = []

    def query() -> None:
        verdicts.append(sea_commands.model_sea(name))

    worker = threading.Thread(target=query, name="picker-query")
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "the child never ran a second time"
        _write_parent(parent, True)
    finally:
        resume.touch()
        worker.join(10)
    assert not worker.is_alive() and verdicts == [None]
    assert runs.read_text(encoding="utf-8") == "xx"
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) == child, "the old verdict is cached under the new parent"
    assert sea_commands.load_sea(child).register_as_model() is True


def test_an_imported_parent_re_executed_while_a_verdict_is_stamped_is_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two lookups of one child, the parent edited between them.  The first found its class
    current and was about to read the revision the parent's module ran from when the second
    re-executed the parent: the first then stamped the old class's verdict with the new
    revision, and the picker never followed the edit.  The revision is read before the class
    is checked, so a re-execution in between leaves the verdict under no revision."""
    name = "stampedpicker"
    module, parent, child = _imported_child(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    _write_parent(parent, True)
    sea_commands.refresh_registry()
    verdicts: list[Path | None] = []

    def query() -> None:
        verdicts.append(sea_commands.model_sea(name))

    worker = threading.Thread(target=query, name="picker-query")

    def stamps_the_parent(frame_locals: dict[str, object]) -> bool:
        return threading.current_thread() is worker and frame_locals["path"] == parent

    with Pause(sea_settings.imported_revision.__code__, stamps_the_parent) as held:
        worker.start()
        try:
            assert held.paused.wait(10), "the first lookup never stamped its verdict"
            _write_parent(parent, False)
            assert sea_commands.model_sea(name) is None  # re-executes the parent
        finally:
            held.release()
            worker.join(10)
    assert not worker.is_alive() and verdicts == [child]
    assert sea_commands.model_sea(name) is None, "the old verdict is cached under the new parent"


@pytest.mark.parametrize("chained", [" from exc", ""], ids=["cause", "context"])
def test_a_missing_parent_behind_the_scripts_own_exception_is_read_once_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chained: str
) -> None:
    """A script that answers the ``ImportError`` of a parent that is not there with an
    exception of its own fails on a missing module all the same: no file names it, so the
    failure is not cached, and the parent created later brings the untouched child in."""
    name = "wrappedpicker"
    module, parent, child = _imported_child(tmp_path, name)
    child.write_text(
        f"try:\n    from {module} import Parent\nexcept ImportError as exc:\n"
        f"    raise RuntimeError('the parent is required'){chained}\n\n\n"
        "class Child(Parent):\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    _write_parent(parent, True)
    importlib.invalidate_caches()  # what ``import`` asks of whoever creates a module
    assert sea_commands.model_sea(name) == child, "the parent created later was not read"


@pytest.mark.parametrize("shape", ["built-in-import", "circular-import", "raised-by-hand"])
def test_a_failure_of_the_files_own_is_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """A failed load is cached unless what it missed may come without one of its files changing.
    A built-in module it is the first to import is no missing module, though the path finder,
    which the picker asks, does not find one: such a load (``traceback`` imports ``_suggestions``
    to format the failure the loader logs) was loaded twice.  ``circular-import``: the parent
    imports a module that imports the parent's class back; the half-imported parent leaves
    ``sys.modules``, as it does when another thread's import of it failed (``own-import-failed``
    above), but that import was this load's own, and every load fails the same way.
    ``raised-by-hand``: no ``from module import name`` missed the name, the script raises that
    error itself, of a module that is in ``sys.modules`` no more than one that left it."""
    name = "brokenpicker"
    module, parent, child = _imported_child(tmp_path, name)
    runs = tmp_path / "runs"  # an ``x`` per load of the child
    script = f"with open({str(runs)!r}, 'a') as f: f.write('x')\n"
    if shape == "circular-import":
        (tmp_path / f"back_{module}.py").write_text(f"from {module} import Parent\n", "utf-8")
        _write_parent(parent, True)
        parent.write_text(f"import back_{module}\n" + parent.read_text("utf-8"), "utf-8")
        script += child.read_text("utf-8")
    elif shape == "raised-by-hand":  # ``traceback`` imports the module to format the failure
        parent.write_text("raise RuntimeError('cannot be imported')\n", encoding="utf-8")
        script += f"raise ImportError('register_as_model', name={module!r}, name_from='Parent')\n"
    else:
        script += "import _symtable\n\nraise RuntimeError('broken before its register_as_model')\n"
        monkeypatch.delitem(sys.modules, "_symtable", raising=False)  # the load imports it anew
    child.write_text(script, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sea_commands.refresh_registry()
    assert not any(sea_commands.model_sea(name) for _ in range(3))
    assert module not in sys.modules
    assert runs.read_text(encoding="utf-8") == "x", "a genuine failure was loaded again"


@pytest.mark.parametrize("broken", [
    "class Parent(:", "raise ImportError(name=['numpy'])",
    # a ``SeaError`` that is its own cause: the picker follows the causes and must come back
    "from kiss.agents.sorcar.sea_settings import SeaError as E; raise (e := E()) from e",
])
def test_an_imported_parent_broken_at_the_first_load_is_re_read_once_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    """``from parent import Parent`` fails at the child's very first load (the parent does
    not parse, or raises an ``ImportError`` naming no module, whose name the picker reads and
    must survive): no earlier verdict named the parent and no ``sea_class`` call read it, so
    the failure was cached on the child's revision alone and the parent's repair never
    brought the child in.  The files the failed load imported are stamped as they were
    before they ran, like the files ``sea_class`` is asked for."""
    name = "firstimportbroken"
    module, parent, child = _imported_child(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    parent.write_text(f"{broken}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    assert sea_commands.model_sea(name) is None, "a broken load is cached until a file changes"
    stamp = child.stat()

    _write_parent(parent, True)
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"
    assert sea_commands.model_seas()[name] == child


@contextlib.contextmanager
def _unimported(name: str) -> Iterator[None]:
    """Take the module *name* out of ``sys.modules``, so that the next ``import`` of it asks
    the finders as the first one of a process does; put back what was there."""
    before = sys.modules.pop(name, None)
    try:
        yield
    finally:
        sys.modules.pop(name, None)
        if before is not None:
            sys.modules[name] = before


class _OwnFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """A library's own finder on ``sys.meta_path``: serves the one module *name*, which has
    no file, or refuses it with the ``ImportError`` *refusal*.  With *once* it leaves
    ``sys.meta_path`` as it answers.  ``asked`` counts the times it was asked for *name*."""

    def __init__(
        self, name: str, refusal: ImportError | None = None, once: bool = False
    ) -> None:
        self.name, self.refusal, self.once, self.asked = name, refusal, once, 0

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.name:
            return None
        self.asked += 1
        if self.refusal is not None:
            raise self.refusal
        if self.once:
            sys.meta_path.remove(self)
        return importlib.util.spec_from_loader(fullname, self)

    def exec_module(self, module: types.ModuleType) -> None:
        module.__dict__["SERVED"] = True


@pytest.mark.parametrize("built_in", [True, False], ids=["built-in", "own-finder"])
def test_a_failed_load_that_first_imports_a_module_without_a_file_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built_in: bool
) -> None:
    """The picker asked the path finder alone whether a module a load imports is there.  A
    built-in module, or one a library's own finder serves, has no file the path finder
    knows: its first import of the process was noted as a module that was not found, and
    the load that then failed on its own file was not cached (``traceback`` imports the
    built-in ``_suggestions`` to format the first ``ImportError`` of a process, so the first
    broken SEA of a daemon was loaded twice, as was the one of whichever test ran first).
    The finders the import asks are asked instead: one lookup loads the child, the next 100
    do not, and the parent's repair is still read."""
    name = "nofilepicker"
    module, parent, child = _imported_child(tmp_path, name)
    first = "errno" if built_in else f"served_{module}"
    if not built_in:  # asked after the path finder, which finds no such file
        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, _OwnFinder(first)])
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_parent(parent, True, "Renamed")  # no ``Parent``: the failure of the child's file
    runs = tmp_path / "runs"  # an ``x`` per load of the child
    child.write_text(
        f"with open({str(runs)!r}, 'a') as f: f.write('x')\nimport {first}\n"
        + child.read_text("utf-8"),
        encoding="utf-8",
    )
    sea_commands.refresh_registry()
    with _unimported(first), _unimported(module):
        assert not any(sea_commands.model_sea(name) for _ in range(101))
        assert first in sys.modules, "the child never imported the module"
        assert runs.read_text(encoding="utf-8") == "x", "the file's own failure was loaded again"

        _write_parent(parent, True)
        assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"


def test_a_failed_load_whose_module_a_finder_refuses_is_read_once_it_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finder may refuse a name with an ``ImportError`` of its own instead of passing it on:
    the module is as absent as one no finder finds and no file names it either, so the failed
    load is not cached, and the untouched child comes in once the finder serves the module."""
    name = "refusedpicker"
    module, parent, child = _imported_child(tmp_path, name)
    served = f"served_{module}"
    finder = _OwnFinder(served, ModuleNotFoundError(f"No module named {served!r}", name=served))
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, finder])
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_parent(parent, True)
    child.write_text(f"import {served}\n" + child.read_text("utf-8"), encoding="utf-8")
    sea_commands.refresh_registry()
    with _unimported(served), _unimported(module):
        assert sea_commands.model_sea(name) is None
        assert finder.asked == 1, "the refusal the import raised was not the one the picker got"
        finder.refusal = None
        assert sea_commands.model_sea(name) == child, "the module served later was not read"


def test_a_finder_that_answers_a_name_once_serves_the_load_as_it_serves_any_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The picker asks the finders after it what a load's import finds, and hands the import
    that answer instead of letting it ask them again.  A finder that answers a name a single
    time (this one leaves ``sys.meta_path`` with its spec) would have nothing left to say to
    a second question, and a SEA that loads everywhere else would fail in the picker with
    ``No module named``: each finder is asked once."""
    name = "oncepicker"
    module, parent, child = _imported_child(tmp_path, name)
    served = f"served_{module}"
    finder = _OwnFinder(served, once=True)
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, finder])
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_parent(parent, True)
    child.write_text(
        f"import {served}\nassert {served}.SERVED\n" + child.read_text("utf-8"), encoding="utf-8"
    )
    sea_commands.refresh_registry()
    with _unimported(served), _unimported(module):
        assert sea_commands.model_sea(name) == child, "the module served once was not found"
        assert finder.asked == 1 and finder not in sys.meta_path


def test_a_load_that_probes_for_an_absent_module_is_answered_none_as_anywhere_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``importlib.util.find_spec()`` answers ``None`` for a module that is not there.  The
    picker passes a name none of the finders resolves on instead of ending the search with
    an error of its own, so a SEA that probes for an optional module that way, and goes on
    without it, loads in the picker as it does anywhere else."""
    name = "probepicker"
    module, parent, child = _imported_child(tmp_path, name)
    absent = f"absent_{module}"
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_parent(parent, True)
    child.write_text(
        f"import importlib.util\nassert importlib.util.find_spec({absent!r}) is None\n"
        + child.read_text("utf-8"),
        encoding="utf-8",
    )
    sea_commands.refresh_registry()
    with _unimported(module):
        assert sea_commands.model_sea(name) == child, "the probe did not answer None"
