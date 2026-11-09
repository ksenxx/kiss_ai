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

import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands, sea_settings
from kiss.core.config import kiss_home
from kiss.tests.agents.sorcar.conftest import isolated_sea_registry


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
        while sea_settings._EXECUTING_WAIT.get(second) != key and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sea_settings._EXECUTING_WAIT.get(second) == key, "the second thread did not wait"
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


def test_an_imported_parent_broken_at_the_first_load_is_re_read_once_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``from parent import Parent`` fails at the child's very first load (the parent does
    not parse): no earlier verdict named the parent and no ``sea_class`` call read it, so
    the failure was cached on the child's revision alone and the parent's repair never
    brought the child in.  The files the failed load imported are stamped as they were
    before they ran, like the files ``sea_class`` is asked for."""
    name = "firstimportbroken"
    module, parent, child = _imported_child(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    parent.write_text("class Parent(:\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert sea_commands.model_sea(name) is None
    assert sea_commands.model_sea(name) is None, "a broken load is cached until a file changes"
    stamp = child.stat()

    _write_parent(parent, True)
    assert (child.stat().st_mtime_ns, child.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert sea_commands.model_sea(name) == child, "the repaired parent was not re-read"
    assert sea_commands.model_seas()[name] == child
