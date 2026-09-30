# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the home-wide ``@``-mention picker.

A tab's picker answers with ``./path`` mentions from its work dir
followed by ``~/path`` mentions from the rest of the home directory
(:class:`kiss.server.file_index.Picker`,
:meth:`FileIndexRegistry.picker_for`), so any file under ``~`` can be
handed to an agent from any tab and the inserted text is already the
right relative path.  Every test drives real directories on disk and a
private registry whose home is a temporary directory.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.server.file_index import HOME_SLOTS, FileIndex, FileIndexRegistry, Picker
from kiss.server.server import VSCodeServer


def _wait(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def _touch(root: Path, *rel: str) -> None:
    for r in rel:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def _texts(items: list[dict[str, str]]) -> list[str]:
    return [i["text"] for i in items]


def _build(reg: FileIndexRegistry, work_dir: str) -> None:
    done = threading.Event()
    assert reg.ensure(work_dir, done.set)
    assert done.wait(10.0)


def _home(tmp_path: Path) -> tuple[Path, FileIndexRegistry]:
    """A home with a project, a sibling directory and a top-level note."""
    home = tmp_path / "home"
    _touch(home, "notes.md", "proj/a.py", "proj/src/b.py", "Documents/notes.md", "other/c.py")
    return home, FileIndexRegistry(home=str(home), cache_dir=tmp_path / "cache")


class TestComplementView:
    def test_complement_holds_everything_outside_the_sub_dir(self, tmp_path: Path) -> None:
        home, _ = _home(tmp_path)
        index = FileIndex.scan(str(home))
        rest = index.view("proj/", complement=True)
        assert not any(p.startswith("proj/") for p in rest.paths)
        assert set(rest.paths) | {"proj/", "proj/a.py", "proj/src/", "proj/src/b.py"} == set(
            index.paths
        )
        assert index.view("proj/", complement=True) is rest, "cached like every view"
        assert index.view("proj/") is not rest
        assert index.view("", complement=True) is index.view(""), (
            "the complement of the whole index is the whole index"
        )

    def test_complement_keeps_the_wide_boundary(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        _touch(home, "proj/a.py", "keep/k.py")
        for i in range(60):
            _touch(home, f"runs/r{i}/out.py")
        index = FileIndex.scan(str(home))
        rest = index.view("proj/", complement=True)
        assert rest.paths[:rest.wide_from] and all(
            not p.startswith("runs/r") for p in rest.paths[:rest.wide_from]
        )
        assert all(p.startswith("runs/r") for p in rest.paths[rest.wide_from:])

    def test_refresh_rebuilds_the_complement_view_on_the_worker(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        assert reg.picker_for(str(home / "proj")) is not None
        time.sleep(0.02)
        _touch(home, "other/d.py")
        _build(reg, str(home / "proj"))
        index = reg._indexes[str(home)]
        assert ("proj/", True) in index.view_keys()
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None and "other/d.py" in picker.home_rest.paths  # type: ignore[union-attr]
        reg.stop()


class TestPickerSearch:
    def test_work_dir_first_then_the_rest_of_home(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        items = picker.search("", {})
        assert _texts(items[:3]) == ["./a.py", "./src/", "./src/b.py"]
        assert [i["type"] for i in items[:3]] == ["file"] * 3
        rest = items[3:]
        assert {i["type"] for i in rest} == {"home"}
        assert set(_texts(rest)) == {
            "~/notes.md", "~/Documents/", "~/Documents/notes.md", "~/other/", "~/other/c.py",
        }
        reg.stop()

    def test_query_matches_anywhere_under_home(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        assert _texts(picker.search("notes", {})) == ["~/notes.md", "~/Documents/notes.md"]
        assert _texts(picker.search("b.py", {})) == ["./src/b.py"]
        assert picker.search("nomatch", {}) == []
        reg.stop()

    def test_tilde_query_searches_all_of_home_only(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        # The work dir is included and spelled from home.
        assert _texts(picker.search("~/proj/src", {})) == ["~/proj/src/", "~/proj/src/b.py"]
        assert _texts(picker.search("~/b.py", {})) == ["~/proj/src/b.py"]
        all_home = picker.search("~", {})
        assert _texts(all_home) == _texts(picker.search("~/", {}))
        assert all(t.startswith("~/") for t in _texts(all_home))
        assert {i["type"] for i in all_home} == {"home"}
        assert "~/proj/a.py" in _texts(all_home)
        reg.stop()

    def test_dot_slash_query_searches_the_work_dir_only(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        assert _texts(picker.search("./", {})) == ["./a.py", "./src/", "./src/b.py"]
        assert _texts(picker.search("./notes", {})) == []
        reg.stop()

    def test_home_keeps_at_least_home_slots_when_the_work_dir_fills_the_list(
        self, tmp_path: Path,
    ) -> None:
        home = tmp_path / "home"
        _touch(home, *[f"proj/f{i:02d}.py" for i in range(30)])
        _touch(home, *[f"elsewhere/f{i:02d}.py" for i in range(10)])
        reg = FileIndexRegistry(home=str(home), cache_dir=tmp_path / "cache")
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        items = picker.search("f", {}, limit=20)
        assert len(items) == 20
        local = [t for t in _texts(items) if t.startswith("./")]
        rest = [t for t in _texts(items) if t.startswith("~/")]
        assert len(local) == 20 - HOME_SLOTS and len(rest) == HOME_SLOTS
        assert items[:len(local)] == [i for i in items if i["text"].startswith("./")], (
            "work-dir items come first"
        )
        # A smaller limit: the same split, work dir 7 + home 5.
        few = picker.search("f0", {}, limit=12)
        assert _texts(few) == [f"./f0{i}.py" for i in range(7)] + [
            f"~/elsewhere/f0{i}.py" for i in range(5)
        ]
        # Both parts together below the limit: nothing is trimmed.
        assert _texts(picker.search("f2", {}, limit=20)) == [f"./f2{i}.py" for i in range(10)]
        # Fewer home matches than slots: the work dir keeps the remainder.
        time.sleep(0.02)  # let the directory mtimes move past the first scan
        _touch(home, *[f"proj/z{i:02d}.py" for i in range(30)])
        _touch(home, "elsewhere/z1.py", "elsewhere/z2.py")
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        mixed = _texts(picker.search("z", {}, limit=20))
        assert len(mixed) == 20 and mixed[18:] == ["~/elsewhere/z1.py", "~/elsewhere/z2.py"]
        reg.stop()

    def test_usage_is_keyed_by_mention_text(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        picker = reg.picker_for(str(home / "proj"))
        assert picker is not None
        usage = {"~/other/c.py": 1, "./src/b.py": 2, "src/b.py": 9, "./notes.md": 5}
        items = picker.search("", usage)
        frequent = [i for i in items if i["type"] == "frequent"]
        assert _texts(frequent) == ["./src/b.py", "~/other/c.py"], (
            "a ./ key counts for the work dir, a ~/ key for home; an unprefixed key "
            "and a ./ key naming a home file count for nothing"
        )
        assert _texts(items).count("./src/b.py") == 1 and _texts(items).count("~/other/c.py") == 1
        reg.stop()

    def test_work_dir_at_home_offers_no_separate_home_part(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        picker = reg.picker_for(str(home))
        assert picker is not None and picker.home_rest is None
        items = picker.search("", {})
        assert all(t.startswith("./") for t in _texts(items))
        assert "./proj/src/b.py" in _texts(items)
        assert _texts(picker.search("~/b.py", {})) == ["~/proj/src/b.py"]
        reg.stop()

    def test_work_dir_outside_home_gets_all_of_home(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        outside = tmp_path / "outside"
        _touch(outside, "x.py")
        _build(reg, str(outside))
        picker = reg.picker_for(str(outside))
        assert picker is not None
        assert picker.home_rest is None and picker.home_all is None, "home not indexed yet"
        assert _texts(picker.search("", {})) == ["./x.py"]
        assert picker.search("~/a", {}) == []
        _build(reg, str(home))
        picker = reg.picker_for(str(outside))
        assert picker is not None and picker.home_rest is picker.home_all
        assert _texts(picker.search("a.py", {})) == ["~/proj/a.py"]
        assert _texts(picker.search("", {}))[0] == "./x.py"
        reg.stop()

    def test_work_dir_above_home_does_not_repeat_home(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _touch(tmp_path, "sibling/s.py")
        _build(reg, str(home))
        _build(reg, str(tmp_path))
        picker = reg.picker_for(str(tmp_path))
        assert picker is not None and picker.home_rest is None
        texts = _texts(picker.search("notes", {}))
        assert texts == ["./home/notes.md", "./home/Documents/notes.md"]
        assert _texts(picker.search("~/notes", {})) == ["~/notes.md", "~/Documents/notes.md"]
        reg.stop()

    def test_unindexed_work_dir_has_no_picker(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        _build(reg, str(home))
        assert reg.picker_for(str(tmp_path / "elsewhere")) is None
        assert reg.picker_for(str(home / "proj")) is not None
        reg.stop()

    def test_picker_without_any_home_view(self) -> None:
        picker = Picker(FileIndex.empty("/nowhere").view(""), None, None)
        assert picker.search("", {}) == []
        assert picker.search("~/x", {}) == []


class TestServerReplies:
    def test_get_files_reply_carries_prefixed_mentions(self, tmp_path: Path) -> None:
        home, reg = _home(tmp_path)
        server = VSCodeServer()
        server.work_dir = str(home / "proj")
        server._file_index.stop()
        server._file_index = reg
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        def record(event: dict[str, Any]) -> None:
            with lock:
                events.append(dict(event))

        server.printer.broadcast = record  # type: ignore[method-assign]
        try:
            server._handle_command({
                "type": "getFiles", "prefix": "notes", "workDir": str(home / "proj"),
                "connId": "c1", "tabId": "t1",
            })
            _wait(lambda: any(e["type"] == "files" and not e.get("loading") for e in events))
            reply = next(e for e in events if e["type"] == "files" and not e.get("loading"))
            assert _texts(reply["files"]) == ["~/notes.md", "~/Documents/notes.md"]
            assert reply["prefix"] == "notes"
            server._handle_command({
                "type": "recordFileUsage", "path": "~/Documents/notes.md",
                "workDir": str(home / "proj"),
            })
            server._handle_command({
                "type": "getFiles", "prefix": "", "workDir": str(home / "proj"),
                "connId": "c1", "tabId": "t1",
            })
            last = [e for e in events if e["type"] == "files"][-1]
            assert last["files"][0] == {"type": "frequent", "text": "~/Documents/notes.md"}
            assert _texts(last["files"])[1:4] == ["./a.py", "./src/", "./src/b.py"]
        finally:
            reg.stop()
