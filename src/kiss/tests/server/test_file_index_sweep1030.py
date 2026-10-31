# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Views of the ``@``-mention file index: sweep 10-30 regressions.

* The view of a directory whose parent is a wide container (50+
  sub-directories) used to list the directory itself as ``""``: its
  entry sits in the index's tail, whose filter lacked the "strictly
  below" test the head had, so a tab opened there offered a blank
  mention.
* Every view a tab ever asked for was rebuilt on each rescan and kept
  for the daemon's lifetime: a rescan carried over all the views the
  previous index held, including those it had itself only carried over.
  Now a rescan prebuilds only the views served since the previous one.

Every test scans real directories on disk.
"""

from __future__ import annotations

import threading
from pathlib import Path

from kiss.server.file_index import WIDE_DIR_MIN_CHILDREN, FileIndex, FileIndexRegistry
from kiss.tests.server.conftest import touch


def _rescan(reg: FileIndexRegistry, root: Path) -> FileIndex:
    """Build *root*'s index anew and return it."""
    done = threading.Event()
    assert reg.ensure(str(root), done.set)
    assert done.wait(10.0)
    return reg._indexes[str(root)]


def test_view_of_a_directory_in_a_wide_container_excludes_itself(tmp_path: Path) -> None:
    touch(tmp_path, *[f"runs/r{i:02d}/out.py" for i in range(WIDE_DIR_MIN_CHILDREN)])
    index = FileIndex.scan(str(tmp_path))
    assert index.paths.index("runs/r00/") >= index.wide_from, "the dir is in the tail"
    view = index.view("runs/r00/")
    assert view.paths == ["out.py"]
    assert [i["text"] for i in view.search("", {})] == ["out.py"], "no blank mention"


def test_a_rescan_prebuilds_only_the_views_served_since_the_previous_one(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    touch(home, "a/x.py", "b/y.py")
    reg = FileIndexRegistry(home=str(home), cache_dir=tmp_path / "cache")
    keys = {("a/", False), ("a/", True)}  # a tab in a/: its files, then the rest of home
    _rescan(reg, home)
    assert reg.view_for(str(home / "a")).paths == ["x.py"]  # type: ignore[union-attr]
    index = _rescan(reg, home)
    assert keys <= set(index._views), "the served views are prebuilt on the worker"
    assert not keys & set(index.view_keys()), "a prebuilt view is not a served one"
    # Nobody asked for them on that index: the next rescan drops them
    # instead of rebuilding them on every rescan for good.
    index = _rescan(reg, home)
    assert not keys & set(index._views)
    assert reg.view_for(str(home / "a")).paths == ["x.py"], "rebuilt on demand"  # type: ignore[union-attr]
    index = _rescan(reg, home)
    assert reg.view_for(str(home / "a")) is index._views[("a/", False)], "prebuilt copy served"
    reg.stop()
