# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9: path traversal through ``tabId`` into the merge data dir.

``_merge_data_dir(tab_id)`` appended the frontend-supplied ``tabId``
verbatim to ``{artifact_root}/merge_dir/``.  The tab id arrives straight
off the wire (``cmd.get("tabId")``) and is only coerced to ``str`` — a
malformed or malicious client could send ``"../victim"`` (or a path with
separators) so that:

* ``_cleanup_merge_data(str(_merge_data_dir(tab_id)))`` — run by
  ``_finish_merge`` on every ``mergeAction all-done`` and by
  ``_close_tab`` on every tab close — ``shutil.rmtree``'d a directory
  OUTSIDE the merge dir (up to and including unrelated artifact
  directories), and
* ``_prepare_and_start_merge`` wrote merge artifacts outside it.

No mocks, patches, fakes, or test doubles: the real path helpers and a
real directory tree under the real artifact root.
"""

from __future__ import annotations

import unittest
import uuid

from kiss.core import config as config_module
from kiss.server.diff_merge import (
    _cleanup_merge_data,
    _merge_data_dir,
    _untracked_base_dir,
)


class TestMergeDataDirTraversal(unittest.TestCase):
    """Frontend tab ids must never escape the merge_dir root."""

    def test_traversal_tab_id_stays_inside_merge_dir(self) -> None:
        base = _merge_data_dir().resolve()
        for evil in ("../evil", "../../evil", "a/../../evil", "..", "."):
            resolved = _merge_data_dir(evil).resolve()
            self.assertTrue(
                resolved == base or base in resolved.parents,
                f"tab_id {evil!r} escaped merge_dir: {resolved}",
            )
            self.assertNotEqual(
                resolved, base.parent,
                f"tab_id {evil!r} resolved to the artifact root's child",
            )

    def test_separator_tab_id_is_single_component(self) -> None:
        base = _merge_data_dir().resolve()
        d = _merge_data_dir("a/b\\c").resolve()
        self.assertEqual(d.parent, base)

    def test_distinct_hostile_ids_do_not_collide(self) -> None:
        self.assertNotEqual(_merge_data_dir("../evil"), _merge_data_dir(".._evil"))
        self.assertNotEqual(_merge_data_dir("a/b"), _merge_data_dir("a_b"))

    def test_normal_uuid_tab_ids_unchanged(self) -> None:
        tab_id = str(uuid.uuid4())
        self.assertEqual(
            _merge_data_dir(tab_id), _merge_data_dir() / tab_id,
        )
        self.assertEqual(
            _untracked_base_dir(tab_id),
            _merge_data_dir() / tab_id / "untracked-base",
        )

    def test_cleanup_with_hostile_tab_id_spares_outside_dirs(self) -> None:
        # A victim directory OUTSIDE merge_dir (sibling, under the
        # artifact root) that a traversal tab id used to rmtree.
        artifact_root = config_module._artifact_root()
        victim = artifact_root / f"victim-{uuid.uuid4().hex[:8]}"
        victim.mkdir(parents=True, exist_ok=True)
        marker = victim / "precious.txt"
        marker.write_text("do not delete")
        try:
            # Exactly what _finish_merge / _close_tab do with the
            # wire-supplied tab id.
            _cleanup_merge_data(str(_merge_data_dir(f"../{victim.name}")))
            self.assertTrue(
                marker.is_file(),
                "traversal tabId deleted a directory outside merge_dir",
            )
        finally:
            if victim.exists():
                import shutil

                shutil.rmtree(victim, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
