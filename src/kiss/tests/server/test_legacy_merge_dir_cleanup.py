# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Legacy diff-review snapshots must be retired at server start.

Releases that still had the interactive diff/merge review snapshotted
dirty and untracked files under ``{artifact_root}/merge_dir/<tab>/``
and deleted them when each review ended.  With the review workflow
removed, nothing writes that tree any more — and without a one-time
cleanup an upgrade (or a restart mid-review) would strand potentially
sensitive file copies forever.  ``VSCodeServer.__init__`` now removes
the whole legacy directory; this test pins that behavior end to end.
"""

from pathlib import Path

from kiss.core import config as config_module
from kiss.server.server import VSCodeServer


def test_server_construction_removes_legacy_merge_dir(
    tmp_path: Path, monkeypatch,
) -> None:
    """A stale ``merge_dir`` tree is deleted when the server starts."""
    monkeypatch.setattr(config_module, "_PROJECT_DIR", tmp_path)
    legacy = config_module._artifact_root() / "merge_dir"
    snapshot = legacy / "tab-1" / "untracked-base" / "secret.txt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("stranded secret copy\n")
    (legacy / "tab-1" / "pending-merge.json").write_text("{}")

    VSCodeServer()

    assert not legacy.exists(), (
        "legacy merge_dir snapshots must be removed at server start"
    )
    # The artifacts root itself (and unrelated content) must survive.
    other = config_module._artifact_root() / "other"
    other.mkdir(parents=True, exist_ok=True)
    VSCodeServer()
    assert other.exists()
