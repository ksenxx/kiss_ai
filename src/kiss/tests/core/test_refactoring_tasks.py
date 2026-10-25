# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for refactoring tasks: the process-wide artifact directory.

Split out of ``tests/agents/third_party_agents/test_refactoring_tasks.py``:
these methods depend only on ``kiss.core.config``, so they belong in
``tests/core`` per the placement invariants.

No mocks, patches, fakes, or any form of test doubles.
"""

from __future__ import annotations

from pathlib import Path

from kiss.core.config import get_artifact_dir


class TestArtifactDir:
    """``get_artifact_dir`` creates the directory lazily, once per process."""

    def test_artifact_dir_is_stable_for_the_process(self) -> None:
        """The artifact directory is resolved once and never changes.

        Replaced ``test_set_artifact_base_dir``: the runtime setter it
        covered was removed because it had no production caller and was
        the only way to make a running agent's trajectory land under a
        different root than the one it started under.
        """
        assert get_artifact_dir() == get_artifact_dir()
        assert Path(get_artifact_dir()).is_dir()
