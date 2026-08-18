# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""main.js must replay persisted ``autocommit_done`` events.

The backend half of this fix (persisting the event to the task history
database) is covered by ``kiss.tests.server.test_autocommit_persistence``;
this file checks the frontend half: ``handleOutputEvent`` in
``media/main.js`` renders the ``autocommit_done`` event when a session is
replayed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class TestMainJsHandlesAutocommitDoneInReplay(unittest.TestCase):
    """main.js handleOutputEvent must handle autocommit_done for replay."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_handle_output_event_has_autocommit_done_case(self) -> None:
        """handleOutputEvent must have a case for autocommit_done."""
        match = re.search(
            r"function handleOutputEvent\(.*?\)\s*\{(.*?)^\s{2}\}",
            self.js,
            re.DOTALL | re.MULTILINE,
        )
        assert match is not None, "handleOutputEvent function not found in main.js"
        body = match.group(1)
        assert "autocommit_done" in body, (
            "handleOutputEvent in main.js must have a case for 'autocommit_done' "
            "so the commit message renders during session replay"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
