# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Keep ``webapp/test/fixtures/bgTabStreamParity.json`` in step with the daemon's recorder.

The JS suite ``bgTabStreamParity.test.js`` replays what the daemon stores
of a run.  ``JsonPrinter._filter_and_coalesce`` is the function
``stop_recording()`` and ``peek_recording()`` run every event list
through on its way to the database, so the fixture's ``recorded`` lists
must be exactly what it makes of the fixture's ``runs``.  When the
recorder changes, regenerate the file with::

    uv run python src/kiss/tests/server/test_bg_tab_stream_parity_fixture.py --regenerate
"""

import json
import sys
from pathlib import Path

from kiss.server.json_printer import JsonPrinter

FIXTURE = (
    Path(__file__).resolve().parents[2] / "webapp" / "test" / "fixtures" / "bgTabStreamParity.json"
)


def recorded_runs(runs: list[list[dict]]) -> list[list[dict]]:
    """Return what the daemon's recorder keeps of each of ``runs``."""
    kept: list[list[dict]] = json.loads(
        json.dumps([JsonPrinter._filter_and_coalesce(run) for run in runs])
    )
    return kept


def test_fixture_matches_the_daemon_recorder() -> None:
    fixture = json.loads(FIXTURE.read_text())
    assert recorded_runs(fixture["runs"]) == fixture["recorded"], (
        f"{FIXTURE} is stale: regenerate it with `uv run python {__file__} --regenerate`"
    )


def regenerate() -> None:
    """Rewrite the fixture's ``recorded`` lists from its ``runs``."""
    fixture = json.loads(FIXTURE.read_text())
    fixture["recorded"] = recorded_runs(fixture["runs"])
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate"]:
        sys.exit(f"usage: {sys.argv[0]} --regenerate")
    regenerate()
