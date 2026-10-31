# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 2, Q2 findings 1 and 2: how the evidence gate reads a package tool's command.

1. The global options before a subcommand were one repeated regex in which
   a ``+a`` word was both a possible value of the option before it and an
   option itself, so ``npm --prefix +a ... test`` had 2**n readings: a
   300-byte command held the GIL, and with it the whole daemon, for
   seconds.  A long run of options with tool names in it (``npm -a npm -a
   ...``) was quadratic.
2. An option value was any run of non-blanks, so a quoted or escaped value
   with blanks (``git -C "my repo" checkout main``) hid the subcommand and
   the write counted as the check of an earlier edit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar import evidence_gate as eg

# Runs writes_files on the commands read from stdin (JSON) while a thread ticks
# every 10 ms; prints the results, the time taken and the longest pause of
# the ticking thread (a regex that runs for seconds holds the GIL that long).
_PROBE = r"""
import json, sys, threading, time
from kiss.agents.sorcar.evidence_gate import writes_files

beats = []
done = threading.Event()


def beat():
    while not done.is_set():
        beats.append(time.monotonic())
        time.sleep(0.01)


thread = threading.Thread(target=beat)
thread.start()
commands = json.load(sys.stdin)
start = time.monotonic()
results = [writes_files({"command": command}) for command in commands]
elapsed = time.monotonic() - start
done.set()
thread.join()
gaps = [later - earlier for earlier, later in zip(beats, beats[1:])]
print(json.dumps({"results": results, "elapsed": elapsed, "max_gap": max(gaps, default=0.0)}))
"""

ADVERSARIAL = {
    "npm " + "--prefix +a " * 2000 + "test": False,
    "npm " + "--prefix +a " * 2000 + "ci": True,
    "npm -a " * 20000 + "test": False,
    "uv " + "-a pip " * 20000 + "test": False,
    "git " + '-c "a b" ' * 20000 + "status": False,
    "cargo " + "+a " * 20000 + "fmt": True,
}


def test_repeated_options_are_read_in_linear_time_without_holding_the_gil() -> None:
    src = str(Path(eg.__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    commands = list(ADVERSARIAL)
    done = subprocess.run(
        [sys.executable, "-c", _PROBE], input=json.dumps(commands),
        capture_output=True, text=True, timeout=120, env=env, check=True,
    )
    report = json.loads(done.stdout)
    assert report["results"] == list(ADVERSARIAL.values())
    assert report["elapsed"] < 30, report
    assert report["max_gap"] < 2, report


def test_short_repeated_options_still_find_the_subcommand() -> None:
    assert not eg.writes_files({"command": "npm " + "--prefix +a " * 25 + "test"})
    assert eg.writes_files({"command": "npm " + "--prefix +a " * 25 + "ci"})
    assert eg.writes_files({"command": "npm --prefix +a ci"})
    assert not eg.writes_files({"command": "npm --prefix +a test"})


QUOTED_WRITES = (
    'npm --prefix "src/web app" run build',
    "npm --prefix src/web\\ app run build",
    "yarn --cwd 'my app' add left-pad",
    'pnpm --dir "my app" remove x',
    'uv --project "my project" sync',
    "uv --directory 'a b' add httpx",
    'git -C "my repo" checkout main',
    "git -C my\\ repo stash",
    "git -c 'user.name=A B' pull",
    'cargo --manifest-path "a b/Cargo.toml" build',
    "cargo --config 'a b' fmt",
    'pip --log "a b.log" uninstall -y x',
    'apt-get -o "Debug::X=a b" purge vim',
    'apt -o "Debug::X=a b" upgrade',
    "git -C repo \\\n  checkout main",
    "bash -c \"npm --prefix 'my app' ci\"",
    'ssh host "cd $DIR && git -C \\"my repo\\" pull"',
    "echo ok && sudo git -C 'my repo' reset --hard",
    "(cd x; npm --prefix \"a b\" i)",
    "x=$(git -C 'a b' rebase main)",
)
QUOTED_CHECKS = (
    'npm --prefix "src/web app" test',
    "npm --prefix src/web\\ app run lint",
    'git -C "my repo" diff',
    "uv --project 'my project' run pytest -q",
    'cargo --manifest-path "a b/Cargo.toml" test',
    "bash -c \"npm --prefix 'my app' test\"",
    "git -C repo\ncheckout main",
)


@pytest.mark.parametrize("command", QUOTED_WRITES)
def test_a_write_with_a_quoted_option_value_leaves_the_edit_unchecked(command: str) -> None:
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {})
    gate.observe("Bash", {"command": command})
    verdict = gate.verdict({"success": True, "summary_in_html": "Done."})
    assert verdict is not None and eg.UNCHECKED_REASON in verdict, command


@pytest.mark.parametrize("command", QUOTED_CHECKS)
def test_a_check_with_a_quoted_option_value_lets_the_finish_through(command: str) -> None:
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {})
    gate.observe("Bash", {"command": command})
    assert gate.verdict({"success": True, "summary_in_html": "Done."}) is None, command
