# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The browser profile guard must be machine-wide, not per-process.

``_launch_browser`` protected the check-then-use sequence
(``_cleanup_stale_escalation_dirs`` → ``_resolve_user_data_dir`` →
``_clean_singleton_locks`` → launch) with a ``threading.RLock``.  The guarded
resource — the Chromium profile directory — is shared by *every* kiss process
on the machine (the ``kiss-web`` daemon, a CLI run, channel agents), so a
process-local lock gives no mutual exclusion at all: two processes both see a
lock-free profile, and the second one deletes the first one's live
``SingletonLock`` before opening the same profile a second time.  Chromium
then has one profile open twice, and the persisted logins the class promises
to preserve are corrupted.

These tests use two real OS processes, a real file lock, and two real
headless Chromium launches against a temp profile — never the user's real
``~/.kiss/browser_profile``.
"""

import subprocess
import sys
import time
from pathlib import Path

#: Driver run in two concurrent processes.  Both wait on a barrier file so
#: they enter the profile-resolution window together, then each reports the
#: profile directory it actually opened.
_BROWSER_DRIVER = '''
import os, sys, time
from pathlib import Path

from kiss.agents.sorcar.web_use_tool import WebUseTool

profile, barrier, tag = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
Path(str(barrier) + "." + tag).write_text("ready")
while not barrier.exists():
    time.sleep(0.02)

tool = WebUseTool(user_data_dir=profile, headless=True)
try:
    tree = tool.go_to_url("data:text/html,<h1>profile lock</h1>")
    print("PROFILE=" + str(tool.effective_user_data_dir))
    print("OK=" + str("profile lock" in tree))
    time.sleep(3)
finally:
    tool.close()
'''

#: Driver that takes the shared inter-process file lock and records the
#: interval it held it, so overlapping intervals prove the lock is broken.
_LOCK_DRIVER = '''
import sys, time
from pathlib import Path

from kiss.agents.sorcar.useful_tools import _file_lock

lock_path, out_path, barrier = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
Path(str(barrier) + "." + out_path.name).write_text("ready")
while not barrier.exists():
    time.sleep(0.02)

with _file_lock(lock_path):
    start = time.time()
    time.sleep(1.0)
    out_path.write_text(f"{start} {time.time()}")
'''


def _spawn(script: Path, *args: str, env_home: Path) -> subprocess.Popen:
    """Start a real child interpreter running *script* with an isolated home."""
    import os

    env = os.environ.copy()
    env["KISS_HOME"] = str(env_home)
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _release_barrier(barrier: Path, tags: list[str]) -> None:
    """Wait until every child signalled readiness, then open the barrier."""
    deadline = time.time() + 60
    while time.time() < deadline:
        if all(Path(f"{barrier}.{tag}").exists() for tag in tags):
            barrier.write_text("go")
            return
        time.sleep(0.02)
    raise AssertionError("children never signalled readiness")


def test_file_lock_excludes_other_processes(tmp_path):
    """The shared helper really serialises two OS processes."""
    script = tmp_path / "lock_driver.py"
    script.write_text(_LOCK_DRIVER, encoding="utf-8")
    lock_path = tmp_path / "shared.lock"
    barrier = tmp_path / "barrier"
    outs = [tmp_path / "a.txt", tmp_path / "b.txt"]

    procs = [
        _spawn(script, str(lock_path), str(out), str(barrier), env_home=tmp_path)
        for out in outs
    ]
    _release_barrier(barrier, [out.name for out in outs])
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 0, stderr or stdout

    spans = []
    for out in outs:
        start, end = (float(v) for v in out.read_text().split())
        spans.append((start, end))
    spans.sort()
    assert spans[0][1] <= spans[1][0], f"lock holders overlapped: {spans}"


def test_two_processes_never_share_one_browser_profile(tmp_path):
    """Concurrent kiss processes must not open the same profile directory."""
    script = tmp_path / "browser_driver.py"
    script.write_text(_BROWSER_DRIVER, encoding="utf-8")
    profile = tmp_path / "browser_profile"
    barrier = tmp_path / "barrier"

    procs = [
        _spawn(script, str(profile), str(barrier), tag, env_home=tmp_path)
        for tag in ("one", "two")
    ]
    _release_barrier(barrier, ["one", "two"])

    profiles = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=300)
        assert proc.returncode == 0, stderr or stdout
        assert "OK=True" in stdout, stdout
        line = next(li for li in stdout.splitlines() if li.startswith("PROFILE="))
        profiles.append(line.split("=", 1)[1])

    assert len(set(profiles)) == 2, (
        f"both processes opened the same Chromium profile: {profiles}"
    )
    assert str(profile) in profiles, profiles
