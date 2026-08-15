# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the trajectory/YAML findings F5 and F6.

* **F5** — importing ``kiss.core.base`` called ``yaml.add_representer``
  without a ``Dumper=``, which mutates PyYAML's *process-global*
  ``yaml.Dumper``.  Every ``yaml.dump`` in the interpreter — including
  third-party code that merely shares the process — changed shape
  because of an unrelated KISS import, and the literal block style was
  applied to every string, keys and one-liners included.
* **F6** — ``Base._save`` truncated the trajectory with ``open("w")``
  and then streamed the dump into it, so the trajectory visualizer
  (which reads the same file out of the jobs root) could observe an
  empty or half-written document.

Real agents, real files, real threads and a real fresh subprocess.  No
mocks, patches, fakes or test doubles, and no LLM calls.
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

import kiss.core.config as config_module
from kiss.core.base import Base
from kiss.core.utils import _try_chmod, atomic_write_text, finish

# The alphabetically last top-level key of a trajectory document: a
# reader that cannot see it is looking at a truncated file.
_TRAILING_KEY = "\ntotal_budget:"

_ISOLATION_SCRIPT = """
import json
import yaml

sample = {"one_line": "svc", "multi_line": "a\\nb\\n"}
before = yaml.dump(sample)
import kiss.core.base  # noqa: F401
after = yaml.dump(sample)
print(json.dumps([before, after]))
"""


@pytest.fixture
def artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect this process's artifact directory into a temp tree."""
    job_dir = tmp_path / "jobs" / "job_test"
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(config_module, "_artifact_dir", str(job_dir))
    yield job_dir


def _large_agent(messages: int) -> Base:
    """Return a real agent carrying a trajectory big enough to tear."""
    agent = Base("trajectory writer")
    agent.model_name = "unknown-test-model"
    for index in range(messages):
        agent._add_message("user", f"message {index}: " + "x" * 400)
    return agent


def test_importing_kiss_does_not_change_global_yaml_dump() -> None:
    """F5: a fresh interpreter's ``yaml.dump`` output must survive the import."""
    completed = subprocess.run(
        [sys.executable, "-c", _ISOLATION_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    before, after = json.loads(completed.stdout)
    assert before == after, f"global yaml.dump changed:\n{before!r}\n{after!r}"


def test_trajectory_yaml_still_uses_literal_blocks_for_multiline(
    artifact_dir: Path,
) -> None:
    """F5: the readable trajectory formatting is kept, just no longer global.

    Multi-line values (prompts, tool results) stay literal blocks so the
    saved trajectory is diffable and human-readable; single-line values
    and keys are plain scalars again instead of being wrapped in ``|-``.
    """
    agent = Base("literal blocks")
    agent._add_message("user", "first line\nsecond line\n")
    agent._save()

    raw = agent.get_trajectory_path().read_text(encoding="utf-8")
    assert "content: |" in raw
    assert "  first line\n" in raw
    assert "name: literal blocks\n" in raw
    assert yaml.safe_load(raw)["messages"][0]["content"] == "first line\nsecond line\n"


def test_finish_result_keeps_its_literal_block_summary() -> None:
    """F5: ``finish()`` must not depend on whether ``base`` was imported.

    Its YAML is handed to the parent agent verbatim, so the readable
    block-scalar shape is part of the contract and has to be requested
    explicitly rather than inherited from a global side effect.
    """
    dumped = finish(True, summary_in_html="<h3>Done</h3>\n<p>All good.</p>\n")

    assert "summary: |" in dumped
    parsed = yaml.safe_load(dumped)
    assert parsed["success"] is True
    assert parsed["summary"] == "<h3>Done</h3>\n<p>All good.</p>\n"


def test_a_failed_write_leaves_the_previous_file_and_no_debris(
    tmp_path: Path,
) -> None:
    """F6: if the write fails, the old content survives and no temp file is left.

    Staging in a temp file means a failure mid-write cannot damage what
    is already on disk — the opposite of ``open("w")``, which destroys
    the previous content before the first byte is produced.
    """
    target = tmp_path / "trajectory.yaml"
    atomic_write_text(target, "first: complete\n")

    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, "lone surrogate: \ud800\n")

    assert target.read_text(encoding="utf-8") == "first: complete\n"
    assert list(tmp_path.iterdir()) == [target]


_SHORT_WRITE_PROBE = """
import resource
import signal
import sys
from pathlib import Path

from kiss.core.utils import atomic_write_text

target = Path(sys.argv[1])
payload = "A" * 4096

# POSIX lets write(2) succeed with a partial count.  A real RLIMIT_FSIZE
# is the portable way to provoke one: with SIGXFSZ ignored the kernel
# writes up to the limit and reports how far it got instead of killing
# the process.
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
try:
    atomic_write_text(target, payload)
except OSError as e:
    print(f"RAISED {type(e).__name__}")
else:
    print("PUBLISHED")
"""


def test_a_short_write_never_publishes_truncated_content(tmp_path: Path) -> None:
    """F6: a partial ``write(2)`` must fail loudly, not publish a stub.

    ``os.write`` may return fewer bytes than it was given.  Ignoring
    that count and then ``os.replace``-ing the staged file turns a
    transient quota/limit/signal condition into a *permanently*
    truncated trajectory that looks perfectly intact — the very damage
    the atomic write exists to prevent.

    The limit is applied in a real child process because it is
    process-wide and irreversible once lowered.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(_SHORT_WRITE_PROBE, encoding="utf-8")
    target = tmp_path / "trajectory.yaml"
    atomic_write_text(target, "first: complete\n")

    result = subprocess.run(
        [sys.executable, str(probe), str(target)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert result.stdout.strip().startswith("RAISED"), (
        f"a short write was reported as success: {result.stdout!r}"
    )
    assert target.read_text(encoding="utf-8") == "first: complete\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["probe.py", "trajectory.yaml"]


def test_atomic_write_applies_the_requested_mode(tmp_path: Path) -> None:
    """F6: the secure variant used for shell RC files keeps 0600."""
    target = tmp_path / "rc"
    atomic_write_text(target, "export X=1\n", mode=0o600)

    assert target.read_text(encoding="utf-8") == "export X=1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_chmod_failure_does_not_break_the_write(tmp_path: Path) -> None:
    """F6: an unchmod-able path degrades instead of raising.

    Some filesystems (and Windows) refuse or ignore permission changes;
    that must never turn a successful write into an exception.
    """
    _try_chmod(str(tmp_path / "does-not-exist"), 0o600)


def test_a_reader_never_sees_a_partial_trajectory(artifact_dir: Path) -> None:
    """F6: concurrent trajectory reads must never observe a torn document.

    ``KISSAgent.run`` saves from a ``finally`` on every run while the
    server serves the very same file to the trajectory visualizer.
    """
    agent = _large_agent(messages=900)
    path = agent.get_trajectory_path()
    agent._save()
    assert path.stat().st_size > 200_000, "trajectory too small to tear"

    stop = threading.Event()
    bad: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                bad.append("trajectory disappeared")
                continue
            if not raw:
                bad.append("empty document")
            elif _TRAILING_KEY not in raw:
                bad.append(f"truncated document ({len(raw)} bytes)")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(15):
            agent._save()
    finally:
        stop.set()
        thread.join(timeout=60)

    assert not bad, bad[:3]
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == "trajectory writer"


def test_concurrent_saves_leave_one_complete_trajectory(artifact_dir: Path) -> None:
    """F6: two agent threads saving at once must not interleave their bytes."""
    agents = [_large_agent(messages=300) for _ in range(2)]
    for agent in agents:
        agent.model_name = f"model-{agent.id}"

    def save_repeatedly(agent: Base) -> None:
        for _ in range(10):
            agent._save()

    threads = [threading.Thread(target=save_repeatedly, args=(a,)) for a in agents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive()

    for agent in agents:
        loaded = yaml.safe_load(agent.get_trajectory_path().read_text(encoding="utf-8"))
        assert loaded["model"] == f"model-{agent.id}"
        assert len(loaded["messages"]) == 300
