# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The agents' temp directory, ``<OS temp dir>/kiss-tmp`` (:mod:`kiss.core.kiss_tmp`).

Every test runs against a real temp directory of its own: the fixture
points ``tempfile.tempdir`` at the test's ``tmp_path`` (the way
``TMPDIR`` would) and puts it back afterwards.

Branches no test here can reach:

* a ``kiss-tmp`` that belongs to another user (it takes a second
  account to make one; the two names being taken is covered with files);
* a machine with no boot id to read, and the boot id of the other
  operating systems (Linux's ``boot_id`` on macOS and the reverse; the
  Windows registry);
* Windows, where the directory is taken as it is.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss
from kiss.agents.sorcar.useful_tools import _clean_env
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.kiss_tmp import SUBAGENT_DIR_PREFIX, _boot_id, kiss_tmp_dir

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and modes")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Iterator[Path]:
    """Make the test's own folder the temp directory of this process for the test."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path)
    try:
        yield tmp_path.resolve()
    finally:
        tempfile.tempdir = previous


def test_directory_is_made_in_the_os_temp_dir_for_the_user_alone(temp_dir: Path) -> None:
    path = kiss_tmp_dir()
    assert path == temp_dir / "kiss-tmp"
    status = path.lstat()
    assert stat.S_ISDIR(status.st_mode)
    assert stat.S_IMODE(status.st_mode) == 0o700
    assert status.st_uid == os.getuid()
    assert kiss_tmp_dir() == path


def test_temp_dir_of_the_environment_is_followed(tmp_path: Path) -> None:
    """A process started with ``TMPDIR`` keeps its files under that directory, and with the
    short ``PATH`` of a cron job it still learns which boot this is."""
    env = {**os.environ, "TMPDIR": str(tmp_path), "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(Path(kiss.__file__).parents[1])}
    shown = subprocess.run(
        [sys.executable, "-c",
         "from kiss.core.kiss_tmp import kiss_tmp_dir; print(kiss_tmp_dir().as_posix())"],
        env=env, capture_output=True, text=True, timeout=120, check=True,
    ).stdout.strip()
    assert shown == (tmp_path.resolve() / "kiss-tmp").as_posix()
    assert (tmp_path / "kiss-tmp" / ".boot").is_file()


def test_command_of_a_sub_agent_names_the_directory_its_private_folder_is_in(
    temp_dir: Path,
) -> None:
    """Its temp directory is that folder: the agents, the test durations and the check cache
    get no second ``kiss-tmp`` in there, which would go with the sub-agent."""
    path = kiss_tmp_dir()
    private = tempfile.mkdtemp(prefix=SUBAGENT_DIR_PREFIX, dir=path)
    env = {**_clean_env(private_dir=private), "PYTHONPATH": str(Path(kiss.__file__).parents[1])}
    env.pop("KISS_TEST_DURATIONS", None)
    code = (
        "from kiss.core.kiss_tmp import kiss_tmp_dir\n"
        "from kiss.scripts import check, durations\n"
        "print(kiss_tmp_dir())\n"
        "print(durations.durations_path().parent)\n"
        "print(check.CACHE_FILE.parent)\n"
    )
    shown = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120,
        check=True,
    ).stdout.splitlines()
    assert shown == [str(path)] * 3
    assert os.listdir(private) == []


def test_content_of_the_current_boot_is_kept(temp_dir: Path) -> None:
    path = kiss_tmp_dir()
    boot = _boot_id()
    assert boot.startswith(("darwin:", "linux:"))
    assert (path / ".boot").read_text() == boot
    (path / "task1").mkdir()
    (path / "task1" / "PROGRESS.md").write_text("notes")
    assert kiss_tmp_dir() == path
    assert (path / "task1" / "PROGRESS.md").read_text() == "notes"


def test_first_use_after_a_reboot_empties_the_directory(temp_dir: Path) -> None:
    path = kiss_tmp_dir()
    (path / "task1" / "tool-out").mkdir(parents=True)
    (path / "task1" / "tool-out" / "a1-1-bash.txt").write_text("old")
    (path / "stray.log").write_text("old")
    kept_file = temp_dir / "kept.txt"
    kept_file.write_text("kept")
    kept_dir = temp_dir / "kept"
    kept_dir.mkdir()
    (kept_dir / "f").write_text("kept")
    (path / "file-link").symlink_to(kept_file)
    (path / "dir-link").symlink_to(kept_dir, target_is_directory=True)
    (path / ".boot").write_text("the boot before this one")

    assert kiss_tmp_dir() == path

    assert sorted(entry.name for entry in path.iterdir()) == [".boot", ".boot.lock"]
    assert (path / ".boot").read_text() == _boot_id()
    # A link is deleted, not followed.
    assert kept_file.read_text() == "kept"
    assert (kept_dir / "f").read_text() == "kept"


@pytest.mark.parametrize("marker", [None, b"", b"\xff\xfe not text"])
def test_directory_without_a_readable_marker_loses_nothing(
    temp_dir: Path, marker: bytes | None,
) -> None:
    """A new directory, or one the system emptied of its marker, belongs to no earlier boot."""
    path = kiss_tmp_dir()
    (path / "task1").mkdir()
    (path / "task1" / "PROGRESS.md").write_text("notes")
    if marker is None:
        (path / ".boot").unlink()
    else:
        (path / ".boot").write_bytes(marker)

    assert kiss_tmp_dir() == path

    assert (path / "task1" / "PROGRESS.md").read_text() == "notes"
    assert (path / ".boot").read_text() == _boot_id()


@pytest.mark.parametrize("unwritable", [".boot", ".boot.lock"])
def test_marker_or_lock_that_cannot_be_written_keeps_the_directory_usable(
    temp_dir: Path, unwritable: str,
) -> None:
    """Nothing in the directory names an earlier boot: nothing is deleted, now or later."""
    path = kiss_tmp_dir()
    (path / ".boot").unlink()
    if unwritable == ".boot":
        (path / ".boot").mkdir()
    else:
        (path / ".boot.lock").chmod(0o400)
    (path / "task1").mkdir()
    assert kiss_tmp_dir() == path
    assert (path / "task1").is_dir()


def test_marker_of_an_earlier_boot_that_cannot_be_replaced_is_deleted(temp_dir: Path) -> None:
    """Left in place, it would have the next call delete what this boot has made since."""
    path = kiss_tmp_dir()
    (path / "stray.log").write_text("old")
    (path / ".boot").write_text("the boot before this one")
    (path / ".boot").chmod(0o400)

    assert kiss_tmp_dir() == path
    assert not (path / ".boot").exists()
    assert not (path / "stray.log").exists()
    (path / "task1").mkdir()
    assert kiss_tmp_dir() == path

    assert (path / "task1").is_dir()
    assert (path / ".boot").read_text() == _boot_id()


def test_marker_that_cannot_be_replaced_is_gone_when_the_lock_is_released(temp_dir: Path) -> None:
    """Deleted after the lock, it would still name the earlier boot to a process that takes
    the lock in between, and that process would empty the directory of what the first one
    makes next.  The marker here takes milliseconds to read, which keeps such a gap open
    long enough for the test, waiting for the lock, to look into it."""
    import fcntl

    path = kiss_tmp_dir()
    marker = path / ".boot"
    marker.write_text("the boot before this one\n" * 1_000_000)
    marker.chmod(0o400)
    (path / "stray.log").write_text("old")
    caller = threading.Thread(target=kiss_tmp_dir)
    caller.start()
    try:
        with open(path / ".boot.lock", "rb") as lock:
            emptied = False
            while not emptied:
                fcntl.flock(lock, fcntl.LOCK_EX)  # Waits while the caller holds the lock.
                try:
                    emptied = not (path / "stray.log").exists()
                    assert not (emptied and marker.exists())
                    assert emptied or caller.is_alive()
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                time.sleep(0.001)
    finally:
        caller.join(timeout=60)
    assert not caller.is_alive()


def test_lock_that_cannot_be_taken_after_a_reboot_is_an_error(temp_dir: Path) -> None:
    """Without the lock the directory is neither emptied nor handed out as it is: a process
    that holds the lock may be emptying it, and a marker deleted now may be its new one."""
    path = kiss_tmp_dir()
    (path / "stray.log").write_text("old")
    (path / ".boot").write_text("the boot before this one")
    (path / ".boot.lock").chmod(0o400)

    with pytest.raises(PermissionError, match=r"kiss-tmp/\.boot\.lock"):
        kiss_tmp_dir()
    assert (path / "stray.log").read_text() == "old"
    assert (path / ".boot").read_text() == "the boot before this one"

    (path / ".boot.lock").chmod(0o600)
    assert kiss_tmp_dir() == path
    assert not (path / "stray.log").exists()
    assert (path / ".boot").read_text() == _boot_id()


@pytest.mark.skipif(not hasattr(os, "chflags"), reason="needs a file its owner cannot delete")
def test_marker_of_an_earlier_boot_that_cannot_be_deleted_either_is_an_error(
    temp_dir: Path,
) -> None:
    """A directory that the call after would empty again is not handed out."""
    path = kiss_tmp_dir()
    marker = path / ".boot"
    marker.write_text("the boot before this one")
    # ``os.chflags`` exists on BSD/macOS only: looked up by name so the type
    # checkers pass on every platform without a platform-dependent ignore.
    chflags = getattr(os, "chflags")
    chflags(marker, stat.UF_IMMUTABLE)
    try:
        with pytest.raises(PermissionError, match=r"kiss-tmp/\.boot"):
            kiss_tmp_dir()
    finally:
        chflags(marker, 0)


@pytest.mark.skipif(not hasattr(os, "chflags"), reason="needs a file its owner cannot delete")
def test_what_cannot_be_deleted_stays_for_the_next_reboot(temp_dir: Path) -> None:
    path = kiss_tmp_dir()
    held = path / "held.txt"
    held.write_text("old")
    (path / "stray.log").write_text("old")
    chflags = getattr(os, "chflags")  # BSD/macOS only, as above
    chflags(held, stat.UF_IMMUTABLE)
    try:
        (path / ".boot").write_text("the boot before this one")
        assert kiss_tmp_dir() == path
        assert held.read_text() == "old"
        assert not (path / "stray.log").exists()
        assert (path / ".boot").read_text() == _boot_id()
    finally:
        chflags(held, 0)


def test_processes_that_start_together_after_a_reboot_keep_each_others_files(
    tmp_path: Path,
) -> None:
    """One of them empties the directory; none deletes what another has made since."""
    base = tmp_path / "t"
    base.mkdir()
    env = {**os.environ, "TMPDIR": str(base), "PYTHONPATH": str(Path(kiss.__file__).parents[1])}
    first = subprocess.run(
        [sys.executable, "-c",
         "from kiss.core.kiss_tmp import kiss_tmp_dir; print(kiss_tmp_dir())"],
        env=env, capture_output=True, text=True, timeout=120, check=True,
    )
    path = Path(first.stdout.strip())
    (path / "old-task").mkdir()
    (path / "old-task" / "PROGRESS.md").write_text("old")
    (path / ".boot").write_text("the boot before this one")
    go = tmp_path / "go"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from kiss.core.kiss_tmp import kiss_tmp_dir\n"
        "while not Path(sys.argv[1]).exists():\n"
        "    time.sleep(0.005)\n"
        "(kiss_tmp_dir() / sys.argv[2]).write_text('new')\n"
    )
    children = [
        subprocess.Popen([sys.executable, "-c", script, str(go), f"task-{index}"], env=env)
        for index in range(6)
    ]
    go.write_text("")
    assert [child.wait(timeout=120) for child in children] == [0] * 6

    assert not (path / "old-task").exists()
    assert sorted(entry.name for entry in path.iterdir() if not entry.name.startswith(".")) == [
        f"task-{index}" for index in range(6)
    ]


def _empty_first(path: Path, boot: str, locked: threading.Event) -> None:
    """Play the process that got the marker's lock first: record *boot*, then make a file."""
    with exclusive_file_lock(path / ".boot.lock"):
        locked.set()
        time.sleep(0.3)  # The test is waiting for the lock by now.
        (path / ".boot").write_text(boot)
        (path / "made-since").write_text("new")


def test_marker_written_while_this_process_waited_for_the_lock_is_trusted(
    temp_dir: Path,
) -> None:
    """Whoever got the lock first has emptied the directory: the next one deletes nothing."""
    path = kiss_tmp_dir()
    (path / ".boot").write_text("the boot before this one")
    locked = threading.Event()
    first = threading.Thread(target=_empty_first, args=(path, _boot_id(), locked))
    first.start()
    try:
        assert locked.wait(timeout=60)
        assert kiss_tmp_dir() == path
    finally:
        first.join(timeout=60)
    assert (path / "made-since").read_text() == "new"


def test_link_planted_under_the_name_is_not_followed(temp_dir: Path) -> None:
    elsewhere = temp_dir / "elsewhere"
    elsewhere.mkdir()
    (temp_dir / "kiss-tmp").symlink_to(elsewhere, target_is_directory=True)
    path = kiss_tmp_dir()
    assert path == temp_dir / f"kiss-tmp-{os.getuid()}"
    assert stat.S_IMODE(path.lstat().st_mode) == 0o700
    assert list(elsewhere.iterdir()) == []


def test_file_under_the_name_is_passed_over(temp_dir: Path) -> None:
    (temp_dir / "kiss-tmp").write_text("not a directory")
    assert kiss_tmp_dir() == temp_dir / f"kiss-tmp-{os.getuid()}"
    assert (temp_dir / "kiss-tmp").read_text() == "not a directory"


def test_both_names_taken_is_an_error_that_names_the_way_out(temp_dir: Path) -> None:
    (temp_dir / "kiss-tmp").write_text("taken")
    (temp_dir / f"kiss-tmp-{os.getuid()}").symlink_to(temp_dir / "nowhere")
    with pytest.raises(PermissionError, match="point TMPDIR at a directory of your own"):
        kiss_tmp_dir()


def test_directory_others_could_enter_is_closed_to_them(temp_dir: Path) -> None:
    (temp_dir / "kiss-tmp").mkdir()
    (temp_dir / "kiss-tmp").chmod(0o755)
    path = kiss_tmp_dir()
    assert path == temp_dir / "kiss-tmp"
    assert stat.S_IMODE(path.lstat().st_mode) == 0o700


def test_directory_others_can_write_to_is_passed_over(temp_dir: Path) -> None:
    """What is in it may be theirs."""
    (temp_dir / "kiss-tmp").mkdir()
    (temp_dir / "kiss-tmp").chmod(0o777)
    assert kiss_tmp_dir() == temp_dir / f"kiss-tmp-{os.getuid()}"
    assert stat.S_IMODE((temp_dir / "kiss-tmp").lstat().st_mode) == 0o777
