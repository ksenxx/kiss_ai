# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the Linux installer.

``scripts/build-linux-run.sh`` is run for real against this checkout
(it builds on macOS and Linux, with tar and gzip) and the installer it
writes is read back: a shell script followed by a gzip-compressed tar
archive, which must hold the payload the Mac package carries, as
``scripts/installer-payload.sh`` stages it.

That installer is then run for real, with ``sh``, in a home directory
of its own on a PATH of the tools it needs and no download tool: it
unpacks the payload into ``~/.kiss/kiss_pkg`` and the real
``install.sh`` starts, which stops at its first check, before it
installs anything.  How the installer hands over (the directory, the
arguments) and what it refuses is read from an installer the same
builder makes of a small checkout whose ``install.sh`` records how it
was started.

Not run here: a whole install.  It downloads uv, Python, the packages
and Chromium and restarts the user's kiss-web daemon;
``scripts/check-linux-run.sh`` runs one in a container.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import NamedTuple

import pytest

from kiss import __version__

# The whole module runs the bash build script and holds the installer's flock.
fcntl = pytest.importorskip("fcntl", reason="POSIX-only: bash build script and flock")

REPO = Path(__file__).resolve().parents[4]
BUILD = REPO / "scripts" / "build-linux-run.sh"
# What the installer runs, and install.sh up to its check for a download tool.
TOOLS = (
    "sh",
    "bash",
    "perl",
    "tar",
    "gzip",
    "tail",
    "rm",
    "mkdir",
    "mv",
    "uname",
    "dirname",
    "sed",
    "grep",
    "head",
    "date",
    "tee",
    "cat",
    "sleep",
    "tr",
)
# The install.sh of the small checkout records what it was run from and with,
# and how it stands with the update lock: whether another open of the lock
# file can lock it ("free") or not ("held"), whether it was told that it runs
# under the lock, the pid the lock file names and its own.
RECORDER = r"""
{ pwd; printf "%s\n" "$@"; } > "$HOME/started.txt"
perl -e 'use Fcntl qw(:flock); open(my $f, ">>", $ARGV[0]) or die;
         print flock($f, LOCK_EX | LOCK_NB) ? "free" : "held"' "$HOME/.kiss/.update.lock" \
    > "$HOME/lock.txt"
holder="$(cat "$HOME/.kiss/.update.lock")"
echo " ${KISS_UPDATE_LOCK_HELD:-untold} ${holder:-nobody} $$" >> "$HOME/lock.txt"
"""
STARTED = "started.txt"
EARLIER = "the payload of an earlier version\n"
UNREADABLE = "the archive in this file cannot be read: the download is incomplete or damaged"


class Installer(NamedTuple):
    """A built installer: the file and the build's output."""

    file: Path
    output: str


def build(*args: str, script: Path = BUILD) -> subprocess.CompletedProcess[str]:
    """Run the build script (of this checkout, or the copy *script*) with *args*.

    A build writes the installer and nothing else: it gets no home
    directory that it could write into.
    """
    bash = shutil.which("bash") or "bash"
    return subprocess.run(
        [bash, str(script), *args],
        env={**os.environ, "HOME": os.devnull},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def files(root: Path) -> dict[str, int]:
    """The files under *root* by relative path, with their sizes."""
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def payload(installer: Path) -> dict[str, int]:
    """The files of the archive that follows the installer's script, with their sizes."""
    data = installer.read_bytes()
    found = re.search(rb"^PAYLOAD_LINE=(\d+)$", data, re.MULTILINE)
    assert found is not None
    archive = data.split(b"\n", int(found.group(1)) - 1)[-1]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return {member.name.removeprefix("./"): member.size for member in tar if member.isfile()}


def small_checkout(root: Path, version_file: str = '__version__ = "1.2.3"\n') -> Path:
    """A small checkout at *root*: this checkout's builder, a tiny app and the recorder."""
    for name in (
        "scripts/build-linux-run.sh",
        "scripts/installer-payload.sh",
        "scripts/linux-run/header.sh",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, root / name)
    (root / "src/kiss/core").mkdir(parents=True)
    (root / "src/kiss/__init__.py").write_text("")
    (root / "src/kiss/core/_version.py").write_text(version_file)
    # installer-payload.sh lists the payload with install.sh's own list of the app's files.
    # The list ends at its closing marker: the builder evaluates the block, not the recorder.
    real = (REPO / "install.sh").read_text()
    end = real.index("\n", real.index("# END: kiss-app-files")) + 1
    (root / "install.sh").write_text(real[real.index("# BEGIN: kiss-app-files") : end] + RECORDER)
    return root


class Machine:
    """A user's machine: a home directory with an earlier payload, and a PATH of its own."""

    def __init__(self, root: Path) -> None:
        self.home = root / "home"
        self.work = root / "work"
        self.bin = root / "bin"
        self.payload = self.home / ".kiss" / "kiss_pkg"
        self.lock = self.home / ".kiss" / ".update.lock"
        for directory in (self.payload, self.work, self.bin):
            directory.mkdir(parents=True)
        (self.payload / "earlier.txt").write_text(EARLIER)
        for tool in TOOLS:
            (self.bin / tool).symlink_to(shutil.which(tool) or tool)

    def run(
        self, installer: Path, *args: str, shell: str = "sh", stdin: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run *installer* with *shell* from the work directory, or pipe it into the shell."""
        return subprocess.run(
            [str(self.bin / shell), *([] if stdin is not None else [str(installer)]), *args],
            cwd=self.work,
            env={"PATH": str(self.bin), "HOME": str(self.home)},
            stdin=stdin if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def refused(self, result: subprocess.CompletedProcess[str], message: str) -> None:
        """Assert the installer stopped with *message* and replaced nothing."""
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"KISS Sorcar: {message}" in result.stderr
        assert files(self.payload) == {"earlier.txt": len(EARLIER)}
        assert not (self.home / STARTED).exists()
        assert not Path(f"{self.payload}.new").exists()


@pytest.fixture(scope="module")
def installer(tmp_path_factory: pytest.TempPathFactory) -> Installer:
    """The installer of this checkout, built once per module."""
    file = tmp_path_factory.mktemp("linux-run") / "out" / "KISS-Sorcar-Linux.run"
    result = build("-o", str(file))
    assert result.returncode == 0, result.stdout + result.stderr
    return Installer(file, result.stdout)


@pytest.fixture(scope="module")
def small_installer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The installer of a small checkout, built once per module where the builder puts it."""
    root = small_checkout(tmp_path_factory.mktemp("small-checkout"))
    result = build(script=root / "scripts" / "build-linux-run.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "To install with it: bash KISS-Sorcar-Linux-1.2.3.run"
    return root / "dist" / "KISS-Sorcar-Linux-1.2.3.run"


@pytest.fixture
def machine(tmp_path: Path) -> Machine:
    """A machine of its own for one test."""
    return Machine(tmp_path)


def test_installer_is_a_script_followed_by_the_payload_of_the_mac_package(
    installer: Installer, tmp_path: Path
) -> None:
    """The file is an executable sh script; its archive holds what ``stage_payload`` stages."""
    assert f"KISS Sorcar {__version__}" in installer.output
    assert os.access(installer.file, os.X_OK)
    script = installer.file.read_bytes().split(b"\x1f\x8b", 1)[0].decode()
    assert script.startswith("#!/bin/sh\n")
    assert f"# KISS Sorcar {__version__} for Linux" in script
    assert not re.search(r"@[A-Z_]+@", script)
    staged = tmp_path / "staged"
    stage = (
        f"PROJECT_DIR={shlex.quote(str(REPO))}\n"
        'die() { echo "$*" >&2; exit 1; }\n'
        '. "$PROJECT_DIR/scripts/installer-payload.sh"\n'
        f"stage_payload {shlex.quote(str(staged))}\n"
    )
    subprocess.run(["bash", "-c", stage], check=True, capture_output=True, timeout=120)
    unpacked = payload(installer.file)
    assert unpacked == files(staged)
    for name in ("install.sh", "scripts/install.sh", "pyproject.toml", "src/kiss/__init__.py"):
        assert name in unpacked, name
    assert not [name for name in unpacked if name.startswith(("installers/", "src/kiss/tests/"))]


def test_run_unpacks_the_payload_and_starts_its_install_sh(
    installer: Installer, machine: Machine
) -> None:
    """The real installer replaces the earlier payload and the real install.sh takes over.

    The machine has no curl and no wget, which install.sh finds out first.
    """
    result = machine.run(installer.file, "--non-interactive")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"KISS Sorcar {__version__}: unpacking into {machine.payload}\n" in result.stdout
    assert "ERROR: curl or wget is required but neither was found." in result.stdout
    assert files(machine.payload) == payload(installer.file)
    assert not Path(f"{machine.payload}.new").exists()


@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_install_sh_gets_the_users_directory_the_arguments_and_the_lock(
    small_installer: Path, machine: Machine, shell: str
) -> None:
    """install.sh runs from the directory the installer was run from, with its arguments.

    It runs under the update lock the installer took before it replaced
    the payload, so that no other install replaces the payload under
    it, and is told so (it does not lock a second time).
    """
    result = machine.run(small_installer, "--non-interactive", "two words", shell=shell)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (machine.home / STARTED).read_text() == (
        f"{machine.work.resolve()}\n--non-interactive\ntwo words\n"
    )
    assert files(machine.payload) == payload(small_installer)
    state, told, holder, pid = (machine.home / "lock.txt").read_text().split()
    assert (state, told, holder) == ("held", "1", pid)


def test_damaged_download_is_refused(small_installer: Path, machine: Machine) -> None:
    """A file that was cut short stops before the earlier payload is touched."""
    cut = machine.work / "cut.run"
    cut.write_bytes(small_installer.read_bytes()[:-100])
    machine.refused(machine.run(cut), UNREADABLE)


def test_missing_tool_is_named(small_installer: Path, machine: Machine) -> None:
    """Without perl, which install.sh locks with, the installer says so."""
    (machine.bin / "perl").unlink()
    machine.refused(machine.run(small_installer), "'perl' is needed and was not found.")


def test_installer_piped_into_a_shell_is_refused(small_installer: Path, machine: Machine) -> None:
    """``sh < file`` cannot find the archive: the shell has no file to read it from."""
    with small_installer.open("rb") as stream:
        result = machine.run(small_installer, stdin=stream.fileno())
    machine.refused(result, UNREADABLE)


def test_run_is_refused_while_another_install_holds_the_lock(
    small_installer: Path, machine: Machine
) -> None:
    """While another install holds the update lock nothing is replaced; afterwards it is."""
    descriptor = os.open(machine.lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        machine.refused(machine.run(small_installer), "an install or update is running right now.")
    finally:
        os.close(descriptor)
    # The lock file stays behind, unlocked.
    result = machine.run(small_installer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (machine.home / STARTED).read_text() == f"{machine.work.resolve()}\n\n"


def test_help_prints_the_usage() -> None:
    """``-h`` prints the options and builds nothing."""
    result = build("-h")
    assert result.returncode == 0
    assert result.stdout.startswith("Build the Linux installer of KISS Sorcar.")
    for option in ("scripts/build-linux-run.sh [-o RUN]", "--output", "--help"):
        assert option in result.stdout
    assert result.stdout.rstrip().endswith("print this text")


@pytest.mark.parametrize(
    ("args", "message"),
    [(["--bogus"], "unknown argument: --bogus"), (["-o"], "-o needs a value")],
)
def test_bad_arguments_are_refused(args: list[str], message: str) -> None:
    """A wrong command line stops with its reason before anything is built."""
    result = build(*args)
    assert result.returncode == 1
    assert f"build-linux-run.sh: {message}" in result.stderr
    assert "Built" not in result.stdout


def test_checkout_without_a_version_builds_nothing(tmp_path: Path) -> None:
    """A version file that names no version stops the build."""
    root = small_checkout(tmp_path / "checkout", version_file="")
    result = build(script=root / "scripts" / "build-linux-run.sh")
    assert result.returncode == 1
    assert "build-linux-run.sh: no version found in" in result.stderr
    assert not (root / "dist").exists()
