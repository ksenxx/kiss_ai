# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the Mac installation package.

``scripts/build-macos-pkg.sh`` is run for real against this checkout and
the package it writes is taken apart with ``pkgutil``: its description
for the Installer, its payload and its scripts.  The package's own
``preinstall`` and ``postinstall``, as extracted from it, are then run
the way the Installer runs them (package path, payload directory, volume
and root as arguments), with a home directory of the test's own.

The payload in those runs is a directory holding a small ``install.sh``
that records how it was started and ends the way the scenario needs.
The real one downloads a gigabyte, installs the app into the home
directory and replaces the user's kiss-web daemon, whose launchd label
does not depend on ``$HOME``; it was run from the package on a Mac
without developer tools when the package was written.

Not run here:

* ``--sign`` with a certificate ``productbuild`` accepts and
  ``--notarize``: they need a Developer ID Installer certificate and
  notarization credentials issued by Apple.  The test passes ``--sign``
  and ``--keychain`` on to ``productbuild`` with an empty keychain and
  checks that it is the one that refuses.
* The dialog of ``alert.sh``: it would open on the screen of whoever
  runs the tests.  Every run here is a command-line install
  (``COMMAND_LINE_INSTALL``), which shows none.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import pytest

from kiss import __version__

# Windows has no fcntl and no pwd; the module is for macOS only (pytestmark below).
fcntl = pytest.importorskip("fcntl", reason="POSIX-only: flock on the update lock")
pwd = pytest.importorskip("pwd", reason="POSIX-only: the user's login shell")

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="pkgbuild, productbuild and pkgutil are part of macOS"
)

REPO = Path(__file__).resolve().parents[4]
BUILD = REPO / "scripts" / "build-macos-pkg.sh"
COMPONENT = "kiss-sorcar.pkg"


class Package(NamedTuple):
    """A built package: the file, what ``pkgutil --expand-full`` made of it, the build's output."""

    file: Path
    expanded: Path
    output: str

    @property
    def payload(self) -> Path:
        """The directory tree the Installer writes to ``~/.kiss/kiss_pkg``."""
        return self.expanded / COMPONENT / "Payload"

    @property
    def scripts(self) -> Path:
        """The directory of the package's preinstall and postinstall."""
        return self.expanded / COMPONENT / "Scripts"


def build(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the build script with *args*."""
    return subprocess.run(
        [str(BUILD), *args], capture_output=True, text=True, timeout=300, check=False
    )


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Package:
    """The package of this checkout, built and expanded once per module."""
    root = tmp_path_factory.mktemp("macos-pkg")
    file = root / "out" / "KISS-Sorcar.pkg"
    result = build("-o", str(file))
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run(["pkgutil", "--expand-full", str(file), str(root / "expanded")], check=True)
    return Package(file, root / "expanded", result.stdout)


def test_package_is_for_one_user_on_both_architectures(package: Package) -> None:
    """The Installer installs it into the home directory, natively on arm64 and Intel."""
    assert f"KISS Sorcar {__version__}" in package.output
    assert "It is not signed" in package.output
    distribution = ET.parse(package.expanded / "Distribution").getroot()
    domains = distribution.find("domains")
    assert domains is not None
    assert domains.attrib == {
        "enable_anywhere": "false",
        "enable_currentUserHome": "true",
        "enable_localSystem": "false",
    }
    options = distribution.find("options")
    assert options is not None
    assert options.get("hostArchitectures") == "arm64,x86_64"
    # The Installer runs this check before anything else; its refusal names
    # the macOS the package needs (seen in the Installer of macOS 13.7).
    check = distribution.find("installation-check")
    assert check is not None
    assert check.get("script") == "macos_is_new_enough()"
    script = distribution.findtext("script") or ""
    assert "compareVersions(system.version.ProductVersion, '14.0') >= 0" in script
    assert "KISS Sorcar needs macOS 14 or later, and this Mac runs macOS" in script
    refs = [ref for ref in distribution.iter("pkg-ref") if ref.get("version")]
    assert [(ref.get("id"), ref.get("version")) for ref in refs] == [
        ("org.kiss-sorcar.pkg", __version__)
    ]
    for text in ("welcome.txt", "conclusion.txt"):
        assert (package.expanded / "Resources" / text).read_text().strip()
    info = ET.parse(package.expanded / COMPONENT / "PackageInfo").getroot()
    assert info.get("install-location") == "/.kiss/kiss_pkg"
    assert info.get("identifier") == "org.kiss-sorcar.pkg"
    assert info.get("version") == __version__


def test_payload_is_the_app_with_its_installer(package: Package) -> None:
    """The payload is what install.sh installs, install.sh itself and the commit; no more."""
    payload = package.payload
    for script in ("install.sh", "scripts/install.sh", "rsorcar", "sorcar-docker"):
        assert os.access(payload / script, os.X_OK), script
    for name in ("pyproject.toml", "uv.lock", "Dockerfile", "src/kiss/__init__.py"):
        assert (payload / name).is_file(), name
    assert (payload / "install.sh").read_bytes() == (REPO / "install.sh").read_bytes()
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    assert (payload / "src/kiss/core/_commit.txt").read_text().strip() == head != ""
    assert not (payload / "src/kiss/tests").exists()
    assert sorted(p.name for p in (payload / "src/kiss/webapp").iterdir()) == ["media"]
    assert sorted(p.name for p in payload.iterdir()) == [
        "Dockerfile",
        "LICENSE",
        "README.md",
        "install.sh",
        "pyproject.toml",
        "rsorcar",
        "scripts",
        "sorcar-docker",
        "src",
        "uv.lock",
    ]
    # No AppleDouble entry ("._name", the extended attributes of the
    # machine that built the package) in the archive or its list of files.
    listed = subprocess.run(
        ["pkgutil", "--payload-files", str(package.file)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "./install.sh" in listed
    assert not [entry for entry in listed if "/._" in entry]
    assert not list(payload.rglob("._*"))
    assert len(listed) - 1 == sum(1 for _ in payload.rglob("*"))
    info = ET.parse(package.expanded / COMPONENT / "PackageInfo").getroot()
    count = info.find("payload")
    assert count is not None
    assert count.get("numberOfFiles") == str(len(listed))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--bogus"], "unknown argument: --bogus"),
        (["-o"], "-o needs a value"),
        (["--sign"], "--sign needs a value"),
        (["--keychain"], "--keychain needs a value"),
        (["--notarize"], "--notarize needs a value"),
        (["--notarize", "profile"], "--notarize needs --sign"),
        (["--keychain", "file"], "--keychain needs --sign"),
    ],
)
def test_bad_arguments_are_refused(args: list[str], message: str) -> None:
    """A wrong command line stops with its reason before anything is built."""
    result = build(*args)
    assert result.returncode == 1
    assert message in result.stderr
    assert "Built" not in result.stdout


def test_help_prints_the_usage() -> None:
    """``--help`` prints the options and builds nothing."""
    result = build("--help")
    assert result.returncode == 0
    assert result.stdout.startswith("Build the Mac installation package of KISS Sorcar.")
    for option in ("--output", "--sign", "--keychain", "--notarize", "--help"):
        assert option in result.stdout
    assert result.stdout.rstrip().endswith("print this text")


def test_sign_and_keychain_reach_productbuild(tmp_path: Path) -> None:
    """``productbuild`` looks for the named certificate in the named keychain."""
    keychain = tmp_path / "empty.keychain-db"
    subprocess.run(["security", "create-keychain", "-p", "test", str(keychain)], check=True)
    output = tmp_path / "signed.pkg"
    try:
        result = build("-o", str(output), "--sign", "No Such Identity", "--keychain", str(keychain))
    finally:
        subprocess.run(["security", "delete-keychain", str(keychain)], check=True)
    assert result.returncode != 0
    assert "No Such Identity" in result.stderr
    assert keychain.name in result.stderr
    assert not output.exists()


class Install(NamedTuple):
    """A home directory for one run of the package's scripts, with its payload."""

    home: Path
    payload: Path
    package: Package

    @property
    def status(self) -> Path:
        """Where postinstall records how install.sh ended."""
        return self.home / ".kiss" / ".pkg-install-status"

    def run(self, script: str, **env: str) -> subprocess.CompletedProcess[str]:
        """Run the package's *script* as the Installer of a command-line install does."""
        sandbox = self.home.parent / "PKInstallSandbox"
        sandbox.mkdir(exist_ok=True)
        return subprocess.run(
            [
                str(self.package.scripts / script),
                str(self.package.file),
                str(self.payload),
                "/",
                "/",
            ],
            env={
                "HOME": str(self.home),
                "PATH": "/bin:/sbin:/usr/bin:/usr/sbin:/usr/libexec",
                "TMPDIR": str(sandbox),
                "INSTALLER_TEMP": str(sandbox),
                "COMMAND_LINE_INSTALL": "1",
                **env,
            },
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def installer(self, ending: str) -> None:
        """Write the payload's install.sh: it records its start, then runs *ending*."""
        self.payload.mkdir(parents=True, exist_ok=True)
        (self.payload / "install.sh").write_text(
            "#!/bin/bash\n"
            '{ echo "arguments: $*"; echo "directory: $PWD"; env; } > "$HOME/install-sh.txt"\n'
            f"{ending}\n"
        )

    def record(self) -> dict[str, str]:
        """How the payload's install.sh was started: its arguments, directory and environment."""
        lines = (self.home / "install-sh.txt").read_text().splitlines()
        record = dict(line.split("=", 1) for line in lines[2:] if "=" in line)
        record["arguments"] = lines[0].removeprefix("arguments: ")
        record["directory"] = lines[1].removeprefix("directory: ")
        return record

    def wait_for_status(self, timeout: float = 30.0) -> str:
        """Return the status file's text once install.sh has ended."""
        deadline = time.monotonic() + timeout
        while not self.status.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        return self.status.read_text()


@pytest.fixture
def install(tmp_path: Path, package: Package) -> Install:
    """A fresh home directory whose payload directory does not exist yet."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    return Install(home, home / ".kiss" / "kiss_pkg", package)


def test_preinstall_removes_the_payload_of_the_version_before(install: Install) -> None:
    """Old payload files do not survive into the new version; other folders are left alone."""
    result = install.run("preinstall")  # a first install: no ~/.kiss at all
    assert result.returncode == 0, result.stderr
    install.installer("exit 0")
    (install.payload / "dropped-in-this-version.py").write_text("old")
    (install.home / ".kiss" / ".update.lock").touch()  # left by an install that has ended
    result = install.run("preinstall")
    assert result.returncode == 0, result.stderr
    assert not install.payload.exists()
    elsewhere = install.home / "elsewhere"
    elsewhere.mkdir()
    result = Install(install.home, elsewhere, install.package).run("preinstall")
    assert result.returncode == 0, result.stderr
    assert elsewhere.is_dir()


def test_preinstall_refuses_while_another_install_runs(install: Install) -> None:
    """The update lock held by a running install stops the package before it writes a file."""
    install.installer("exit 0")
    lock = install.home / ".kiss" / ".update.lock"
    with lock.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        result = install.run("preinstall")
    assert result.returncode == 1
    assert "being installed or updated right now" in result.stderr
    assert (install.payload / "install.sh").is_file()


def test_postinstall_runs_the_payload_installer(install: Install) -> None:
    """install.sh runs unattended, from the home directory, in an environment of its own."""
    install.installer("echo 'the remote-access password is hunter2'\nexit 0")
    result = install.run("postinstall", KISS_HOME="/elsewhere")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "KISS Sorcar is installed.\n"
    assert install.status.read_text() == "0\n\n"
    record = install.record()
    assert record["arguments"] == "--non-interactive"
    assert record["directory"] == str(install.home)
    assert record["HOME"] == str(install.home)
    assert record["KISS_NO_REVEAL"] == "1"
    # The user's login shell (zsh on a Mac): install.sh puts ~/.local/bin,
    # where the sorcar command is, on the PATH in that shell's rc file.
    assert record["SHELL"] == pwd.getpwuid(os.getuid()).pw_shell
    # Nothing of the Installer's environment: its temporary directory is
    # deleted when the script returns.
    assert os.path.isdir(record["TMPDIR"])
    assert "PKInstallSandbox" not in record["TMPDIR"]
    for name in ("INSTALLER_TEMP", "COMMAND_LINE_INSTALL", "KISS_HOME"):
        assert name not in record


def test_postinstall_keeps_the_directory_the_daemon_serves(install: Install) -> None:
    """Installing over a copy whose daemon serves a project leaves it serving that project."""
    install.installer("exit 0")
    project = install.home / "work" / "project"
    project.mkdir(parents=True)
    agent = install.home / "Library" / "LaunchAgents" / "com.kiss.web-server.plist"
    agent.parent.mkdir(parents=True)
    agent.write_bytes(plistlib.dumps({"WorkingDirectory": str(project)}))
    assert install.run("postinstall").returncode == 0
    assert install.record()["directory"] == str(project)
    project.rmdir()  # the project was deleted since: back to the home directory
    assert install.run("postinstall").returncode == 0
    assert install.record()["directory"] == str(install.home)


@pytest.mark.parametrize(
    ("ending", "status", "reason"),
    [
        # The error install.sh recorded for the daemon's notification.
        (
            'printf "WARNING: no brew\\nERROR: the disk is full\\n"'
            ' > "$HOME/.kiss/.install-problems"\nexit 3',
            3,
            "ERROR: the disk is full",
        ),
        # What it wrote before it had a log: another install holds the lock.
        (
            "echo noise >&2\necho 'another install is running' >&2\nexit 1",
            1,
            "another install is running",
        ),
        ("exit 7", 7, "install.sh stopped without saying why."),
    ],
)
def test_postinstall_reports_a_failed_install(
    install: Install, ending: str, status: int, reason: str
) -> None:
    """The Installer is told the install failed, with the reason install.sh gave."""
    install.installer(ending)
    result = install.run("postinstall")
    assert result.returncode == 1
    assert result.stderr == f"KISS Sorcar could not be installed: {reason}\n"
    assert install.status.read_text() == f"{status}\n{reason}\n"
    assert not list(install.status.parent.glob(".pkg-install-status.*"))


def test_postinstall_returns_before_the_installer_kills_it(install: Install) -> None:
    """Past its deadline postinstall returns and the install finishes on its own."""
    install.installer('while [ ! -e "$HOME/downloads-done" ]; do sleep 0.2; done\nexit 0')
    result = install.run("postinstall", KISS_PKG_WAIT_SECONDS="1")
    assert result.returncode == 0, result.stderr
    assert "still installing in the background" in result.stdout
    assert not install.status.exists()
    (install.home / "downloads-done").touch()
    assert install.wait_for_status() == "0\n\n"


def test_postinstall_without_a_payload_fails(install: Install) -> None:
    """A payload the Installer did not write is an error, not a silent success."""
    result = install.run("postinstall")
    assert result.returncode == 1
    assert "no install.sh in the package's payload" in result.stderr
    assert not install.status.exists()
