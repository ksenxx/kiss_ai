# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``kiss-runtime-setup`` block of ``install.sh``.

That block finishes, without a VS Code window, the runtime setup the
extension's ``DependencyInstaller`` otherwise performs on first
activation: uv, ``uv sync`` of every installed copy of the bundled
``kiss_project``, the ``sorcar`` CLI wrapper, Playwright Chromium,
cloudflared, ``~/.local/bin`` on PATH, and the kiss-web daemon as the
user's service.

Every test extracts the block verbatim from ``install.sh`` (between
``# BEGIN: kiss-runtime-setup`` and the matching END marker) and runs it
under a real bash with a sandboxed ``$HOME`` and a PATH made of stub
executables (``uv``, ``curl``, ``systemctl``, ``launchctl``, ``lsof``,
``fuser``, ``sudo``, ``uname``, ``brew``) that record their calls, plus
symlinks to the real coreutils.  The "daemon" started by the service
stubs is a real process that writes the endpoint file and listens on a
port, exactly as kiss-web does, so the block's "is it up" logic is
exercised for real.  The active-task probe talks to a real fake daemon
over ``wss://`` (``kiss.tests.local_ws.fake_daemon``).

Branches that need root (``playwright install-deps`` run as uid 0) are
not reachable without privileges and are not covered.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path

import pytest
from websockets.asyncio.server import ServerConnection

from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import fake_daemon

pytestmark = posix_only("runs the runtime-setup block of install.sh under bash")

REPO = Path(__file__).resolve().parents[3]
INSTALL = REPO / "install.sh"
VERSION = "9.9.9"
_BEGIN = "# BEGIN: kiss-runtime-setup"
_END = "# END: kiss-runtime-setup"
# Real utilities the block needs; everything else on PATH is a stub.
_TOOLS = (
    "bash",
    "sh",
    "cat",
    "sleep",
    "ps",
    "id",
    "uname",
    "tr",
    "sed",
    "grep",
    "tail",
    "head",
    "mkdir",
    "mv",
    "rm",
    "chmod",
    "dirname",
    "nohup",
    "tar",
    "env",
    "sort",
    "touch",
    "gzip",
    "true",
    "false",
    "find",
)


def _runtime_block() -> str:
    """Return the verbatim ``kiss-runtime-setup`` block of ``install.sh``."""
    src = INSTALL.read_text(encoding="utf-8")
    assert _BEGIN in src, f"install.sh missing '{_BEGIN}'"
    begin = src.index("\n", src.index(_BEGIN)) + 1
    return src[begin : src.index(_END, begin)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Sandbox:
    """A fake ``$HOME``, a stub PATH and a bash runner for the block."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.stubs = tmp_path / "stubs"
        self.stubs.mkdir()
        self.tools = tmp_path / "tools"
        self.tools.mkdir()
        for tool in _TOOLS:
            real = shutil.which(tool)
            if real:
                (self.tools / tool).symlink_to(real)
        self.calls = tmp_path / "calls.log"
        self.pids = tmp_path / "daemon.pids"
        self.port = _free_port()
        self.workdir = tmp_path / "work"
        self.workdir.mkdir()

    def stub(self, name: str, body: str, log: bool = True) -> Path:
        """Install an executable bash stub; with *log* it records its argv in ``calls``."""
        path = self.stubs / name
        logline = f'echo "{name} $*" >> {str(self.calls)!r}\n' if log else ""
        path.write_text("#!/bin/bash\n" + logline + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)
        return path

    def uname(self, system: str, machine: str = "x86_64") -> None:
        """Pin what ``uname -s`` / ``uname -m`` report (never the real host's)."""
        self.stub("uname", f"case $1 in -s) echo {system};; -m) echo {machine};; esac\n", log=False)

    def logged_calls(self) -> list[str]:
        if not self.calls.exists():
            return []
        return self.calls.read_text(encoding="utf-8").splitlines()

    def project(self, root: str, version: str = VERSION, venv: bool = True) -> Path:
        """Create an installed ``kiss_project`` copy under extension root *root*.

        With *venv*, ``.venv/bin/python`` is the test's own interpreter
        (it imports ``kiss``) and ``.venv/bin/kiss-web`` is a stand-in
        daemon: it records its working directory, writes the endpoint
        file under its home and listens on ``$KISS_WEB_PORT``.
        """
        project = self.home / root / f"ksenxx.kiss-sorcar-{version}" / "kiss_project"
        (project / "src" / "kiss").mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname='kiss'\n", encoding="utf-8")
        (project / "src" / "kiss" / "a.py").write_text("x = 1\n", encoding="utf-8")
        if venv:
            bin_dir = project / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            python = bin_dir / "python"
            python.write_text(f'#!/bin/bash\nexec {sys.executable!r} "$@"\n', encoding="utf-8")
            python.chmod(0o755)
            daemon = bin_dir / "kiss-web"
            daemon.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/bash
                    home="${{KISS_HOME:-$HOME/.kiss}}"
                    mkdir -p "$home"
                    echo "$PWD" > "$home/kiss-web.cwd"
                    echo "$$" >> {str(self.pids)!r}
                    printf '{{"url":"wss://127.0.0.1:1/","token":"t"}}' > "$home/sorcar-local.json"
                    exec "$(dirname "$0")/python" -c 'import os, socket, time
                    s = socket.socket()
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", int(os.environ["KISS_WEB_PORT"])))
                    s.listen()
                    time.sleep(120)'
                    """
                ),
                encoding="utf-8",
            )
            daemon.chmod(0o755)
        return project

    def run(
        self,
        commands: str,
        env: dict[str, str] | None = None,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        """Run *commands* after the block, under bash, in the sandbox."""
        script = (
            "set -eo pipefail\n"
            f"export HOME={str(self.home)!r}\n"
            f"export PATH={str(self.stubs)!r}:{str(self.tools)!r}\n"
            f"PROJECT_DIR={str(REPO)!r}\n"
            f"export KISS_WEB_PORT={self.port}\n"
            'export KISS_WEB_START_TIMEOUT="${KISS_WEB_START_TIMEOUT:-20}"\n'
            + _runtime_block()
            + "\n"
            + textwrap.dedent(commands)
        )
        full_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("KISS_HOME", "KISS_SORCAR_LOCAL", "KISS_WEB_PORT")
        }
        full_env["SHELL"] = "/bin/bash"
        full_env.update(env or {})
        return subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
            check=False,
        )

    def kill_daemons(self) -> None:
        if not self.pids.exists():
            return
        for line in self.pids.read_text(encoding="utf-8").split():
            try:
                os.kill(int(line), 9)
            except (OSError, ValueError):
                pass


@pytest.fixture
def sandbox(tmp_path: Path):
    box = Sandbox(tmp_path)
    try:
        yield box
    finally:
        box.kill_daemons()


def _wait_port(port: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------
# installed_kiss_projects
# --------------------------------------------------------------------------


def test_installed_kiss_projects_lists_every_copy_desktop_first(sandbox: Sandbox) -> None:
    server = sandbox.project(".vscode-server/extensions", venv=False)
    code_server = sandbox.project(".local/share/code-server/extensions", venv=False)
    desktop = sandbox.project(".vscode/extensions", venv=False)
    # An extension directory without the bundled project is not a copy.
    (
        sandbox.home
        / ".vscode-insiders/extensions"
        / f"ksenxx.kiss-sorcar-{VERSION}"
        / "kiss_project"
    ).mkdir(parents=True)
    sandbox.project(".vscode-oss/extensions", version="1.0.0", venv=False)
    res = sandbox.run(f'installed_kiss_projects "{VERSION}"')
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == [str(desktop), str(code_server), str(server)]
    res = sandbox.run('installed_kiss_projects ""; echo "rc=$?"')
    assert res.stdout == "rc=0\n"


# --------------------------------------------------------------------------
# ensure_uv
# --------------------------------------------------------------------------


def test_ensure_uv_prefers_the_uv_on_path(sandbox: Sandbox) -> None:
    uv = sandbox.stub("uv", "")
    res = sandbox.run("ensure_uv")
    assert res.stdout.strip() == str(uv)
    assert "Installing uv" not in res.stderr


def test_ensure_uv_finds_the_local_bin_copy(sandbox: Sandbox) -> None:
    local = sandbox.home / ".local" / "bin" / "uv"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/bash\n", encoding="utf-8")
    local.chmod(0o755)
    res = sandbox.run("ensure_uv")
    assert res.stdout.strip() == str(local)


def test_ensure_uv_installs_with_the_official_installer(sandbox: Sandbox) -> None:
    """``curl … | sh`` runs the installer; its PATH edits are turned off."""
    installer = sandbox.root / "installer.sh"
    env_file = sandbox.root / "installer.env"
    installer.write_text(
        textwrap.dedent(
            f"""\
            mkdir -p "$HOME/.local/bin"
            printf '#!/bin/bash\\n' > "$HOME/.local/bin/uv"
            chmod 755 "$HOME/.local/bin/uv"
            echo "modify_path=${{UV_NO_MODIFY_PATH:-unset}}" > {str(env_file)!r}
            """
        ),
        encoding="utf-8",
    )
    sandbox.stub("curl", f"cat {str(installer)!r}\n")
    res = sandbox.run("ensure_uv")
    assert res.stdout.strip() == str(sandbox.home / ".local/bin/uv")
    assert "Installing uv" in res.stderr
    assert "curl -LsSf https://astral.sh/uv/install.sh" in sandbox.logged_calls()[0]
    assert (sandbox.root / "installer.env").read_text() == "modify_path=1\n"


def test_ensure_uv_fails_when_the_download_fails(sandbox: Sandbox) -> None:
    sandbox.stub("curl", "exit 22\n")
    res = sandbox.run('ensure_uv || echo "rc=$?"')
    assert res.stdout == "rc=1\n"
    assert not (sandbox.home / ".local/bin/uv").exists()


def test_ensure_uv_fails_when_the_installer_leaves_no_binary(sandbox: Sandbox) -> None:
    sandbox.stub("curl", "echo 'true'\n")
    res = sandbox.run('ensure_uv || echo "rc=$?"')
    assert res.stdout == "rc=1\n"


# --------------------------------------------------------------------------
# add_local_bin_to_shell_rc
# --------------------------------------------------------------------------


def test_shell_rc_gets_local_bin_once(sandbox: Sandbox) -> None:
    rc = sandbox.home / ".bashrc"
    rc.write_text("alias ll='ls -l'", encoding="utf-8")  # no trailing newline
    res = sandbox.run("add_local_bin_to_shell_rc; add_local_bin_to_shell_rc")
    assert res.returncode == 0, res.stderr
    assert rc.read_text(encoding="utf-8") == (
        "alias ll='ls -l'\nexport PATH=\"$HOME/.local/bin:$PATH\"\n"
    )
    assert res.stdout.count("Added ~/.local/bin to PATH") == 1


def test_shell_rc_existing_tilde_entry_is_respected(sandbox: Sandbox) -> None:
    rc = sandbox.home / ".zshrc"
    rc.write_text('export PATH="~/.local/bin:$PATH"\n', encoding="utf-8")
    res = sandbox.run("add_local_bin_to_shell_rc", env={"SHELL": "/bin/zsh"})
    assert res.stdout == ""
    assert rc.read_text(encoding="utf-8") == 'export PATH="~/.local/bin:$PATH"\n'


def test_shell_rc_fish_uses_fish_add_path(sandbox: Sandbox) -> None:
    res = sandbox.run(
        "add_local_bin_to_shell_rc; add_local_bin_to_shell_rc", env={"SHELL": "/usr/bin/fish"}
    )
    rc = sandbox.home / ".config" / "fish" / "config.fish"
    assert rc.read_text(encoding="utf-8") == 'fish_add_path "$HOME/.local/bin"\n'
    assert res.stdout.count("Added ~/.local/bin to PATH") == 1


# --------------------------------------------------------------------------
# install_cloudflared
# --------------------------------------------------------------------------


def test_cloudflared_on_path_or_in_local_bin_is_kept(sandbox: Sandbox) -> None:
    sandbox.stub("cloudflared", "")
    assert sandbox.run("install_cloudflared").stdout == ""
    os.remove(sandbox.stubs / "cloudflared")
    local = sandbox.home / ".local/bin/cloudflared"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/bash\n", encoding="utf-8")
    local.chmod(0o755)
    assert sandbox.run("install_cloudflared").stdout == ""
    assert "curl" not in "".join(sandbox.logged_calls())


def test_cloudflared_linux_download(sandbox: Sandbox) -> None:
    sandbox.uname("Linux", "x86_64")
    sandbox.stub("curl", 'while [ $# -gt 1 ]; do [ "$1" = -o ] && echo bin > "$2"; shift; done\n')
    res = sandbox.run("install_cloudflared")
    assert "Installing cloudflared" in res.stdout
    target = sandbox.home / ".local/bin/cloudflared"
    assert target.read_text() == "bin\n"
    assert target.stat().st_mode & 0o777 == 0o755
    assert any("cloudflared-linux-amd64" in c for c in sandbox.logged_calls())


def test_cloudflared_linux_download_failure_is_a_warning(sandbox: Sandbox) -> None:
    sandbox.uname("Linux", "aarch64")
    sandbox.stub("curl", 'shift; : > "$2"; exit 22\n')  # curl -o leaves an empty file
    res = sandbox.run("install_cloudflared; echo rc=$?")
    assert "WARNING: cloudflared download failed" in res.stdout
    assert res.stdout.endswith("rc=0\n")
    assert any("cloudflared-linux-arm64" in c for c in sandbox.logged_calls())
    assert not (sandbox.home / ".local/bin/cloudflared").exists()


def test_cloudflared_unsupported_architecture(sandbox: Sandbox) -> None:
    sandbox.uname("Linux", "riscv64")
    res = sandbox.run("install_cloudflared")
    assert "unsupported architecture riscv64" in res.stdout
    assert not any(c.startswith("curl") for c in sandbox.logged_calls())


def test_cloudflared_macos_prefers_homebrew(sandbox: Sandbox) -> None:
    sandbox.uname("Darwin", "arm64")
    sandbox.stub("brew", "")
    res = sandbox.run("install_cloudflared")
    assert res.returncode == 0, res.stderr
    assert "brew install cloudflared" in sandbox.logged_calls()
    assert not any(c.startswith("curl") for c in sandbox.logged_calls())


def test_cloudflared_macos_falls_back_to_the_release_tarball(sandbox: Sandbox) -> None:
    sandbox.uname("Darwin", "x86_64")
    sandbox.stub("brew", "exit 1\n")
    payload = sandbox.root / "cloudflared"
    payload.write_text("#!/bin/bash\n", encoding="utf-8")
    tgz = sandbox.root / "cloudflared.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(payload, arcname="cloudflared")
    sandbox.stub("curl", f"cat {str(tgz)!r}\n")
    res = sandbox.run("install_cloudflared")
    assert res.returncode == 0, res.stderr
    target = sandbox.home / ".local/bin/cloudflared"
    assert target.read_text() == "#!/bin/bash\n"
    assert target.stat().st_mode & 0o777 == 0o755
    assert any("cloudflared-darwin-amd64.tgz" in c for c in sandbox.logged_calls())


def test_cloudflared_macos_tarball_failure_is_a_warning(sandbox: Sandbox) -> None:
    sandbox.uname("Darwin", "arm64")
    sandbox.stub("curl", "exit 22\n")
    res = sandbox.run("install_cloudflared; echo rc=$?")
    assert "WARNING: cloudflared download failed" in res.stdout
    assert res.stdout.endswith("rc=0\n")


# --------------------------------------------------------------------------
# setup_bundled_runtime
# --------------------------------------------------------------------------

_UV_STUB = """\
case "$1" in
    sync) touch .synced; [ -f .sync-fails ] && exit 1; exit 0 ;;
    run) [ -f .playwright-fails ] && exit 1; exit 0 ;;
esac
"""


def test_setup_without_an_installed_copy_is_a_no_op(sandbox: Sandbox) -> None:
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"; echo "primary=$KISS_PRIMARY_PROJECT"')
    assert "No installed copy of the extension found" in res.stdout
    assert res.stdout.endswith("primary=\n")


def test_setup_without_uv_warns_and_continues(sandbox: Sandbox) -> None:
    sandbox.project(".vscode/extensions", venv=False)
    sandbox.stub("curl", "exit 22\n")
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"; echo "primary=$KISS_PRIMARY_PROJECT"')
    assert "WARNING: uv is not available" in res.stdout
    assert res.stdout.endswith("primary=\n")


def test_setup_syncs_every_copy_and_installs_the_cli(sandbox: Sandbox) -> None:
    """Both copies are synced; the CLI binds to the first that succeeded."""
    desktop = sandbox.project(".vscode/extensions", venv=False)
    server = sandbox.project(".vscode-server/extensions", venv=False)
    (desktop / ".sync-fails").touch()
    uv = sandbox.stub("uv", _UV_STUB)
    sandbox.stub("sudo", "")
    sandbox.stub("cloudflared", "")
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"; echo "primary=$KISS_PRIMARY_PROJECT"')
    assert res.returncode == 0, res.stderr
    assert (desktop / ".synced").exists() and (server / ".synced").exists()
    assert f"WARNING: uv sync failed in {desktop}" in res.stdout
    assert res.stdout.endswith(f"primary={server}\n")
    wrapper = sandbox.home / ".local/bin/sorcar"
    assert wrapper.read_text(encoding="utf-8") == (
        "#!/bin/bash\n"
        "# Installed by KISS Sorcar VS Code extension\n"
        'export KISS_WORKDIR="$PWD"\n'
        f'exec "{uv}" run --directory "{server}" sorcar "$@"\n'
    )
    assert wrapper.stat().st_mode & 0o777 == 0o755
    assert 'export PATH="$HOME/.local/bin:$PATH"' in (sandbox.home / ".bashrc").read_text()
    calls = sandbox.logged_calls()
    assert "uv run python -m playwright install chromium" in calls
    if platform.system() == "Linux":
        assert "sudo -n true" in calls
        assert "uv run python -m playwright install-deps chromium" in calls


def test_setup_uses_the_brand_product_name_in_the_cli_comment(sandbox: Sandbox) -> None:
    sandbox.project(".vscode/extensions", venv=False)
    sandbox.stub("uv", _UV_STUB)
    sandbox.stub("sudo", "exit 1\n")
    sandbox.stub("cloudflared", "")
    res = sandbox.run(f'BRAND_PRODUCT_NAME="Seamless Loop"; setup_bundled_runtime "{VERSION}"')
    assert res.returncode == 0, res.stderr
    assert "# Installed by Seamless Loop VS Code extension" in (
        sandbox.home / ".local/bin/sorcar"
    ).read_text(encoding="utf-8")


@pytest.mark.skipif(platform.system() != "Linux", reason="install-deps is Linux-only")
def test_setup_hints_at_install_deps_when_sudo_would_prompt(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions", venv=False)
    sandbox.stub("uv", _UV_STUB)
    sandbox.stub("sudo", "exit 1\n")
    sandbox.stub("cloudflared", "")
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"')
    assert f"sudo {project}/.venv/bin/python -m playwright install-deps chromium" in res.stdout
    assert "install-deps" not in "".join(c for c in sandbox.logged_calls() if c.startswith("uv"))


@pytest.mark.skipif(platform.system() != "Linux", reason="install-deps is Linux-only")
def test_setup_install_deps_failure_is_a_warning(sandbox: Sandbox) -> None:
    sandbox.project(".vscode/extensions", venv=False)
    sandbox.stub("uv", 'case "$*" in *install-deps*) exit 1;; esac\n' + _UV_STUB)
    sandbox.stub("sudo", "")
    sandbox.stub("cloudflared", "")
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"; echo rc=$?')
    assert "WARNING: Chromium system libraries not installed" in res.stdout
    assert res.stdout.endswith("rc=0\n")


def test_setup_playwright_failure_is_a_warning(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions", venv=False)
    (project / ".playwright-fails").touch()
    sandbox.stub("uv", _UV_STUB)
    sandbox.stub("sudo", "")
    sandbox.stub("cloudflared", "")
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"; echo rc=$?')
    assert "WARNING: Playwright Chromium install failed" in res.stdout
    assert "install-deps" not in "".join(sandbox.logged_calls())
    assert res.stdout.endswith("rc=0\n")


def test_setup_with_every_sync_failing_installs_no_cli(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions", venv=False)
    (project / ".sync-fails").touch()
    sandbox.stub("uv", _UV_STUB)
    res = sandbox.run(f'setup_bundled_runtime "{VERSION}"; echo "primary=$KISS_PRIMARY_PROJECT"')
    assert res.stdout.endswith("primary=\n")
    assert not (sandbox.home / ".local/bin/sorcar").exists()
    assert "playwright" not in "".join(sandbox.logged_calls())


# --------------------------------------------------------------------------
# kiss_web_fingerprint: must equal the extension's computeKissWebFingerprint
# --------------------------------------------------------------------------

_TS_FINGERPRINT_JS = """
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const [project, bin, workDir] = process.argv.slice(1);
const hash = crypto.createHash('sha256');
hash.update(fs.readFileSync(bin));
hash.update(workDir);
let latest = BigInt(0);
const walk = (dir) => {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    if (entry.name === '__pycache__' || entry.name === 'tests') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (entry.isFile() && entry.name.endsWith('.py')) {
      const st = fs.statSync(full, {bigint: true});
      if (st.mtimeNs > latest) latest = st.mtimeNs;
    }
  }
};
walk(path.join(project, 'src', 'kiss'));
hash.update(latest.toString());
process.stdout.write(hash.digest('hex'));
"""


def test_fingerprint_matches_the_extension_algorithm(sandbox: Sandbox) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to run the extension's algorithm")
    project = sandbox.project(".vscode/extensions")
    src = project / "src" / "kiss"
    (src / "tests").mkdir()
    (src / "__pycache__").mkdir()
    (src / "sub").mkdir()
    newest = src / "sub" / "b.py"
    newest.write_text("y = 2\n", encoding="utf-8")
    os.utime(src / "a.py", ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
    os.utime(newest, ns=(1_800_000_000_987_654_321, 1_800_000_000_987_654_321))
    # Newer files in the excluded directories must not count.
    for excluded in (src / "tests" / "t.py", src / "__pycache__" / "c.py"):
        excluded.write_text("", encoding="utf-8")
        os.utime(excluded, ns=(1_900_000_000_000_000_000, 1_900_000_000_000_000_000))
    (src / "sub" / "data.txt").write_text("", encoding="utf-8")
    os.utime(src / "sub" / "data.txt", ns=(1_900_000_000_000_000_000, 1_900_000_000_000_000_000))
    workdir = "/some/work dir"
    res = sandbox.run(f'kiss_web_fingerprint "{project}" "{workdir}"')
    assert res.returncode == 0, res.stderr
    expected = subprocess.run(
        [
            node,
            "-e",
            _TS_FINGERPRINT_JS,
            str(project),
            str(project / ".venv/bin/kiss-web"),
            workdir,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len(expected) == 64
    assert res.stdout.strip() == expected
    # A different work dir is a different runtime.
    other = sandbox.run(f'kiss_web_fingerprint "{project}" "/elsewhere"').stdout.strip()
    assert other != expected and len(other) == 64


# --------------------------------------------------------------------------
# stop_stray_kiss_web
# --------------------------------------------------------------------------


def _spawn_named(name: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(["bash", "-c", f"exec -a {name} sleep 60"])


def test_stop_stray_kills_only_kiss_web_listeners(sandbox: Sandbox) -> None:
    stray = _spawn_named("kiss-web")
    other = _spawn_named("not-a-daemon")
    try:
        sandbox.stub("lsof", f"echo {stray.pid}; echo {other.pid}\n")
        res = sandbox.run("stop_stray_kiss_web")
        assert res.returncode == 0, res.stderr
        assert stray.wait(timeout=10) != 0
        assert other.poll() is None
        assert f"lsof -t -iTCP:{sandbox.port} -sTCP:LISTEN" in sandbox.logged_calls()
    finally:
        other.kill()
        stray.kill()


def test_stop_stray_uses_fuser_without_lsof(sandbox: Sandbox) -> None:
    stray = _spawn_named("kiss-web")
    try:
        sandbox.stub("fuser", f"echo ' {stray.pid}'\n")
        res = sandbox.run("stop_stray_kiss_web")
        assert res.returncode == 0, res.stderr
        assert stray.wait(timeout=10) != 0
        assert f"fuser {sandbox.port}/tcp" in sandbox.logged_calls()
    finally:
        stray.kill()


def test_stop_stray_without_lsof_or_fuser_does_nothing(sandbox: Sandbox) -> None:
    res = sandbox.run("stop_stray_kiss_web; echo rc=$?")
    assert res.stdout == "rc=0\n"


# --------------------------------------------------------------------------
# kiss_web_is_idle / start_kiss_web_daemon
# --------------------------------------------------------------------------

_SYSTEMD_STUB = """\
case "$*" in
    "--user show-environment") exit 0 ;;
    "--user start kiss-web.service")
        unit="$HOME/.config/systemd/user/kiss-web.service"
        bin="$(sed -n 's/^ExecStart=//p' "$unit")"
        wd="$(sed -n 's/^WorkingDirectory=//p' "$unit")"
        kh="$(sed -n 's/^Environment=KISS_HOME=//p' "$unit")"
        [ -n "$kh" ] && export KISS_HOME="$kh"
        (cd "$wd" && exec nohup "$bin" > /dev/null 2>&1 < /dev/null) &
        ;;
esac
exit 0
"""


async def _busy_handler(ws: ServerConnection) -> None:
    """Answer the active-tasks probe like a daemon with one task running."""
    await ws.recv()
    await ws.send(json.dumps({"type": "activeTasksResponse", "count": 1, "tabs": ["t"]}))


async def _run_with_busy_daemon(
    sandbox: Sandbox, endpoint: Path, commands: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run *commands* while a busy fake daemon is reachable through *endpoint*."""
    daemon_dir = sandbox.root / "busy"
    daemon_dir.mkdir()
    async with fake_daemon(daemon_dir, _busy_handler, endpoint_file=endpoint):
        return await asyncio.to_thread(sandbox.run, commands, env)


def test_start_skips_when_kiss_web_is_not_built(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions", venv=False)
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert "kiss-web not built" in res.stdout
    assert sandbox.logged_calls() == []


def test_start_gives_up_quietly_when_the_runtime_python_is_broken(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    python = project / ".venv/bin/python"
    python.unlink()
    python.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    python.chmod(0o755)
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"; echo rc=$?')
    assert res.stdout == "rc=0\n"
    assert sandbox.logged_calls() == []


def test_start_leaves_a_healthy_daemon_with_the_same_fingerprint_alone(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    fp = sandbox.run(f'kiss_web_fingerprint "{project}" "{sandbox.workdir}"').stdout.strip()
    kiss_home = sandbox.home / ".kiss"
    kiss_home.mkdir()
    (kiss_home / ".kiss-web.fingerprint").write_text(fp + "\n", encoding="utf-8")
    (kiss_home / "sorcar-local.json").write_text("{}", encoding="utf-8")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", sandbox.port))
        listener.listen()
        res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert "already serves this runtime; leaving it running" in res.stdout
    assert sandbox.logged_calls() == []


def test_start_defers_while_the_daemon_has_tasks_in_flight(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    endpoint = sandbox.home / ".kiss" / "sorcar-local.json"
    endpoint.parent.mkdir()
    res = asyncio.run(
        _run_with_busy_daemon(
            sandbox, endpoint, f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"; echo rc=$?'
        )
    )
    assert "kiss-web has tasks in flight; restart deferred" in res.stdout
    assert res.stdout.endswith("rc=0\n")
    assert sandbox.logged_calls() == []
    assert not (sandbox.home / ".config/systemd/user/kiss-web.service").exists()


def test_start_defers_for_a_busy_daemon_of_the_stock_home_after_a_brand_switch(
    sandbox: Sandbox,
) -> None:
    """The new runtime's home is empty, but ~/.kiss still owns a busy daemon."""
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    endpoint = sandbox.home / ".kiss" / "sorcar-local.json"
    endpoint.parent.mkdir()
    res = asyncio.run(
        _run_with_busy_daemon(
            sandbox,
            endpoint,
            f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"; echo rc=$?',
            env={"KISS_HOME": str(sandbox.home / ".brand")},
        )
    )
    assert "restart deferred" in res.stdout
    assert sandbox.logged_calls() == []


def test_start_registers_and_starts_the_systemd_user_service(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    sandbox.stub("loginctl", "")
    kiss_home = sandbox.home / ".kiss"
    kiss_home.mkdir()
    (kiss_home / ".kiss-web.restart-pending").touch()
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert res.returncode == 0, res.stderr
    assert "Restarting the kiss-web systemd user service" in res.stdout
    assert "kiss-web is up" in res.stdout and f"http://localhost:{sandbox.port}" in res.stdout
    unit = (sandbox.home / ".config/systemd/user/kiss-web.service").read_text(encoding="utf-8")
    assert f"ExecStart={project}/.venv/bin/kiss-web\n" in unit
    assert f"WorkingDirectory={sandbox.workdir}\n" in unit
    assert "Description=KISS Sorcar Remote Web Server\n" in unit
    assert f"Environment=PATH={sandbox.home}/.local/bin:/usr/local/bin:/usr/bin:/bin\n" in unit
    assert "KISS_HOME" not in unit
    assert f"StandardOutput=append:{kiss_home}/kiss-web-stdout.log\n" in unit
    assert "WantedBy=default.target" in unit
    user = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    assert sandbox.logged_calls() == [
        "systemctl --user show-environment",
        "systemctl --user daemon-reload",
        "systemctl --user enable kiss-web.service",
        "systemctl --user stop kiss-web.service",
        "systemctl --user start kiss-web.service",
        f"loginctl enable-linger {user}",
    ]
    # The daemon really runs: in the work dir, with the endpoint rewritten.
    assert (kiss_home / "kiss-web.cwd").read_text().strip() == str(sandbox.workdir)
    assert _wait_port(sandbox.port)
    fp = sandbox.run(f'kiss_web_fingerprint "{project}" "{sandbox.workdir}"').stdout
    assert (kiss_home / ".kiss-web.fingerprint").read_text(encoding="utf-8") == fp
    assert not (kiss_home / ".kiss-web.restart-pending").exists()
    assert not (kiss_home / ".kiss-web.restart-stamp").exists()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"_KISS_HOST_SERVICE": "kiss-web.service", "_KISS_INTERACTIVE": "1"}, True),
        ({"_KISS_HOST_SERVICE": "kiss-web.service", "_KISS_INTERACTIVE": "0"}, False),
        ({"_KISS_INTERACTIVE": "1"}, False),
    ],
)
def test_start_warns_when_kiss_web_hosts_the_terminal(
    sandbox: Sandbox, env: dict[str, str], expected: bool
) -> None:
    """The webapp's Terminal tab is a shell kiss-web forks, so the restart
    hangs it up: a script that left the daemon's cgroup (see the
    ``kiss-service-cgroup-escape`` block) and talks to a human says so
    before stopping the daemon.  The Update button (non-interactive)
    and a plain terminal get no such notice."""
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    sandbox.stub("loginctl", "")
    log = sandbox.home / "install.log"
    res = sandbox.run(
        f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"',
        env={**env, "LOG_FILE": str(log)},
    )
    assert res.returncode == 0, res.stderr
    assert "kiss-web is up" in res.stdout
    notice = (
        "   kiss-web hosts this terminal, so it closes with the daemon;\n"
        f"   the install carries on and its remaining output is in {log}.\n"
    )
    assert (notice in res.stdout) is expected
    calls = sandbox.logged_calls()
    assert calls.index("systemctl --user stop kiss-web.service") < calls.index(
        "systemctl --user start kiss-web.service"
    )
    if expected:
        assert res.stdout.index(notice) < res.stdout.index("kiss-web is up")


def test_start_propagates_kiss_home_to_the_service(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    sandbox.stub("loginctl", "")
    custom = sandbox.home / "custom-home"
    res = sandbox.run(
        f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"', env={"KISS_HOME": str(custom)}
    )
    assert res.returncode == 0, res.stderr
    assert "kiss-web is up" in res.stdout
    unit = (sandbox.home / ".config/systemd/user/kiss-web.service").read_text(encoding="utf-8")
    assert (
        f"Environment=KISS_HOME={custom}\nStandardOutput=append:{custom}/kiss-web-stdout.log\n"
        in unit
    )
    assert (custom / ".kiss-web.fingerprint").exists()
    assert (custom / "sorcar-local.json").exists()


def test_start_falls_back_to_a_detached_process_without_systemd(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", "exit 1\n")
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert res.returncode == 0, res.stderr
    assert "Starting kiss-web as a detached background process" in res.stdout
    assert "kiss-web is up" in res.stdout
    assert sandbox.logged_calls() == ["systemctl --user show-environment"]
    assert (sandbox.home / ".kiss" / "kiss-web.cwd").read_text().strip() == str(sandbox.workdir)
    assert (sandbox.home / ".kiss" / ".kiss-web.fingerprint").exists()


def test_start_registers_the_launch_agent_on_macos(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Darwin", "arm64")
    sandbox.stub(
        "launchctl",
        f'case "$1" in kickstart) (cd {str(sandbox.workdir)!r} && exec nohup '
        f"{str(project)!r}/.venv/bin/kiss-web > /dev/null 2>&1 < /dev/null) & ;; esac\nexit 0\n",
    )
    custom = sandbox.home / "custom-home"
    res = sandbox.run(
        f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"', env={"KISS_HOME": str(custom)}
    )
    assert res.returncode == 0, res.stderr
    assert "Restarting the kiss-web LaunchAgent" in res.stdout
    assert "kiss-web is up" in res.stdout
    plist = (sandbox.home / "Library/LaunchAgents/com.kiss.web-server.plist").read_text(
        encoding="utf-8"
    )
    assert f"<string>{project}/.venv/bin/kiss-web</string>" in plist
    assert f"<key>WorkingDirectory</key>\n    <string>{sandbox.workdir}</string>" in plist
    assert f"<string>{custom}/kiss-web-stderr.log</string>" in plist
    assert (
        f"</string>\n        <key>KISS_HOME</key>\n        <string>{custom}</string>\n    </dict>"
        in plist
    )
    uid = os.getuid()
    assert sandbox.logged_calls() == [
        f"launchctl bootout gui/{uid}/com.kiss.web-server",
        f"launchctl bootstrap gui/{uid} {sandbox.home}/Library/LaunchAgents/"
        "com.kiss.web-server.plist",
        f"launchctl kickstart -k gui/{uid}/com.kiss.web-server",
    ]
    assert (custom / ".kiss-web.fingerprint").exists()


def test_start_macos_without_kiss_home_omits_the_env_entry(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Darwin", "arm64")
    sandbox.stub("launchctl", '[ "$1" = kickstart ] && exit 1\nexit 0\n')
    res = sandbox.run(
        f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"',
        env={"KISS_WEB_START_TIMEOUT": "2"},
    )
    assert "WARNING: launchctl kickstart failed" in res.stdout
    plist = (sandbox.home / "Library/LaunchAgents/com.kiss.web-server.plist").read_text(
        encoding="utf-8"
    )
    assert "KISS_HOME" not in plist
    assert "<string>/opt/homebrew/bin:" in plist


def test_start_reports_a_daemon_that_never_comes_up(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", '[ "$*" = "--user start kiss-web.service" ] && exit 1\nexit 0\n')
    sandbox.stub("loginctl", "")
    kiss_home = sandbox.home / ".kiss"
    kiss_home.mkdir()
    (kiss_home / "sorcar-local.json").write_text("{}", encoding="utf-8")
    res = sandbox.run(
        f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"; echo rc=$?',
        env={"KISS_WEB_START_TIMEOUT": "2"},
    )
    assert "WARNING: systemctl start failed" in res.stdout
    assert "WARNING: kiss-web did not come up within 2s" in res.stdout
    assert res.stdout.endswith("rc=0\n")
    assert not (kiss_home / ".kiss-web.fingerprint").exists()
    assert not (kiss_home / ".kiss-web.restart-stamp").exists()
    # The stale endpoint file of the old daemon is left in place.
    assert (kiss_home / "sorcar-local.json").read_text() == "{}"


# --------------------------------------------------------------------------
# write_kiss_web_plist / write_kiss_web_unit: escaping as the extension does
# --------------------------------------------------------------------------


def test_plist_escapes_xml_special_characters(sandbox: Sandbox) -> None:
    plist = sandbox.root / "agent.plist"
    res = sandbox.run(
        f'write_kiss_web_plist "{plist}" "/opt/a&b/kiss-web" "/work/<dir>" "/home/\\"q\\""',
        env={"KISS_HOME": "/home/it's"},
    )
    assert res.returncode == 0, res.stderr
    text = plist.read_text(encoding="utf-8")
    assert "<string>/opt/a&amp;b/kiss-web</string>" in text
    assert "<string>/work/&lt;dir&gt;</string>" in text
    assert "<string>/home/&quot;q&quot;/kiss-web-stdout.log</string>" in text
    assert "<key>KISS_HOME</key>\n        <string>/home/it&apos;s</string>" in text


def test_unit_escapes_percent_and_backslash(sandbox: Sandbox) -> None:
    unit = sandbox.root / "kiss-web.service"
    res = sandbox.run(
        f'write_kiss_web_unit "{unit}" "/opt/100%/kiss-web" "/work/back\\\\slash" "/home/x"',
        env={"KISS_HOME": "/home/50%"},
    )
    assert res.returncode == 0, res.stderr
    text = unit.read_text(encoding="utf-8")
    assert "ExecStart=/opt/100%%/kiss-web\n" in text
    assert "WorkingDirectory=/work/back\\\\slash\n" in text
    assert (
        "Environment=KISS_HOME=/home/50%%\nStandardOutput=append:/home/x/kiss-web-stdout.log\n"
        in text
    )


def test_unit_escapes_newlines_like_the_extension(sandbox: Sandbox) -> None:
    unit = sandbox.root / "kiss-web.service"
    res = sandbox.run(
        f'write_kiss_web_unit "{unit}" "/opt/kiss-web" "$(printf \'/work/two\\nlines\')" "/home/x"'
    )
    assert res.returncode == 0, res.stderr
    assert "WorkingDirectory=/work/two\\nlines\n" in unit.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Review-driven cases: symlinks, the restart lock, previous homes, old
# daemons, best-effort failures
# --------------------------------------------------------------------------


def test_fingerprint_ignores_symlinked_sources_like_the_extension(sandbox: Sandbox) -> None:
    """``Dirent.isFile()`` is false for symlinks, so the twin must skip them too."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to run the extension's algorithm")
    project = sandbox.project(".vscode/extensions")
    src = project / "src" / "kiss"
    external = sandbox.root / "external.py"
    external.write_text("z = 3\n", encoding="utf-8")
    os.utime(external, ns=(1_900_000_000_000_000_000, 1_900_000_000_000_000_000))
    (src / "link.py").symlink_to(external)
    (src / "linked_dir").symlink_to(sandbox.root)
    res = sandbox.run(f'kiss_web_fingerprint "{project}" "/w"')
    expected = subprocess.run(
        [node, "-e", _TS_FINGERPRINT_JS, str(project), str(project / ".venv/bin/kiss-web"), "/w"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert res.stdout.strip() == expected


def test_start_yields_to_a_live_restart_lock_holder(sandbox: Sandbox) -> None:
    """The extension's ``.kiss-web.restart.lock`` with a live owner wins."""
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    holder = subprocess.Popen(["sleep", "60"])
    kiss_home = sandbox.home / ".kiss"
    kiss_home.mkdir()
    lock = kiss_home / ".kiss-web.restart.lock"
    lock.write_text(json.dumps({"pid": holder.pid, "token": "window"}), encoding="utf-8")
    try:
        res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    finally:
        holder.kill()
    assert "A VS Code window is restarting kiss-web" in res.stdout
    assert sandbox.logged_calls() == []
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "window"


def test_start_breaks_a_lock_left_by_a_dead_process_and_releases_its_own(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    sandbox.stub("loginctl", "")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kiss_home = sandbox.home / ".kiss"
    kiss_home.mkdir()
    lock = kiss_home / ".kiss-web.restart.lock"
    lock.write_text(json.dumps({"pid": dead.pid, "token": "gone"}), encoding="utf-8")
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert "kiss-web is up" in res.stdout, res.stdout
    assert not lock.exists()
    # A lock without a readable owner may be one the extension has just
    # created and not yet written: it is respected for two minutes.
    lock.write_text("", encoding="utf-8")
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert "A VS Code window is restarting kiss-web" in res.stdout
    assert lock.exists()
    stale = time.time() - 3 * 60
    os.utime(lock, (stale, stale))
    res = sandbox.run(f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"')
    assert "already serves this runtime" in res.stdout
    assert not lock.exists()


def test_start_asks_the_daemon_of_the_previous_service_home(sandbox: Sandbox) -> None:
    """A busy daemon that an earlier install ran under another KISS_HOME is probed."""
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    old_home = sandbox.home / "old%home"
    old_home.mkdir()
    unit = sandbox.home / ".config/systemd/user/kiss-web.service"
    unit.parent.mkdir(parents=True)
    # Written the way the extension writes it: an argument after the
    # launcher (older units) and a %-containing home escaped as %%.
    unit.write_text(
        f"[Service]\nExecStart=/nowhere/kiss-web --workdir /x\n"
        f"Environment=KISS_HOME={str(old_home).replace('%', '%%')}\n",
        encoding="utf-8",
    )
    res = asyncio.run(
        _run_with_busy_daemon(
            sandbox,
            old_home / "sorcar-local.json",
            f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"',
        )
    )
    assert "restart deferred" in res.stdout
    assert sandbox.logged_calls() == []


def test_start_asks_the_daemon_of_the_previous_runtime_brand_home(sandbox: Sandbox) -> None:
    """The old service's runtime resolves its own home (a brand's, say) — probed too."""
    project = sandbox.project(".vscode/extensions")
    old_project = sandbox.project(".vscode-server/extensions", version="1.2.3")
    old_project = old_project.rename(old_project.parent / "kiss&project")
    sandbox.uname("Darwin")
    sandbox.stub("launchctl", "")
    brand_home = sandbox.home / ".oldbrand"
    brand_home.mkdir()
    # The old runtime's python reports the brand home when KISS_HOME is unset.
    python = old_project / ".venv/bin/python"
    python.write_text(f"#!/bin/bash\necho {str(brand_home)!r}\n", encoding="utf-8")
    plist = sandbox.home / "Library/LaunchAgents/com.kiss.web-server.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text(
        "<plist><dict>\n    <key>ProgramArguments</key>\n    <array>\n"
        f"        <string>{str(old_project).replace('&', '&amp;')}/.venv/bin/kiss-web</string>\n"
        "    </array>\n</dict></plist>\n",
        encoding="utf-8",
    )
    res = asyncio.run(
        _run_with_busy_daemon(
            sandbox,
            brand_home / "sorcar-local.json",
            f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"',
        )
    )
    assert "restart deferred" in res.stdout
    assert sandbox.logged_calls() == []


async def _old_daemon_handler(ws: ServerConnection) -> None:
    await ws.recv()
    await ws.send(json.dumps({"type": "error", "text": "Unknown command: activeTasksQuery"}))


def test_start_defers_for_a_daemon_too_old_to_report_its_tasks(sandbox: Sandbox) -> None:
    project = sandbox.project(".vscode/extensions")
    sandbox.uname("Linux")
    sandbox.stub("systemctl", _SYSTEMD_STUB)
    endpoint = sandbox.home / ".kiss" / "sorcar-local.json"
    endpoint.parent.mkdir()

    async def go() -> subprocess.CompletedProcess[str]:
        daemon_dir = sandbox.root / "old"
        daemon_dir.mkdir()
        async with fake_daemon(daemon_dir, _old_daemon_handler, endpoint_file=endpoint):
            return await asyncio.to_thread(
                sandbox.run, f'start_kiss_web_daemon "{project}" "{sandbox.workdir}"'
            )

    res = asyncio.run(go())
    assert "restart deferred" in res.stdout
    assert sandbox.logged_calls() == []


def test_setup_reports_an_unwritable_shell_rc_and_cli_dir_and_continues(sandbox: Sandbox) -> None:
    """Filesystem failures are warnings; the install must go on."""
    sandbox.project(".vscode/extensions", venv=False)
    sandbox.stub("uv", _UV_STUB)
    sandbox.stub("sudo", "exit 1\n")
    sandbox.stub("cloudflared", "")
    rc = sandbox.home / ".bashrc"
    rc.write_text("# read only\n", encoding="utf-8")
    rc.chmod(0o444)
    local_bin = sandbox.home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    local_bin.chmod(0o555)
    try:
        res = sandbox.run(
            f'setup_bundled_runtime "{VERSION}" || echo "WARNING: wrapped"; echo INSTALL_CONTINUED'
        )
    finally:
        local_bin.chmod(0o755)
        rc.chmod(0o644)
    assert res.stdout.endswith("INSTALL_CONTINUED\n"), res.stdout
    assert "WARNING: could not install the sorcar CLI" in res.stdout
    assert "WARNING: could not add ~/.local/bin to PATH" in res.stdout
    assert rc.read_text(encoding="utf-8") == "# read only\n"


def test_unit_escapes_a_trailing_newline(sandbox: Sandbox) -> None:
    res = sandbox.run("printf '[%s]' \"$(unit_escape $'/work/end\\n')\"")
    assert res.stdout == "[/work/end\\n]"


def test_restart_lock_is_released_by_the_exit_trap_on_abort(sandbox: Sandbox) -> None:
    """``install.sh``'s EXIT trap releases a held lock (Ctrl-C twice exits 130)."""
    src = INSTALL.read_text(encoding="utf-8")
    trap_line = next(
        line for line in src.splitlines() if line.startswith('trap \'rm -f "$PROGRESS_FILE"')
    )
    assert "release_kiss_web_restart_lock" in trap_line
    lock = sandbox.home / ".kiss-web.restart.lock"
    res = sandbox.run(
        f'PROGRESS_FILE="{sandbox.root}/progress"\n{trap_line}\n'
        f'acquire_kiss_web_restart_lock "{lock}"\n'
        '[ -f "$KISS_WEB_RESTART_LOCK_HELD" ] && echo held\nexit 130\n'
    )
    assert res.returncode == 130
    assert res.stdout == "held\n"
    assert not lock.exists()
