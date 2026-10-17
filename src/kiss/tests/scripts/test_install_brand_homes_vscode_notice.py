# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ROOT ``install.sh``'s brand-home passes, the
VS Code notice and the password box text.

Every test runs the real ``install.sh`` from a seeded fake checkout with
a fake HOME, the whitelisted PATH and the stubs of
``test_install_update_lock`` (git/curl fail offline, ``uv sync`` lays
out a fake venv).  This module's ``uv`` stub also answers the
remote-password command (it writes ``remote_password`` into the home's
config.json and prints it) so the password box can be checked, and its
``kiss-web`` stub records the KISS_HOME and the app it was started from.

Covered:

* a ``$HOME/.<brand>/post-install.d/<hook>`` is discovered and gets a
  nested install (``KISS_HOME`` = that home), its hook runs with that
  home, and the daemon restart uses the brand app when the existing
  service unit runs from that home (with that home's password in the box
  and its remote-url.json for the webapp step);
* the ``_KISS_BRAND_PASS=1`` recursion guard (two brand homes get exactly
  one pass each; a guarded run discovers nothing and prints no box);
* the VS Code notice with ``TERM_PROGRAM=vscode``, with an installed
  ``ksenxx.kiss-sorcar-*`` extension directory, and absent otherwise;
* the new second line of the password box.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from kiss.tests.scripts.test_install_update_lock import (
    STUB_BREW,
    STUB_CLOUDFLARED,
    STUB_CURL,
    STUB_GIT,
    STUB_UNAME,
    WHITELISTED_TOOLS,
    _alive,
    _seed_project,
)

pytest.importorskip("fcntl", reason="POSIX-only: bash install.sh and flock")

# ``python`` answers the runtime-home import with KISS_HOME (what the
# real kiss.core.config does), keeps the remote password of KISS_HOME's
# config.json or sets it as ``pw-<home name>`` when there is none (the
# keep-unless-empty rule of the real module) and otherwise runs the real
# interpreter.  ``kiss-web`` records how it was started and lingers like
# a daemon until the test kills it.
STUB_UV = """#!/bin/bash
case "$1" in
    --version) echo "uv 0.9.0 (test stub)" ;;
    sync)
        app="$3"
        mkdir -p "$app/.venv/bin"
        cat > "$app/.venv/bin/python" <<EOF
#!/bin/bash
home="\\${KISS_HOME:-\\$HOME/.kiss}"
case "\\$2" in
    "from kiss.core.config import kiss_home; print(kiss_home())") echo "\\$home" ;;
    kiss.core.remote_password)
        pw="\\$(sed -n 's/^[[:space:]]*"remote_password":[[:space:]]*"\\([^"]*\\)".*/\\1/p' \\
            "\\$home/config.json" 2>/dev/null | head -n 1)"
        if [ -z "\\$pw" ] && [ "\\$3" != --show ]; then
            pw="pw-\\$(basename "\\$home")"
            mkdir -p "\\$home"
            printf '{\\n  "remote_password": "%s"\\n}\\n' "\\$pw" > "\\$home/config.json"
        fi
        echo "\\$pw" ;;
    *) exec "$KISS_TEST_PYTHON" "\\$@" ;;
esac
EOF
        cat > "$app/.venv/bin/kiss-web" <<'EOF'
#!/bin/bash
echo "$$" >> "$KISS_TEST_MARK_DIR/daemon"
echo "KISS_HOME=${KISS_HOME:-unset} APP=$0" >> "$KISS_TEST_MARK_DIR/daemon-start"
pwd >> "$KISS_TEST_MARK_DIR/daemon-pwd"
exec sleep 60
EOF
        chmod 755 "$app/.venv/bin/python" "$app/.venv/bin/kiss-web"
        ;;
esac
exit 0
"""

HOOK = """#!/bin/bash
echo "KISS_HOME=$KISS_HOME APP=$KISS_APP_DIR" >> "$KISS_TEST_MARK_DIR/hooks"
"""

# The editor CLI (``KISS_CODE_CLI``): records every call; an uninstall
# succeeds, removing the extension directories, unless the test leaves a
# ``code-fails`` mark, when it fails like a CLI with no window to talk to.
STUB_CODE = """#!/bin/bash
echo "$*" >> "$KISS_TEST_MARK_DIR/code"
[ ! -e "$KISS_TEST_MARK_DIR/code-fails" ] || exit 1
if [ "$1" = --uninstall-extension ]; then
    /bin/rm -rf "$HOME"/.vscode/extensions/"$2"-* "$HOME"/.vscode-server/extensions/"$2"-*
fi
exit 0
"""

NESTED_PASS = "Installing into brand home "
NOTICE = "VS Code is no longer supported by KISS Sorcar."
NOTICE_REMOVED = "The VS Code extension has been removed and will never be updated."
NOTICE_UNINSTALL = "Uninstall it from the Extensions view of VS Code (ksenxx.kiss-sorcar)."
NOTICE_URL = "Use the web app instead: http://127.0.0.1:8787"
NOTICE_FILE = "VSCODE-EXTENSION-REMOVED.txt"
PASSWORD_HINT = "Type it in the web app on another device; it is also shown in Settings."


class BrandHomesAndVscodeNoticeTest(unittest.TestCase):
    """install.sh installs into every brand home, restarts the brand's daemon.

    It also warns VS Code users.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-install-brand-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.marks = self.tmp / "marks"
        self.marks.mkdir()
        tools = self.tmp / "tools"
        tools.mkdir()
        for name in (*WHITELISTED_TOOLS, "nohup", "chmod", "id", "cp", "mv", "env", "basename"):
            real = shutil.which(name)
            self.assertIsNotNone(real, f"{name} is required by this test")
            assert real is not None
            os.symlink(real, tools / name)
        self.stubs = self.tmp / "stubs"
        self.stubs.mkdir()
        for name, body in (
            ("git", STUB_GIT),
            ("curl", STUB_CURL),
            ("brew", STUB_BREW),
            ("uv", STUB_UV),
            ("uname", STUB_UNAME),
            ("cloudflared", STUB_CLOUDFLARED),
            ("code", STUB_CODE),
        ):
            (self.stubs / name).write_text(body)
            (self.stubs / name).chmod(0o755)
        self.project = self.tmp / "project"
        _seed_project(self.project)
        (self.project / "src" / "kiss" / "core" / "models").mkdir(parents=True)
        (self.project / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json").write_text("{}\n")
        # The git/curl stubs hold until these exist; nothing here waits.
        (self.tmp / "release").touch()
        (self.tmp / "release2").touch()
        python = shutil.which("python3")
        assert python is not None
        # No DISPLAY and no systemctl: a remote Linux box whose daemon is a
        # detached process; the webapp step prints the tunnel URL instead
        # of opening a browser.  KISS_WEB_START_TIMEOUT=0: the stub daemon
        # never opens the port.
        self.env = {
            "PATH": f"{self.stubs}:{tools}",
            "HOME": str(self.home),
            "KISS_TEST_MARK_DIR": str(self.marks),
            "KISS_TEST_RELEASE": str(self.tmp / "release"),
            "KISS_TEST_RELEASE2": str(self.tmp / "release2"),
            "KISS_TEST_PYTHON": python,
            "KISS_WEB_START_TIMEOUT": "0",
            "LANG": "C",
        }
        self.log = self.home / ".kiss" / "install.log"

    def tearDown(self) -> None:
        daemon = self.marks / "daemon"
        if daemon.exists():
            for pid in daemon.read_text().split():
                if _alive(int(pid)):
                    os.kill(int(pid), signal.SIGKILL)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, extra_env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
        """Run install.sh non-interactively from *cwd* (the scratch
        directory by default); return its combined output."""
        proc = subprocess.run(
            ["bash", str(self.project / "install.sh"), "--non-interactive"],
            cwd=cwd or self.tmp,
            env={**self.env, **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc.stdout + proc.stderr

    def _brand_home(self, name: str) -> Path:
        """Create ``$HOME/.<name>/post-install.d/hook`` (executable) and return the home."""
        home = self.home / f".{name}"
        (home / "post-install.d").mkdir(parents=True)
        (home / "post-install.d" / "hook").write_text(HOOK)
        (home / "post-install.d" / "hook").chmod(0o755)
        return home

    def _write_service_unit(self, kiss_home: Path) -> None:
        unit = self.home / ".config" / "systemd" / "user" / "kiss-web.service"
        unit.parent.mkdir(parents=True)
        unit.write_text(
            "[Service]\n"
            f"Environment=KISS_HOME={kiss_home}\n"
            f"ExecStart={kiss_home}/{kiss_home.name[1:]}/.venv/bin/kiss-web\n"
        )

    def _marks(self, name: str) -> list[str]:
        path = self.marks / name
        return path.read_text().splitlines() if path.exists() else []

    def _daemon_starts(self) -> list[str]:
        """The ``daemon-start`` marks, once the detached daemon has written one.

        install.sh backgrounds ``nohup kiss-web`` and returns without
        waiting for it, so the stub's mark can land a moment after
        ``_run`` (macOS spawns it visibly later than Linux does).
        """
        deadline = time.monotonic() + 10.0
        while not self._marks("daemon-start"):
            self.assertLess(time.monotonic(), deadline, "the daemon was not started")
            time.sleep(0.05)
        return self._marks("daemon-start")

    def test_brand_homes_get_a_nested_pass_and_the_branded_daemon_is_restarted(self) -> None:
        brandx = self._brand_home("brandx")
        brandy = self._brand_home("brandy")
        # A directory with a hook that is not executable is not a brand home.
        (self.home / ".notabrand" / "post-install.d").mkdir(parents=True)
        (self.home / ".notabrand" / "post-install.d" / "hook").write_text(HOOK)
        self._write_service_unit(brandx)
        # The brand daemon's URL file, so the webapp step reads the brand
        # home's tunnel rather than waiting on ~/.kiss/remote-url.json.
        (brandx / "remote-url.json").write_text(
            '{\n  "loopback": "http://127.0.0.1:8787",\n'
            '  "tunnel": "https://brandx.trycloudflare.com"\n}\n'
        )

        out = self._run()

        # One nested pass per brand home, none for ~/.notabrand or ~/.kiss,
        # and none from inside a nested pass (the recursion guard).
        self.assertEqual(out.count(NESTED_PASS), 2, out)
        self.assertIn(f"{NESTED_PASS}~/.brandx...", out)
        self.assertIn(f"{NESTED_PASS}~/.brandy...", out)
        self.assertNotIn(".notabrand", out)
        for home in (brandx, brandy):
            self.assertIn(f"   Installed into brand home {home}", out)
            # The app is named after its home: ~/.brandx/brandx.
            self.assertTrue((home / home.name[1:] / "src" / "kiss" / "__init__.pyc").exists(), home)
            self.assertFalse((home / "kiss").exists(), home)
            self.assertTrue((home / "MODEL_INFO.json").exists(), home)
            self.assertIn(
                f'"remote_password": "pw-{home.name}"', (home / "config.json").read_text()
            )
        self.assertIn(
            '"remote_password": "pw-.kiss"', (self.home / ".kiss" / "config.json").read_text()
        )
        self.assertEqual(
            sorted(self._marks("hooks")),
            sorted([
                f"KISS_HOME={brandx} APP={brandx / 'brandx'}",
                f"KISS_HOME={brandy} APP={brandy / 'brandy'}",
            ]),
        )
        # The service runs from ~/.brandx: that app is restarted, with its
        # home, and the box shows that home's password.
        self.assertIn(
            f"The kiss-web service runs from brand home {brandx}; restarting that app.", out
        )
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME={brandx} APP={brandx / 'brandx' / '.venv' / 'bin' / 'kiss-web'}"],
        )
        self.assertIn("https://brandx.trycloudflare.com", out)
        self.assertIn("*  Remote access password:  pw-.brandx", out)
        self.assertNotIn("pw-.kiss", out.split("=== App install complete ===")[-1])
        self.assertIn(f"*  {PASSWORD_HINT}  *", out)
        self.assertEqual(out.count("Remote access password:"), 1, "a nested pass printed a box")
        self.assertEqual(out.count("=== App install complete ==="), 1, out)
        self.assertIn(f"=== Brand home {brandx}: app install complete ===", out)
        self.assertEqual(out.count("Log saved to"), 1, out)
        # Nested passes write to the outer tee: every line is logged once.
        log = self.log.read_text()
        self.assertEqual(log.count(">>> [5/5] Finishing the setup..."), 3, log)
        self.assertNotIn("pw-", log)
        self.assertNotIn(NOTICE, out)

    def test_without_brand_homes_the_stock_app_is_restarted(self) -> None:
        (self.home / ".kiss").mkdir()
        (self.home / ".kiss" / "remote-url.json").write_text(
            '{\n  "loopback": "http://127.0.0.1:8787",\n'
            '  "tunnel": "https://stock.trycloudflare.com"\n}\n'
        )
        out = self._run()
        self.assertNotIn(NESTED_PASS, out)
        self.assertNotIn("runs from brand home", out)
        app = self.home / ".kiss" / "kiss"
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME=unset APP={app / '.venv' / 'bin' / 'kiss-web'}"],
        )
        self.assertIn("https://stock.trycloudflare.com", out)
        self.assertIn("   Set a new remote-access password (shown at the end).", out)
        self.assertIn("*  Remote access password:  pw-.kiss", out)
        self.assertIn(f"*  {PASSWORD_HINT}  *", out)

        # An update: the password the user's devices know is kept, and
        # the box at the end still shows it.
        config = self.home / ".kiss" / "config.json"
        config.write_text('{\n  "max_budget": 9,\n  "remote_password": "kept-by-user"\n}\n')
        out = self._run()
        self.assertIn("   Kept the existing remote-access password (shown at the end).", out)
        self.assertNotIn("Set a new remote-access password", out)
        self.assertIn("*  Remote access password:  kept-by-user", out)
        self.assertIn('"remote_password": "kept-by-user"', config.read_text())

    def test_service_home_without_hooks_is_a_brand_home_too(self) -> None:
        # The existing service runs from a home that has no hooks (its
        # brand layer was removed): it is still installed into and kept
        # as the daemon's home, so the restart does not switch apps.
        plain = self.home / ".plain"
        plain.mkdir()
        self._write_service_unit(plain)
        (plain / "remote-url.json").write_text('{\n  "tunnel": "https://plain.trycloudflare.com"\n}\n')
        out = self._run()
        self.assertEqual(out.count(NESTED_PASS), 1, out)
        self.assertIn(f"{NESTED_PASS}~/.plain...", out)
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME={plain} APP={plain / 'plain' / '.venv' / 'bin' / 'kiss-web'}"],
        )

    def test_brand_home_the_service_does_not_name_becomes_the_daemon(self) -> None:
        # No service definition names a brand home (a dev daemon, or a
        # brand installed with --no-restart): a stock install still ends
        # with the brand's daemon, the first brand home installed, so the
        # brand the user put on the machine is what runs after the hook
        # has branded its app.
        brandx = self._brand_home("brandx")
        brandy = self._brand_home("brandy")
        (brandx / "remote-url.json").write_text(
            '{\n  "loopback": "http://127.0.0.1:8787",\n'
            '  "tunnel": "https://brandx.trycloudflare.com"\n}\n'
        )
        out = self._run()
        self.assertEqual(out.count(NESTED_PASS), 2, out)
        self.assertIn(
            f"   Brand home {brandx} is installed and branded; starting the daemon from that app,"
            f" not from {self.home / '.kiss' / 'kiss'}.",
            out,
        )
        self.assertNotIn("The kiss-web service runs from brand home", out)
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME={brandx} APP={brandx / 'brandx' / '.venv' / 'bin' / 'kiss-web'}"],
        )
        self.assertTrue((brandy / "brandy" / "src" / "kiss" / "__init__.pyc").exists())
        self.assertTrue((self.home / ".kiss" / "kiss" / "src" / "kiss" / "__init__.pyc").exists())
        self.assertIn("https://brandx.trycloudflare.com", out)
        self.assertIn("*  Remote access password:  pw-.brandx", out)
        self.assertEqual(out.count("Remote access password:"), 1, out)

    def test_install_into_a_custom_home_starts_its_own_daemon(self) -> None:
        # An install into a home that is not the stock ~/.kiss (a brand's
        # own installer, KISS_HOME set by hand) installs into the other
        # brand homes too but the daemon is its own (no service names
        # another brand home here).
        brandx = self._brand_home("brandx")
        custom = self.home / ".custom"
        custom.mkdir()
        (custom / "remote-url.json").write_text(
            '{\n  "tunnel": "https://custom.trycloudflare.com"\n}\n'
        )
        out = self._run({"KISS_HOME": str(custom)})
        self.assertEqual(out.count(NESTED_PASS), 1, out)
        self.assertIn(f"{NESTED_PASS}~/.brandx...", out)
        self.assertTrue((brandx / "brandx" / "src" / "kiss" / "__init__.pyc").exists())
        self.assertNotIn("is installed and branded", out)
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME={custom} APP={custom / 'custom' / '.venv' / 'bin' / 'kiss-web'}"],
        )
        self.assertIn("https://custom.trycloudflare.com", out)
        self.assertIn("*  Remote access password:  pw-.custom", out)

    def test_failing_nested_pass_is_a_warning(self) -> None:
        brandx = self._brand_home("brandx")
        # The nested pass fails at its app-files step: the brand home's
        # app directory is a file, so the copy into it cannot succeed.
        (brandx / "brandx").write_text("not a directory\n")
        out = self._run({"KISS_SKIP_LAUNCH": "1"})
        self.assertIn(f"{NESTED_PASS}~/.brandx...", out)
        self.assertIn(f"WARNING: the install into brand home {brandx} failed (status ", out)
        self.assertIn("=== App install complete ===", out)
        self.assertIn("*  Remote access password:  pw-.kiss", out)

        # A brand home whose pass failed has no app to run: the stock
        # app is the daemon.
        (self.home / ".kiss" / "remote-url.json").write_text(
            '{\n  "tunnel": "https://stock.trycloudflare.com"\n}\n'
        )
        out = self._run()
        self.assertIn(f"WARNING: the install into brand home {brandx} failed (status ", out)
        self.assertNotIn("is installed and branded", out)
        app = self.home / ".kiss" / "kiss"
        self.assertEqual(
            self._daemon_starts(),
            [f"KISS_HOME=unset APP={app / '.venv' / 'bin' / 'kiss-web'}"],
        )
        self.assertIn("*  Remote access password:  pw-.kiss", out)

    def test_guarded_run_discovers_no_brand_home_and_prints_no_box(self) -> None:
        brandx = self._brand_home("brandx")
        self._brand_home("brandy")
        out = self._run({
            "KISS_HOME": str(brandx),
            "KISS_SKIP_LAUNCH": "1",
            "KISS_UPDATE_LOCK_HELD": "1",
            "_KISS_BRAND_PASS": "1",
            "TERM_PROGRAM": "vscode",
        })
        self.assertNotIn(NESTED_PASS, out)
        self.assertNotIn("brandy", out)
        self.assertEqual(self._marks("hooks"), [f"KISS_HOME={brandx} APP={brandx / 'brandx'}"])
        self.assertIn(
            f"Set a new remote-access password for {brandx} (shown in its Settings).", out
        )
        self.assertNotIn("Remote access password:", out)
        self.assertNotIn("pw-.brandx", out)
        self.assertNotIn(NOTICE, out)
        self.assertNotIn("=== App install complete ===", out)
        self.assertIn(f"=== Brand home {brandx}: app install complete ===", out)
        self.assertNotIn("Log saved to", out)
        # The brand home's second pass (an update) keeps its password.
        out = self._run({
            "KISS_HOME": str(brandx),
            "KISS_SKIP_LAUNCH": "1",
            "KISS_UPDATE_LOCK_HELD": "1",
            "_KISS_BRAND_PASS": "1",
        })
        self.assertIn(
            f"Kept the remote-access password of {brandx} (shown in its Settings).", out
        )
        self.assertNotIn("Set a new remote-access password", out)
        self.assertIn('"remote_password": "pw-.brandx"', (brandx / "config.json").read_text())

    def test_vscode_notice_for_a_vscode_terminal(self) -> None:
        out = self._run({"KISS_SKIP_LAUNCH": "1", "TERM_PROGRAM": "vscode"})
        self.assertIn(f"*  {NOTICE}", out)
        self.assertIn(f"*  {NOTICE_URL}", out)
        self.assertLess(
            out.index("Remote access password:"), out.index(NOTICE), "the notice is last"
        )
        log = self.log.read_text()
        self.assertIn(f"*  {NOTICE}", log)
        self.assertIn(f"*  {NOTICE_URL}", log)
        self.assertEqual(out.count(NOTICE), 1)

    def test_vscode_notice_for_an_ipc_variable(self) -> None:
        out = self._run(
            {"KISS_SKIP_LAUNCH": "1", "VSCODE_GIT_IPC_HANDLE": "/tmp/vscode-git-1.sock"}
        )
        self.assertIn(NOTICE, out)
        out = self._run({"KISS_SKIP_LAUNCH": "1", "VSCODE_IPC_HOOK_CLI": "/tmp/vscode-ipc-1.sock"})
        self.assertIn(NOTICE, out)

    def test_vscode_notice_for_an_installed_extension(self) -> None:
        (self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-1.0.0").mkdir(parents=True)
        out = self._run({"KISS_SKIP_LAUNCH": "1", "KISS_WEB_PORT": "9999"})
        self.assertIn(f"*  {NOTICE}", out)
        self.assertIn(f"*  {NOTICE_REMOVED}", out)
        self.assertIn("*  Use the web app instead: http://127.0.0.1:9999", out)

    def test_vscode_notice_for_code_server_extension(self) -> None:
        (self.home / ".local" / "share" / "code-server" / "extensions"
         / "ksenxx.kiss-sorcar-2.3.4").mkdir(parents=True)
        out = self._run({"KISS_SKIP_LAUNCH": "1"})
        self.assertIn(NOTICE, out)

    def _daemon_pwd(self) -> Path:
        """The resolved directory the detached daemon was started in."""
        return Path(self._wait_mark("daemon-pwd", 1)[0]).resolve()

    def _wait_mark(self, name: str, count: int) -> list[str]:
        """The lines of mark *name* once it has *count* of them (a call the
        installer detaches lands a moment after it exits)."""
        deadline = time.monotonic() + 10.0
        while len(self._marks(name)) < count:
            self.assertLess(time.monotonic(), deadline, f"mark {name} never reached {count}")
            time.sleep(0.05)
        return self._marks(name)

    def test_installed_extension_is_uninstalled_and_the_web_app_told(self) -> None:
        # The retired extension's own Update button: a VS Code terminal
        # with two copies of the extension on the machine.
        local = self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12"
        remote = self.home / ".vscode-server" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12"
        local.mkdir(parents=True)
        remote.mkdir(parents=True)
        out = self._run({"KISS_SKIP_LAUNCH": "1", "TERM_PROGRAM": "vscode"})
        self.assertIn(f"*  {NOTICE_REMOVED}", out)
        self.assertNotIn(NOTICE_UNINSTALL, out)
        self.assertFalse(local.exists())
        self.assertFalse(remote.exists())
        notice_file = self.home / ".kiss" / NOTICE_FILE
        calls = self._wait_mark("code", 2)
        self.assertEqual(calls[0], "--uninstall-extension ksenxx.kiss-sorcar")
        self.assertEqual(calls[1], f"--reuse-window {notice_file}")
        notice = notice_file.read_text()
        self.assertIn("KISS Sorcar has left VS Code.", notice)
        self.assertIn(NOTICE_REMOVED.replace("The VS Code", "The KISS Sorcar VS Code"), notice)
        self.assertIn("This update uninstalled it", notice)
        self.assertIn(NOTICE_URL, notice)
        # The web app's notice flag and the old extension's reload marker.
        self.assertTrue((self.home / ".kiss" / ".vscode-retired").is_file())
        self.assertTrue((self.home / ".kiss" / ".extension-updated").is_file())
        # The progress step carried the whole notice for the old toast.
        log = self.log.read_text()
        self.assertIn(f"*  {NOTICE_REMOVED}", log)

    def test_failed_uninstall_tells_the_user_to_do_it(self) -> None:
        # No VS Code terminal, no running editor: the daemon ran this
        # update for an extension user, and the CLI cannot uninstall.
        ext = self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12"
        ext.mkdir(parents=True)
        (self.marks / "code-fails").touch()
        out = self._run({"KISS_SKIP_LAUNCH": "1"})
        self.assertIn(f"*  {NOTICE_REMOVED}", out)
        self.assertIn(f"*  {NOTICE_UNINSTALL}", out)
        self.assertTrue(ext.is_dir())
        self.assertEqual(self._marks("code"), ["--uninstall-extension ksenxx.kiss-sorcar"])
        notice = (self.home / ".kiss" / NOTICE_FILE).read_text()
        self.assertIn(NOTICE_UNINSTALL, notice)
        self.assertNotIn("This update uninstalled it", notice)
        self.assertTrue((self.home / ".kiss" / ".vscode-retired").is_file())
        self.assertTrue((self.home / ".kiss" / ".extension-updated").is_file())

    def test_vscode_terminal_without_the_extension_only_gets_the_notice(self) -> None:
        out = self._run({"KISS_SKIP_LAUNCH": "1", "TERM_PROGRAM": "vscode"})
        self.assertIn(f"*  {NOTICE_REMOVED}", out)
        self.assertNotIn(NOTICE_UNINSTALL, out)
        notice_file = self.home / ".kiss" / NOTICE_FILE
        self.assertEqual(self._wait_mark("code", 1), [f"--reuse-window {notice_file}"])
        notice = notice_file.read_text()
        self.assertNotIn("uninstall", notice.lower())
        # A developer's VS Code shell is not an extension user: no web
        # app notice, nothing for an old extension to reload on.
        self.assertFalse((self.home / ".kiss" / ".vscode-retired").exists())
        self.assertFalse((self.home / ".kiss" / ".extension-updated").exists())

    def test_every_editor_cli_uninstalls_and_the_first_opens_the_notice(self) -> None:
        # Four editors on PATH: each uninstalls; the first alone opens
        # the notice (picking it must not SIGPIPE the list under pipefail).
        for name in ("code-insiders", "cursor", "code-server"):
            os.symlink(self.stubs / "code", self.stubs / name)
        ext = self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12"
        ext.mkdir(parents=True)
        out = self._run({"KISS_SKIP_LAUNCH": "1", "TERM_PROGRAM": "vscode"})
        self.assertIn(f"*  {NOTICE_REMOVED}", out)
        calls = self._wait_mark("code", 5)
        self.assertEqual(calls[:4], ["--uninstall-extension ksenxx.kiss-sorcar"] * 4)
        self.assertEqual(calls[4], f"--reuse-window {self.home / '.kiss' / NOTICE_FILE}")
        self.assertTrue((self.home / ".kiss" / ".extension-updated").is_file())

    def test_no_editor_cli_leaves_the_notice_file_and_flag(self) -> None:
        ext = self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12"
        ext.mkdir(parents=True)
        out = self._run({"KISS_SKIP_LAUNCH": "1", "TERM_PROGRAM": "vscode", "KISS_CODE_CLI": ""})
        self.assertIn(f"*  {NOTICE_UNINSTALL}", out)
        self.assertEqual(self._marks("code"), [])
        self.assertTrue(ext.is_dir())
        self.assertTrue((self.home / ".kiss" / NOTICE_FILE).is_file())
        self.assertTrue((self.home / ".kiss" / ".vscode-retired").is_file())

    def test_the_notice_flag_lands_in_the_restarted_brand_daemons_home(self) -> None:
        brandx = self._brand_home("brandx")
        (self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12").mkdir(parents=True)
        self._write_service_unit(brandx)
        self._run({"KISS_SKIP_LAUNCH": "1"})
        self.assertTrue((brandx / ".vscode-retired").is_file())
        self.assertFalse((self.home / ".kiss" / ".vscode-retired").exists())

    def test_the_notice_names_the_brand_whose_daemon_is_restarted(self) -> None:
        # The brand's hook brands the app the nested pass installed; the
        # stock clone's run then restarts that daemon, so the terminal box
        # and the VS Code notice file name the brand, as the web app will.
        brandx = self._brand_home("brandx")
        hook = brandx / "post-install.d" / "hook"
        hook.write_text(
            HOOK + 'media="$KISS_APP_DIR/src/kiss/webapp/media"\n'
            'mkdir -p "$media"\n'
            'echo \'{"product_name": "Brand X"}\' > "$media/brand.json"\n'
        )
        (self.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.10.12").mkdir(parents=True)
        self._write_service_unit(brandx)
        out = self._run({"KISS_SKIP_LAUNCH": "1", "TERM_PROGRAM": "vscode"})
        self.assertIn("*  VS Code is no longer supported by Brand X.", out)
        self.assertNotIn(NOTICE, out)
        notice = (self.home / ".kiss" / NOTICE_FILE).read_text()
        self.assertIn("Brand X has left VS Code.", notice)
        self.assertIn("The Brand X VS Code extension has been removed", notice)
        self.assertNotIn("KISS Sorcar", notice)

    def test_daemon_serves_the_directory_the_installer_ran_from(self) -> None:
        self._run({"KISS_WEBAPP_WAIT_SECS": "0"})
        self.assertEqual(self._daemon_pwd(), self.tmp.resolve())

    def test_daemon_serves_home_when_run_from_the_bootstrap_clone(self) -> None:
        # The daemon's update and the old extension's Update button run
        # the installer from ~/.kiss/kiss_ai; that clone must not become
        # the user's work dir.
        clone = self.home / ".kiss" / "kiss_ai"
        clone.mkdir(parents=True)
        self._run({"KISS_WEBAPP_WAIT_SECS": "0"}, cwd=clone)
        self.assertEqual(self._daemon_pwd(), self.home.resolve())

    def test_daemon_serves_home_when_run_from_a_symlinked_stock_clone(self) -> None:
        # ~/.kiss is a symlink and the install goes into another home:
        # the clone under the stock directory is still not a work dir.
        stock = self.tmp / "stock-state"
        (stock / "kiss_ai").mkdir(parents=True)
        os.symlink(stock, self.home / ".kiss")
        brand = self.tmp / "brand-state"
        brand.mkdir()
        self._run(
            {"KISS_HOME": str(brand), "KISS_WEBAPP_WAIT_SECS": "0"},
            cwd=self.home / ".kiss" / "kiss_ai",
        )
        self.assertEqual(self._daemon_pwd(), self.home.resolve())

    def test_daemon_serves_home_when_run_from_a_custom_kiss_home(self) -> None:
        state = self.tmp / "state"
        inside = state / "kiss_ai"
        inside.mkdir(parents=True)
        self._run({"KISS_HOME": str(state), "KISS_WEBAPP_WAIT_SECS": "0"}, cwd=inside)
        self.assertEqual(self._daemon_pwd(), self.home.resolve())

    def test_no_vscode_notice_otherwise(self) -> None:
        # Another extension, and a file (not a directory) with the name,
        # do not make a VS Code user.
        (self.home / ".vscode" / "extensions" / "ms-python.python-1.0.0").mkdir(parents=True)
        (self.home / ".cursor" / "extensions").mkdir(parents=True)
        (self.home / ".cursor" / "extensions" / "ksenxx.kiss-sorcar-1.0.0").write_text("")
        out = self._run({"KISS_SKIP_LAUNCH": "1"})
        self.assertNotIn("VS Code", out)
        self.assertNotIn("VS Code", self.log.read_text())
        self.assertIn("*  Remote access password:  pw-.kiss", out)


if __name__ == "__main__":
    unittest.main()
