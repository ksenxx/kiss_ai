# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``scripts/reinstall-kiss-web-when-idle.sh``.

The helper re-runs ``install.sh`` from a detached job once the kiss-web
daemon reports no in-flight task.  On 2026-10-28 an ad-hoc copy of it was
started with ``launchctl submit``, which keeps the program alive: every time
it finished, launchd restarted it, so it reinstalled and restarted kiss-web
every ~20 s whenever no task was running.  The macOS test here is the
regression test for that incident: it launches the REAL helper through
launchd and checks that the worker ran exactly once.

Every test runs the real script copied into a throwaway checkout whose
``install.sh`` and ``scripts/check-kiss-web-active-tasks.py`` are small
stand-ins driven by files in that checkout (how many probes stay busy, what
each install run prints), against a throwaway ``$HOME`` whose
``.kiss/kiss/.venv/bin/python`` is this interpreter.  So nothing installs
anything, restarts the real daemon or touches the real ``~/.kiss``.  The
Linux launcher branches run on any OS through a ``PATH`` of symlinks to the
real tools plus stub ``uname`` / ``systemd-run`` / ``setsid`` binaries,
exactly as the install.sh tests stub ``git`` and ``curl``.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "reinstall-kiss-web-when-idle.sh"

# The probe exits 1 (busy) when its 0-based index is listed in ``busy-probes``
# (or that file says ``all``), else 0 (idle).
CHECK_STUB = """\
import pathlib, sys
root = pathlib.Path(__file__).resolve().parent.parent
count = root / "probe-count"
n = int(count.read_text() or 0)
count.write_text(str(n + 1))
busy = (root / "busy-probes").read_text().split()
sys.exit(1 if busy == ["all"] or str(n) in busy else 0)
"""

# Each install run pops the first line of ``install-outcomes`` and prints it;
# a line of the form ``exit:N`` exits N without printing a status.
INSTALL_STUB = """\
#!/bin/bash
root="$(cd "$(dirname "$0")" && pwd)"
outcome="$(head -n 1 "$root/install-outcomes")"
tail -n +2 "$root/install-outcomes" > "$root/install-outcomes.next"
mv "$root/install-outcomes.next" "$root/install-outcomes"
echo "run $(wc -l < "$root/install-outcomes") remaining; args: $*"
case "$outcome" in
    exit:*) exit "${outcome#exit:}" ;;
    *) echo "   $outcome" ;;
esac
"""

TOOLS = ["bash", "date", "dirname", "basename", "sed", "id", "mkdir", "grep", "sleep", "ps",
         "tr", "env", "head", "tail", "mv", "wc", "cat", "nohup"]


class ReinstallWhenIdleTest(unittest.TestCase):
    """Run the real helper against a throwaway checkout and ``$HOME``."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="kiss-reinstall-test-"))
        self.checkout = self.root / "checkout"
        (self.checkout / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.checkout / "scripts" / SCRIPT.name)
        (self.checkout / "scripts" / "check-kiss-web-active-tasks.py").write_text(CHECK_STUB)
        install = self.checkout / "install.sh"
        install.write_text(INSTALL_STUB)
        install.chmod(0o755)
        (self.checkout / "probe-count").write_text("0")
        self.home = self.root / "home"
        venv_bin = self.home / ".kiss" / "kiss" / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(sys.executable)
        self.label = f"kiss-reinstall-test-{os.getpid()}-{time.time_ns()}"
        self.script = self.checkout / "scripts" / SCRIPT.name
        self.logdir = self.checkout / "tmp" / "reinstall-kiss-web"
        self.log = self.logdir / f"{self.label}.log"
        self.env = {
            "HOME": str(self.home),
            "PATH": os.environ["PATH"],
            "KISS_REINSTALL_LABEL": self.label,
            "KISS_REINSTALL_POLL_SECONDS": "0.1",
        }

    def tearDown(self):
        # A worker a failed test left behind would otherwise poll a deleted
        # checkout until its 4 h deadline.
        if platform.system() == "Darwin":
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{self.label}"],
                           capture_output=True, check=False)
        pidfile = self.checkout / "worker-pid"
        if pidfile.exists():
            subprocess.run(["kill", pidfile.read_text().strip()], capture_output=True, check=False)
        shutil.rmtree(self.root, ignore_errors=True)

    def arrange(self, busy_probes, outcomes):
        """Set which probes (0-based, or ``"all"``) report busy and what each install run prints."""
        busy = busy_probes if busy_probes == "all" else " ".join(str(i) for i in busy_probes)
        (self.checkout / "busy-probes").write_text(busy)
        (self.checkout / "install-outcomes").write_text("".join(o + "\n" for o in outcomes))

    def run_worker(self, **env):
        return subprocess.run(["/bin/bash", str(self.script), "--worker"], env={**self.env, **env},
                              capture_output=True, text=True, timeout=60)

    def wait_for_log(self, needle, timeout=30):
        """Return the worker log once ``needle`` appears in it."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.log.exists() and needle in self.log.read_text():
                return self.log.read_text()
            time.sleep(0.1)
        text = self.log.read_text() if self.log.exists() else "<no log>"
        self.fail(f"{needle!r} not in worker log after {timeout}s:\n{text}")

    def probes(self):
        return int((self.checkout / "probe-count").read_text())

    def transcripts(self):
        return sorted(p.name for p in self.logdir.glob(f"{self.label}.install-*.log"))

    def test_worker_waits_for_idle_retries_after_deferral_then_succeeds(self):
        # busy, idle, busy (resets the streak), idle x3 -> attempt 1.
        self.arrange(busy_probes=[0, 2], outcomes=[
            "kiss-web has tasks in flight; restart deferred (re-run this script once they finish).",
            "kiss-web is up (1s): http://localhost:8787",
        ])
        result = self.run_worker()
        log = self.log.read_text()
        self.assertEqual(result.returncode, 0, log)
        self.assertEqual(result.stdout + result.stderr, "", "worker must log to its file only")
        self.assertIn("running install.sh (attempt 1)", log)
        self.assertIn("tasks became active again; waiting for an idle window and retrying.", log)
        self.assertIn("running install.sh (attempt 2)", log)
        self.assertIn("kiss-web restarted on the new code; done.", log)
        self.assertEqual(self.transcripts(),
                         [f"{self.label}.install-1.log", f"{self.label}.install-2.log"])
        # 6 probes before attempt 1 (see above), 3 idle probes before attempt 2.
        self.assertEqual(self.probes(), 9)
        transcript = (self.logdir / f"{self.label}.install-1.log").read_text()
        self.assertIn("args: --non-interactive", transcript)

    def test_worker_accepts_an_unchanged_runtime_as_success(self):
        self.arrange(busy_probes=[],
                     outcomes=["kiss-web already serves this runtime; leaving it running."])
        result = self.run_worker()
        self.assertEqual(result.returncode, 0, self.log.read_text())
        self.assertIn("done.", self.log.read_text())
        self.assertEqual(self.transcripts(), [f"{self.label}.install-1.log"])

    def test_worker_uses_python3_when_the_app_venv_is_absent(self):
        self.arrange(busy_probes=[], outcomes=["kiss-web is up (1s): http://localhost:8787"])
        (self.home / ".kiss" / "kiss" / ".venv" / "bin" / "python").unlink()
        bindir = self.root / "python3-bin"
        bindir.mkdir()
        (bindir / "python3").symlink_to(sys.executable)
        result = self.run_worker(PATH=f"{bindir}:{os.environ['PATH']}")
        self.assertEqual(result.returncode, 0, self.log.read_text())
        self.assertEqual(self.probes(), 3)

    def test_worker_gives_up_when_install_fails_without_a_restart_status(self):
        self.arrange(busy_probes=[], outcomes=["exit:3"])
        result = self.run_worker()
        self.assertEqual(result.returncode, 3)
        self.assertIn("ended without a restart status (exit=3); giving up.", self.log.read_text())

    def test_worker_gives_up_when_install_succeeds_without_a_restart_status(self):
        self.arrange(busy_probes=[], outcomes=["Installed nothing of note"])
        result = self.run_worker()
        self.assertEqual(result.returncode, 1)
        self.assertIn("ended without a restart status (exit=0); giving up.", self.log.read_text())

    def test_worker_gives_up_at_the_deadline_while_tasks_stay_in_flight(self):
        self.arrange(busy_probes="all", outcomes=["kiss-web is up (1s): http://localhost:8787"])
        result = self.run_worker(KISS_REINSTALL_DEADLINE_SECONDS="1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("deadline reached without an idle window; giving up.", self.log.read_text())
        self.assertEqual(self.transcripts(), [])

    # ---- launcher branches -------------------------------------------------

    def stub_path(self, stubs):
        """A PATH of the real tools the script needs plus the given stubs.

        ``systemd-run`` and ``setsid`` are present only when stubbed, so the
        launcher's branch choice depends on the stubs alone, on any OS.
        """
        bindir = self.root / "bin"
        bindir.mkdir()
        for tool in TOOLS:
            real = shutil.which(tool)
            self.assertIsNotNone(real, tool)
            (bindir / tool).symlink_to(str(real))
        for name, body in stubs.items():
            stub = bindir / name
            stub.write_text(body)
            stub.chmod(0o755)
        return str(bindir)

    def run_launcher(self, path):
        return subprocess.run(["/bin/bash", str(self.script)], env={**self.env, "PATH": path},
                              capture_output=True, text=True, timeout=60)

    def stubs_for(self, os_name, **stubs):
        """A PATH whose ``uname`` reports ``os_name`` plus the given stub binaries."""
        return self.stub_path({"uname": f"#!/bin/bash\necho {os_name}\n", **stubs})

    def test_linux_launcher_runs_the_worker_in_a_one_shot_transient_unit(self):
        self.arrange(busy_probes=[0], outcomes=["kiss-web is up (1s): http://localhost:8787"])
        # The stub records its arguments, applies the --setenv entries and
        # runs the trailing command detached, as systemd-run would.
        systemd_run = (
            "#!/bin/bash\n"
            f"printf '%s\\n' \"$@\" > '{self.checkout}/systemd-run-args'\n"
            "exports=()\n"
            "while [ $# -gt 0 ]; do\n"
            "  case \"$1\" in\n"
            "    --setenv=*) exports+=(\"${1#--setenv=}\") ;;\n"
            "    --*) ;;\n"
            "    *) break ;;\n"
            "  esac\n"
            "  shift\n"
            "done\n"
            "nohup env \"${exports[@]}\" \"$@\" </dev/null >/dev/null 2>&1 &\n"
            f"echo $! > '{self.checkout}/worker-pid'\n"
        )
        result = self.run_launcher(self.stubs_for("Linux", **{"systemd-run": systemd_run}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Detached job {self.label}", result.stdout)
        args = (self.checkout / "systemd-run-args").read_text().splitlines()
        self.assertEqual(args[:3], ["--user", f"--unit={self.label}", "--collect"])
        self.assertIn(f"--setenv=HOME={self.home}", args)
        self.assertIn(f"--setenv=KISS_REINSTALL_LABEL={self.label}", args)
        self.assertIn("--setenv=KISS_REINSTALL_POLL_SECONDS=0.1", args)
        self.assertEqual(args[-3:], ["/bin/bash", str(self.script), "--worker"])
        self.assertTrue(os.path.isabs(args[-2]), "a transient unit starts in $HOME: absolute path")
        log = self.wait_for_log("done.")
        self.assertEqual(log.count("worker started"), 1)

    def test_launcher_falls_back_to_setsid_without_launchd_or_systemd(self):
        self.arrange(busy_probes=[], outcomes=["kiss-web is up (1s): http://localhost:8787"])
        setsid = ("#!/bin/bash\n[ \"$1\" = -f ] && shift\n\"$@\" &\n"
                  f"echo $! > '{self.checkout}/worker-pid'\n")
        result = self.run_launcher(self.stubs_for("Linux", setsid=setsid))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Detached job {self.label}", result.stdout)
        log = self.wait_for_log("done.")
        self.assertEqual(log.count("worker started"), 1)

    def test_launcher_reports_a_detacher_that_fails(self):
        self.arrange(busy_probes=[], outcomes=["kiss-web is up (1s): http://localhost:8787"])
        failing = "#!/bin/bash\necho \"$(basename \"$0\") refused\" >&2\nexit 1\n"
        detachers = [("Darwin", "launchctl"), ("Linux", "systemd-run"), ("Linux", "setsid")]
        for os_name, detacher in detachers:
            with self.subTest(detacher=detacher):
                result = self.run_launcher(self.stubs_for(os_name, **{detacher: failing}))
                shutil.rmtree(self.root / "bin")
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{detacher} refused", result.stderr)
                self.assertNotIn("Detached job", result.stdout)
                self.assertFalse(self.log.exists())

    def test_launcher_refuses_without_a_way_to_detach(self):
        self.arrange(busy_probes=[], outcomes=["kiss-web is up (1s): http://localhost:8787"])
        result = self.run_launcher(self.stubs_for("Linux"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("need launchctl (macOS), systemd-run or setsid", result.stderr)
        self.assertFalse(self.log.exists())

    @unittest.skipUnless(platform.system() == "Darwin", "launchd is macOS only")
    def test_macos_launcher_runs_the_worker_exactly_once_and_unloads_its_label(self):
        """Regression test for the 2026-10-28 reinstall loop.

        ``launchctl submit`` kept the helper alive, so launchd restarted it
        after every exit and it reinstalled every ~20 s.  The launcher must
        load a one-shot job: after the worker reports ``done.`` there must be
        no second ``worker started``, no further probes, and no label left
        in ``launchctl list``.
        """
        domain = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}"],
                                capture_output=True, check=False)
        if domain.returncode != 0:
            self.skipTest("no launchd GUI domain for this user (not a logged-in session)")
        self.arrange(busy_probes=[0], outcomes=["kiss-web is up (1s): http://localhost:8787"])
        result = self.run_launcher(os.environ["PATH"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Detached job {self.label}", result.stdout)
        log = self.wait_for_log("done.")
        self.assertIn("worker started", log)
        probes_at_done = self.probes()
        time.sleep(3)  # launchd restarted the submit-job within a second
        log = self.log.read_text()
        self.assertEqual(log.count("worker started"), 1, log)
        self.assertEqual(self.probes(), probes_at_done, "a relaunched worker would keep probing")
        self.assertEqual(self.transcripts(), [f"{self.label}.install-1.log"])
        listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True,
                                check=True).stdout
        self.assertNotIn(self.label, listed, "the one-shot job must unload its own label")


if __name__ == "__main__":
    unittest.main()
