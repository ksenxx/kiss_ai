# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Verify the root ``conftest.py`` raises ``RLIMIT_NOFILE`` for test sessions.

macOS ships with a default soft file-descriptor limit of 256, which the full
test suite exhausts (many concurrent UDS sockets and subprocesses), surfacing
as ``OSError: [Errno 24] Too many open files`` inside asyncio
``socket.accept``. The root ``conftest.py`` raises the soft limit to at least
4096 at import time so every pytest invocation gets the bigger cap without
relying on a shell-level ``ulimit`` setting. This test pins that behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


class TestRLimitNoFileBump(unittest.TestCase):
    """End-to-end checks for the conftest-level FD soft-limit bump."""

    def test_current_pytest_session_has_at_least_4096_open_files_soft_limit(
        self,
    ) -> None:
        """The currently running pytest process must already have soft >= 4096.

        This validates the in-process effect of the root ``conftest.py``: by
        the time any test runs, the bump has already executed at module
        import.
        """
        if sys.platform == "win32":
            self.skipTest("RLIMIT_NOFILE bump is POSIX-only.")
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        self.assertGreaterEqual(
            soft,
            4096,
            f"Root conftest.py should have raised RLIMIT_NOFILE soft limit to "
            f">= 4096 for this pytest session, but it is {soft}.",
        )

    def test_fresh_interpreter_loading_conftest_bumps_low_soft_limit(self) -> None:
        """A fresh Python with macOS-default soft=256 ends up with soft==4096.

        Spawns ``bash -c 'ulimit -Sn 256 && exec python <script>'`` so the
        child starts with the historical macOS default: soft=256 while the
        hard cap stays at the inherited (essentially unlimited) value. The
        child imports the root ``conftest.py`` exactly the way pytest does
        (via ``runpy.run_path``) and then prints its own ``RLIMIT_NOFILE``
        soft limit. We assert the printed value is at least 4096, proving the
        bump is effective end-to-end even when the parent shell has the bad
        default.

        Note: ``ulimit -Sn`` is used (not bare ``ulimit -n``) because the
        latter also lowers the hard cap, which unprivileged processes cannot
        raise back — that simulates a stricter sandbox than the macOS default
        and is not what we want to verify here.
        """
        if sys.platform == "win32":
            self.skipTest("RLIMIT_NOFILE bump is POSIX-only.")

        # tests/test_rlimit_nofile_bump.py -> tests -> kiss -> src -> repo root
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        conftest_path = os.path.join(repo_root, "conftest.py")
        self.assertTrue(
            os.path.isfile(conftest_path),
            f"Expected root conftest.py at {conftest_path}",
        )

        child_program = textwrap.dedent(
            f"""
            import resource, runpy, sys
            before, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            # Execute the root conftest.py as a module the same way pytest
            # loads it: top-level statements run, raising RLIMIT_NOFILE.
            runpy.run_path({conftest_path!r}, run_name="conftest")
            after, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"BEFORE={{before}} AFTER={{after}}")
            sys.exit(0 if after >= 4096 else 1)
            """
        ).lstrip()

        # Write the child program to a tempfile to bypass all shell-quoting
        # hazards (the script contains newlines, single quotes, parens, etc.).
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(child_program)
            child_script = tf.name
        try:
            cmd = f"ulimit -Sn 256 && exec {sys.executable} {child_script}"
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                cwd=repo_root,
            )
        finally:
            os.unlink(child_script)

        combined = result.stdout + "\n" + result.stderr
        self.assertEqual(
            result.returncode,
            0,
            f"Child interpreter run failed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )
        self.assertIn(
            "AFTER=",
            combined,
            f"Did not find AFTER= marker in child output:\n{combined}",
        )
        after_value = int(
            combined.split("AFTER=", 1)[1].split()[0].strip().rstrip(",;")
        )
        self.assertGreaterEqual(
            after_value,
            4096,
            f"Child interpreter started with ulimit -n 256 but conftest.py "
            f"bump left RLIMIT_NOFILE soft={after_value} (expected >= 4096). "
            f"Full output:\n{combined}",
        )
        # And sanity: BEFORE must really have been 256, otherwise the test
        # isn't exercising the macOS-default starting point.
        before_value = int(
            combined.split("BEFORE=", 1)[1].split()[0].strip().rstrip(",;")
        )
        self.assertEqual(
            before_value,
            256,
            f"Child interpreter should have started with soft=256 (set via "
            f"`ulimit -Sn 256`) but reported BEFORE={before_value}. "
            f"Full output:\n{combined}",
        )


if __name__ == "__main__":
    unittest.main()
