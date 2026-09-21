# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-process race: legacy-RC key migration vs. concurrent key deletion.

``_migrate_legacy_rc_keys()`` used to make its migrate-or-not decision
(read the canonical store, scan/source the shell RCs) under only the
process-local ``_config_lock``; the cross-process sidecar flock was
acquired only later, around the store edit that is now
``_edit_api_keys_env_file_locked()``, when the stale snapshot was
already committed to.  A second process sharing the
same ``$KISS_HOME`` could therefore complete a full ``save_api_key(key,
"")`` deletion — canonical store scrubbed under the sidecar flock, RC
assignment removed under the RC flock — inside that window, after which
the migrating process wrote its stale RC observation back into the
canonical store, resurrecting the key the user just deleted.

Reproduction (all real, no mocks): two subprocesses share an isolated
``HOME``/``KISS_HOME``.  The ``.bashrc`` assembles the key indirectly
(``export GEMINI_API_KEY="${V}-value"``) so the textual scan must skip
it (``_shell_would_expand``) and the migration falls into the
RC-*sourcing* path, whose duration is the real race window; a ``sleep``
inside the RC widens it deterministically.  The RC also touches a
sentinel file before sleeping, proving process A has committed to its
snapshot before process B starts.  B waits for the sentinel and runs the
complete public deletion.  Before the fix, A then appended
``export GEMINI_API_KEY=legacy-value`` to the store B had just scrubbed.

The fix holds the canonical store's sidecar flock across the whole
read-scan-source-write migration transaction, and across
``save_api_key``'s whole store-edit-plus-RC-scrub critical section, so
the two operations serialize in either order: migration first — the
deletion then removes the migrated key; deletion first — the migration
finds nothing to import.  Either way the key stays deleted.

Branch coverage: the fix adds no new branches — it widens the span of
existing lock acquisitions so that the store edit,
``_edit_api_keys_env_file_locked``, runs only inside a caller-held
``_config_lock`` + store-flock section (the former standalone wrapper
that took those locks itself had no callers and was removed).  That
locked helper is covered here (via ``load_api_keys``/``save_api_key``)
and by the pre-existing ``test_vscode_config.py`` suites, which drive
every store-edit branch through it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase

from kiss.tests.conftest import posix_only

# The race window is a bash ``~/.bashrc`` sourced through ``/bin/bash``
# with ``touch``/``sleep`` inside it: a POSIX-shell mechanism.  Windows
# has no shell RC to migrate from (and ``Path.home()`` ignores ``HOME``).
pytestmark = posix_only("legacy shell-RC key migration sources ~/.bashrc via bash")

_MIGRATOR = r"""
from kiss.core.vscode_config import load_api_keys
load_api_keys()
"""

_DELETER = r"""
import os, sys, time
from kiss.core.vscode_config import save_api_key
sentinel = sys.argv[1]
deadline = time.time() + 20
while not os.path.exists(sentinel):
    if time.time() > deadline:
        sys.exit(2)
    time.sleep(0.005)
save_api_key("GEMINI_API_KEY", "")
sys.exit(0)
"""


class TestMigrationCannotResurrectDeletedKey(TestCase):
    """A completed cross-process deletion must survive a racing migration."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_racing_delete_is_not_resurrected(self) -> None:
        home = Path(self.tmpdir) / "home"
        kiss_home = Path(self.tmpdir) / "kiss"
        home.mkdir()
        kiss_home.mkdir()
        sentinel = Path(self.tmpdir) / "sourcing-started"
        # Indirect assembly forces the sourcing path (textual scan skips
        # expansion-dependent values); the sentinel + sleep pins process A
        # inside its snapshot window while B completes the deletion.
        (home / ".bashrc").write_text(
            f"touch {sentinel}\n"
            "sleep 2\n"
            "V=legacy\n"
            'export GEMINI_API_KEY="${V}-value"\n',
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["HOME"] = str(home)
        env["KISS_HOME"] = str(kiss_home)
        env["SHELL"] = "/bin/bash"
        env.pop("GEMINI_API_KEY", None)

        migrator = subprocess.Popen(
            [sys.executable, "-c", _MIGRATOR],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deleter = subprocess.Popen(
            [sys.executable, "-c", _DELETER, str(sentinel)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            m_out, m_err = migrator.communicate(timeout=60)
            d_out, d_err = deleter.communicate(timeout=60)
        finally:
            for proc in (migrator, deleter):
                if proc.poll() is None:
                    proc.kill()
        self.assertEqual(
            migrator.returncode, 0, f"migrator failed:\n{m_out}\n{m_err}",
        )
        self.assertEqual(
            deleter.returncode, 0, f"deleter failed:\n{d_out}\n{d_err}",
        )
        # Give a straggling non-atomic observer no excuse: both processes
        # have exited; the store's final content is settled.
        time.sleep(0.05)
        store = kiss_home / "api_keys.env"
        store_text = store.read_text(encoding="utf-8") if store.exists() else ""
        self.assertNotIn(
            "GEMINI_API_KEY",
            store_text,
            "deleted key was resurrected into the canonical store: "
            f"{store_text!r}",
        )
        rc_text = (home / ".bashrc").read_text(encoding="utf-8")
        self.assertNotIn("export GEMINI_API_KEY", rc_text)
