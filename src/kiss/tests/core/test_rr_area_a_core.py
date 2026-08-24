# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the area-A redundancy and race fixes in kiss.core.

Covers:

* A-R3 — ``_NON_RETRYABLE_ERROR_TYPES`` deduplication: after dropping the
  shadowed ``"PermissionDeniedError"`` entry, exceptions whose type name
  contains ``PermissionDenied`` (including ``PermissionDeniedError``
  itself) must still classify as non-retryable.
* A-R4 — ``apply_config_to_env`` now delegates its value coercion to
  ``sanitize_config``; every junk shape (bool, non-numeric string,
  NaN/Infinity) must still fall back to the default budget and every
  sane shape must still be applied.
* A-RC2 — ``save_api_key_to_shell`` takes the shared config flock around
  its read-modify-replace of the shell RC file, so concurrent savers in
  separate processes cannot lose each other's keys.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.vscode_config as vscode_config
from kiss.core.kiss_agent import _is_retryable_error
from kiss.core.vscode_config import DEFAULTS, apply_config_to_env


class PermissionDeniedError(Exception):
    """Same type name a provider SDK raises on a permission failure."""


class PermissionDenied(Exception):  # noqa: N818 -- mirrors the provider SDK's type name
    """The shorter provider type name for the same failure."""


class AuthenticationError(Exception):
    """Provider type name for an authentication failure."""


class FlakyTransportError(Exception):
    """A transient failure that must remain retryable."""


class TestNonRetryableClassification:
    """A-R3: dropping the shadowed entry must not change behavior."""

    def test_permission_denied_error_type_is_non_retryable(self) -> None:
        """The longer type name still matches via the ``PermissionDenied`` substring."""
        assert _is_retryable_error(PermissionDeniedError("boom")) is False

    def test_permission_denied_type_is_non_retryable(self) -> None:
        """The shorter provider type name matches directly."""
        assert _is_retryable_error(PermissionDenied("boom")) is False

    def test_authentication_error_type_is_non_retryable(self) -> None:
        """The other listed type name is unaffected by the deduplication."""
        assert _is_retryable_error(AuthenticationError("boom")) is False

    def test_unrelated_error_stays_retryable(self) -> None:
        """A transient error with a clean message must remain retryable."""
        assert _is_retryable_error(FlakyTransportError("connection reset")) is True


@pytest.fixture()
def _restore_default_budget() -> Iterator[None]:
    """Snapshot and restore ``DEFAULT_CONFIG.max_budget`` around a test."""
    from kiss.core import config as config_module

    saved = config_module.DEFAULT_CONFIG.max_budget
    yield
    config_module.DEFAULT_CONFIG.max_budget = saved


@pytest.mark.usefixtures("_restore_default_budget")
class TestApplyConfigToEnv:
    """A-R4: coercion now shared with ``sanitize_config``, behavior kept."""

    def _applied(self, value: object) -> float:
        """Apply a config with the given ``max_budget`` and read the result."""
        from kiss.core import config as config_module

        apply_config_to_env({"max_budget": value})
        return config_module.DEFAULT_CONFIG.max_budget

    def test_numeric_value_is_applied(self) -> None:
        """A finite number is used as-is."""
        assert self._applied(12.5) == 12.5

    def test_numeric_string_is_coerced(self) -> None:
        """A numeric string from a hand-edited config.json is accepted."""
        assert self._applied("7.25") == 7.25

    def test_boolean_falls_back_to_default(self) -> None:
        """``True`` must not silently become a 1.0 budget."""
        assert self._applied(True) == float(DEFAULTS["max_budget"])

    def test_non_numeric_string_falls_back_to_default(self) -> None:
        """Junk text cannot crash the handler nor poison the budget."""
        assert self._applied("plenty") == float(DEFAULTS["max_budget"])

    def test_nan_falls_back_to_default(self) -> None:
        """NaN survives json.load but must not disable budget checks."""
        assert self._applied(float("nan")) == float(DEFAULTS["max_budget"])

    def test_infinity_falls_back_to_default(self) -> None:
        """Infinity would make every ``cost > max_budget`` check False."""
        assert self._applied(float("inf")) == float(DEFAULTS["max_budget"])

    def test_missing_key_uses_default(self) -> None:
        """A config without the key applies the default budget."""
        from kiss.core import config as config_module

        apply_config_to_env({})
        assert config_module.DEFAULT_CONFIG.max_budget == float(
            DEFAULTS["max_budget"]
        )

    def test_result_is_finite_float(self) -> None:
        """Whatever the input, the applied budget is a finite float."""
        for junk in (True, "x", float("nan"), float("-inf"), None, [1]):
            value = self._applied(junk)
            assert isinstance(value, float)
            assert math.isfinite(value)


_SAVER_SCRIPT = """
import sys
from kiss.core.vscode_config import save_api_key_to_shell

key_name, key_value = sys.argv[1], sys.argv[2]
save_api_key_to_shell(key_name, key_value)
"""


class TestSaveApiKeyToShellCrossProcessRace:
    """A-RC2: concurrent RC savers must not lose each other's keys.

    Before the fix, ``save_api_key_to_shell`` read the RC file, edited
    the lines in memory, and atomically replaced the file with no
    cross-process guard: two processes that read the same snapshot
    each published a file missing the other's export line.  The fix
    serializes the read-modify-replace under an fcntl flock keyed to
    the RC file itself (a ``<rc>.kiss.lock`` sidecar), so savers agree
    on the lock no matter which ``KISS_HOME`` they run under.

    This is a real end-to-end reproduction: several *processes* invoke
    the real function against one shared HOME at the same time, and
    every export line must survive.
    """

    def test_concurrent_process_savers_all_keys_survive(
        self, tmp_path: Path,
    ) -> None:
        """Every concurrently saved key ends up in the RC file."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        env = {
            **os.environ,
            "HOME": str(fake_home),
            "KISS_HOME": str(fake_home / ".kiss"),
            "SHELL": "/bin/bash",
        }
        keys = [f"GEMINI_API_KEY_{i}" for i in range(6)]

        for round_no in range(3):
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", _SAVER_SCRIPT, key, f"v{round_no}-{key}"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for key in keys
            ]
            for proc in procs:
                _, stderr = proc.communicate(timeout=120)
                assert proc.returncode == 0, stderr.decode()

            rc_text = (fake_home / ".bashrc").read_text(encoding="utf-8")
            for key in keys:
                assert f"export {key}=v{round_no}-{key}" in rc_text, (
                    f"round {round_no}: key {key} was lost by a concurrent saver"
                )
            # Replacement, not accumulation: exactly one line per key.
            for key in keys:
                assert rc_text.count(f"export {key}=") == 1

    def test_savers_with_different_kiss_homes_share_one_rc_lock(
        self, tmp_path: Path,
    ) -> None:
        """Two daemons with distinct KISS_HOMEs edit one RC without losses.

        The RC file is selected from ``$HOME``, so daemons that share a
        HOME but run with different ``KISS_HOME`` values write the SAME
        file.  Before the fix each of them flocked its own
        ``<KISS_HOME>/.config.lock``, which excluded nobody: both read
        one RC snapshot and the later atomic replace erased the earlier
        writer's export line.  The lock is now a sidecar keyed to the RC
        itself, so the KISS_HOME value must not matter.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        keys = [f"MOONSHOT_API_KEY_{i}" for i in range(6)]

        for round_no in range(5):
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", _SAVER_SCRIPT, key, f"kh{round_no}-{key}"],
                    env={
                        **os.environ,
                        "HOME": str(fake_home),
                        # Every saver gets its own KISS_HOME, like two
                        # daemons deployed side by side for one user.
                        "KISS_HOME": str(fake_home / f".kiss-{index}"),
                        "SHELL": "/bin/bash",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for index, key in enumerate(keys)
            ]
            for proc in procs:
                _, stderr = proc.communicate(timeout=120)
                assert proc.returncode == 0, stderr.decode()

            rc_text = (fake_home / ".bashrc").read_text(encoding="utf-8")
            for key in keys:
                assert f"export {key}=kh{round_no}-{key}" in rc_text, (
                    f"round {round_no}: key {key} was lost by a saver "
                    "holding a different KISS_HOME's lock"
                )
                assert rc_text.count(f"export {key}=") == 1

    def test_rc_file_stays_private(self, tmp_path: Path) -> None:
        """The RC file holding keys is written with mode 0600."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        env = {
            **os.environ,
            "HOME": str(fake_home),
            "KISS_HOME": str(fake_home / ".kiss"),
            "SHELL": "/bin/bash",
        }
        done = subprocess.run(
            [sys.executable, "-c", _SAVER_SCRIPT, "OPENAI_API_KEY", "sk-x"],
            env=env,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr.decode()
        assert ((fake_home / ".bashrc").stat().st_mode & 0o077) == 0

    def test_in_process_threads_all_keys_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concurrent threads in one process are serialized by the in-process lock."""
        import threading

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setitem(vars(vscode_config), "CONFIG_DIR", fake_home / ".kiss")
        monkeypatch.setitem(
            vars(vscode_config), "CONFIG_PATH", fake_home / ".kiss" / "config.json",
        )
        keys = [f"ZAI_API_KEY_{i}" for i in range(8)]
        for key in keys:
            monkeypatch.delenv(key, raising=False)

        threads = [
            threading.Thread(
                target=vscode_config.save_api_key_to_shell, args=(key, f"tv-{key}"),
            )
            for key in keys
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        rc_text = (fake_home / ".bashrc").read_text(encoding="utf-8")
        for key in keys:
            assert f"export {key}=tv-{key}" in rc_text
