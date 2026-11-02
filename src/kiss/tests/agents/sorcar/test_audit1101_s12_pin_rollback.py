"""A re-registration that fails after replacing the pin leaves the old machine usable (S12 F1).

``add_machine`` re-registering an existing name publishes the new
daemon's certificate at the old machine's pin path and only then
rewrites the registry.  A Stop (the runner's injected
``KeyboardInterrupt``) or a registry-write error between the two left
the old URL and password in the registry with the new daemon's
certificate pinned, so the machine that was healthy a moment ago
failed TLS verification on every reconnect.  The old pin (or its
absence) is now restored whenever the new entry did not make it into
the registry.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.pause import Pause

ADDER = "s12-adder"


def _on_adder(_locals: dict[str, Any]) -> bool:
    return threading.current_thread().name == ADDER


def _daemon(tmp_path: Path, name: str) -> RemoteDaemon:
    return start_remote_daemon(tmp_path / name, password=f"pw-{name}", env={"HOME": str(tmp_path)})


def _register_in_thread(name: str, url: str, password: str) -> tuple[threading.Thread, dict]:
    """Run ``add_machine`` on a thread called :data:`ADDER`; return it and its outcome."""
    outcome: dict[str, Any] = {}

    def register() -> None:
        try:
            outcome["machine"] = mach.add_machine(name, url, password)
        except BaseException as exc:
            outcome["error"] = type(exc).__name__

    return threading.Thread(target=register, name=ADDER, daemon=True), outcome


def test_stop_after_the_pin_is_replaced_restores_the_old_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop between the pin replacement and the registry write keeps the old machine reachable."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    old = _daemon(tmp_path, "old")
    new: RemoteDaemon | None = None
    try:
        registered = mach.add_machine("box", old.url, "pw-old")
        pin = registered.cert_file().read_bytes()
        assert mach.machine_status(registered) == "reachable"
        new = _daemon(tmp_path, "new")
        adder, outcome = _register_in_thread("box", new.url, "pw-new")
        # Hold the re-registration at the registry write: the new
        # daemon's certificate is already at the old machine's pin path.
        with Pause(mach.save_machines.__code__, _on_adder) as held:
            adder.start()
            assert held.paused.wait(30), "the registry write was not reached"
            assert registered.cert_file().read_bytes() != pin
            assert inject_keyboard_interrupt(adder.ident or 0) == 1
        adder.join(10)
        assert not adder.is_alive()
        assert outcome == {"error": "KeyboardInterrupt"}
        assert mach.load_machines() == [registered]
        assert registered.cert_file().read_bytes() == pin
        assert mach.machine_status(registered) == "reachable"
    finally:
        mach.close_links()
        if new is not None:
            new.stop()
        old.stop()


@posix_only("chmod-based permission denial")
@pytest.mark.skipif(is_root(), reason="root ignores directory permissions")
def test_registry_write_failure_restores_the_old_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry that cannot be written leaves the old pin and the old entry in place."""
    home = tmp_path / "local"
    monkeypatch.setenv("KISS_HOME", str(home))
    old = _daemon(tmp_path, "old")
    new: RemoteDaemon | None = None
    try:
        registered = mach.add_machine("box", old.url, "pw-old")
        pin = registered.cert_file().read_bytes()
        new = _daemon(tmp_path, "new")
        # The home holds the registry and its staging file; the pins'
        # directory under it stays writable, so the pin is replaced
        # before the registry write fails.
        home.chmod(0o500)
        try:
            with pytest.raises(OSError):
                mach.add_machine("box", new.url, "pw-new")
        finally:
            home.chmod(0o700)
        assert mach.load_machines() == [registered]
        assert registered.cert_file().read_bytes() == pin
        assert mach.machine_status(registered) == "reachable"
    finally:
        mach.close_links()
        if new is not None:
            new.stop()
        old.stop()


def test_stop_after_the_registry_write_keeps_the_new_entry_and_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the entry is published, a Stop in the cache cleanup that follows rolls nothing back."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    old = _daemon(tmp_path, "old")
    new: RemoteDaemon | None = None
    try:
        registered = mach.add_machine("box", old.url, "pw-old")
        pin = registered.cert_file().read_bytes()
        new = _daemon(tmp_path, "new")
        adder, outcome = _register_in_thread("box", new.url, "pw-new")
        # ``save_machines`` retires the old entry's links after writing
        # the registry: hold the re-registration there.
        with Pause(mach.close_links.__code__, _on_adder) as held:
            adder.start()
            assert held.paused.wait(30), "the cache cleanup was not reached"
            assert inject_keyboard_interrupt(adder.ident or 0) == 1
        adder.join(10)
        assert outcome == {"error": "KeyboardInterrupt"}
        replacement = mach.Machine("box", new.url, "pw-new")
        assert mach.load_machines() == [replacement]
        assert replacement.cert_file().read_bytes() != pin
        assert mach.machine_status(replacement) == "reachable"
    finally:
        mach.close_links()
        if new is not None:
            new.stop()
        old.stop()


def test_a_first_registration_that_fails_leaves_no_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no pin to restore, a registration stopped at the registry write removes the new one."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    daemon = _daemon(tmp_path, "only")
    try:
        adder, outcome = _register_in_thread("box", daemon.url, "pw-only")
        with Pause(mach.save_machines.__code__, _on_adder) as held:
            adder.start()
            assert held.paused.wait(30), "the registry write was not reached"
            assert inject_keyboard_interrupt(adder.ident or 0) == 1
        adder.join(10)
        assert outcome == {"error": "KeyboardInterrupt"}
        assert mach.load_machines() == []
        assert not mach.Machine("box").cert_file().exists()
        assert not any((mach.kiss_home() / mach.MACHINE_CERTS_DIR).glob(".*.new"))
    finally:
        mach.close_links()
        daemon.stop()
