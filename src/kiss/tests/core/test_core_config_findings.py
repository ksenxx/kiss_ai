# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the core configuration findings F2, F3, F4 and F7.

* **F2** — ``save_api_key_to_shell`` rebuilt ``DEFAULT_CONFIG`` from
  scratch, silently reverting the ``max_budget`` that
  ``apply_config_to_env`` had just applied.
* **F3** — ``is_parallel`` was a ``DEFAULTS`` key with no reader; it was
  persisted and broadcast to every client forever.
* **F4** — the ``max_budget`` default was spelled ``200.0`` in
  ``config.py`` and ``100`` in ``vscode_config.py``.
* **F7** — ``save_config`` could only ever write keys listed in
  ``DEFAULTS``, silently dropping ``tunnel_token`` / ``email`` /
  ``skill_permissions`` / ``mcp_permissions`` passed *in*.

Everything here uses the real functions against a real temp config file
and (for the concurrency test) real separate OS processes.  No mocks,
patches, fakes or test doubles, and no LLM calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.config as config_module
import kiss.core.vscode_config as vscode_config
from kiss.core.vscode_config import (
    API_KEY_ENV_VARS,
    DEFAULTS,
    RETIRED_KEYS,
    apply_config_to_env,
    load_config,
    save_api_key_to_shell,
    save_config,
)

_PROBE_PROCESSES = 6
_PROBE_WRITES = 15


@pytest.fixture
def config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Point every config and shell-RC write at a temp dir.

    Redirects ``HOME`` (so the real ``_shell_rc_path`` writes into the
    temp tree instead of the developer's ``~/.zshrc``) and
    ``CONFIG_DIR`` / ``CONFIG_PATH``.  ``DEFAULT_CONFIG`` is restored
    field by field rather than by rebinding the name: the functions
    under test update the singleton in place, so a later test making a
    real model call would otherwise inherit this test's placeholder API
    keys and get a 401.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setitem(vars(vscode_config), "CONFIG_DIR", fake_home / ".kiss")
    monkeypatch.setitem(
        vars(vscode_config), "CONFIG_PATH", fake_home / ".kiss" / "config.json",
    )
    for key in API_KEY_ENV_VARS:
        monkeypatch.setitem(os.environ, key, os.environ.get(key, ""))
    saved_fields = config_module.DEFAULT_CONFIG.model_dump()
    try:
        yield fake_home / ".kiss" / "config.json"
    finally:
        for field, value in saved_fields.items():
            setattr(config_module.DEFAULT_CONFIG, field, value)


def test_saving_an_api_key_does_not_revert_max_budget(config_file: Path) -> None:
    """F2: an API key saved in the same settings round-trip must not reset the budget.

    The settings handler calls ``apply_config_to_env`` and then
    ``save_api_key_to_shell`` for every key in the same payload, so a
    user who changes the budget *and* pastes a key in one panel close
    used to end up with the live budget silently back at its default.
    """
    save_config({"max_budget": 5})
    apply_config_to_env(load_config())
    assert config_module.DEFAULT_CONFIG.max_budget == 5.0

    save_api_key_to_shell("ANTHROPIC_API_KEY", "sk-f2-probe")

    assert config_module.DEFAULT_CONFIG.max_budget == 5.0
    assert config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY == "sk-f2-probe"


def test_refreshing_config_still_picks_up_new_keys(config_file: Path) -> None:
    """F2: the refresh must keep doing its actual job — importing new env keys."""
    save_api_key_to_shell("OPENAI_API_KEY", "sk-f2-openai")
    assert config_module.DEFAULT_CONFIG.OPENAI_API_KEY == "sk-f2-openai"

    save_api_key_to_shell("OPENAI_API_KEY", "sk-f2-openai-rotated")
    assert config_module.DEFAULT_CONFIG.OPENAI_API_KEY == "sk-f2-openai-rotated"


def test_retired_is_parallel_key_is_forgotten(config_file: Path) -> None:
    """F3: a legacy ``is_parallel`` entry is neither read back nor re-persisted.

    Every previous release wrote the key, so simply dropping it from
    ``DEFAULTS`` would leave it being loaded, echoed in ``configData``
    and written back forever.  It must go through ``RETIRED_KEYS``.
    """
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps({"is_parallel": False, "max_budget": 7}), encoding="utf-8",
    )

    loaded = load_config()
    assert "is_parallel" not in loaded
    assert loaded["max_budget"] == 7

    save_config(loaded)
    on_disk = json.loads(config_file.read_text(encoding="utf-8"))
    assert "is_parallel" not in on_disk
    assert on_disk["max_budget"] == 7


def test_a_client_cannot_reintroduce_a_retired_key(config_file: Path) -> None:
    """F3: a stale client sending ``is_parallel`` in ``saveConfig`` is ignored."""
    save_config({"is_parallel": True, "max_budget": 9})
    assert "is_parallel" not in json.loads(config_file.read_text(encoding="utf-8"))
    assert "is_parallel" not in load_config()
    assert "is_parallel" in RETIRED_KEYS


def test_max_budget_default_agrees_across_the_two_config_layers(
    config_file: Path,
) -> None:
    """F4: a fresh install must report one budget default, not two.

    ``Config.max_budget`` is what channel-agent CLI runs default to and
    ``load_config()['max_budget']`` is what the settings panel and the
    daemon use.  With no ``config.json`` on disk these are the same
    product decision and must produce the same number.
    """
    assert not config_file.exists()
    assert config_module.Config().max_budget == float(load_config()["max_budget"])


def test_max_budget_default_has_a_single_source(config_file: Path) -> None:
    """F4: both layers must derive the default from one named constant."""
    authoritative = config_module.DEFAULT_MAX_BUDGET
    assert config_module.Config().max_budget == authoritative
    assert float(DEFAULTS["max_budget"]) == authoritative
    assert float(load_config()["max_budget"]) == authoritative


def test_junk_budget_falls_back_to_the_same_single_default(
    config_file: Path,
) -> None:
    """F4: the junk-value fallback path lands on the authoritative default too."""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps({"max_budget": "not-a-number"}), encoding="utf-8",
    )
    apply_config_to_env(load_config())
    assert config_module.DEFAULT_CONFIG.max_budget == config_module.DEFAULT_MAX_BUDGET


def test_save_config_round_trips_a_live_non_default_key(config_file: Path) -> None:
    """F7: extension-owned keys passed to ``save_config`` must reach the disk.

    ``tunnel_token`` (read by ``_resolve_tunnel_settings``) and
    ``skill_permissions`` (read by ``load_permission_rules``) are live
    settings that used to be writable only by hand-editing the file:
    ``save_config`` accepted them, returned without error, and dropped
    them, so the loss only surfaced after a daemon restart.
    """
    save_config({
        "tunnel_token": "tok-abc",
        "skill_permissions": {"Bash": "ask"},
        "max_budget": 7,
    })

    loaded = load_config()
    assert loaded["max_budget"] == 7
    assert loaded["tunnel_token"] == "tok-abc"
    assert loaded["skill_permissions"] == {"Bash": "ask"}


def test_save_config_still_preserves_untouched_non_default_keys(
    config_file: Path,
) -> None:
    """F7: writing new keys must not disturb keys only present on disk."""
    save_config({"email": "user@example.com"})
    save_config({"max_budget": 12})

    loaded = load_config()
    assert loaded["email"] == "user@example.com"
    assert loaded["max_budget"] == 12


def test_save_config_never_persists_api_keys(config_file: Path) -> None:
    """F7 guard: widening the write must not start leaking secrets to disk."""
    save_config({"ANTHROPIC_API_KEY": "sk-must-not-persist", "max_budget": 3})

    raw = config_file.read_text(encoding="utf-8")
    assert "sk-must-not-persist" not in raw
    assert "ANTHROPIC_API_KEY" not in json.loads(raw)


_WORKER = textwrap.dedent(
    """
    import sys
    from kiss.core.vscode_config import save_config

    index, writes = int(sys.argv[1]), int(sys.argv[2])
    for _ in range(writes):
        save_config({f"probe_{index}": index, "max_budget": 10 + index})
    """
)


def test_concurrent_processes_never_lose_a_non_default_key(
    tmp_path: Path,
) -> None:
    """F7 under contention: real parallel daemons must not clobber each other.

    ``save_config`` guards its read-merge-write with an ``flock``, so the
    file must end up holding every process's key.  Run as separate OS
    processes (not threads) because the ``flock`` is what makes this
    safe, and the in-process lock would mask a failure.
    """
    kiss_home = tmp_path / "kiss"
    kiss_home.mkdir()
    env = {**os.environ, "KISS_HOME": str(kiss_home)}

    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(index), str(_PROBE_WRITES)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(_PROBE_PROCESSES)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=180)
        assert process.returncode == 0, stderr

    stored = json.loads((kiss_home / "config.json").read_text(encoding="utf-8"))
    for index in range(_PROBE_PROCESSES):
        assert stored[f"probe_{index}"] == index, stored
    assert stored["max_budget"] in {10 + i for i in range(_PROBE_PROCESSES)}
