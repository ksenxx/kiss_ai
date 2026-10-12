"""``enrolled_services()`` answers from the vault on disk and starts no daemon.

The sidebar channel status (``auth_status.vault_services``) asks which
services are enrolled.  That used to be a ``status`` op, and every op
first calls ``ensure_daemon()``: a mere status probe from a process whose
default home was the stock ``~/.kiss`` spawned a ``muse_auth.daemon``
there as a side effect.  The daemon's own ``status`` answer is the
``vault/*.json`` listing, so the client reads it directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir
from kiss.agents.third_party_agents.muse_auth.client import enrolled_services
from kiss.tests.agents.seas.test_audit1012_sea_lint_no_side_effects import muse_daemons_of


def test_enrolled_services_lists_the_vault_without_spawning_a_daemon(
    isolated_kiss_home: Path,
) -> None:
    home = Path(os.environ["KISS_HOME"])
    vault = muse_auth_dir() / "vault"
    assert enrolled_services() == []
    vault.mkdir(parents=True)
    (vault / "slack-acme-0123abcd.json").write_text("{}")
    (vault / "github.json").write_text("{}")
    (vault / "notes.txt").write_text("not a credential")

    assert enrolled_services() == ["github", "slack-acme-0123abcd"]

    assert not (muse_auth_dir() / "authd.sock").exists(), "a daemon socket appeared"
    assert muse_daemons_of(home) == [], "a status probe spawned muse_auth.daemon"


@pytest.mark.parametrize("missing", ["vault", "muse_auth"])
def test_enrolled_services_is_empty_when_the_vault_is_absent(
    isolated_kiss_home: Path, missing: str
) -> None:
    if missing == "vault":
        muse_auth_dir().mkdir(parents=True, exist_ok=True)
    assert enrolled_services() == []
    assert not (muse_auth_dir() / "authd.sock").exists()
