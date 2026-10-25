# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A refused ``setWorkDir`` answers its caller instead of going silent.

``ServerApi.dispatch`` blanks a filesystem root to ``""`` before the
handler runs, and ``_apply_new_work_dir`` refuses a root that reaches
it anyway.  Before the fix ``_cmd_set_work_dir`` returned early on
both paths without the ``configData`` a ``reply: true`` caller waits
for: a daemon proxying a pick for a machine-bound tab
(``sidebar_panels.machine_work_dirs``) waited out its 8 s link timeout
and reported a timeout instead of the refusal, and a local picker got
no feedback at all.  Now every refusal tells the requesting connection
with a ``notice`` and still answers ``reply: true`` with the
(unchanged) ``configData``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.tests.server.conftest import (
    clear_agent_states,
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)

pytestmark = pytest.mark.usefixtures("kiss_home")


@pytest.fixture
def server(kiss_home: Path) -> Iterator[tuple[Any, list[dict[str, Any]], str]]:
    """A capturing ``AgentServer`` on a scratch home, plus a real directory to adopt."""
    clear_agent_states()
    tmpdir = tempfile.mkdtemp()
    saved = redirect_persistence(tmpdir)
    srv, events = make_capturing_server()
    yield srv, events, tmpdir
    clear_agent_states()
    restore_persistence(saved)
    shutil.rmtree(tmpdir, ignore_errors=True)


def _by_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("type") == kind]


@pytest.mark.parametrize("work_dir", [os.path.abspath(os.sep), ""])
def test_refused_work_dir_is_reported_and_still_replies(
    server: tuple[Any, list[dict[str, Any]], str], kiss_home: Path, work_dir: str,
) -> None:
    """A root (or the ``""`` dispatch blanks it to) keeps the old directory, says so, replies."""
    srv, events, _ = server
    before = srv.work_dir

    srv._handle_command({
        "type": "setWorkDir", "workDir": work_dir, "reply": True, "connId": "conn-pick",
    })

    assert srv.work_dir == before
    assert not (kiss_home / "config.json").exists()
    (notice,) = _by_type(events, "notice")
    assert notice["connId"] == "conn-pick"
    assert "cannot be used as the working directory" in notice["text"]
    (config,) = _by_type(events, "configData")
    assert config["connId"] == "conn-pick"
    assert config["config"]["work_dir"] == before
    assert _by_type(events, "workDirChanged") == []


def test_refusal_without_reply_only_notifies(
    server: tuple[Any, list[dict[str, Any]], str],
) -> None:
    srv, events, _ = server

    srv._handle_command({"type": "setWorkDir", "workDir": "", "connId": "conn-pick"})

    assert [e["type"] for e in events] == ["notice"]


def test_adopted_work_dir_replies_without_a_notice(
    server: tuple[Any, list[dict[str, Any]], str], kiss_home: Path,
) -> None:
    """The accepted path is unchanged: ``workDirChanged`` and the reply, no notice."""
    srv, events, tmpdir = server

    srv._handle_command({
        "type": "setWorkDir", "workDir": tmpdir, "reply": True, "connId": "conn-pick",
    })

    assert srv.work_dir == tmpdir
    assert _by_type(events, "notice") == []
    assert [e["workDir"] for e in _by_type(events, "workDirChanged")] == [tmpdir]
    (config,) = _by_type(events, "configData")
    assert config["config"]["work_dir"] == tmpdir
    assert (kiss_home / "config.json").exists()
