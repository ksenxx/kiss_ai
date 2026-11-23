# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for projects: storage, commands, history page and run-time settings.

A project is a row of ``history.db``'s ``projects`` table and a chat is
in at most one (``chat_summaries.project_id``).  The tests send the real
commands to an ``AgentServer`` whose broadcasts are captured, on a
scratch database.  For the settings a task of a project runs with, real
runs go through ``_cmd_run`` with an agent whose ``run`` records what
the runner started it with (and what a real shell command of it finds
in its environment) instead of calling a model; a project's own
Anthropic key is what makes the run's model available, so no key of the
environment is needed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import projects, sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.chrome_bridge import CONFIG_FILE_NAME, ChromeBridge, extension_dir
from kiss.agents.sorcar.machines import HISTORY_MAX_PAGE, HISTORY_PAGE
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import config as config_module
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import environ_without
from kiss.core.models.model_info import get_available_models
from kiss.core.secret_redaction import redact_secrets
from kiss.core.user_config import API_KEY_ENV_VARS, get_current_api_keys, load_config
from kiss.core.utils import finish
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.commands import _PROJECTS_PUBLISH_LOCK
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.server.web_server import WebPrinter
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer
from kiss.tests.server.conftest import (
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

_PAPER_KEY = "sk-ant-paper-0123456789abcdef"
_THESIS_KEY = "sk-ant-thesis-0123456789abcdef"
# Characters 8 to 12 of the key in the command's environment ("paper",
# "thesi"): enough to tell the keys apart, and no secret to redact.
_KEY_PROBE = "printenv ANTHROPIC_API_KEY | cut -c8-12"
_WAIT_S = 60.0


@pytest.fixture(autouse=True)
def _scratch_history(tmp_path: Path) -> Iterator[None]:
    """Run each test on a history database of its own."""
    saved = redirect_persistence(tmp_path)
    yield
    th._close_thread_db()
    restore_persistence(saved)


def _send(server: AgentServer, events: list[dict[str, Any]], cmd: dict[str, Any]) -> list[dict]:
    """Send *cmd* to *server* and return the events it broadcast."""
    events.clear()
    server._handle_command(cmd)
    return list(events)


def _product_form() -> dict[str, Any]:
    """The settings form of a new project: every field holds the product's setting."""
    config = load_config()
    return {key: config[key] for key in projects.CONFIG_KEYS} | {
        "api_keys": dict(get_current_api_keys()),
    }


def _stored(server: AgentServer, events: list[dict[str, Any]], settings: object) -> dict[str, Any]:
    """Save a new project with *settings* and return what was stored of them."""
    saved = _send(server, events, {"type": "saveProject", "project": {
        "name": "Probe", "settings": settings,
    }})[0]
    project = th._get_project(saved["id"])
    assert project is not None
    return dict(project["settings"])


def test_a_project_is_saved_listed_pinned_and_deleted() -> None:
    server, events = make_capturing_server()
    config = load_config()
    form = _product_form()
    # A project needs a name; the refusal goes to the panel that asked.
    assert _send(server, events, {"type": "saveProject", "connId": "c1", "requestId": "r0",
                                  "project": {"name": "  ", "settings": form}}) == [
        {"type": "projectError", "id": "", "requestId": "r0", "connId": "c1",
         "text": "A project needs a name."}]
    assert _send(server, events, {"type": "saveProject", "project": "junk"})[0]["type"] == (
        "projectError")
    assert th._list_projects() == []

    os.chmod(th._DB_PATH, 0o644)  # a database other accounts can read
    own = {"max_budget": config["max_budget"] + 5, "use_memory": not config["use_memory"],
           "default_work_dir": "/srv/paper", "api_keys": {"ANTHROPIC_API_KEY": _PAPER_KEY}}
    saved, listed = _send(server, events, {
        "type": "saveProject", "connId": "c1", "requestId": "r1",
        "project": {"name": "  Paper  ", "description": " The draft ", "isPinned": True,
                    "settings": form | own | {"api_keys": form["api_keys"] | own["api_keys"]}},
    })
    project_id = saved["id"]
    assert saved == {"type": "projectSaved", "id": project_id, "requestId": "r1", "connId": "c1"}
    # Every page gets the list; the chats' projects travel with getProjects only.
    assert set(listed) == {"type", "projects"}
    (project,) = listed["projects"]
    assert (project["id"], project["name"], project["description"], project["isPinned"]) == (
        project_id, "Paper", "The draft", True)
    assert project["settings"] == own
    assert project["createdAt"] == project["updatedAt"] > 0
    # Its key is a secret of tool output from now on.
    assert redact_secrets(f"key={_PAPER_KEY}") == "key=ANTHROPIC_API_KEY"
    # The database that holds the key is its owner's alone, like the key
    # store, sidecars included (Windows has no such modes).  The branch of
    # persistence._make_db_private for a file it cannot chmod (a sidecar
    # that is not there, a file of another account) needs a database that
    # is not in WAL mode or a second account: no test reaches it.
    if sys.platform != "win32":
        assert [stat.S_IMODE(os.stat(f"{th._DB_PATH}{suffix}").st_mode)
                for suffix in ("", "-wal", "-shm")] == [0o600] * 3

    # A save without settings (a rename) keeps them, and the pin.
    saved, listed = _send(server, events, {"type": "saveProject", "requestId": "r2", "project": {
        "id": project_id, "name": "Paper v2", "description": 7}})
    assert saved == {"type": "projectSaved", "id": project_id, "requestId": "r2"}
    (project,) = listed["projects"]
    assert (project["name"], project["description"], project["isPinned"], project["settings"]) == (
        "Paper v2", "", True, own)
    # A form that holds the product's settings again stores nothing.
    _send(server, events, {"type": "saveProject", "project": {
        "id": project_id, "name": "x" * 500, "description": "y" * 5000, "isPinned": False,
        "settings": form}})
    project = th._get_project(project_id)
    assert project is not None and project["settings"] == {} and project["isPinned"] is False
    assert (len(project["name"]), len(project["description"])) == (
        projects.NAME_MAX, projects.DESCRIPTION_MAX)
    assert _send(server, events, {"type": "saveProject", "requestId": "r3", "project": {
        "id": "gone", "name": "Ghost"}}) == [{
            "type": "projectError", "id": "gone", "requestId": "r3",
            "text": "This project no longer exists."}]

    pinned = _send(server, events, {"type": "setProjectPinned", "projectId": project_id,
                                    "pinned": True})
    assert [(e["type"], e["projects"][0]["isPinned"]) for e in pinned] == [("projects", True)]
    assert _send(server, events, {"type": "setProjectPinned", "projectId": "gone",
                                  "pinned": True}) == []
    assert _send(server, events, {"type": "deleteProject", "projectId": project_id}) == [
        {"type": "projects", "projects": []}]
    assert _send(server, events, {"type": "deleteProject", "projectId": project_id}) == []
    # A new project saved without settings overrides nothing.
    _, listed = _send(server, events, {"type": "saveProject", "project": {"name": "Bare"}})
    assert [project["settings"] for project in listed["projects"]] == [{}]


def _note_redacted(seen: list[str], text: str) -> None:
    """Record *text* as tool output would show it at this moment."""
    seen.append(redact_secrets(text))


def test_a_projects_key_is_a_secret_of_tool_output_before_its_row_is_written() -> None:
    server, events = make_capturing_server()
    # A key no other test saves: only this save can have made it a secret.
    key = "sk-ant-at-write-0123456789abcdef"
    # A trigger of the connection the save writes through reads the row
    # as it goes in: what a running task would get from the database then.
    at_write: list[str] = []
    db = th._get_db()
    db.create_function("note_redacted", 1, partial(_note_redacted, at_write))
    db.execute(
        "CREATE TEMP TRIGGER note_new_project AFTER INSERT ON projects "
        "BEGIN SELECT note_redacted(NEW.settings); END",
    )
    assert _stored(server, events, {"api_keys": {"ANTHROPIC_API_KEY": key}}) == {
        "api_keys": {"ANTHROPIC_API_KEY": key}}
    assert at_write == ['{"api_keys": {"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"}}']


def test_only_what_differs_from_the_products_settings_is_stored() -> None:
    server, events = make_capturing_server()
    config = load_config()
    assert _stored(server, events, _product_form()) == {}
    # Values of the wrong type, unknown keys and empty texts are dropped.
    assert _stored(server, events, {
        "max_budget": "12", "use_memory": "yes", "compaction": 1, "default_work_dir": 7,
        "unknown": True, "victory_sound": "loud",
        "api_keys": {"NOT_A_KEY": "x" * 20, "ANTHROPIC_API_KEY": 5, "GEMINI_API_KEY": "   "},
    }) == {}
    junk_forms: tuple[object, ...] = (
        "nope", None, {"max_budget": float("inf")}, {"max_budget": True},
        {"max_budget": 10**400}, {"default_work_dir": "   "}, {"api_keys": []})
    for junk in junk_forms:
        assert _stored(server, events, junk) == {}
    assert _stored(server, events, {
        "max_budget": int(config["max_budget"]) + 1, "victory_sound": False,
        "default_work_dir": "  /srv/x  ", "chrome_bridge": not config["chrome_bridge"],
        "api_keys": {"GEMINI_API_KEY": "  gm-project-key-000001  "},
    }) == {
        "max_budget": float(int(config["max_budget"]) + 1), "victory_sound": False,
        "default_work_dir": "/srv/x", "chrome_bridge": not config["chrome_bridge"],
        "api_keys": {"GEMINI_API_KEY": "gm-project-key-000001"},
    }


def _handle(server: AgentServer, cmd: dict[str, Any]) -> None:
    """Thread body: have *server* handle *cmd*, then close the thread's database connection."""
    try:
        server._handle_command(cmd)
    finally:
        th._close_thread_db()


def send_past(
    server: AgentServer, held: Pause, first: dict[str, Any], second: dict[str, Any],
) -> None:
    """Send *second* while *first* is held where *held* pauses it, then let both finish.

    *second* gets half a second to finish ahead of *first*: a command
    that has to wait for the held one is still waiting then.
    """
    threads = [threading.Thread(target=_handle, args=(server, cmd)) for cmd in (first, second)]
    with held:
        threads[0].start()
        assert held.paused.wait(_WAIT_S), "the first command never reached its pause"
        threads[1].start()
        threads[1].join(0.5)
    for thread in threads:
        thread.join(_WAIT_S)
        assert not thread.is_alive()


def _projects_shown(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The projects and the chats' projects a page shows after *frames*, as main.js applies them.

    A ``projects`` event replaces the list, and the chats' projects when
    it carries them; a chat of a project no longer listed is in none.
    """
    listed: list[dict[str, Any]] = []
    of_chats: dict[str, str] = {}
    for frame in frames:
        if frame["type"] == "projects":
            listed = frame["projects"]
            ids = {project["id"] for project in listed}
            of_chats = {
                chat: project for chat, project in frame.get("chatProjects", of_chats).items()
                if project in ids
            }
        elif frame["type"] == "chatProject":
            of_chats[frame["chatId"]] = frame["projectId"]
    return listed, of_chats


def _reply_to_reader(entered: dict[str, Any]) -> bool:
    """Whether the ``_broadcast_to_conn`` just entered sends to the connection ``reader``."""
    return bool(entered["conn_id"] == "reader")


def test_a_project_list_read_before_a_change_is_not_sent_after_it() -> None:
    """A ``getProjects`` reply held between its read and its send does not overtake a change.

    The page replaces what it shows with every list it is sent: a list
    read before a save, a pin, a chat's move or a delete and sent after
    the change's own event was the page's last word, and a stale one.
    """
    server, events = make_capturing_server()
    project_id = th._save_project("", "Old name", "", {})
    _, chat = th._add_task("a chat for the project")
    changes: list[dict[str, Any]] = [
        {"type": "saveProject", "project": {"id": project_id, "name": "New name"}},
        {"type": "setProjectPinned", "projectId": project_id, "pinned": True},
        {"type": "setChatProject", "chatId": chat, "projectId": project_id},
        {"type": "deleteProject", "projectId": project_id},
    ]
    for change in changes:
        events.clear()
        held = Pause(AgentServer._broadcast_to_conn.__code__, _reply_to_reader)
        send_past(server, held, {"type": "getProjects", "connId": "reader"}, change)
        assert _projects_shown(events) == (th._list_projects(), th._chat_projects()), change
    assert th._list_projects() == []


def _renames(entered: dict[str, Any]) -> bool:
    """Whether the ``_save_project`` just entered writes the name ``Renamed``."""
    return bool(entered["name"] == "Renamed")


def test_a_rename_does_not_undo_settings_saved_meanwhile() -> None:
    """A save without settings keeps the stored ones, also those another page saves as it runs.

    The rename read the settings to keep and wrote them back with the
    name: settings another page saved in between were replaced by the
    older ones, an older budget or API key back in force.
    """
    server, events = make_capturing_server()
    budget = load_config()["max_budget"]
    older, newer = {"max_budget": float(budget + 5)}, {"max_budget": float(budget + 7)}
    project_id = th._save_project("", "Paper", "", older)
    rename = {"type": "saveProject", "project": {"id": project_id, "name": "Renamed"}}
    resave = {"type": "saveProject", "project": {
        "id": project_id, "name": "Paper", "settings": _product_form() | newer}}
    send_past(server, Pause(th._save_project.__code__, _renames), rename, resave)
    assert [event["type"] for event in events].count("projectSaved") == 2
    project = th._get_project(project_id)
    assert project is not None and project["settings"] == newer


# A second process on the history of its KISS_HOME (another daemon, a command
# of a terminal) saving a project's settings: the project's id and the
# settings as JSON are its arguments.
_SAVES_SETTINGS = """
import json
import sys

from kiss.agents.sorcar import persistence

persistence._save_project(sys.argv[1], "Paper", "", json.loads(sys.argv[2]))
persistence._close_thread_db()
"""


def test_a_rename_does_not_undo_settings_another_process_saved_meanwhile() -> None:
    """A save without settings does not write them: those another process saves as it runs stay.

    The daemon's lock orders its own project commands, not a second
    process on the same history.  The rename read the settings to keep
    and wrote them back with the name, over the ones that process had
    saved since: the older budget was back in force.
    """
    server = AgentServer()
    budget = load_config()["max_budget"]
    older, newer = {"max_budget": float(budget + 5)}, {"max_budget": float(budget + 7)}
    project_id = th._save_project("", "Paper", "", older)
    rename = {"type": "saveProject", "project": {"id": project_id, "name": "Renamed"}}
    renamer = threading.Thread(target=_handle, args=(server, rename))
    with Pause(th._save_project.__code__, _renames) as held:
        renamer.start()
        assert held.paused.wait(_WAIT_S), "the rename never reached its write"
        subprocess.run(
            [sys.executable, "-c", _SAVES_SETTINGS, project_id, json.dumps(newer)],
            env={**os.environ, "KISS_HOME": str(th._KISS_DIR)}, check=True, timeout=_WAIT_S,
        )
        # The other process's save is in, and the rename still on its way.
        project = th._get_project(project_id)
        assert project is not None
        assert (project["name"], project["settings"]) == ("Paper", newer)
    renamer.join(_WAIT_S)
    assert not renamer.is_alive()
    project = th._get_project(project_id)
    assert project is not None
    assert (project["name"], project["settings"]) == ("Renamed", newer)


def test_a_project_deleted_under_a_save_is_refused() -> None:
    """A save whose project is deleted between its read and its write writes nothing.

    Deleted in the database, as another process does: the daemon's own
    commands wait for the save.
    """
    server, events = make_capturing_server()
    project_id = th._save_project("", "Paper", "", {})
    rename = {"type": "saveProject", "requestId": "r1", "project": {
        "id": project_id, "name": "Renamed"}}
    saver = threading.Thread(target=_handle, args=(server, rename))
    with Pause(th._save_project.__code__, _renames) as held:
        saver.start()
        assert held.paused.wait(_WAIT_S), "the save never reached its write"
        assert th._delete_project(project_id)
    saver.join(_WAIT_S)
    assert not saver.is_alive()
    assert events == [{
        "type": "projectError", "id": project_id, "requestId": "r1",
        "text": "This project no longer exists."}]
    assert th._list_projects() == []


def test_a_save_the_database_fails_is_answered_with_the_failure() -> None:
    """A create or an update the database fails is refused: its panel awaits an answer.

    The form of a new project is locked until its create is answered.
    The table is renamed away over a connection of its own, as in a
    ``history.db`` put back from a copy that predates projects.
    """
    server, events = make_capturing_server()
    project_id = th._save_project("", "Paper", "", {})
    con = sqlite3.connect(th._DB_PATH)
    con.execute("ALTER TABLE projects RENAME TO projects_away")
    con.commit()
    for asked in ("", project_id):
        assert _send(server, events, {"type": "saveProject", "requestId": "r1", "project": {
            "id": asked, "name": "Renamed"}}) == [{
                "type": "projectError", "id": asked, "requestId": "r1",
                "text": "Could not save the project: no such table: projects."}]
    con.execute("ALTER TABLE projects_away RENAME TO projects")
    con.commit()
    con.close()
    assert [project["name"] for project in th._list_projects()] == ["Paper"]


def _history(server: AgentServer, events: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """Ask for a history page and return the reply."""
    (page,) = [e for e in _send(server, events, {"type": "getHistory", **fields})
               if e["type"] == "history"]
    return page


def test_chats_are_put_in_projects_and_a_project_page_lists_only_its_chats() -> None:
    server, events = make_capturing_server()
    paper = th._save_project("", "Paper", "", {})
    time.sleep(0.002)  # the list is ordered by creation time
    other = th._save_project("", "Other", "", {}, True)
    first, chat = th._add_task("write the intro", extra={"startTs": 1_000})
    th._save_task_extra({"endTs": 2_000}, task_id=first)
    second, _ = th._add_task("polish the abstract", chat_id=chat, extra={"startTs": 3_000})
    th._add_task("unrelated")
    _, rowless = th._add_task("still running")

    assert _send(server, events, {"type": "setChatProject", "chatId": chat,
                                  "projectId": paper}) == [
        {"type": "chatProject", "chatId": chat, "projectId": paper}]
    # No task of this chat has finished, so it has no row yet: the project makes it.
    _send(server, events, {"type": "setChatProject", "chatId": rowless, "projectId": other})
    # A project that is gone is an error to the sender, and nothing is written.
    assert _send(server, events, {"type": "setChatProject", "chatId": rowless,
                                  "projectId": "gone", "connId": "c9"}) == [
        {"type": "error", "text": "This project no longer exists.",
         "command": "setChatProject", "connId": "c9"}]
    for malformed in ({"chatId": "", "projectId": paper}, {"chatId": chat, "projectId": None}):
        assert _send(server, events, {"type": "setChatProject", **malformed}) == []
    assert _send(server, events, {"type": "getProjects", "connId": "c1"}) == [{
        "type": "projects", "connId": "c1", "projects": th._list_projects(),
        "chatProjects": {chat: paper, rowless: other}}]
    assert [(p["name"], p["isPinned"]) for p in th._list_projects()] == [
        ("Paper", False), ("Other", True)]

    # The task finishing recomputes the chat's row; the chat stays in its project.
    th._save_task_extra({"endTs": 4_000}, task_id=second)
    assert th._chat_project(chat) == paper
    page = _history(server, events, project=paper, generation=3)
    assert (page["project"], page["generation"]) == (paper, 3)
    assert [s["title"] for s in page["sessions"]] == ["polish the abstract", "write the intro"]
    page = _history(server, events)
    assert "project" not in page and len(page["sessions"]) == 4
    # A project's page holds more rows than a page of everything; the next continues it.
    for index in range(projects.HISTORY_PROJECT_PAGE):
        th._add_task(f"step {index}", chat_id=chat)
    assert [len(_history(server, events, project=paper, offset=offset)["sessions"])
            for offset in (0, projects.HISTORY_PROJECT_PAGE)] == [projects.HISTORY_PROJECT_PAGE, 2]
    # A panel that loaded both pages asks for its rows again in one page of
    # that many rows; the reply says how many were asked for, so a shorter
    # page is known to be the last.  A limit that is no count is the default.
    rows = projects.HISTORY_PROJECT_PAGE + 2
    assert [(page["limit"], len(page["sessions"])) for page in (
        _history(server, events, project=paper, limit=rows),
        _history(server, events, project=paper, limit=rows + 5),
        _history(server, events, project=paper, limit=-3),
        _history(server, events, project=paper, limit="many"),
    )] == [(rows, rows), (rows + 5, rows)] + [(projects.HISTORY_PROJECT_PAGE,) * 2] * 2
    # A page of everything holds as many rows as it is asked for too (the list
    # asks for the rows that fill it in one request), up to a bound; here are
    # the project's rows and the two tasks of no project.
    assert [(page["limit"], len(page["sessions"])) for page in (
        _history(server, events, limit=3),
        _history(server, events, offset=rows, limit=3),
        _history(server, events),
        _history(server, events, limit=rows),
        _history(server, events, limit=10**6),
    )] == [(3, 3), (3, 2), (HISTORY_PAGE,) * 2, (rows, rows), (HISTORY_MAX_PAGE, rows + 2)]
    # The page holds the tasks that ran on this machine: a row a history
    # sync copied in from another one is listed with everything, not here.
    synced, _ = th._add_task("ran on the box", chat_id=chat)
    db = th._get_db()
    db.execute("UPDATE task_history SET machine = 'box' WHERE id = ?", (synced,))
    db.commit()
    page = _history(server, events, project=paper, limit=rows + 5)
    titles = [s["title"] for s in page["sessions"]]
    assert len(titles) == rows and "ran on the box" not in titles
    assert _history(server, events)["sessions"][0]["title"] == "ran on the box"

    # Deleting a project keeps its chats, in no project.
    assert _send(server, events, {"type": "deleteProject", "projectId": paper}) == [
        {"type": "projects", "projects": [th._get_project(other)]}]
    assert th._chat_projects() == {rowless: other}
    assert len(_history(server, events)["sessions"]) == HISTORY_PAGE


class _RecordingAgent(ChatSorcarAgent):
    """Real agent whose ``run`` records what the runner started it with, calling no model."""

    def __init__(self, gate: threading.Barrier | None = None) -> None:
        super().__init__("Sorcar")
        self.gate = gate
        self.runs: list[dict[str, Any]] = []

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate the task row, wait for the other run, record the run's settings."""
        del args
        prompt = str(kwargs.get("prompt_template", ""))
        task_id, self._chat_id = th._add_task(prompt, chat_id=self._chat_id or "")
        with self._task_id_lock:
            self._last_task_id = task_id
        if self.gate is not None:
            self.gate.wait(timeout=_WAIT_S)
        tools = UsefulTools(work_dir=str(kwargs["work_dir"]))
        self.runs.append({
            "budget": kwargs["max_budget"],
            "web": kwargs["web_tools"],
            "memory": kwargs["use_memory"],
            "key": DEFAULT_CONFIG.ANTHROPIC_API_KEY,
            "compaction": DEFAULT_CONFIG.tool_output_compaction,
            "codex_subscription": DEFAULT_CONFIG.codex_use_subscription,
            "cli_env": environ_without(("PATH",)).get("ANTHROPIC_API_KEY", ""),
            "shell": tools.Bash(_KEY_PROBE, "Part of the key in the command's environment"),
            "parallel": tools.run_commands_parallel(json.dumps([_KEY_PROBE])),
        })
        return finish(True, summary_in_html=f"<p>done {prompt}</p>")


class _CapturePrinter(JsonPrinter):
    """Real printer that records every broadcast."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then broadcast it."""
        self.events.append(dict(event))
        super().broadcast(event)


@pytest.fixture
def daemon() -> Iterator[tuple[AgentServer, _CapturePrinter]]:
    """A server for runs on tabs named ``proj-…``, cleaned up afterwards."""
    printer = _CapturePrinter()
    yield AgentServer(printer=printer), printer
    for state in agent_state.snapshot():
        if state.tab_id.startswith("proj-"):
            agent_state.unregister(state.task_id)
    ChromeBridge.shutdown()
    (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)


def _new_project(server: AgentServer, name: str, own: dict[str, Any]) -> str:
    """Create project *name* whose settings form differs from the product's by *own*."""
    form = _product_form()
    settings = form | own | {"api_keys": form["api_keys"] | own.get("api_keys", {})}
    server._handle_command(
        {"type": "saveProject", "project": {"name": name, "settings": settings}},
    )
    return next(str(p["id"]) for p in th._list_projects() if p["name"] == name)


def _claude_model() -> str:
    """A model that an Anthropic key, and nothing else, makes available."""
    with config_module.run_overlay() as overlay:
        overlay["ANTHROPIC_API_KEY"] = _PAPER_KEY
        return next(name for name in get_available_models() if name.startswith("claude-"))


def _start(
    server: AgentServer, agent: _RecordingAgent, tab_id: str, work_dir: Path, **fields: Any,
) -> None:
    """Start a run of *agent* on *tab_id* through the real ``_cmd_run``."""
    # ``_cmd_run`` carries the agent of the tab's previous state over to the
    # new run; a finished run leaves its state without one.
    state = agent_state.find_by_tab(tab_id)
    if state is None:
        agent_state.register(
            AgentState(f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True),
        )
    else:
        state.agent = agent
    server._cmd_run({
        "type": "run", "tabId": tab_id, "prompt": f"task {len(agent.runs)}",
        "model": _claude_model(), "workDir": str(work_dir), "useWorktree": False,
        "autoCommit": False, "classifyTasks": False, **fields,
    })


def _wait_idle(tab_id: str) -> None:
    """Wait until the run started on *tab_id* has ended."""
    deadline = time.time() + _WAIT_S
    while time.time() < deadline:
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            thread = state.task_thread if state is not None else None
        if thread is None:
            return
        thread.join(timeout=max(0.0, deadline - time.time()))
    raise AssertionError(f"tab {tab_id} never went idle")


def _run_next(
    daemon: tuple[AgentServer, _CapturePrinter], agent: _RecordingAgent, work_dir: Path,
    **fields: Any,
) -> list[dict[str, Any]]:
    """Run the next task of tab ``proj-one``; return the chat moves it broadcast."""
    server, printer = daemon
    printer.events.clear()
    _start(server, agent, "proj-one", work_dir, **fields)
    _wait_idle("proj-one")
    return [e for e in printer.events if e["type"] == "chatProject"]


def _set_stored_settings(project_id: str, text: str) -> None:
    """Write *text* as the settings of *project_id*, as an edit by hand would."""
    con = sqlite3.connect(th._DB_PATH)
    con.execute("UPDATE projects SET settings = ? WHERE id = ?", (text, project_id))
    con.commit()
    con.close()


def test_two_tasks_of_two_projects_run_at_once_each_with_its_own_settings(
    daemon: tuple[AgentServer, _CapturePrinter], tmp_path: Path,
) -> None:
    server, printer = daemon
    config = load_config()
    daemon_key = DEFAULT_CONFIG.ANTHROPIC_API_KEY
    daemon_compaction = DEFAULT_CONFIG.tool_output_compaction
    daemon_codex = DEFAULT_CONFIG.codex_use_subscription
    daemon_environment = {name: os.environ.get(name) for name in API_KEY_ENV_VARS}
    daemon_key_store = dict(get_current_api_keys())
    paper = _new_project(server, "Paper", {
        "max_budget": config["max_budget"] + 5, "use_web_browser": not config["use_web_browser"],
        "use_memory": not config["use_memory"], "compaction": not config["compaction"],
        "codex_use_subscription": not config["codex_use_subscription"],
        "api_keys": {"ANTHROPIC_API_KEY": _PAPER_KEY},
    })
    thesis = _new_project(server, "Thesis", {
        "max_budget": config["max_budget"] + 9, "api_keys": {"ANTHROPIC_API_KEY": _THESIS_KEY},
    })
    # Each run waits inside its agent for the other: they overlap.
    gate = threading.Barrier(2)
    paper_agent, thesis_agent = _RecordingAgent(gate), _RecordingAgent(gate)
    _start(server, paper_agent, "proj-paper", tmp_path, projectId=paper)
    _start(server, thesis_agent, "proj-thesis", tmp_path, projectId=thesis)
    _wait_idle("proj-paper")
    _wait_idle("proj-thesis")

    (paper_run,), (thesis_run,) = paper_agent.runs, thesis_agent.runs
    shell_output = {key: paper_run.pop(key) for key in ("shell", "parallel")}
    assert paper_run == {
        "budget": config["max_budget"] + 5, "web": not config["use_web_browser"],
        "memory": not config["use_memory"], "key": _PAPER_KEY,
        "compaction": not config["compaction"],
        "codex_subscription": not config["codex_use_subscription"], "cli_env": _PAPER_KEY,
    }
    # The commands of the task find the project's key in their environment.
    assert all("paper" in output for output in shell_output.values()), shell_output
    shell_output = {key: thesis_run.pop(key) for key in ("shell", "parallel")}
    # What a project does not name follows the product's settings.
    assert thesis_run == {
        "budget": config["max_budget"] + 9, "web": config["use_web_browser"], "memory": None,
        "key": _THESIS_KEY, "compaction": daemon_compaction, "codex_subscription": daemon_codex,
        "cli_env": _THESIS_KEY,
    }
    assert all("thesi" in output for output in shell_output.values()), shell_output
    # Each run put its new chat in the project its command named.
    moves = [(e["chatId"], e["projectId"]) for e in printer.events if e["type"] == "chatProject"]
    assert sorted(moves) == sorted(
        [(paper_agent.chat_id, paper), (thesis_agent.chat_id, thesis)])
    assert th._chat_projects() == {paper_agent.chat_id: paper, thesis_agent.chat_id: thesis}
    # Nothing of either run is left on the daemon's own settings.
    assert config_module.current_run_overlay() is None
    assert DEFAULT_CONFIG.ANTHROPIC_API_KEY == daemon_key
    assert {name: os.environ.get(name) for name in API_KEY_ENV_VARS} == daemon_environment
    assert get_current_api_keys() == daemon_key_store
    assert environ_without(()).get("ANTHROPIC_API_KEY") == daemon_environment[
        "ANTHROPIC_API_KEY"]


def _say_ok(agent: KISSAgent, key: str) -> None:
    """Run one model turn of *agent* as a task whose project has API key *key*."""
    with config_module.run_overlay() as overlay:
        overlay["ANTHROPIC_API_KEY"] = key
        agent.run(model_name=_claude_model(), prompt_template="Say ok.", is_agentic=False,
                  verbose=False)


def test_an_agent_that_runs_again_sends_the_key_of_the_run_it_is_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real SDK, sent to a server of this test instead of Anthropic's.
    with ScriptedAnthropicServer() as api:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", api.base_url)
        agent = KISSAgent("Runs again")
        _say_ok(agent, _PAPER_KEY)
        paper_model = agent.model
        # The same settings again: the model and its connection are kept.
        _say_ok(agent, _PAPER_KEY)
        assert agent.model is paper_model
        # Another project's run, then the first one's again: a model keeps
        # the key it was built with, so each gets one built with its own.
        _say_ok(agent, _THESIS_KEY)
        _say_ok(agent, _PAPER_KEY)
        sent = [headers["x-api-key"] for headers in api.request_headers]
    assert sent == [_PAPER_KEY, _PAPER_KEY, _THESIS_KEY, _PAPER_KEY]


def test_a_runs_settings_follow_its_command_and_its_chats_project(
    daemon: tuple[AgentServer, _CapturePrinter], tmp_path: Path,
) -> None:
    server, printer = daemon
    config = load_config()
    daemon_key = DEFAULT_CONFIG.ANTHROPIC_API_KEY
    paper = _new_project(server, "Paper", {
        "max_budget": config["max_budget"] + 5, "api_keys": {"ANTHROPIC_API_KEY": _PAPER_KEY},
    })
    thesis = _new_project(server, "Thesis", {
        "max_budget": config["max_budget"] + 9, "use_memory": not config["use_memory"],
        "api_keys": {"ANTHROPIC_API_KEY": _THESIS_KEY},
    })
    agent = _RecordingAgent()

    _run_next(daemon, agent, tmp_path, projectId=paper)
    chat = agent.chat_id
    assert (agent.runs[-1]["key"], th._chat_project(chat)) == (_PAPER_KEY, paper)
    # The run's end names its chat: a page that does not have the task's
    # tab still finds the project the task was in (its sound setting).
    (ended,) = [e for e in printer.events if e["type"] == "task_done"]
    assert (ended["tabId"], ended["chatId"]) == ("proj-one", chat)
    # The command names another project: the chat moves there and every page is told.
    assert _run_next(daemon, agent, tmp_path, projectId=thesis) == [
        {"type": "chatProject", "chatId": chat, "projectId": thesis, "tabId": ""}]
    assert (agent.runs[-1]["key"], agent.runs[-1]["budget"], agent.runs[-1]["memory"]) == (
        _THESIS_KEY, config["max_budget"] + 9, not config["use_memory"])
    # No project on the command, the chat's own, a project that is gone, junk: the chat
    # stays where it is and no page is told.
    for fields in ({}, {"projectId": thesis}, {"projectId": "gone"}, {"projectId": 7}):
        assert _run_next(daemon, agent, tmp_path, **fields) == []
        assert (agent.runs[-1]["key"], th._chat_project(chat)) == (_THESIS_KEY, thesis)
    # What the run names itself comes before its project's settings.
    _run_next(daemon, agent, tmp_path, maxBudget=1.5,
              useWebTools=not config["use_web_browser"], useMemory=config["use_memory"])
    assert [agent.runs[-1][key] for key in ("budget", "web", "memory")] == [
        1.5, not config["use_web_browser"], config["use_memory"]]

    # A row edited by hand: a value of the wrong type is ignored, the rest applies,
    # and a project that drives the user's Chrome has the bridge started for its task.
    _set_stored_settings(thesis, json.dumps({
        "max_budget": "lots", "use_web_browser": not config["use_web_browser"],
        "chrome_bridge": True, "api_keys": {"ANTHROPIC_API_KEY": _THESIS_KEY, "NOPE": "x"},
    }))
    assert ChromeBridge._instance is None
    _run_next(daemon, agent, tmp_path)
    assert [agent.runs[-1][key] for key in ("budget", "web", "memory", "key")] == [
        config["max_budget"], not config["use_web_browser"], None, _THESIS_KEY]
    assert ChromeBridge._instance is not None
    # Settings that are not a JSON object read as none.
    for text in ("not json", "[1]"):
        _set_stored_settings(paper, text)
        assert [p["settings"] for p in th._list_projects() if p["id"] == paper] == [{}]

    # Taken out of its project, the chat is back on the product's settings: the
    # daemon's own key, or no model at all when it has none.
    done = len(agent.runs)
    assert _run_next(daemon, agent, tmp_path, projectId="") == [
        {"type": "chatProject", "chatId": chat, "projectId": "", "tabId": ""}]
    assert th._chat_project(chat) == ""
    assert [r["key"] for r in agent.runs[done:]] == ([daemon_key] if daemon_key else [])


def _sends_projects(entered: dict[str, Any]) -> bool:
    """Whether the ``_broadcast_to_conn`` just entered sends a ``projects`` list."""
    return bool(entered["event"]["type"] == "projects")


def _sends_chat_project(entered: dict[str, Any]) -> bool:
    """Whether the ``broadcast`` just entered sends a ``chatProject``."""
    return bool(entered["event"].get("type") == "chatProject")


def _note_entry(entered: threading.Event, _locals: dict[str, Any]) -> bool:
    """Note that the watched function was entered; its thread is not held."""
    entered.set()
    return False


def _hold_entry(_locals: dict[str, Any]) -> bool:
    """Hold the thread that entered the watched function."""
    return True


def _in_call_of(code: CodeType, frame: FrameType | None) -> bool:
    """Whether *frame*, or a caller of it, runs *code*."""
    while frame is not None and frame.f_code is not code:
        frame = frame.f_back
    return frame is not None


def _waits_for_the_database(thread: threading.Thread) -> bool:
    """Whether *thread* waits for the history database's lock."""
    assert thread.ident is not None
    return _in_call_of(th._RWLock._await.__code__, sys._current_frames().get(thread.ident))


def _ends_a_chat_move(_locals: dict[str, Any]) -> bool:
    """Whether the lock release just entered ends a chat's move: its row is stored."""
    return _in_call_of(th._set_chat_project.__code__, sys._getframe())


def _no_teller_left() -> bool:
    """Whether every thread that tells of a cut chat move has ended."""
    return all(thread.name != "kiss-chat-move" for thread in threading.enumerate())


class TestARunsProjectOnTheWire(TwoClientHarness):
    """Pages connected to one daemon, and a run that names the project of its chat.

    The run is a command's help, which the daemon answers itself (no
    model) once the chat is in its project.
    """

    def setUp(self) -> None:
        super().setUp()
        assert sea_commands.get_command("sh") is not None, "the run would be a task for a model"
        self.paper = th._save_project("", "Paper", "", {})

    def _run_in_paper(self, page: Client) -> None:
        """Start a run from *page* whose command names the project."""
        page.send({
            "type": "run", "tabId": page.tab_id, "prompt": "/sh help", "projectId": self.paper,
            "workDir": self.repo, "useWorktree": False, "autoCommit": False,
            "classifyTasks": False,
        })

    def test_a_project_list_read_before_a_runs_chat_move_is_not_sent_after_it(self) -> None:
        """A run naming a project moves its chat as a ``setChatProject`` does: after a list held.

        The run's thread moved the chat and sent its ``chatProject``
        outside the lock a ``getProjects`` reply is read and sent
        under: a reply read before the move and sent after it left the
        asking page with the chat in no project.
        """
        reader, runner = self.open_client("reader"), self.open_client("runner")
        held = Pause(AgentServer._broadcast_to_conn.__code__, _sends_projects)
        moving = threading.Event()
        with Pauses(held, Pause(AgentServer._enter_project.__code__, partial(_note_entry, moving))):
            reader.send({"type": "getProjects"})
            assert held.paused.wait(_WAIT_S), "the list never reached its send"
            self._run_in_paper(runner)
            assert moving.wait(_WAIT_S), "the run never reached its project"
            # Half a second to move the chat ahead of the held list: a run
            # that has to wait for it is still waiting then.
            time.sleep(0.5)
        self.wait_event(runner, "status", tabId=runner.tab_id, running=False)
        self.wait_for(
            lambda: reader.of_type("projects") and reader.of_type("chatProject"),
            what="the list and the chat's move sent to the page that asked",
        )
        frames = [frame for frame in reader.events if frame["type"] in ("projects", "chatProject")]
        assert list(th._chat_projects().values()) == [self.paper]
        assert _projects_shown(frames) == (th._list_projects(), th._chat_projects())

    def test_a_stop_between_a_runs_chat_move_and_its_event_does_not_part_them(self) -> None:
        """A run stopped once its chat is moved still tells the pages of the move.

        The Stop's interrupt is raised in the run's thread wherever it
        is: between the move and its ``chatProject`` it left the chat
        in the project and every page showing it in none.
        """
        page = self.open_client("page")
        with Pause(WebPrinter.broadcast.__code__, _sends_chat_project) as sending:
            self._run_in_paper(page)
            assert sending.paused.wait(_WAIT_S), "the chat's move never reached its event"
            page.send({"type": "stop", "tabId": page.tab_id})
            self.wait_event(page, "stop_ack", tabId=page.tab_id, accepted=True)
            # The interrupt of the Stop's watchdog, now rather than after its grace.
            state = agent_state.find_by_tab(page.tab_id)
            assert state is not None and state.task_thread is not None
            assert state.task_thread.ident is not None
            assert inject_keyboard_interrupt(state.task_thread.ident) == 1
        self.wait_event(page, "status", tabId=page.tab_id, running=False)
        assert list(th._chat_projects().values()) == [self.paper]
        assert page.of_type("chatProject", projectId=self.paper)

    def test_a_stop_reaches_a_run_that_waits_to_move_its_chat(self) -> None:
        """A run waiting for the projects' publication lock is stopped while it waits.

        The Stop's interrupt was held back for the whole move, the wait
        for the lock included: the run ended only once the lock's holder
        (a project being saved, a list on its way to a page) let go.
        """
        page = self.open_client("page")
        moving = threading.Event()
        entry = Pause(AgentServer._move_chat_to_project.__code__, partial(_note_entry, moving))
        with _PROJECTS_PUBLISH_LOCK, entry:
            self._run_in_paper(page)
            assert moving.wait(_WAIT_S), "the run never reached its chat's move"
            page.send({"type": "stop", "tabId": page.tab_id})
            self.wait_event(page, "stop_ack", tabId=page.tab_id, accepted=True)
            # The interrupt of the Stop's watchdog, now rather than after its grace.
            state = agent_state.find_by_tab(page.tab_id)
            assert state is not None and state.task_thread is not None
            assert state.task_thread.ident is not None
            assert inject_keyboard_interrupt(state.task_thread.ident) == 1
            self.wait_event(page, "status", timeout=10, tabId=page.tab_id, running=False)
        assert th._chat_projects() == {}

    def test_a_stop_reaches_a_run_that_waits_for_the_database_to_move_its_chat(self) -> None:
        """A run waiting for the history database's lock is stopped while it waits.

        The Stop's interrupt was still held back for the move under the
        publication lock, its wait for the database included: the run
        ended only once the database's writer let go.  The move it cut
        wrote nothing, so no page is told of one.
        """
        page = self.open_client("page")
        with Pause(th._set_chat_project.__code__, _hold_entry) as storing:
            self._run_in_paper(page)
            assert storing.paused.wait(_WAIT_S), "the run never reached its chat's move"
            state = agent_state.find_by_tab(page.tab_id)
            assert state is not None and state.task_thread is not None
            assert state.task_thread.ident is not None
            with th._rw_lock.write_lock():
                storing.release()
                self.wait_for(
                    partial(_waits_for_the_database, state.task_thread),
                    what="the run waiting for the history database",
                )
                page.send({"type": "stop", "tabId": page.tab_id})
                self.wait_event(page, "stop_ack", tabId=page.tab_id, accepted=True)
                # The interrupt of the Stop's watchdog, now rather than after its grace.
                assert inject_keyboard_interrupt(state.task_thread.ident) == 1
                self.wait_event(page, "status", timeout=10, tabId=page.tab_id, running=False)
        self.wait_for(_no_teller_left, what="the thread that looked for the cut move ended")
        assert th._chat_projects() == {}
        assert page.of_type("chatProject") == []

    def test_a_stop_as_a_runs_chat_move_is_stored_still_tells_the_pages(self) -> None:
        """A run stopped as its chat's move is stored still tells the pages of the move.

        The move waits for the history database with no Stop held back,
        so the interrupt can be raised once the row is written and
        before its ``chatProject`` is handed over: a thread no Stop
        reaches sends it then, once.
        """
        page = self.open_client("page")
        with Pause(th._RWLock._teardown.__code__, _ends_a_chat_move) as stored:
            self._run_in_paper(page)
            assert stored.paused.wait(_WAIT_S), "the chat's move was never stored"
            page.send({"type": "stop", "tabId": page.tab_id})
            self.wait_event(page, "stop_ack", tabId=page.tab_id, accepted=True)
            # The interrupt of the Stop's watchdog, now rather than after its grace.
            state = agent_state.find_by_tab(page.tab_id)
            assert state is not None and state.task_thread is not None
            assert state.task_thread.ident is not None
            assert inject_keyboard_interrupt(state.task_thread.ident) == 1
        self.wait_event(page, "status", tabId=page.tab_id, running=False)
        self.wait_event(page, "chatProject", projectId=self.paper)
        self.wait_for(_no_teller_left, what="the thread that told of the move ended")
        assert list(th._chat_projects().values()) == [self.paper]
        assert len(page.of_type("chatProject")) == 1
