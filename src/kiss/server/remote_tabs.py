# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chat tabs whose tasks run on a registered remote machine.

The machine drop-down at the top of a chat webview chooses where the
tab's tasks run; a ``submit`` carries the choice as ``machine`` (empty
for this machine).  A click on a remote machine's row of the CHATS
panel binds a tab the same way: its ``resumeSession`` carries the
row's ``machine`` (and the task as ``title``), the machine's daemon
replays the chat into the tab, and the tab's next prompts run there.
Every remote machine is thus managed from this daemon's own surfaces;
the remote's web UI is never opened.  :class:`RemoteTabs` keeps one :class:`RemoteTab`
per tab bound to a remote machine: an authenticated WebSocket to that
machine's daemon (:meth:`MachineLink.open_connection`) over which the
tab's commands are forwarded unchanged, same ``tabId``, so the remote
daemon runs, queues and stops the tab's tasks exactly as it would for
its own webview.  Every frame the remote stamps with the tab id is
handed to this daemon's clients as it is, bypassing the printer's
task bookkeeping: the task, its events and its history row live on the
remote machine only (the CHATS panel shows them through the merged
history with that machine's chip).

What else the stream carries:

* **Sub-agent tabs.**  A ``new_tab`` the remote announces for a task
  of the bound tab is relayed; every surface then creates its own
  sub-agent tab and asks for the transcript with ``resumeSession
  {taskId, tabId}``, which is forwarded (the sub-agent tab joins the
  stream's tabs, so its frames and its ``subagentDone`` are relayed and
  its ``stop``/``closeTab`` reach the remote).  A sub-agent that
  finished before the chat was opened here announced no ``new_tab``
  to this daemon: the fan-out panel's resume names its
  ``parentTabId``, whose stream the sub-agent tab joins.
* **Replays.**  A client's ``ready`` (a reload, a surface opening)
  replays every bound tab to that connection: a ``resumeSession`` of
  the tab's remote chat is forwarded and the ``task_events`` it
  answers go to that connection alone, as the daemon does for its own
  chats.
* **The binding.**  ``tabMachine {tabId, machine}`` tells every surface
  which machine a tab is bound to (on bind, on unbind, and on
  ``ready``), so the chat header's drop-down agrees across surfaces.

A tab stays bound while it exists: a later ``submit`` or
``resumeSession`` for another machine re-binds it, an empty
``machine`` brings it home, a ``resumeSession`` of a local chat or
``closeTab`` unbinds it
(``closeTab`` is forwarded too, so the remote drops its tab; its task
keeps running there as a closed local tab's would).  Losing the socket
ends the tab's running state with an ``error`` naming the machine.

A ``run`` (the API command :func:`kiss.agents.sorcar.daemon_client.run`
sends, e.g. for ``run_agent(..., machine="box")``) with a ``machine``
binds its ``api-…`` tab the same way, keeping its ``workDir`` (the
dispatcher sends the path on the machine).  Forwarded with its
``parentTaskId``/``parentTabId``, the remote runs it as a sub-agent
run of the LOCAL caller: no top-level tab on either daemon, and the
``new_tab`` it broadcasts names the caller's tab as parent, which the
stream relays once the run's own frames have told its task id
(:attr:`RemoteTab.subagent`, :attr:`RemoteTab.pending_spawns`: the
remote announces every sub-agent run to every client, siblings
dispatched to the same machine included) so every surface here opens
the nested sub-agent tab under the caller and closes it on the
relayed ``subagentDone``, as for a sub-agent running here; a surface
connecting while it runs is told the same ``new_tab``
(:attr:`RemoteTab.spawn`, :meth:`RemoteTabs.replay_to`).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar.machines import (
    REQUEST_TIMEOUT,
    Machine,
    MachineError,
    link_for,
    load_machines,
)

logger = logging.getLogger(__name__)

FORWARDED_COMMANDS = frozenset({
    "submit", "run", "stop", "userAnswer", "appendUserMessage", "interruptTool",
    "selectModel",
})
"""Tab commands a bound tab sends to its machine instead of running here."""

BINDING_COMMANDS = frozenset({"submit", "run"})
"""The commands whose ``machine`` field binds (or unbinds) their tab."""

ROUTED_COMMANDS = FORWARDED_COMMANDS | {"closeTab", "newChat", "resumeSession"}
"""The commands :meth:`RemoteTabs.route` looks at (``ready`` goes to :meth:`replay_to`)."""

_DROPPED_FIELDS = ("connId", "machine", "workDir", "tabScopeWorkDir")
"""Fields of a forwarded command that only mean something on this daemon.

``workDir`` is a path of this machine: without it the remote daemon
runs the task in its own configured working directory.  A ``run``
keeps it: its sender names the directory on the machine (empty for
the machine's configured one).
"""

_DROPPED_RUN_FIELDS = tuple(f for f in _DROPPED_FIELDS if f != "workDir")


class RemoteTab:
    """One bound tab's stream to the daemon of a remote machine.

    Attributes:
        tab_id: The bound (root) tab.
        machine: The machine the stream talks to.
        tab_ids: The root tab plus the sub-agent tabs this daemon's
            clients opened for its tasks; their frames are relayed.
        task_ids: The ids of the sub-agent tasks the remote announced
            (``new_tab``); a ``resumeSession`` naming one is forwarded.
        subagent: Whether the bound tab's run is a sub-agent run of a
            task of this daemon (a forwarded ``run`` with a
            ``parentTaskId``): the ``new_tab`` the remote announces
            for it is relayed, under the caller's tab or parentless
            as the remote broadcast it.  False for a tab the user bound.
        parent_tab_id: The caller's tab (the ``parentTabId`` of that
            ``run``), "" when the caller has none or for a user-bound tab.
        task_id: The run's persisted task id, learnt from the first
            frame of the bound tab that carries one; "" until then.
        pending_spawns: The ``new_tab`` frames announced before the
            run's task id was known, by task id: the remote announces
            every sub-agent run to every client, so a sibling's may
            arrive here too; the run's own is relayed once its id is.
        spawn: The run's own ``new_tab`` frame while the run lasts (""
            before it, and after its terminal ``status``): a surface
            connecting meanwhile gets it from :meth:`RemoteTabs.replay_to`.
        chat_id: The remote chat of the root tab: the chat a
            CHATS-panel click resumed in it, else the one learnt from
            the first relayed frame that carries one; "" until then.
        shown_task_id: The task a CHATS-panel click pinned the root
            tab to (the resume's ``taskId``, as the registry keeps it
            for a local chat), "" while the tab tracks the chat's
            latest task: a surface connecting later is replayed at it.
            Cleared by a submit, whose task becomes the latest; never
            set by a relayed frame, whose replay may predate a submit.
        replay_sinks: Per tab, the connections waiting for a
            ``task_events`` replay, one per forwarded ``resumeSession``,
            answered in order (the socket keeps the remote's order).
        replay_conn: Per tab, the connection its last replay went to:
            the ``openSubagentTab`` announcements and child transcripts
            a replay goes on with reach that connection alone.
        retired: Set (under ``lock``) once the owner took the stream
            out of service; frames still in flight are dropped.
    """

    def __init__(
        self, tab_id: str, machine: Machine, owner: RemoteTabs,
        subagent: bool = False, parent_tab_id: str = "",
    ) -> None:
        """Connect to *machine* and start relaying its frames for *tab_id*.

        Raises:
            MachineError: When the machine is unreachable or rejects the
                password.
        """
        self.tab_id = tab_id
        self.machine = machine
        self.tab_ids: set[str] = {tab_id}
        self.task_ids: set[str] = set()
        self.subagent = subagent
        self.parent_tab_id = parent_tab_id
        self.task_id = ""
        self.pending_spawns: dict[str, str] = {}
        self.spawn = ""
        self.chat_id = ""
        self.shown_task_id = ""
        self.replay_sinks: dict[str, list[str]] = {}
        self.replay_conn: dict[str, str] = {}
        self.retired = False
        self.lock = threading.Lock()
        self._owner = owner
        self._closed = False
        self._ws: ClientConnection = link_for(machine).open_connection()
        self._thread = threading.Thread(
            target=self._relay, name=f"remote-tab-{tab_id}", daemon=True,
        )
        self._thread.start()

    def send(self, cmd: dict[str, Any]) -> None:
        """Forward *cmd* to the machine's daemon, without this daemon's own fields.

        Bounded by :data:`~kiss.agents.sorcar.machines.REQUEST_TIMEOUT`:
        a plain ``ws.send`` has no deadline and holds the connection's
        write lock, so one stuck on a daemon that stopped reading would
        block the client's command loop for good and wedge the
        stream's :meth:`close` (``closeTab``, daemon shutdown) behind
        it.  A write that misses the deadline closes the socket
        underneath, which ends the stream as a lost connection.

        Raises:
            MachineError: When the socket is gone or the daemon did not
                accept the command in time.
        """
        dropped = _DROPPED_RUN_FIELDS if cmd.get("type") == "run" else _DROPPED_FIELDS
        text = json.dumps({k: v for k, v in cmd.items() if k not in dropped})
        try:
            local_endpoint.send(self._ws, text, timeout=REQUEST_TIMEOUT)
        except (ConnectionClosed, OSError) as exc:
            raise MachineError(f"{self.machine.name}: the connection is lost: {exc}") from exc

    def close(self) -> None:
        """Close the socket; the relay thread then ends without an error."""
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            logger.debug("closing the stream of tab %s failed", self.tab_id, exc_info=True)

    def _relay(self) -> None:
        """Hand every frame of this tab's tasks to the daemon's clients."""
        try:
            for raw in self._ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    self._relay_frame(msg, text)
        except (ConnectionClosed, OSError):
            pass
        finally:
            self._owner.dropped(self, lost=not self._closed)

    def _relay_frame(self, msg: dict[str, Any], text: str) -> None:
        """Deliver one decoded frame when it belongs to this stream's tabs.

        A ``task_events`` of a tab with replay sinks answers the oldest
        one; an ``openSubagentTab`` (a replay announcing a running
        sub-agent) adopts the sub-agent tab and follows its parent's
        replay to that connection; everything else goes to every client.
        """
        kind = msg.get("type")
        parent = str(msg.get("parent_tab_id") or "")
        sub_tab = str(msg.get("tab_id") or "")
        to: list[str] | None = []  # [] = every client, [conn] = that one, None = nobody
        adopt = ""
        spawn = ""  # the run's own new_tab, learnt to be ours by this frame
        with self.lock:
            if self.retired:
                return
            if kind == "new_tab" and parent in self.tab_ids:
                if msg.get("task_id"):
                    self.task_ids.add(str(msg["task_id"]))
            elif kind == "new_tab" and self.subagent:
                # The remote announces every sub-agent run to every
                # client, this stream's own among those of its
                # siblings: only the one whose task is this run's is
                # this stream's to relay (kept, as ``spawn``, for the
                # surfaces that connect while it runs).  Until the run's
                # own frames have told its task id, the candidates wait.
                spawned = str(msg.get("task_id") or "")
                if not self.task_id:
                    self.pending_spawns[spawned] = text
                    to = None
                elif spawned == self.task_id:
                    self.task_ids.add(spawned)
                    self.spawn = text
                else:
                    to = None
            elif kind == "openSubagentTab" and parent in self.tab_ids and sub_tab:
                # Announced by a replay: the sub-agent tab's own (a
                # client resumed it: the oldest sink gets it, and keeps
                # waiting for the transcript) or its parent's (a client
                # that just got the parent's replay learns of a running
                # sub-agent: its transcript follows).
                if sub_tab not in self.tab_ids:
                    self.tab_ids.add(sub_tab)
                    adopt = sub_tab
                if self.replay_sinks.get(sub_tab):
                    to = [self.replay_sinks[sub_tab][0]]
                elif self.replay_conn.get(parent, ""):
                    conn = self.replay_conn[parent]
                    to = [conn]
                    self.replay_sinks.setdefault(sub_tab, []).append(conn)
            elif kind == "subagentDone" and sub_tab in self.tab_ids:
                pass
            elif msg.get("tabId") in self.tab_ids:
                tab_id = str(msg["tabId"])
                if tab_id == self.tab_id and not self.chat_id and msg.get("chat_id"):
                    self.chat_id = str(msg["chat_id"])
                if tab_id == self.tab_id and self.subagent:
                    spawn = self._own_run_frame(kind, msg)
                if kind == "task_events" and self.replay_sinks.get(tab_id):
                    conn = self.replay_sinks[tab_id].pop(0)
                    self.replay_conn[tab_id] = conn
                    to = [conn]
            else:
                to = None
        if to is None:
            return
        if adopt:
            self._owner.adopt(self, adopt)  # takes the owner's lock: outside ours
        # Delivered under the lock retirement takes: a retired stream
        # delivers nothing more (the sends themselves are scheduled, not
        # awaited, so nothing blocks here).
        with self.lock:
            if self.retired:
                return
            if spawn:
                self._owner.deliver("", spawn)
            if to:
                self._owner.deliver_to(to[0], text)
            else:
                # A frame without a tab (``new_tab``, ``subagentDone``)
                # is global, as the daemon's own broadcast of it is: a
                # peer that only addressed other tabs still gets it.
                self._owner.deliver(str(msg.get("tabId") or ""), text)

    def _own_run_frame(self, kind: Any, msg: dict[str, Any]) -> str:
        """Note what a frame of the bound tab tells about the sub-agent run (under ``lock``).

        The first frame carrying the run's persisted task id (every
        frame but ``status``, whose ``taskId`` is the run token) names
        the run among the ``new_tab`` announcements held back so far;
        the terminal ``status running=false`` ends what a connecting
        surface is told.

        Returns:
            The run's own ``new_tab`` frame when this frame singled it
            out of the held-back ones, to be delivered now; else "".
        """
        if kind == "status":
            if not msg.get("running"):
                self.spawn = ""
            return ""
        if self.task_id or not msg.get("taskId"):
            return ""
        self.task_id = str(msg["taskId"])
        spawn = self.pending_spawns.pop(self.task_id, "")
        self.pending_spawns.clear()
        if spawn:
            self.task_ids.add(self.task_id)
            self.spawn = spawn
        return spawn


class RemoteTabs:
    """The tabs of this daemon that run on remote machines."""

    def __init__(self, backend: Any) -> None:
        """*backend* is the daemon (``ServerBackend``): its printer and agent server."""
        self._backend = backend
        # Every tab a stream relays (the bound tab and its sub-agent tabs).
        self._tabs: dict[str, RemoteTab] = {}
        # Bumped by every unbind of a tab: a bind whose connect spanned
        # an unbind of the same tab stands down (its tab is gone).
        self._unbinds: dict[str, int] = {}
        self._lock = threading.Lock()

    def machine_of(self, tab_id: str) -> str:
        """Return the name of the machine *tab_id* is bound to ("" when it runs here)."""
        with self._lock:
            tab = self._tabs.get(tab_id)
        return tab.machine.name if tab is not None else ""

    def route(self, cmd: dict[str, Any]) -> bool:
        """Send *cmd* to its tab's machine when the tab runs there.

        A ``submit`` or ``run`` binds the tab to the machine it names
        (re-binding when the name changed, unbinding when it is empty);
        every other forwarded command follows the tab's binding.
        A ``resumeSession`` naming a machine binds the tab to it too
        (the machine's chat is replayed into the tab); one of a
        sub-agent task joins its tab to the stream; ``resumeSession``
        of anything else, ``newChat`` and ``closeTab`` unbind the tab
        and are still handled here.

        Returns:
            True when the command was consumed (forwarded, or refused
            with an ``error`` on the tab); False when this daemon
            handles it.
        """
        name = cmd["type"]
        tab_id = str(cmd.get("tabId", "") or "")
        conn_id = str(cmd.get("connId", "") or "")
        if not tab_id:
            return False
        if name == "resumeSession":
            return self._resume(tab_id, str(cmd.get("taskId", "") or ""), conn_id, cmd)
        if name in ("closeTab", "newChat"):
            self._unbind(tab_id, forward_close=name == "closeTab")
            return False
        if name in BINDING_COMMANDS:
            machine_name = str(cmd.get("machine", "") or "")
            if not machine_name:
                # A submit brings its tab home; a run's tab is fresh
                # (nothing to unbind, and nothing to count for it).
                if name == "submit":
                    self._unbind(tab_id)
                return False
            tab = self._bind(tab_id, machine_name, cmd, str(cmd.get("prompt", "") or ""))
            if tab is None:
                return True
            with tab.lock:
                # The new task becomes the chat's latest: no pin to replay.
                tab.shown_task_id = ""
        else:
            with self._lock:
                tab = self._tabs.get(tab_id)
            if tab is None:
                return False
        try:
            tab.send(cmd)
        except MachineError as exc:
            self._backend._vscode_server._refuse_run(tab_id, str(exc))
        return True

    def _bind(
        self, tab_id: str, machine_name: str, cmd: dict[str, Any], title: str,
    ) -> RemoteTab | None:
        """Bind *tab_id* to the registered machine *machine_name* (None when refused).

        *title* is the tab's title in the registry (the prompt of a
        submit, the task of a resumed chat).  Connecting blocks (up to
        the link's timeout) outside the lock, so the other tabs'
        commands keep flowing meanwhile; what
        happened to the tab in between decides what the new stream is
        worth: a ``closeTab`` (``_unbinds`` moved) stands it down, a
        bind to the same machine that won the race is kept instead of
        it, one to another machine is retired for it.
        """
        machine = next((m for m in load_machines() if m.name == machine_name), None)
        if machine is None:
            self._backend._vscode_server._refuse_run(
                tab_id, f"{machine_name} is not a registered machine",
            )
            return None
        with self._lock:
            tab = self._tabs.get(tab_id)
            unbinds = self._unbinds.get(tab_id, 0)
        if tab is not None and tab.machine != machine:
            self._unbind(tab_id)
            tab = None
            unbinds += 1
        announce = False
        if tab is None:
            try:
                tab = RemoteTab(
                    tab_id, machine, self, subagent=bool(cmd.get("parentTaskId")),
                    parent_tab_id=str(cmd.get("parentTabId", "") or ""),
                )
            except MachineError as exc:
                self._backend._vscode_server._refuse_run(tab_id, str(exc))
                return None
            displaced: RemoteTab | None = None
            with self._lock:
                if self._unbinds.get(tab_id, 0) != unbinds:
                    tab.close()
                    return None
                winner = self._tabs.get(tab_id)
                if winner is not None and winner.machine == machine:
                    tab.close()
                    tab = winner
                else:
                    if winner is not None:
                        displaced = self._retire(winner)
                    self._tabs[tab_id] = tab
            if displaced is not None:
                displaced.close()
            announce = tab is not winner
        # Every surface shows the tab (same registry entry a local run
        # publishes); its chat and task stay unbound here, they are the
        # remote daemon's.  Registered before the binding is announced:
        # a local client's delivery is narrowed to the tabs it can show.
        # A sub-agent run (``parentTaskId``) gets no top-level tab, as
        # one running here does not: the surfaces open its nested tab
        # under the caller on the relayed ``new_tab``.
        if not cmd.get("parentTaskId"):
            self._backend._vscode_server._registry_update_tab(
                tab_id,
                title=title,
                work_dir=str(cmd.get("workDir", "") or ""),
                task_id="",
                create=True,
            )
        if announce:
            self._announce(tab_id, machine.name)
        return tab

    def _retire(self, tab: RemoteTab) -> RemoteTab:
        """Take a stream out of service (under the lock): its frames dropped, its tabs unmapped."""
        with tab.lock:
            tab.retired = True
            tab_ids = list(tab.tab_ids)
        for tid in tab_ids:
            if self._tabs.get(tid) is tab:
                del self._tabs[tid]
        return tab

    def _unbind(self, tab_id: str, forward_close: bool = False) -> None:
        """Take *tab_id* off its stream: a bound tab's stream closes, a sub-agent tab leaves it.

        Counted even for a tab without a stream, so a bind still
        connecting for it stands down.  With *forward_close* the remote
        daemon closes the tab too.
        """
        with self._lock:
            self._unbinds[tab_id] = self._unbinds.get(tab_id, 0) + 1
            tab = self._tabs.get(tab_id)
            if tab is None:
                return
            if tab.tab_id == tab_id:
                self._retire(tab)
            else:
                del self._tabs[tab_id]
                with tab.lock:
                    tab.tab_ids.discard(tab_id)
                    tab.replay_sinks.pop(tab_id, None)
                    tab.replay_conn.pop(tab_id, None)
        if forward_close:
            try:
                tab.send({"type": "closeTab", "tabId": tab_id})
            except MachineError:
                pass
        if tab.tab_id == tab_id:
            tab.close()
            self._announce(tab_id, "")

    def adopt(self, tab: RemoteTab, sub_tab_id: str) -> None:
        """Map the sub-agent tab *sub_tab_id* a replay announced to *tab*'s stream."""
        with self._lock:
            if self._tabs.get(tab.tab_id) is tab:
                self._tabs[sub_tab_id] = tab

    def _resume(self, tab_id: str, task_id: str, conn_id: str, cmd: dict[str, Any]) -> bool:
        """Route a ``resumeSession``: a machine's chat binds the tab, a sub-agent task's joins it.

        A resume naming a ``machine`` (a click on that machine's row of
        the History panel) binds the tab to the machine as a submit
        would and forwards the resume: the machine's daemon replays its
        chat into this tab, which stays bound, so the user reads and
        continues the chat here instead of in the machine's own web UI.
        The forwarded ``title`` names the tab meanwhile (the replay
        then sets it, as a local chat's would).  A resume of a
        sub-agent task joins its tab to the task's stream: the stream
        that announced the task (``new_tab``), else the stream of the
        ``parentTabId`` the resume names (a finished sub-agent of a
        replayed chat, whose ``new_tab`` this daemon never saw).  Any
        other resume into a bound tab shows a local chat there: the tab
        is unbound and the resume handled here.
        """
        machine_name = str(cmd.get("machine", "") or "")
        if machine_name:
            tab = self._bind(tab_id, machine_name, cmd, str(cmd.get("title", "") or ""))
            if tab is None:
                return True
            with tab.lock:
                # The tab shows this chat, at this task, from here on
                # (a later ``ready`` replays those); taken from the
                # click, not from the replay it gets, which can cross a
                # later submit on the socket.
                tab.chat_id = str(cmd.get("chatId") or cmd.get("id") or "")
                tab.shown_task_id = str(cmd.get("taskId") or "")
                if cmd.get("scoped") and conn_id:
                    tab.replay_sinks.setdefault(tab_id, []).append(conn_id)
            try:
                tab.send(cmd)
            except MachineError as exc:
                self._backend._vscode_server._refuse_run(tab_id, str(exc))
            return True
        with self._lock:
            tab = next(
                (t for t in self._tabs.values() if task_id and task_id in t.task_ids), None,
            )
            if tab is None and task_id:
                tab = self._tabs.get(str(cmd.get("parentTabId", "") or ""))
            if tab is None:
                bound = self._tabs.get(tab_id)
                unbind = bound is not None and bound.tab_id == tab_id
            else:
                unbind = False
                self._tabs[tab_id] = tab
                with tab.lock:
                    tab.tab_ids.add(tab_id)
                    tab.task_ids.add(task_id)
                    tab.replay_sinks.setdefault(tab_id, []).append(conn_id)
        if unbind:
            self._unbind(tab_id)
        if tab is None:
            return False
        try:
            tab.send(cmd)
        except MachineError as exc:
            self._backend._vscode_server._refuse_run(tab_id, str(exc))
        return True

    def replay_to(self, conn_id: str, single_tab_id: str) -> None:
        """Replay every bound tab to *conn_id*, with its binding.

        Called after a client's ``ready`` was handled, so the client has
        the tabs (the ``tabs_state`` snapshot) before their transcripts
        arrive; *single_tab_id* (an editor-tab panel's one tab) narrows
        the replays as it does for the daemon's own chats.  A stream
        carrying a sub-agent run of a local task has no tab of its own
        to replay: the client gets the run's ``new_tab`` again (if it
        is still running) and opens and resumes the nested tab itself.
        """
        with self._lock:
            tabs = {t.tab_id: t for t in self._tabs.values()}
        for tab_id, tab in tabs.items():
            if tab.subagent:
                if not single_tab_id or single_tab_id == tab.parent_tab_id:
                    # Under the lock retirement takes (as a relayed frame
                    # is): a stream retired meanwhile announces nothing.
                    with tab.lock:
                        if tab.spawn and not tab.retired:
                            self.deliver_to(conn_id, tab.spawn)
                continue
            if single_tab_id and tab_id != single_tab_id:
                continue
            self.deliver_to(conn_id, _tab_machine_frame(tab_id, tab.machine.name))
            with tab.lock:
                chat_id = tab.chat_id
                task_id = tab.shown_task_id
                if chat_id:
                    tab.replay_sinks.setdefault(tab_id, []).append(conn_id)
            if not chat_id:
                continue
            resume: dict[str, Any] = {"type": "resumeSession", "chatId": chat_id, "tabId": tab_id}
            if task_id:
                resume["taskId"] = task_id
            try:
                tab.send(resume)
            except MachineError:
                pass

    def _announce(self, tab_id: str, machine_name: str) -> None:
        """Tell every client which machine *tab_id* is bound to ("" when none)."""
        self.deliver(tab_id, _tab_machine_frame(tab_id, machine_name))

    def deliver(self, tab_id: str, text: str) -> None:
        """Send one frame of the remote daemon, stamped with *tab_id*, to every client."""
        self._backend._printer._send_to_ws_clients(text, tab_id)

    def deliver_to(self, conn_id: str, text: str) -> None:
        """Send one frame of the remote daemon to the connection *conn_id* only."""
        self._backend._printer._send_to_conn(conn_id, text)

    def dropped(self, tab: RemoteTab, lost: bool) -> None:
        """A stream ended: forget its tabs, and when it was *lost* end the tab's task.

        A stream already retired (replaced, unbound) is nobody's loss.
        """
        with self._lock:
            if self._tabs.get(tab.tab_id) is tab:
                self._retire(tab)
            else:
                lost = False
        if lost:
            self._backend._vscode_server._refuse_run(
                tab.tab_id, f"lost the connection to {tab.machine.name}",
            )
            self._announce(tab.tab_id, "")

    def close_all(self) -> None:
        """Close every stream (daemon shutdown)."""
        with self._lock:
            tabs = list({id(t): t for t in self._tabs.values()}.values())
            for tab in tabs:
                self._retire(tab)
        for tab in tabs:
            tab.close()


def _tab_machine_frame(tab_id: str, machine_name: str) -> str:
    """The ``tabMachine`` frame announcing *tab_id*'s binding."""
    return json.dumps({"type": "tabMachine", "tabId": tab_id, "machine": machine_name})
