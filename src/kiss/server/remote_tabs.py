# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chat tabs whose tasks run on a registered remote machine.

The machine drop-down at the top of a chat page chooses where the
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
its own chat page.  Every frame the remote stamps with the tab id is
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
  The binding is also told to :class:`~kiss.server.remote_views.RemoteViews`,
  which keeps a view stream to the machine open while a tab is bound
  there: the Browser tab an agent of the machine opens for the user
  (``show_browser``, a sign-in hand-off) reaches this daemon's
  surfaces over it.

A tab stays bound while it exists: a later ``submit`` or
``resumeSession`` for another machine re-binds it, an empty
``machine`` brings it home, a ``resumeSession`` of a local chat or
``closeTab`` unbinds it
(``closeTab`` is forwarded too, so the remote drops its tab; its task
keeps running there as a closed local tab's would).  Losing the socket
ends the tab's running state with an ``error`` naming the machine.
Either way the stream's sub-agent tabs end with a ``subagentDone`` of
this daemon's: no frame of the machine reaches them any more.

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
import threading
import uuid
from typing import Any

from kiss.agents.sorcar.machines import Machine, MachineError, load_machines
from kiss.server.machine_stream import DROPPED_FIELDS, MachineStream
from kiss.server.remote_views import RemoteViews

FORWARDED_COMMANDS = frozenset(
    {
        "submit",
        "run",
        "stop",
        "userAnswer",
        "appendUserMessage",
        "interruptTool",
        "selectModel",
    }
)
"""Tab commands a bound tab sends to its machine instead of running here."""

BINDING_COMMANDS = frozenset({"submit", "run"})
"""The commands whose ``machine`` field binds (or unbinds) their tab."""

ROUTED_COMMANDS = FORWARDED_COMMANDS | {"closeTab", "newChat", "resumeSession"}
"""The commands :meth:`RemoteTabs.route` looks at (``ready`` goes to :meth:`replay_to`)."""

_DROPPED_FIELDS = DROPPED_FIELDS + ("workDir", "tabScopeWorkDir")
"""Fields of a forwarded command that only mean something on this daemon.

``workDir`` is a path of this machine: without it the remote daemon
runs the task in its own configured working directory.  A ``run``
keeps it: its sender names the directory on the machine (empty for
the machine's configured one).
"""

_DROPPED_RUN_FIELDS = tuple(f for f in _DROPPED_FIELDS if f != "workDir")


class RemoteTab(MachineStream):
    """One bound tab's stream to the daemon of a remote machine.

    Attributes:
        tab_id: The bound (root) tab.
        tab_ids: The root tab plus the sub-agent tabs this daemon's
            clients opened for its tasks; their frames are relayed.
        task_ids: The ids of the sub-agent tasks the remote announced
            (``new_tab``); a ``resumeSession`` naming one is forwarded,
            and frames addressed to its ``{parent}__sub_{task_id}`` tab
            are relayed before any client resumed it
            (:meth:`_owns_sub_tab`).
        announced: By task id, the tab ``{parent}__sub_{task_id}`` the
            surfaces opened on a relayed ``new_tab`` and no client has
            resumed yet (a resume maps the tab, an ``openSubagentTab``
            adopts it, its ``subagentDone`` ends it): a stream lost
            meanwhile ends those tabs too, or they would spin for good.
        subagent: Whether the bound tab's run is a sub-agent run of a
            task of this daemon (a forwarded ``run`` with a
            ``parentTaskId``): the ``new_tab`` the remote announces
            for it is relayed, under the caller's tab or parentless
            as the remote broadcast it.  False for a tab the user bound.
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
        replay_sinks: By replay id, the connection waiting for the
            answer to the forwarded ``resumeSession`` stamped with it
            (:meth:`stamp_replay`).  The remote echoes the id on every
            frame of the answer (the ``task_events`` transcript, or
            ``status running:false`` alone for a chat it has no row
            for and nothing running, or ``subagentDone`` for a
            sub-agent that is over, and the ``openSubagentTab``
            announcements and child transcripts a replay goes on
            with) and ends it with ``replayDone``, which releases the
            sink.  A frame of the tab carrying no id (the task's own
            lifecycle, a transcript nobody here asked for) goes to
            every client, whatever replays are pending.
    """

    def __init__(
        self,
        tab_id: str,
        machine: Machine,
        owner: RemoteTabs,
        subagent: bool = False,
    ) -> None:
        """Connect to *machine* and start relaying its frames for *tab_id*.

        Raises:
            MachineError: When the machine is unreachable or rejects the
                password.
        """
        self.tab_id = tab_id
        self.tab_ids: set[str] = {tab_id}
        self.task_ids: set[str] = set()
        self.announced: dict[str, str] = {}
        self.subagent = subagent
        self.task_id = ""
        self.pending_spawns: dict[str, str] = {}
        self.spawn = ""
        self.chat_id = ""
        self.shown_task_id = ""
        self.replay_sinks: dict[str, str] = {}
        super().__init__(machine, owner, f"remote-tab-{tab_id}")

    def _dropped_fields(self, cmd: dict[str, Any]) -> tuple[str, ...]:
        """:data:`_DROPPED_FIELDS`, except that a ``run`` keeps its ``workDir``."""
        return _DROPPED_RUN_FIELDS if cmd.get("type") == "run" else _DROPPED_FIELDS

    def stamp_replay(self, cmd: dict[str, Any], conn_id: str) -> None:
        """Stamp the ``resumeSession`` *cmd* with a replay id whose answer is *conn_id*'s alone.

        Called under ``lock``, before the command is forwarded: the
        remote echoes the id on every frame of the answer
        (:attr:`replay_sinks`).

        Args:
            cmd: The ``resumeSession`` command about to be forwarded
                (mutated in place).
            conn_id: The connection the answer goes to.
        """
        replay_id = uuid.uuid4().hex
        self.replay_sinks[replay_id] = conn_id
        cmd["replayId"] = replay_id

    def _relay_frame(self, msg: dict[str, Any], text: str) -> None:
        """Deliver one decoded frame when it belongs to this stream's tabs.

        A frame echoing the id of a pending replay goes to the
        connection that asked for it (``replayDone`` releases the sink
        and reaches nobody); an ``openSubagentTab`` (a replay
        announcing a running sub-agent) adopts the sub-agent tab;
        everything else goes to every client.
        """
        kind = msg.get("type")
        parent = str(msg.get("parent_tab_id") or "")
        sub_tab = str(msg.get("tab_id") or "")
        replay_id = str(msg.get("replayId") or "")
        to: list[str] | None = []  # [] = every client, [conn] = that one, None = nobody
        adopt = ""
        spawn = ""  # the run's own new_tab, learnt to be ours by this frame
        with self.lock:
            if self.retired:
                return
            if replay_id in self.replay_sinks:
                to = [self.replay_sinks[replay_id]]
            if kind == "replayDone":
                self.replay_sinks.pop(replay_id, None)
                to = None
            elif kind == "new_tab" and self._owns_sub_tab(parent):
                if msg.get("task_id"):
                    self.task_ids.add(str(msg["task_id"]))
                    self.announced[str(msg["task_id"])] = _sub_tab_id(msg)
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
                    self.announced[spawned] = _sub_tab_id(msg)
                    self.spawn = text
                else:
                    to = None
            elif kind == "openSubagentTab" and parent in self.tab_ids and sub_tab:
                # Announced by a replay: the sub-agent tab's own (a
                # client resumed it: the transcript follows) or its
                # parent's (a client that just got the parent's replay
                # learns of a running sub-agent).
                if sub_tab not in self.tab_ids:
                    self.tab_ids.add(sub_tab)
                    self.announced.pop(_sub_task_id(sub_tab), None)
                    adopt = sub_tab
            elif kind == "subagentDone" and self._owns_sub_tab(sub_tab):
                self.announced.pop(_sub_task_id(sub_tab), None)
            elif msg.get("tabId") in self.tab_ids:
                tab_id = str(msg["tabId"])
                if tab_id == self.tab_id and not self.chat_id and msg.get("chat_id"):
                    self.chat_id = str(msg["chat_id"])
                if tab_id == self.tab_id and self.subagent:
                    spawn = self._own_run_frame(kind, msg)
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

    def _owns_sub_tab(self, tab_id: str) -> bool:
        """Whether *tab_id* is a tab of this stream's tasks (under ``lock``).

        A tab a client joined (``tab_ids``), or the sub-agent tab
        ``{parent}__sub_{task_id}`` of a task the stream announced
        (``task_ids``) that no client has resumed yet: a sub-agent that
        fans out at once names that derived id as the parent of its
        children's ``new_tab``, and a sub-agent's ``subagentDone`` may
        name it, before any surface's ``resumeSession`` for the tab has
        reached this daemon.  A local run broadcasts those frames to
        every client, so the stream relays them too, or the surfaces
        would never open the nested tabs.
        """
        return tab_id in self.tab_ids or _sub_task_id(tab_id) in self.task_ids

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
            self.announced[self.task_id] = _sub_tab_id(json.loads(spawn))
            self.spawn = spawn
        return spawn


class RemoteTabs:
    """The tabs of this daemon that run on remote machines."""

    def __init__(self, backend: Any, views: RemoteViews) -> None:
        """*backend* is the daemon (``ServerBackend``): its printer and agent server.

        *views* is told every binding (:meth:`RemoteViews.bind`).
        """
        self._backend = backend
        self._views = views
        # Every tab a stream relays (the bound tab and its sub-agent tabs).
        self._tabs: dict[str, RemoteTab] = {}
        # Per tab, the binds connecting for it, and the unbinds of it
        # since the first of those started: a bind whose connect
        # spanned an unbind stands down (its tab is gone).  Both
        # entries go once no bind connects for the tab, so unbinds of
        # the tabs of a long session leave nothing behind.
        self._connecting: dict[str, int] = {}
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
        self._forward(tab, tab_id, cmd)
        return True

    def _forward(self, tab: RemoteTab, tab_id: str, cmd: dict[str, Any]) -> None:
        """Send *cmd* down *tab*'s stream; a lost stream fails the command on a sub-agent *tab_id*.

        A failed send ends the stream, whose relay thread then runs
        :meth:`dropped` as lost: that refuses the run on the root tab
        and ends its sub-agent tabs, so only a command of a sub-agent
        tab (which gets no refusal there) is answered here.
        """
        try:
            tab.send(cmd)
        except MachineError as exc:
            if tab_id != tab.tab_id:
                self._backend._agent_server._refuse_run(tab_id, str(exc))

    def _bind(
        self,
        tab_id: str,
        machine_name: str,
        cmd: dict[str, Any],
        title: str,
    ) -> RemoteTab | None:
        """Bind *tab_id* to the registered machine *machine_name* (None when refused).

        *title* is the tab's title in the registry (the prompt of a
        submit, the task of a resumed chat).  Connecting blocks (up to
        the link's timeout) outside the lock, so the other tabs'
        commands keep flowing meanwhile; what
        happened to the tab in between decides what the new stream is
        worth: an unbind (``_unbinds`` moved, counted while this bind
        is in ``_connecting``) stands it down, a bind to the same
        machine that won the race is kept instead of it, one to
        another machine is retired for it (its sub-agent tabs ended),
        and a stream the remote dropped meanwhile (its relay thread
        ran :meth:`dropped` before this registration) refuses the run
        instead of binding.
        """
        machine = next((m for m in load_machines() if m.name == machine_name), None)
        if machine is None:
            self._backend._agent_server._refuse_run(
                tab_id,
                f"{machine_name} is not a registered machine",
            )
            return None
        with self._lock:
            tab = self._tabs.get(tab_id)
            self._connecting[tab_id] = self._connecting.get(tab_id, 0) + 1
            unbinds = self._unbinds.get(tab_id, 0)
        try:
            if tab is not None and tab.machine != machine:
                self._unbind(tab_id)
                tab = None
                unbinds += 1
            announce = False
            if tab is None:
                try:
                    tab = RemoteTab(
                        tab_id,
                        machine,
                        self,
                        subagent=bool(cmd.get("parentTaskId")),
                    )
                except MachineError as exc:
                    self._backend._agent_server._refuse_run(tab_id, str(exc))
                    return None
                displaced: RemoteTab | None = None
                orphans: list[str] = []
                same: RemoteTab | None = None  # a bind to the same machine that won
                lost = False
                with self._lock:
                    stood_down = self._unbinds.get(tab_id, 0) != unbinds
                    winner = self._tabs.get(tab_id)
                    if winner is not None and winner.machine == machine:
                        same = winner
                    elif not stood_down:
                        with tab.lock:
                            lost = tab.retired  # dropped before it could be registered
                        if not lost:
                            if winner is not None:
                                orphans = self._retire(winner)
                                displaced = winner
                            self._tabs[tab_id] = tab
                            self._views.bind(tab_id, machine.name)
                if displaced is not None:
                    self._end_stream(displaced, orphans)
                if stood_down or same is not None or lost:
                    tab.close()
                if stood_down:
                    return None
                if lost:
                    self._backend._agent_server._refuse_run(
                        tab_id,
                        f"lost the connection to {machine.name}",
                    )
                    return None
                if same is not None:
                    tab = same
                announce = same is None
            # Every surface shows the tab (same registry entry a local run
            # publishes); its chat and task stay unbound here, they are the
            # remote daemon's.  Registered before the binding is announced:
            # a local client's delivery is narrowed to the tabs it can show.
            # A sub-agent run (``parentTaskId``) gets no top-level tab, as
            # one running here does not: the surfaces open its nested tab
            # under the caller on the relayed ``new_tab``.
            if not cmd.get("parentTaskId"):
                self._backend._agent_server._registry_update_tab(
                    tab_id,
                    title=title,
                    work_dir=str(cmd.get("workDir", "") or ""),
                    task_id="",
                    create=True,
                )
            if announce:
                self._announce(tab_id, tab)
            # The run is forwarded only once the machine's pages are watched
            # (a stream that could not be opened for an earlier bind is
            # tried again): the page its agent opens for the user must not
            # slip by before the stream is subscribed.
            self._views.open_for(machine.name)
            return tab
        finally:
            with self._lock:
                self._connecting[tab_id] -= 1
                if not self._connecting[tab_id]:
                    del self._connecting[tab_id]
                    self._unbinds.pop(tab_id, None)

    def _retire(self, tab: RemoteTab) -> list[str]:
        """Take a stream out of service (under the lock): its frames dropped, its tabs unmapped.

        Returns:
            The tab ids the stream was mapped to until now, then the
            tabs of the sub-agents it announced that no client resumed
            (``announced``) and no other stream has since mapped.
        """
        with tab.lock:
            tab.retired = True
            tab_ids = list(tab.tab_ids)
            announced = list(tab.announced.values())
        unmapped = [tid for tid in tab_ids if self._tabs.get(tid) is tab]
        for tid in unmapped:
            del self._tabs[tid]
        if tab.tab_id in unmapped:
            self._views.bind(tab.tab_id, "")
        return unmapped + [tid for tid in announced if tid not in self._tabs]

    def _detach(self, tab_id: str) -> tuple[RemoteTab | None, list[str]]:
        """Take *tab_id* off its stream, under the lock.

        Counted even for a tab without a stream, so a bind still
        connecting for it stands down.  A bound tab's stream is retired
        and returned with the sub-agent tabs it leaves behind, for
        :meth:`_end_stream` outside the lock; a sub-agent tab just
        leaves its stream (``(None, [])``, as for a tab without one).
        """
        if tab_id in self._connecting:
            self._unbinds[tab_id] = self._unbinds.get(tab_id, 0) + 1
        tab = self._tabs.get(tab_id)
        if tab is None:
            return None, []
        if tab.tab_id == tab_id:
            return tab, self._retire(tab)
        del self._tabs[tab_id]
        with tab.lock:
            tab.tab_ids.discard(tab_id)
        return None, []

    def _end_stream(self, tab: RemoteTab, tab_ids: list[str]) -> None:
        """Finish off the retired stream *tab* outside the lock.

        *tab_ids* is what :meth:`_retire` returned.  The socket is
        closed here (a close handshake can wait on the peer, up to the
        socket's close timeout, and the other tabs' commands must
        not), the sub-agent tabs it relayed end with
        ``subagentDone`` as a sub-agent running here ends (the ones a
        client resumed and the ones only announced so far: the
        sub-agent may still run on the machine, but no frame of it
        reaches this daemon's surfaces any more, and nothing else would
        ever stop their spinners), the tab's binding is announced gone
        (dropped by :meth:`_announce` when another stream has the tab
        by now) and the machine's view stream is released.

        A sub-agent tab is ended under the lock and only while the
        registry still has no stream for it: a client that bound it
        meanwhile (a History-panel click into the machine's chat
        while the close handshake waited) owns it now, and a
        ``subagentDone`` of the old stream would close its live tab
        on every surface.
        """
        tab.close()
        with self._lock:
            for sub_tab_id in tab_ids:
                if sub_tab_id != tab.tab_id and sub_tab_id not in self._tabs:
                    self.deliver(
                        "",
                        json.dumps(
                            {"type": "subagentDone", "tab_id": sub_tab_id, "tabId": ""},
                        ),
                    )
        self._announce(tab.tab_id, None)
        self._views.release(tab.machine.name)

    def _unbind(self, tab_id: str, forward_close: bool = False) -> None:
        """Take *tab_id* off its stream: a bound tab's stream ends, a sub-agent tab leaves it.

        With *forward_close* the remote daemon closes the tab too.
        """
        with self._lock:
            tab = self._tabs.get(tab_id)
            ended, orphans = self._detach(tab_id)
        if forward_close and tab is not None:
            try:
                tab.send({"type": "closeTab", "tabId": tab_id})
            except MachineError:
                pass
        if ended is not None:
            self._end_stream(ended, orphans)

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
                    tab.stamp_replay(cmd, conn_id)
            self._forward(tab, tab_id, cmd)
            return True
        ended: RemoteTab | None = None
        orphans: list[str] = []
        with self._lock:
            tab = next(
                (t for t in self._tabs.values() if task_id and task_id in t.task_ids),
                None,
            )
            if tab is None and task_id:
                tab = self._tabs.get(str(cmd.get("parentTabId", "") or ""))
            previous = self._tabs.get(tab_id)
            if tab is None:
                if previous is not None and previous.tab_id == tab_id:
                    ended, orphans = self._detach(tab_id)
            else:
                # One stream per tab: a tab bound to a machine, or on
                # another run's stream, leaves it (its own stream ends)
                # before it joins the task's stream.
                if previous is not None and previous is not tab:
                    ended, orphans = self._detach(tab_id)
                self._tabs[tab_id] = tab
                with tab.lock:
                    tab.tab_ids.add(tab_id)
                    tab.task_ids.add(task_id)
                    tab.announced.pop(task_id, None)
                    if conn_id:
                        tab.stamp_replay(cmd, conn_id)
        if ended is not None:
            self._end_stream(ended, orphans)
        if tab is None:
            return False
        self._forward(tab, tab_id, cmd)
        return True

    def replay_to(self, conn_id: str) -> None:
        """Replay every bound tab to *conn_id*, with its binding.

        Called after a client's ``ready`` was handled, so the client has
        the tabs (the ``tabs_state`` snapshot) before their transcripts
        arrive.  A stream carrying a sub-agent run of a local task has no tab of its own
        to replay: the client gets the run's ``new_tab`` again (if it
        is still running) and opens and resumes the nested tab itself.
        """
        with self._lock:
            tabs = {t.tab_id: t for t in self._tabs.values()}
        for tab_id, tab in tabs.items():
            if tab.subagent:
                # Under the lock retirement takes (as a relayed frame
                # is): a stream retired meanwhile announces nothing.
                with tab.lock:
                    if tab.spawn and not tab.retired:
                        self.deliver_to(conn_id, tab.spawn)
                continue
            self._replay_binding(conn_id, tab_id, tab)
            resume: dict[str, Any] = {"type": "resumeSession", "tabId": tab_id}
            with tab.lock:
                if not tab.chat_id:
                    continue
                resume["chatId"] = tab.chat_id
                if tab.shown_task_id:
                    resume["taskId"] = tab.shown_task_id
                tab.stamp_replay(resume, conn_id)
            try:
                tab.send(resume)
            except MachineError:
                pass

    def _replay_binding(self, conn_id: str, tab_id: str, tab: RemoteTab) -> None:
        """Tell *conn_id* that *tab_id* is bound to *tab*'s machine, while the registry says so.

        As :meth:`_announce`: a binding the registry moved past since
        the replay copied it is not replayed (the change announced
        itself, to this connection too), so the replay never lands
        after a later unbind's frame.
        """
        with self._lock:
            if self._tabs.get(tab_id) is tab:
                self.deliver_to(conn_id, _tab_machine_frame(tab_id, tab.machine.name))

    def _announce(self, tab_id: str, tab: RemoteTab | None) -> None:
        """Tell every client that *tab_id* is bound to *tab*'s machine (None: to none).

        Under the lock, and only while the registry still says so: the
        announcement of a binding that changed again meanwhile (an
        unbind under a bind still announcing, a re-bind under an
        unbind) is dropped, as the later change announced its own, so
        no client ends on a binding the registry does not have.  The
        views learn of a binding as the registry changes (:meth:`_bind`,
        :meth:`_retire`), not here: a bind that finds the tab registered
        by another must find its stream wanted already.
        """
        with self._lock:
            if self._tabs.get(tab_id) is not tab:
                return
            machine_name = tab.machine.name if tab is not None else ""
            self.deliver(tab_id, _tab_machine_frame(tab_id, machine_name))

    def deliver(self, tab_id: str, text: str) -> None:
        """Send one frame of the remote daemon, stamped with *tab_id*, to every client."""
        self._backend._printer._send_to_ws_clients(text, tab_id)

    def deliver_to(self, conn_id: str, text: str) -> None:
        """Send one frame of the remote daemon to the connection *conn_id* only."""
        self._backend._printer._send_to_conn(conn_id, text)

    def dropped(self, tab: RemoteTab, lost: bool) -> None:
        """A stream ended: forget its tabs, and when it was *lost* end the tab's task.

        The tab's run is refused (the surfaces show the loss) and the
        stream finished off as an unbound one is (:meth:`_end_stream`:
        its sub-agent tabs end too).  A stream already retired
        (replaced, unbound) is nobody's loss.
        """
        with self._lock:
            # Retired even when not (or no longer) registered: a stream
            # lost while _bind is still about to register it must not be
            # installed; _bind checks the flag.
            registered = self._tabs.get(tab.tab_id) is tab
            tab_ids = self._retire(tab)
        if lost and registered:
            self._backend._agent_server._refuse_run(
                tab.tab_id,
                f"lost the connection to {tab.machine.name}",
            )
            self._end_stream(tab, tab_ids)

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


def _sub_tab_id(new_tab: dict[str, Any]) -> str:
    """The tab the surfaces open for the sub-agent a ``new_tab`` frame announces.

    ``{parent_tab_id}__sub_{task_id}``, as the webapp derives it (and
    as the daemon names a sub-agent's tab when it replays its chat).
    """
    return f"{new_tab.get('parent_tab_id') or 'task'}__sub_{new_tab.get('task_id') or ''}"


def _sub_task_id(tab_id: str) -> str:
    """The task id a ``{parent}__sub_{task_id}`` tab id names ("" for any other id)."""
    return tab_id.rpartition("__sub_")[2] if "__sub_" in tab_id else ""
