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
* **A path-only prompt.**  A ``submit`` whose prompt is just the path
  of a file on the machine starts no task there: the machine answers
  ``promptOpened`` (the tab takes back the title the prompt displaced
  in the registry) and the file's ``fileContent``, relayed stamped
  ``machine`` like the reply to a forwarded ``openFile``, so the
  editor tab the surface opens saves and reloads the file on the
  machine.
* **Replays.**  A client's ``ready`` (a reload, a surface opening)
  replays every bound tab to that connection: a ``resumeSession`` of
  the tab's remote chat is forwarded and the ``task_events`` it
  answers go to that connection alone, as the daemon does for its own
  chats.
* **The binding.**  ``tabMachine {tabId, machine}`` tells every surface
  which machine a tab is bound to (on bind, on unbind, and on
  ``ready``), so the chat header's drop-down agrees across surfaces.
  A sub-agent tab that joins a stream (its ``resumeSession``, or an
  ``openSubagentTab`` a replay announced) is told the same way: the
  drop-down shows the machine for the sub-agent tab as for its
  parent, not this machine.  The binding is also told to
  :class:`~kiss.server.remote_views.RemoteViews`, which keeps a view
  stream to the machine open while a tab is bound
  there: the Browser tab an agent of the machine opens for the user
  (``show_browser``, a sign-in hand-off) reaches this daemon's
  surfaces over it.

A tab stays bound while it exists: a later ``submit`` or
``resumeSession`` for another machine re-binds it, an empty
``machine`` brings it home, a ``resumeSession`` of a local chat or
``closeTab`` unbinds it
(``closeTab`` is forwarded too, so the remote drops its tab; its task
keeps running there as a closed local tab's would).  A machine's chat
shows in one tab, as a local chat does: a ``resumeSession`` of it into
another tab displaces the tab that showed it, closed on every surface
(:meth:`RemoteTabs._displace_locked`).  Losing the socket
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
import logging
import threading
import uuid
from typing import Any

from kiss.agents.sorcar.machines import Machine, MachineError
from kiss.agents.sorcar.persistence import _record_steer_input
from kiss.server.machine_stream import DROPPED_FIELDS, MachineStream
from kiss.server.remote_views import RemoteViews, registered_machine

logger = logging.getLogger(__name__)

FORWARDED_COMMANDS = frozenset(
    {
        "submit",
        "run",
        "stop",
        "userAnswer",
        "appendUserMessage",
        "interruptTool",
        "worktreeAction",
    }
)
"""Tab commands a bound tab sends to its machine instead of running here.

The Merge/Discard/Do-nothing bar of a bound tab (``worktreeAction``)
is the machine's: its task left the worktree there, and the machine's
``worktree_result`` comes back stamped with the tab.

A model pick (``selectModel``) is not one: it is for the machine the
chat header's drop-down shows, which the page names in its ``machine``
field (:func:`~kiss.server.sidebar_panels.machine_models_reply`).
"""

BINDING_COMMANDS = frozenset({"submit", "run"})
"""The commands whose ``machine`` field binds (or unbinds) their tab."""

TYPED_COMMANDS = frozenset({"submit", "appendUserMessage"})
"""Forwarded commands whose ``prompt`` the user typed into the composer.

Their text is remembered here too (``steer_inputs``, see
:func:`_record_steer_input`) when it is handed to the machine: the
machine keeps the task row or the steer echo, and this daemon's
composer history (its ArrowUp list, ghost completions and ``/`` popup
order) reads only its own database, so it would forget a prompt or a
``/ask`` line typed into a bound tab.  A ``run`` is an API caller's
text and a ``userAnswer`` an answer, not prompt history.
"""

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
        finished: By sub-agent tab, the ``subagentDone`` relayed for it
            while the tab is still the stream's (no client closed it
            yet): a surface connecting later is told it again
            (:meth:`RemoteTabs.replay_to`), as a page that was away when
            the child finished keeps its tab (the registry snapshot
            never listed it) and would show it spinning for good.  Kept
            across the ``closeTab`` a page online answers the
            ``subagentDone`` with (the tab is closed on every surface
            online; the ones away still need the end); forgotten when
            the tab shows something else again (a resume into it, a
            fan-out panel reopening the finished child, a new chat, a
            bind: :meth:`RemoteTabs._forget_finished`) or the stream
            retires.
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
        displaced_titles: By tab (the root tab or a sub-agent tab the
            stream relays), what the latest forwarded ``submit`` of
            that tab did to the registry: the id the submit was
            stamped with (``promptId``, echoed by the machine's
            ``promptOpened``), the title the prompt replaced ("" when
            the tab had none) and the generation token of that
            publication.  A prompt that is just the path of a file on
            the machine starts no task there: its ``promptOpened`` puts
            the title back (:meth:`_relay_frame`), as a local tab's
            page drops the claim it stamped for such a prompt --
            provided the id is the pending submit's and nothing
            republished the tab since (a later real submit's title
            must not be undone by an earlier file prompt's late answer).
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
        self.revoked: set[str] = set()
        self.orphan_generations: dict[str, int] = {}
        self.children: dict[str, str] = {}  # live announcements, without replay stamps
        self.finished: dict[str, str] = {}
        self.subagent = subagent
        self.task_id = ""
        self.pending_spawns: dict[str, str] = {}
        self.spawn = ""
        self.chat_id = ""
        self.shown_task_id = ""
        self.displaced_titles: dict[str, tuple[str, str, int]] = {}
        self.replay_sinks: dict[str, str] = {}
        self.last_status: dict[str, str] = {}
        """The latest ``status`` frame relayed for each of the stream's tabs, by tab id.

        What a client connecting later is told about the tab's running
        state instead of the transcript (:meth:`RemoteTabs.replay_to`).
        """
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

        A path-only ``submit`` (a prompt that is just the path of a
        file on the machine) starts no task there: the machine answers
        ``promptOpened``, which puts the registry title the prompt
        displaced back, and the file's ``fileContent``, which is
        stamped with the machine before delivery, as the reply to a
        forwarded ``openFile`` is
        (:func:`~kiss.server.sidebar_panels.machine_workspace_reply`):
        the webapp binds the editor tab to the machine named in the
        stamp, so its ``saveFile`` and reload go back there instead of
        to this daemon, which has no such file.
        """
        kind = msg.get("type")
        parent = str(msg.get("parent_tab_id") or "")
        sub_tab = str(msg.get("tab_id") or "")
        addressed = str(msg.get("tabId") or "")  # the frame's tab, "" for a global frame
        replay_id = str(msg.get("replayId") or "")
        to: list[str] | None = []  # [] = every client, [conn] = that one, None = nobody
        adopt = ""
        learnt = False  # the frame told the bound tab's chat
        spawn = ""  # the run's own new_tab, learnt to be ours by this frame
        restore: tuple[str, str, int] | None = None  # a path-only prompt's displaced title
        with self.lock:
            if self.retired:
                return
            if replay_id in self.replay_sinks:
                to = [self.replay_sinks[replay_id]]
            if kind == "replayDone":
                self.replay_sinks.pop(replay_id, None)
                to = None
            elif kind == "new_tab" and self._owns_sub_tab(parent):
                if _sub_tab_id(msg) in self.revoked:
                    return
                if msg.get("task_id"):
                    self.task_ids.add(str(msg["task_id"]))
                    self.announced[str(msg["task_id"])] = _sub_tab_id(msg)
                    self.children[_sub_tab_id(msg)] = json.dumps(_without_replay_id(msg))
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
            elif kind == "openSubagentTab" and self._owns_sub_tab(parent) and sub_tab:
                # Announced by a replay: the sub-agent tab's own (a
                # client resumed it: the transcript follows) or its
                # parent's (a client that just got the parent's replay
                # learns of a running sub-agent).
                if sub_tab in self.revoked:
                    return
                adopt = sub_tab
            elif kind == "subagentDone" and self._owns_sub_tab(sub_tab):
                self.announced.pop(_sub_task_id(sub_tab), None)
                self.children.pop(sub_tab, None)
                self.last_status.pop(sub_tab, None)
                self.finished[sub_tab] = json.dumps(_without_replay_id(msg))
            elif msg.get("tabId") in self.tab_ids:
                tab_id = str(msg["tabId"])
                if tab_id == self.tab_id and not self.chat_id and msg.get("chat_id"):
                    self.chat_id = str(msg["chat_id"])
                    learnt = True
                if tab_id == self.tab_id and self.subagent:
                    spawn = self._own_run_frame(kind, msg)
                if kind == "status":
                    # The tab's state as last told (a replay's status is
                    # addressed to one client; the state it carries is
                    # the tab's).
                    self.last_status[tab_id] = (
                        json.dumps(_without_replay_id(msg)) if replay_id else text
                    )
                if kind == "fileContent":
                    msg["machine"] = self.machine.name
                    text = json.dumps(msg)
                elif kind == "promptOpened":
                    pending = self.displaced_titles.get(tab_id)
                    if pending and pending[0] == str(msg.get("promptId") or ""):
                        del self.displaced_titles[tab_id]
                        restore = (tab_id, pending[1], pending[2])
            else:
                to = None
        if to is None:
            return
        if restore:
            self._owner.retitle(*restore)
        if adopt and not self._owner.adopt(self, adopt):
            return  # the child left while this announcement was in flight
        # Delivered under the lock retirement takes: a retired stream
        # delivers nothing more (the sends themselves are scheduled, not
        # awaited, so nothing blocks here), nor does a tab that left the
        # stream meanwhile (a sub-agent tab a local chat was resumed
        # into, :meth:`RemoteTabs._detach`) get the frame found eligible
        # for it before: it would land in the chat the tab shows now,
        # or, a ``subagentDone``, close the tab that shows it.
        with self.lock:
            if self.retired or (addressed and addressed not in self.tab_ids):
                return
            if kind in ("subagentDone", "openSubagentTab") and not self._owns_sub_tab(sub_tab):
                return
            if kind == "new_tab" and _sub_tab_id(msg) in self.revoked:
                return
            if kind == "openSubagentTab" and not msg.get("isDone"):
                self.children[sub_tab] = json.dumps(_without_replay_id(msg))
            if spawn:
                self._owner.deliver("", spawn)
            if to:
                self._owner.deliver_to(to[0], text)
            else:
                # A frame without a tab (``new_tab``, ``subagentDone``)
                # is global, as the daemon's own broadcast of it is: a
                # peer that only addressed other tabs still gets it.
                self._owner.deliver(addressed, text)
        if learnt:
            # After the frame: closing the chat's other tab can wait on
            # its socket's close handshake, the run's output must not.
            self._owner.displace(self)  # takes the owner's lock: outside ours

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
        return tab_id not in self.revoked and (
            tab_id in self.tab_ids or _sub_task_id(tab_id) in self.task_ids
        )

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
        # Set by :meth:`close_all`: a bind still connecting then stands
        # down instead of registering a stream the shutdown never closes.
        self._closing = False
        self._lock = threading.Lock()

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
        if name in TYPED_COMMANDS:
            try:
                _record_steer_input(str(cmd.get("prompt", "") or ""))
            except Exception:
                logger.warning("typed text not recorded", exc_info=True)
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
            with self._lock:
                if tab_id != tab.tab_id and self._tabs.get(tab_id) is tab:
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
        submit, the task of a resumed chat).  A tab bound to another
        machine is taken off its stream first, in the same critical
        section that counts this bind as connecting: a concurrent bind
        to the same machine then finds the tab already unbound instead
        of unbinding it again, which would stand this bind down (see
        :meth:`_detach`).  Connecting blocks (up to the link's timeout)
        outside the lock, so the other tabs' commands keep flowing
        meanwhile; what happened to the tab in between decides what
        the new stream is worth: an unbind (``_unbinds`` moved, counted
        while this bind is in ``_connecting``) stands it down, a bind
        to the same machine that won the race is kept instead of it,
        one to another machine is retired for it (its sub-agent tabs
        ended), and a stream the remote dropped meanwhile (its relay
        thread ran :meth:`dropped` before this registration) refuses
        the run instead of binding.  An unbind landing after the
        registration stands the bind down at its publication
        (:meth:`_publish`).  The machine's view stream (its pages and
        terminals, :class:`~kiss.server.remote_views.RemoteViews`) is
        connected meanwhile, not after the tab's stream: a chat opened
        from the machine waits for one round of handshakes, not two.
        """
        machine = registered_machine(machine_name)
        if machine is None:
            self._backend._agent_server._refuse_run(
                tab_id,
                f"{machine_name} is not a registered machine",
            )
            return None
        with self._lock:
            if self._closing:
                return None
            tab = self._tabs.get(tab_id)
            displaced: RemoteTab | None = None
            orphans: list[str] = []
            if tab is not None and tab.machine != machine:
                displaced, orphans = self._detach(tab_id)
                tab = None
            self._connecting[tab_id] = self._connecting.get(tab_id, 0) + 1
            unbinds = self._unbinds.get(tab_id, 0)
        if displaced is not None:
            self._end_stream(displaced, orphans)
        warming = self._views.warm(machine.name)
        try:
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
                    with self._lock:
                        if (not self._closing and self._tabs.get(tab_id) is None
                                and self._unbinds.get(tab_id, 0) == unbinds):
                            self._backend._agent_server._refuse_run(tab_id, str(exc))
                    return None
                displaced = None
                orphans = []
                same: RemoteTab | None = None  # a bind to the same machine that won
                lost = False
                with self._lock:
                    stood_down = self._closing or self._unbinds.get(tab_id, 0) != unbinds
                    winner = self._tabs.get(tab_id)
                    if winner is not None and winner.machine == machine:
                        same = winner
                    elif not stood_down:
                        with tab.lock:
                            lost = tab.retired  # dropped before it could be registered
                        if lost:
                            if winner is None:
                                self._backend._agent_server._refuse_run(
                                    tab_id, f"lost the connection to {machine.name}",
                                )
                        else:
                            if winner is not None:
                                orphans = self._retire(winner)
                                displaced = winner
                            self._tabs[tab_id] = tab
                            self._forget_finished(tab_id)
                            self._views.bind(tab_id, machine.name)
                if displaced is not None:
                    self._end_stream(displaced, orphans)
                if stood_down or same is not None or lost:
                    tab.close()
                if stood_down:
                    return None
                if lost:
                    return None
                if same is not None:
                    tab = same
                announce = same is None
            # Registered before the binding is announced: a local client's
            # delivery is narrowed to the tabs it can show.  A sub-agent
            # run (``parentTaskId``) gets no top-level tab, as one running
            # here does not: the surfaces open its nested tab under the
            # caller on the relayed ``new_tab``.
            if not cmd.get("parentTaskId") and not self._publish(tab_id, tab, cmd, title):
                return None
            if announce:
                self._announce(tab_id, tab)
        finally:
            # A stream warmed for a bind that was refused or stood down
            # has no binding to keep it: closed when idle.
            warming.join()
            self._views.release(machine.name)
            with self._lock:
                self._connecting[tab_id] -= 1
                if not self._connecting[tab_id]:
                    del self._connecting[tab_id]
                    self._unbinds.pop(tab_id, None)
        # The run is forwarded only once the machine's pages are watched
        # (a stream that could not be opened for an earlier bind is
        # tried again): the page its agent opens for the user must not
        # slip by before the stream is subscribed.
        self._views.open_for(machine.name)
        return tab

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
            tab.announced.clear()
            tab.children.clear()
            tab.finished.clear()
        unmapped = [tid for tid in tab_ids if self._tabs.get(tid) is tab]
        for tid in unmapped:
            del self._tabs[tid]
        if tab.tab_id in unmapped:
            self._views.bind(tab.tab_id, "")
        orphans = unmapped + [tid for tid in announced if tid not in self._tabs]
        registry = self._backend._agent_server.tab_registry
        tab.orphan_generations.update((tid, registry.generation(tid)) for tid in orphans)
        return orphans

    def _forget_finished(self, tab_id: str) -> None:
        """Drop the ``subagentDone`` kept for *tab_id* on every stream, under the lock.

        The tab shows something else from here on (a chat resumed into
        it, a new chat, a bind): the end of the sub-agent it showed
        (:attr:`RemoteTab.finished`) must not close it on a surface
        connecting later.
        """
        for stream in {id(t): t for t in self._tabs.values()}.values():
            with stream.lock:
                stream.finished.pop(tab_id, None)

    def _detach(self, tab_id: str, closed: bool = False) -> tuple[RemoteTab | None, list[str]]:
        """Take *tab_id* off its stream, under the lock.

        Counted even for a tab without a stream, so a bind still
        connecting for it stands down.  A bound tab's stream is retired
        and returned with the sub-agent tabs it leaves behind, for
        :meth:`_end_stream` outside the lock; a sub-agent tab just
        leaves its stream (``(None, [])``, as for a tab without one),
        with the task it was derived from: the stream must not claim
        the tab again (:meth:`RemoteTab._owns_sub_tab`), or the
        sub-agent's later ``subagentDone`` would close it on every
        surface while it shows something else (a local chat).  With
        *closed* (a ``closeTab``: the tab is gone from every surface
        online, not shown again) the ``subagentDone`` kept for a
        sub-agent tab stays, to end it on the surfaces that were away
        (:attr:`RemoteTab.finished`); otherwise the tab shows something
        else from here on and the end goes, whether or not the tab is
        still the stream's: a closed tab left the stream (and a viewer
        tab's id need not name it) with its end kept, so the local
        chat resumed into the id now is what the end would close.
        """
        if tab_id in self._connecting:
            self._unbinds[tab_id] = self._unbinds.get(tab_id, 0) + 1
        tab = self._tabs.get(tab_id)
        for stream in {id(t): t for t in self._tabs.values()}.values():
            with stream.lock:
                if not closed:
                    stream.finished.pop(tab_id, None)
                if tab_id in stream.tab_ids or tab_id.startswith(f"{stream.tab_id}__sub_"):
                    stream.revoked.add(tab_id)
                    stream.children.pop(tab_id, None)
                    stream.last_status.pop(tab_id, None)
                    stream.task_ids.discard(_sub_task_id(tab_id))
                    stream.announced.pop(_sub_task_id(tab_id), None)
                    if tab_id != stream.tab_id:
                        stream.tab_ids.discard(tab_id)
        if tab is None:
            return None, []
        if tab.tab_id == tab_id:
            return tab, self._retire(tab)
        del self._tabs[tab_id]
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

        Child completion is qualified by the registry generation captured
        at retirement: a local chat published during the close handshake
        cannot receive a stale ``subagentDone``, even without a remote mapping.
        """
        tab.close()
        registry = self._backend._agent_server.tab_registry
        with self._lock, registry._lock:
            for tid in dict.fromkeys(tab_ids):
                if (tid != tab.tab_id and tid not in self._tabs
                        and registry.generation(tid) == tab.orphan_generations.get(tid)):
                    done = {"type": "subagentDone", "tab_id": tid, "tabId": ""}
                    self.deliver("", json.dumps(done))
        self._announce(tab.tab_id, None)
        self._views.release(tab.machine.name)

    def _unbind(self, tab_id: str, forward_close: bool = False) -> None:
        """Take *tab_id* off its stream: a bound tab's stream ends, a sub-agent tab leaves it.

        With *forward_close* the remote daemon closes the tab too.
        """
        with self._lock:
            tab = self._tabs.get(tab_id)
            ended, orphans = self._detach(tab_id, closed=forward_close)
        if forward_close and tab is not None:
            try:
                tab.send({"type": "closeTab", "tabId": tab_id})
            except MachineError:
                pass
        if ended is not None:
            self._end_stream(ended, orphans)

    def _displace_locked(self, tab: RemoteTab) -> list[tuple[RemoteTab, list[str], int]]:
        """Take the other tabs showing *tab*'s machine's chat off their streams (under the lock).

        One tab per chat, as the registry keeps for a local chat
        (:meth:`~kiss.server.tab_registry.TabRegistry.update_tab`
        displaces the tab bound to the chat before): a machine's chat
        resumed into a second tab (another surface's CHATS-panel click,
        landing before the first tab's replay told that surface the
        chat is on screen) displaces the tab that showed it, as the
        machine's daemon did on its side when it replayed the chat
        into the new tab.  A stream carrying a sub-agent run of a local
        task has no chat tab: it neither displaces (the machine
        publishes no tab for the run either) nor is displaced (the
        run's relay would end), and a stream that lost its tab
        meanwhile displaces nothing.

        Returns:
            The displaced streams, each with the tabs it leaves behind
            (what :meth:`_detach` returned) and the registry generation
            its tab carries (the publication the stream owned: made
            under this lock by :meth:`_publish`, re-stamped by anything
            published for the tab since), for :meth:`_close_displaced`
            outside the lock.
        """
        with tab.lock:
            chat_id = tab.chat_id
        if tab.subagent or not chat_id or self._tabs.get(tab.tab_id) is not tab:
            return []
        registry = self._backend._agent_server.tab_registry
        displaced: list[tuple[RemoteTab, list[str], int]] = []
        for other in list({id(t): t for t in self._tabs.values()}.values()):
            if other is tab or other.subagent or other.machine.name != tab.machine.name:
                continue
            with other.lock:
                same_chat = other.chat_id == chat_id
            if same_chat:
                ended, orphans = self._detach(other.tab_id)
                if ended is not None:
                    displaced.append((ended, orphans, registry.generation(other.tab_id)))
        return displaced

    def _close_displaced(self, displaced: list[tuple[RemoteTab, list[str], int]]) -> None:
        """Finish off the streams :meth:`_displace_locked` retired and close their tabs.

        Each stream ends as an unbound one does (:meth:`_end_stream`:
        its sub-agent tabs end, its binding is announced gone) and its
        tab leaves the registry as a ``closeTab`` of it would (every
        surface closes it) -- unless something republished the tab
        since it was displaced: the stream's end can wait on its
        socket's close handshake, and a chat another surface resumed
        into the tab meanwhile (a local chat: the tab is off its stream
        by then) owns it now, so the close stands down
        (:meth:`~kiss.server.server.AgentServer._close_tab_if_generation`).
        """
        for other, orphans, generation in displaced:
            self._end_stream(other, orphans)
            self._backend._agent_server._close_tab_if_generation(other.tab_id, generation)

    def _publish(self, tab_id: str, tab: RemoteTab, cmd: dict[str, Any], title: str) -> bool:
        """Publish *tab_id*, bound to *tab*'s stream, in the shared tab registry.

        Every surface shows the tab (the registry entry a local run
        publishes) under *title*, with no chat or task: they are the
        machine's.  The local chat the tab showed before goes, with the
        chat view, the task pin and the printer subscriptions that went
        with it: opening that chat in another tab does not displace this
        one, and neither a run of it there nor a local task the tab was
        streaming (still running in another tab) is fanned out to it.
        Published in the critical section that checks the tab is still
        on *tab*'s stream, so an unbind from another connection
        (:meth:`_detach`: a ``closeTab``, ``newChat``, or local-chat
        resume) lands wholly before it (nothing is published: a closed
        tab stays closed, a resumed chat stays shown and subscribed) or
        wholly after it.  A submit is stamped with an id
        (``promptId``) the machine echoes on ``promptOpened``: a
        path-only prompt is no task, and the tab takes back the title
        it displaced if nothing republished it meanwhile.

        Returns:
            False when the tab left *tab*'s stream before the publication.
        """
        server = self._backend._agent_server
        old_title = self._registry_title(tab_id)
        with self._lock:
            if self._tabs.get(tab_id) is not tab:
                return False
            # No chat id: nothing is displaced (one tab per local chat).
            changed, _, generation = server.tab_registry.update_tab(
                tab_id,
                chat_id="",
                title=title,
                work_dir=str(cmd.get("workDir", "") or ""),
                task_id="",
                create=True,
            )
            with server._state_lock:
                server._tab_chat_views.pop(tab_id, None)
                server._tab_opened_task_ids.pop(tab_id, None)
                server._printer_cleanup_tab(tab_id)
        if changed:
            server._publish_tabs_state()
        if cmd.get("type") == "submit":
            prompt_id = uuid.uuid4().hex
            cmd["promptId"] = prompt_id
            with tab.lock:
                tab.displaced_titles[tab_id] = (prompt_id, old_title, generation)
        return True

    def _registry_title(self, tab_id: str) -> str:
        """The title the shared tab registry holds for *tab_id* ("" when it has no entry)."""
        registry = self._backend._agent_server.tab_registry
        return next(
            (str(e.get("title", "")) for e in registry.snapshot() if e.get("tabId") == tab_id),
            "",
        )

    def retitle(self, tab_id: str, title: str, generation: int) -> None:
        """Put *title* back as *tab_id*'s registry title on every surface.

        Called by a tab's stream when the machine answered its
        forwarded ``submit`` with ``promptOpened``: the prompt was the
        path of a file, no task started, and the title the bind gave
        the tab (the prompt) goes back to what it was -- unless a later
        publication (another submit, a resume) stamped the tab since,
        whose title stands
        (:meth:`~kiss.server.tab_registry.TabRegistry.retitle_if_generation`),
        or the tab left its stream meanwhile (unbound, or displaced by
        another tab of its chat): the restore is a publication of the
        stream's own, and one landing after the displacement captured
        the tab's generation would keep the tab from the close that
        follows (:meth:`_close_displaced`).  Decided under the lock
        the displacement takes.

        Args:
            tab_id: The bound tab.
            title: The title the prompt displaced ("" restores nothing:
                the tab had no registry entry before).
            generation: The token of the bind's publication.
        """
        server = self._backend._agent_server
        with self._lock:
            if tab_id not in self._tabs:
                return
            restored = bool(title) and server.tab_registry.retitle_if_generation(
                tab_id, title, generation,
            )
        if restored:
            server._publish_tabs_state()

    def adopt(self, tab: RemoteTab, sub_tab_id: str) -> bool:
        """Map the sub-agent tab *sub_tab_id* a replay announced to *tab*'s stream.

        The tab's binding is announced as a bound tab's is: the
        surfaces show the machine for the sub-agent tab too. Return False
        when the child was detached or a different stream owns it. Membership
        and mapping change together, before any announcement is delivered.
        """
        with self._lock, tab.lock:
            if (self._tabs.get(tab.tab_id) is not tab or tab.retired
                    or sub_tab_id in tab.revoked
                    or self._tabs.get(sub_tab_id) not in (None, tab)):
                return False
            if self._tabs.get(sub_tab_id) is not tab:
                self._tabs[sub_tab_id] = tab
                tab.tab_ids.add(sub_tab_id)
                tab.announced.pop(_sub_task_id(sub_tab_id), None)
                self.deliver("", _tab_machine_frame(sub_tab_id, tab.machine.name))
            return True

    def displace(self, tab: RemoteTab) -> None:
        """Displace the other tabs showing the chat *tab*'s stream just learnt it carries.

        A tab bound by a submit learns its chat from the run's first
        frame: the machine ran the task in the chat the tab already
        had there, which another tab here may show meanwhile (a resume
        of the chat).  The run's tab keeps the chat, as a local run's
        does when its task starts (:meth:`_displace_locked`).
        """
        with self._lock:
            displaced = self._displace_locked(tab)
        self._close_displaced(displaced)

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
        replayed chat, whose ``new_tab`` this daemon never saw).  A
        ``scoped`` resume of the chat a bound tab shows is the tab's
        own fetch of the transcript :meth:`replay_to` deferred (the
        client answering the ``replay_deferred`` offer when the user
        opens the tab): forwarded to the machine, the answer to that
        connection alone.  Any other resume into a bound tab, or one
        whose bind is still connecting, shows a local chat there: the
        tab is unbound (the bind stands down) and the resume handled
        here.  A sub-agent tab on a stream leaves it for a chat
        (``chatId``) the same way, but keeps it on a resume of a task
        the stream never saw: that is the tab's own fetch, answered
        here with nothing running, while the sub-agent runs on.
        """
        machine_name = str(cmd.get("machine", "") or "")
        if machine_name:
            tab = self._bind(tab_id, machine_name, cmd, str(cmd.get("title", "") or ""))
            if tab is None:
                return True
            with self._lock:
                with tab.lock:
                    # The tab shows this chat, at this task, from here on
                    # (a later ``ready`` replays those); taken from the
                    # click, not from the replay it gets, which can cross a
                    # later submit on the socket.
                    tab.chat_id = str(cmd.get("chatId") or cmd.get("id") or "")
                    tab.shown_task_id = str(cmd.get("taskId") or "")
                    if cmd.get("scoped") and conn_id:
                        tab.stamp_replay(cmd, conn_id)
                # One tab per chat, as for a local chat: the tab that
                # showed the chat is displaced in the critical section
                # that claims it, so two resumes of the chat racing
                # each other leave exactly one tab.
                displaced = self._displace_locked(tab)
            self._close_displaced(displaced)
            self._forward(tab, tab_id, cmd)
            return True
        ended: RemoteTab | None = None
        orphans: list[str] = []
        unjoined = False  # a sub-agent tab left its stream for a local chat
        with self._lock:
            tab = next(
                (t for t in self._tabs.values() if task_id and task_id in t.task_ids),
                None,
            )
            if tab is None and task_id:
                tab = self._tabs.get(str(cmd.get("parentTabId", "") or ""))
            previous = self._tabs.get(tab_id)
            own_fetch = False
            if tab is None and previous is not None and previous.tab_id == tab_id:
                # The bound tab's own fetch of the chat it shows (the
                # answer to a deferred replay's offer): nothing to bind
                # or unbind, the machine replays it to this connection.
                with previous.lock:
                    own_fetch = bool(
                        cmd.get("scoped") and conn_id and previous.chat_id
                        and str(cmd.get("chatId") or cmd.get("id") or "") == previous.chat_id
                    )
                    if own_fetch:
                        previous.stamp_replay(cmd, conn_id)
            if own_fetch:
                tab = previous
            elif tab is None:
                # A local chat shows in the tab: it leaves its own stream,
                # and a bind still connecting for it stands down, as on a
                # closeTab or newChat (both counted by _detach).  A
                # sub-agent tab leaves its parent's stream for a chat
                # only (shown here whatever this daemon has of it); the
                # stream's frames for the tab would land in that chat.
                if previous is None or previous.tab_id == tab_id or cmd.get("chatId"):
                    ended, orphans = self._detach(tab_id)
                    unjoined = previous is not None and previous.tab_id != tab_id
            else:
                # One stream per tab: a tab bound to a machine, or on
                # another run's stream, leaves it (its own stream ends)
                # before it joins the task's stream.
                if previous is not None and previous is not tab:
                    ended, orphans = self._detach(tab_id)
                self._tabs[tab_id] = tab
                self._forget_finished(tab_id)
                with tab.lock:
                    tab.revoked.discard(tab_id)
                    tab.tab_ids.add(tab_id)
                    tab.task_ids.add(task_id)
                    tab.announced.pop(task_id, None)
                    if conn_id:
                        tab.stamp_replay(cmd, conn_id)
                    # The tab is the stream's machine's from here on (a
                    # sub-agent tab joining its parent's stream, or a
                    # root tab moved to another): told before its
                    # transcript, as a bound tab's binding precedes its
                    # frames, so the chat header's drop-down shows the
                    # machine for the sub-agent tab as for its parent.
                    # Delivered under the stream's lock that just made
                    # the tab one of its own: a frame of the tab already
                    # on the stream (a history click's replay of a
                    # finished sub-agent's transcript, which the machine
                    # sends to every window after the parent's) is
                    # relayed under that lock too, so it follows the
                    # binding instead of landing in the gap before an
                    # announcement made after the locks were released.
                    # To every client, as :meth:`_announce` sends it.
                    self.deliver("", _tab_machine_frame(tab_id, tab.machine.name))
        if own_fetch and tab is not None:
            self._forward(tab, tab_id, cmd)
            return True
        if ended is not None:
            self._end_stream(ended, orphans)
        if tab is None:
            if unjoined:
                # This machine's from here on: told before the local
                # chat's transcript, as a bound tab's unbinding is.
                self._announce(tab_id, None)
            return False
        self._forward(tab, tab_id, cmd)
        return True

    def replay_to(self, conn_id: str, shown_tab_id: str = "") -> None:
        """Replay the bound tab *conn_id* shows to it; offer every other bound tab.

        Called after a client's ``ready`` was handled, so the client has
        the tabs (the ``tabs_state`` snapshot) before their transcripts
        arrive.  A stream carrying a sub-agent run of a local task has no tab of its own
        to replay: the client gets the run's ``new_tab`` again (if it
        is still running) and opens and resumes the nested tab itself.
        The sub-agent tabs on a stream (:attr:`RemoteTab.tab_ids`) are
        bound to its machine too.  Every binding (a bound tab's, then
        its sub-agent tabs': the order they were mapped in) is told
        before any transcript is asked for: a replayed transcript
        (and the ``openSubagentTab`` it carries for a sub-agent tab the
        stream already knows, which announces nothing) never reaches
        the client ahead of the binding its file links are checked on.

        Only the tab the client is looking at (*shown_tab_id*, the
        ``ready``'s ``tabId``) gets its transcript, as a local tab does
        (``web_server._handle_ready``).  Every other bound tab is
        deferred the same way: the client gets the tab's running state
        (the last ``status`` the stream relayed for it,
        :attr:`RemoteTab.last_status`) and a ``replay_deferred`` offer,
        and asks for the transcript with a ``scoped`` ``resumeSession``
        when the user opens the tab (:meth:`_resume` forwards it to the
        machine).  The sub-agents that finished on the stream end again
        (:attr:`RemoteTab.finished`), shown tab or not, their tabs
        closed by a page online since or not: a page that was away when
        the child finished still has its tab, and the parent's
        transcript, when the page asks for it, announces the child done
        but closes nothing.
        Replaying every bound tab used to send each
        connecting page every machine's transcripts, sub-agents
        included (megabytes, ahead of the page's CHATS rows on a slow
        link), for tabs the user may never open.

        Args:
            conn_id: The connection that sent the ``ready``.
            shown_tab_id: The tab it shows (``""`` for none).
        """
        with self._lock:
            mapped = list(self._tabs.items())
        tabs = {t.tab_id: t for _, t in mapped}
        for tab_id, tab in mapped:
            # A stream carrying a local task's sub-agent run has no
            # tab of its own, only the sub-agent tabs that joined it.
            if tab_id != tab.tab_id or not tab.subagent:
                self._replay_binding(conn_id, tab_id, tab)
        for tab_id, tab in tabs.items():
            with tab.lock:
                if tab.retired:
                    continue
                if tab.subagent and tab.spawn:
                    self.deliver_to(conn_id, tab.spawn)
                if tab_id != shown_tab_id:
                    for child, announcement in tab.children.items():
                        self.deliver_to(conn_id, announcement)
                        status = tab.last_status.get(child)
                        if status:
                            self.deliver_to(conn_id, status)
                for done in tab.finished.values():
                    self.deliver_to(conn_id, done)
            if tab.subagent:
                continue
            resume: dict[str, Any] = {"type": "resumeSession", "tabId": tab_id}
            with tab.lock:
                if not tab.chat_id:
                    continue
                if tab_id != shown_tab_id:
                    if tab.retired:
                        continue
                    status = tab.last_status.get(tab_id)
                    if status:
                        self.deliver_to(conn_id, status)
                    self.deliver_to(
                        conn_id,
                        _replay_deferred_frame(tab_id, tab.chat_id, tab.shown_task_id),
                    )
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

        A binding is a registry event, as ``new_tab`` and
        ``subagentDone`` are: every client gets it, whatever tabs it
        addressed so far (a surface shows the machine of every tab,
        and a peer that addresses the tab later must already hold the
        binding when the tab's frames start reaching it).
        """
        with self._lock:
            if self._tabs.get(tab_id) is not tab:
                return
            machine_name = tab.machine.name if tab is not None else ""
            self.deliver("", _tab_machine_frame(tab_id, machine_name))

    def deliver(self, tab_id: str, text: str) -> None:
        """Send one frame of the remote daemon, stamped with *tab_id*, to every client."""
        self._backend._printer._send_to_ws_clients(text, tab_id)

    def deliver_to(self, conn_id: str, text: str) -> None:
        """Send one frame of the remote daemon to the connection *conn_id* only."""
        self._backend._printer._send_to_conn(conn_id, text)

    def dropped(self, tab: RemoteTab, lost: bool) -> None:
        """A stream ended: forget its tabs, and when it was *lost* end the tab's task.

        The tab's run is refused (the surfaces show the loss; the
        terminal status names the machine as ``lost``, as the task may
        still be running there) and the stream finished off as an
        unbound one is (:meth:`_end_stream`: its sub-agent tabs end
        too).  A stream already retired (replaced, unbound) is nobody's
        loss.
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
                    lost=tab.machine.name,
                )
        if lost and registered:
            self._end_stream(tab, tab_ids)

    def close_all(self) -> None:
        """Close every stream (daemon shutdown); a bind still connecting stands down."""
        with self._lock:
            self._closing = True
            tabs = list({id(t): t for t in self._tabs.values()}.values())
            for tab in tabs:
                self._retire(tab)
        for tab in tabs:
            tab.close()


def _tab_machine_frame(tab_id: str, machine_name: str) -> str:
    """The ``tabMachine`` frame announcing *tab_id*'s binding."""
    return json.dumps({"type": "tabMachine", "tabId": tab_id, "machine": machine_name})


def _without_replay_id(msg: dict[str, Any]) -> dict[str, Any]:
    """*msg* without the ``replayId`` stamp of the replay it came in."""
    return {key: value for key, value in msg.items() if key != "replayId"}


def _replay_deferred_frame(tab_id: str, chat_id: str, task_id: str) -> str:
    """The ``replay_deferred`` offer for *tab_id*, as the daemon makes it for a local tab.

    The client answers it with a ``scoped`` ``resumeSession`` of the
    chat (and task) when the user opens the tab
    (:meth:`RemoteTabs._resume` forwards that one to the machine).
    """
    return json.dumps({
        "type": "replay_deferred", "tabId": tab_id, "chatId": chat_id, "taskId": task_id,
    })


def _sub_tab_id(new_tab: dict[str, Any]) -> str:
    """The tab the surfaces open for the sub-agent a ``new_tab`` frame announces.

    ``{parent_tab_id}__sub_{task_id}``, as the webapp derives it (and
    as the daemon names a sub-agent's tab when it replays its chat).
    """
    return f"{new_tab.get('parent_tab_id') or 'task'}__sub_{new_tab.get('task_id') or ''}"


def _sub_task_id(tab_id: str) -> str:
    """The task id a ``{parent}__sub_{task_id}`` tab id names ("" for any other id)."""
    return tab_id.rpartition("__sub_")[2] if "__sub_" in tab_id else ""
