# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
# test_explorer_scm_commands and re-exported for pytest's collector)
"""A remote webapp opened mid-task shows the running chat's spinner at once.

Scenario: a task is started from VS Code (a local-endpoint client, the
transport the extension uses) and is still running when a user opens the
remote webapp in a browser.  The new client knows nothing about the task
yet; its ``ready`` must make the daemon replay the running state so the
chat's tab appears WITH the header spinner immediately, with no click,
and the spinner must go away on that same client when the task ends.

Everything runs for real: the production ``RemoteAccessServer`` over
https/wss (:class:`ExplorerHarness`), the daemon's ``run`` pipeline,
tab registry and replay, and the real ``chat.html`` / ``main.js`` in a
headless Chromium.  Only the executor LLM loop (``KISSAgent.run``) is
replaced, by a function that blocks until the test releases it, so the
task stays running exactly as long as the test needs.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from playwright.sync_api import Browser, BrowserContext, sync_playwright
from websockets.asyncio.client import connect

from kiss.agents.sorcar import local_endpoint
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.server.test_explorer_scm_commands import (
    ExplorerHarness,
    _no_verify_ssl,
    harness,  # noqa: F401  (module fixture)
)

PROMPT = "long running task for the remote-connect spinner check"


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


class BlockingExecutor:
    """Replaces the executor LLM loop with one that waits to be released."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._original = KISSAgent.run

    def install(self) -> None:
        """Swap ``KISSAgent.run`` for the blocking stub."""
        executor = self

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""  # follow-up proposer and other one-shot sessions
            arguments = dict(kwargs.get("arguments") or {})
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            if "task_description" not in arguments:
                return "result: prior progress\n"
            executor.entered.set()
            executor.release.wait(timeout=120)
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            kwargs["printer"].print(
                raw, type="result", step_count=1, total_tokens=1, cost="$0.0001",
            )
            return raw

        cast(Any, KISSAgent).run = stub_run

    def restore(self) -> None:
        """Release the task if still blocked and put the real loop back."""
        self.release.set()
        cast(Any, KISSAgent).run = self._original


class VsCodeClient:
    """The VS Code side: a local-endpoint connection that starts the task."""

    def __init__(self, harness: ExplorerHarness) -> None:
        self.tab_id = f"vscode-{uuid.uuid4().hex[:8]}"
        self.ws = local_endpoint.connect(
            Path(harness.tmpdir) / "sorcar-local.json", open_timeout=60,
        )
        self.work_dir = str(harness.work_dir)

    def start_task(self) -> dict[str, Any]:
        """Send ``ready`` + ``run`` and return the ``status running:true`` event."""
        self.ws.send(json.dumps({"type": "ready", "tabId": self.tab_id}))
        self.ws.send(json.dumps({
            "type": "run",
            "prompt": PROMPT,
            "tabId": self.tab_id,
            "taskId": uuid.uuid4().hex,
            "workDir": self.work_dir,
            "model": "",
            "useWorktree": False,
            "useWebTools": False,
            "autoCommit": False,
        }))
        return self.wait_for_status(running=True)

    def wait_for_status(self, *, running: bool, timeout: float = 60) -> dict[str, Any]:
        """Read events until this tab's ``status`` with the given flag."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.1, deadline - time.monotonic())
            event: dict[str, Any] = json.loads(self.ws.recv(timeout=remaining))
            if (
                event.get("type") == "status"
                and event.get("tabId") == self.tab_id
                and bool(event.get("running")) is running
            ):
                return event

    def close(self) -> None:
        """Close the local connection."""
        self.ws.close()


async def _remote_ready_replies(harness: ExplorerHarness) -> list[dict[str, Any]]:
    """Connect a fresh remote client, send ``ready``, drain the replies."""
    async with connect(harness.ws_url, ssl=_no_verify_ssl()) as ws:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        while json.loads(await asyncio.wait_for(ws.recv(), 30)).get("type") != "auth_ok":
            pass
        await ws.send(json.dumps({"type": "ready", "tabId": "remote-fresh-tab"}))
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(json.loads(await asyncio.wait_for(ws.recv(), 2)))
            except TimeoutError:
                return events


def _assert_ready_replay(
    events: list[dict[str, Any]], tab: str, start_ts: int,
) -> None:
    """Check the ``ready`` replies a mid-run remote client receives.

    The tab snapshot must precede the tab's ``status running:true``;
    every start stamp the client is given (status, replayed row
    ``extra``, ``task_settings`` event) must be the one the originating
    client's status carried, and no ``running:false`` may follow.
    """
    # Failure messages list only event types: the drained replies
    # include ``configData``, which carries the user's API keys.
    types = [e.get("type") for e in events]
    snapshot_index = next(
        (
            i for i, e in enumerate(events)
            if e.get("type") == "tabs_state"
            and any(t.get("tabId") == tab for t in e.get("tabs", []))
        ),
        None,
    )
    assert snapshot_index is not None, f"no tabs_state with {tab}; got {types}"
    statuses = [
        (i, e) for i, e in enumerate(events)
        if e.get("type") == "status" and e.get("tabId") == tab
    ]
    assert statuses, f"no status event for {tab}; got {types}"
    first_index, first_status = statuses[0]
    assert first_status["running"] is True
    assert first_status["startTs"] == start_ts > 0
    assert snapshot_index < first_index, (
        "the tab must exist (tabs_state) before its running status arrives"
    )
    assert all(e["running"] is True for _, e in statuses), (
        f"a still-running task must never be reported stopped: {statuses}"
    )
    replay = next(
        e for e in events
        if e.get("type") == "task_events" and e.get("tabId") == tab
    )
    assert replay["task"] == PROMPT
    # ``main.js`` rebases its timer on these two after the status: a
    # provisional row stamp here would leave the remote timer behind.
    assert json.loads(replay["extra"])["startTs"] == start_ts
    settings = next(
        e for e in replay["events"] if e.get("type") == "task_settings"
    )
    assert settings["settings"]["start_ts"] == start_ts


def test_remote_client_opened_mid_run_gets_status_replay_and_spinner(
    browser: Browser, harness: ExplorerHarness,
) -> None:
    """Wire replay and rendered spinner for a client that connects mid-task."""
    executor = BlockingExecutor()
    vscode: VsCodeClient | None = None
    context: BrowserContext | None = None
    try:
        executor.install()
        vscode = VsCodeClient(harness)
        started = vscode.start_task()
        assert executor.entered.wait(timeout=60), "the task's executor never ran"
        tab = vscode.tab_id
        start_ts = int(started["startTs"])

        # 1. Wire level: what a fresh remote client's ``ready`` gets.
        _assert_ready_replay(harness.run(_remote_ready_replies(harness)), tab, start_ts)

        # 2. Rendered: the real remote page (fresh context, no saved
        #    tabs) shows the chat tab with the header spinner right after
        #    boot, without any interaction, and its timer counts from the
        #    task's real start.
        context = browser.new_context(
            ignore_https_errors=True, viewport={"width": 1280, "height": 900},
            service_workers="block",
        )
        page = context.new_page()
        page.goto(harness.base_url + "/")
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        # The chat's header is its entry on the main tab row; the group
        # strip under it (hidden for a chat without sub-agent or file
        # tabs) repeats the active chat, so the probe names the row.
        tab_selector = f'#main-tab-list .chat-tab[data-tab-id="{tab}"]'
        page.wait_for_selector(f"{tab_selector} .chat-tab-spinner", timeout=15000)
        assert page.locator(tab_selector).count() == 1
        assert page.locator(f"{tab_selector} .chat-tab-spinner").is_visible()
        assert page.locator(f"{tab_selector} .chat-tab-status").count() == 0
        # Tab titles are shortened with an ellipsis; the visible part is
        # the prompt's beginning.
        label = page.locator(f"{tab_selector} .chat-tab-label").inner_text()
        assert PROMPT.startswith(label.rstrip("\u2026")), label
        # The label floors the elapsed seconds and refreshes once a
        # second, so it is read together with the browser clock and
        # retried: within one tick it must equal floor(now - startTs)
        # or the previous second (a label about to refresh).  A timer
        # rebased on a stamp a second or more late never satisfies
        # this; the exact wire assertions above cover smaller drifts.
        page.wait_for_function(
            "(startTs) => {"
            "  const m = /^Running (\\d+)s$/.exec("
            "    document.getElementById('status-text').textContent);"
            "  if (!m) return false;"
            "  const want = Math.floor((Date.now() - startTs) / 1000);"
            "  return want - Number(m[1]) <= 1 && Number(m[1]) <= want;"
            "}",
            arg=start_ts,
            timeout=15000,
        )

        # 3. The task ends: the already-connected remote client drops the
        #    spinner and shows the tick; VS Code sees the task stop too.
        executor.release.set()
        vscode.wait_for_status(running=False)
        page.wait_for_selector(f"{tab_selector} .chat-tab-ok", timeout=15000)
        assert page.locator(f"{tab_selector} .chat-tab-spinner").count() == 0
    finally:
        executor.restore()
        if vscode is not None:
            vscode.close()
        if context is not None:
            context.close()
