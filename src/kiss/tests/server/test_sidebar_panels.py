# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the right sidebar's Schedule, Apps and Spend data.

* :func:`kiss.server.sidebar_panels.cron_jobs_report` reads the real
  cron store under the session ``$KISS_HOME``.
* :func:`kiss.server.sidebar_panels.spend_report` sums the task
  history's ``cost`` / ``tokens`` columns per local day and model from
  a real SQLite ``task_history`` (in a temporary ``.kiss`` dir).
* :mod:`kiss.agents.third_party_agents.auth_status` probes the real
  channel agents: a Brave Search key written to the store turns that
  app "connected", every other app stays "not connected".
* :func:`kiss.server.sidebar_panels.apps_status` runs the probe
  subprocess and caches its answer.
* ``getCronJobs`` / ``getAppsStatus`` / ``getSpendReport`` are
  answered over the UDS transport of a real :class:`RemoteAccessServer`.

Not covered in-process: the probe-failure branch of ``_probe_apps``
(the subprocess crashing, timing out or printing no JSON) cannot be
forced without replacing the interpreter or the probe module;
``auth_status.main`` only runs as the probe subprocess (it ends in
``os._exit``) and is exercised through ``apps_status``; and the
Muse-auth-on branch of ``vault_services`` needs the credential daemon,
which the suite switches off (``KISS_MUSE_AUTH=0`` in conftest).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.agent_dispatch import available_channels
from kiss.agents.third_party_agents import auth_status
from kiss.core.config import kiss_home
from kiss.server import sidebar_panels
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.test_history_date_range import (
    _redirect,
    _restore,
    _set_timestamp,
)
from kiss.tests.conftest import requires_unix_sockets


def _brave_config() -> Path:
    return kiss_home() / "third_party_agents" / "brave_search" / "config.json"


def _expire_apps_cache() -> None:
    """Forget the apps-status probe cache so the next call probes again.

    Any earlier test in the process that opened a webapp sidebar may have
    probed within the TTL; resetting only ``_apps_checked_at`` would then
    serve that cached answer stamped ``checked_at == 0``.
    """
    sidebar_panels._apps_attempted_at = 0.0
    sidebar_panels._apps_checked_at = 0.0
    sidebar_panels._apps_cache = []


class _StoreTestCase(unittest.TestCase):
    """Back up and restore the cron store and the Brave Search config."""

    def setUp(self) -> None:
        self.saved_jobs = cron_agent.load_jobs()
        brave = _brave_config()
        self.saved_brave = brave.read_text() if brave.exists() else None
        _expire_apps_cache()

    def tearDown(self) -> None:
        cron_agent.save_jobs(self.saved_jobs)
        brave = _brave_config()
        if self.saved_brave is None:
            brave.unlink(missing_ok=True)
        else:
            brave.write_text(self.saved_brave)
        _expire_apps_cache()

    def write_brave_key(self) -> None:
        brave = _brave_config()
        brave.parent.mkdir(parents=True, exist_ok=True)
        brave.write_text(json.dumps({"api_key": "test-key"}))


class TestCronJobsReport(_StoreTestCase):
    def test_rows_are_ordered_and_complete(self) -> None:
        now = time.time()
        cron_agent.save_jobs([
            {"id": "late", "name": "Late digest", "schedule": "0 9 * * *",
             "prompt": "summarize email", "enabled": True,
             "next_run_at": now + 7200, "last_run_at": now - 600,
             "last_status": "ok", "work_dir": "/tmp/w"},
            {"id": "paused", "schedule": "every 1h", "command": "echo hi",
             "enabled": False, "next_run_at": now + 60},
            {"id": "soon", "name": "Soon", "schedule": "every 5m",
             "command": "date", "next_run_at": now + 300},
            {"id": "never", "name": "Never", "schedule": "0 0 30 2 *",
             "prompt": "impossible"},
        ])
        rows = sidebar_panels.cron_jobs_report()
        self.assertEqual([r["id"] for r in rows], ["soon", "late", "never", "paused"])
        soon, late, never, paused = rows
        self.assertEqual(soon["kind"], "command")
        self.assertEqual(soon["what"], "date")
        self.assertTrue(soon["enabled"])
        self.assertFalse(soon["running"])
        self.assertEqual(late["kind"], "prompt")
        self.assertEqual(late["what"], "summarize email")
        self.assertEqual(late["lastStatus"], "ok")
        self.assertEqual(late["workDir"], "/tmp/w")
        # Epoch milliseconds: no time-zone ambiguity for the browser.
        self.assertEqual(late["nextRunAt"], int((now + 7200) * 1000))
        self.assertEqual(late["lastRunAt"], int((now - 600) * 1000))
        self.assertEqual(never["nextRunAt"], 0)
        self.assertEqual(never["lastRunAt"], 0)
        # A job without a name is listed under its id.
        self.assertEqual(paused["name"], "paused")
        self.assertFalse(paused["enabled"])
        self.assertNotIn("_next", paused)

    def test_empty_store(self) -> None:
        cron_agent.save_jobs([])
        self.assertEqual(sidebar_panels.cron_jobs_report(), [])


def _noon(days_ago: int) -> tuple[str, float]:
    """The local calendar day *days_ago* days back: its key and its noon."""
    day = datetime.date.today() - datetime.timedelta(days=days_ago)
    noon = datetime.datetime.combine(day, datetime.time(12)).timestamp()
    return day.isoformat(), noon


def _spent(
    model: str, cost: float, tokens: int, ts: float, parent: str = ""
) -> str:
    """Add a finished task row with *model*, *cost* and *tokens* at *ts*."""
    extra: dict[str, object] = {"model": model, "cost": cost, "tokens": tokens}
    if parent:
        extra["parent_task_id"] = parent
    task_id, _ = th._add_task(f"task {model} {cost}", "", extra)
    _set_timestamp(task_id, ts)
    return task_id


class _SpendDbTestCase(unittest.TestCase):
    """Point the persistence DB at an empty temporary ``.kiss`` dir."""

    def setUp(self) -> None:
        self.db_dir = tempfile.mkdtemp()
        self.saved_db = _redirect(self.db_dir)

    def tearDown(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved_db)
        shutil.rmtree(self.db_dir, ignore_errors=True)


class TestSpendReport(_SpendDbTestCase):
    """``spend_report`` sums the history per day and model."""

    def test_empty_history(self) -> None:
        self.assertEqual(th._spend_by_day_and_model(), [])
        self.assertEqual(sidebar_panels.spend_report(), {
            "total": {"cost": 0.0, "tokens": 0, "tasks": 0},
            "days": [],
            "totalByModel": [],
            "daysByModel": {},
        })

    def test_sums_per_local_day_and_model_without_subagent_rows(self) -> None:
        old_key, old_noon = _noon(40)
        new_key, new_noon = _noon(1)
        parent = _spent("claude-fable-5-1", 1.5, 1000, old_noon)
        _spent("gpt-6-astra", 0.5, 200, old_noon + 3600)
        _spent("claude-fable-5-1", 2.0, 3000, new_noon)
        # An empty model is reported as "unknown"; a sub-agent's usage
        # is already in its parent's totals and is not counted again.
        _spent("", 0.25, 10, new_noon + 60)
        _spent("gpt-6-astra", 100.0, 999999, new_noon + 120, parent=parent)

        self.assertEqual(th._spend_by_day_and_model(), [
            {"date": old_key, "model": "claude-fable-5-1", "cost": 1.5, "tokens": 1000, "tasks": 1},
            {"date": old_key, "model": "gpt-6-astra", "cost": 0.5, "tokens": 200, "tasks": 1},
            {"date": new_key, "model": "claude-fable-5-1", "cost": 2.0, "tokens": 3000, "tasks": 1},
            {"date": new_key, "model": "unknown", "cost": 0.25, "tokens": 10, "tasks": 1},
        ])
        report = sidebar_panels.spend_report()
        self.assertEqual(report["total"], {"cost": 4.25, "tokens": 4210, "tasks": 4})
        self.assertEqual(report["days"], [
            {"date": old_key, "cost": 2.0, "tokens": 1200, "tasks": 2},
            {"date": new_key, "cost": 2.25, "tokens": 3010, "tasks": 2},
        ])
        self.assertEqual(report["totalByModel"], [
            {"model": "claude-fable-5-1", "cost": 3.5, "tokens": 4000, "tasks": 2},
            {"model": "gpt-6-astra", "cost": 0.5, "tokens": 200, "tasks": 1},
            {"model": "unknown", "cost": 0.25, "tokens": 10, "tasks": 1},
        ])
        self.assertEqual(report["daysByModel"], {
            old_key: [
                {"model": "claude-fable-5-1", "cost": 1.5, "tokens": 1000, "tasks": 1},
                {"model": "gpt-6-astra", "cost": 0.5, "tokens": 200, "tasks": 1},
            ],
            new_key: [
                {"model": "claude-fable-5-1", "cost": 2.0, "tokens": 3000, "tasks": 1},
                {"model": "unknown", "cost": 0.25, "tokens": 10, "tasks": 1},
            ],
        })
        # The reply is JSON-serialisable as the daemon sends it.
        json.dumps(report)

    def test_a_day_boundary_is_the_local_one(self) -> None:
        """A task at 23:30 and one at 00:30 the next day are two days."""
        key, noon = _noon(3)
        next_key = (datetime.date.fromisoformat(key) + datetime.timedelta(days=1)).isoformat()
        _spent("m", 1.0, 1, noon + 11.5 * 3600)
        _spent("m", 1.0, 1, noon + 12.5 * 3600)
        self.assertEqual(
            [d["date"] for d in sidebar_panels.spend_report()["days"]], [key, next_key]
        )


class TestAuthStatusProbe(_StoreTestCase):
    def test_every_channel_is_reported_with_its_state(self) -> None:
        self.write_brave_key()
        statuses = auth_status.all_channel_statuses()
        by_name = {s["name"]: s for s in statuses}
        self.assertEqual(sorted(by_name), sorted(available_channels()))
        self.assertIs(by_name["brave"]["authenticated"], True)
        self.assertEqual(by_name["brave"]["label"], "Brave Search")
        self.assertEqual(by_name["github"]["label"], "GitHub")
        self.assertEqual(by_name["homeassistant"]["label"], "Home Assistant")
        for status in statuses:
            self.assertEqual(set(status), {"name", "label", "authenticated", "error"})

    def test_a_broken_channel_reports_unknown(self) -> None:
        status = auth_status.channel_status("no_such_channel")
        self.assertIsNone(status["authenticated"])
        self.assertIn("ModuleNotFoundError", status["error"])
        self.assertEqual(status["label"], "No_such_channel")

    def test_a_module_without_an_agent_class_reports_unknown(self) -> None:
        # The module lives in an extra directory of the package's import
        # path, so channel discovery (a scan of the package directory)
        # never lists it for concurrently running tests.
        import kiss.agents.third_party_agents as pkg

        with tempfile.TemporaryDirectory() as extra:
            Path(extra, "zz_empty_probe").mkdir()
            Path(extra, "zz_empty_probe", "__init__.py").write_text("")
            Path(extra, "zz_empty_probe", "zz_empty_probe_sea.py").write_text(
                '"""A channel module that defines no agent."""\n'
            )
            pkg.__path__.append(extra)
            try:
                status = auth_status.channel_status("zz_empty_probe")
            finally:
                pkg.__path__.remove(extra)
        self.assertIsNone(status["authenticated"])
        self.assertIn("defines no channel agent", status["error"])

    def test_a_vault_enrollment_counts_as_connected(self) -> None:
        # No credential files: only the vault knows these services.
        self.assertIs(auth_status.channel_status("slack")["authenticated"], False)
        self.assertIs(auth_status.channel_status("slack", {"slack"})["authenticated"], True)
        # Brave Search enrolls under its own service name.
        self.assertIs(auth_status.channel_status("brave", {"brave"})["authenticated"], False)
        self.assertIs(
            auth_status.channel_status("brave", {"brave_search"})["authenticated"], True
        )
        # Google Workspace channels connect through Composio now: a stale
        # vault enrollment left by an older version does not count.
        self.assertIs(
            auth_status.channel_status("gcal", {"google_calendar"})["authenticated"], False
        )
        by_name = {
            s["name"]: s for s in auth_status.all_channel_statuses(enrolled={"discord"})
        }
        self.assertIs(by_name["discord"]["authenticated"], True)

    def test_vault_services_is_none_with_muse_auth_off(self) -> None:
        # The test suite runs with KISS_MUSE_AUTH=0 (conftest).
        self.assertEqual(os.environ.get("KISS_MUSE_AUTH"), "0")
        self.assertIsNone(auth_status.vault_services())

    def test_probes_past_the_deadline_are_reported_as_timed_out(self) -> None:
        statuses = auth_status.all_channel_statuses(timeout=0)
        self.assertEqual(len(statuses), len(available_channels()))
        timed_out = [s for s in statuses if s["error"] == "status check timed out"]
        self.assertTrue(timed_out)
        for status in timed_out:
            self.assertIsNone(status["authenticated"])

    def test_class_derived_labels(self) -> None:
        class ExampleThingAgent:
            pass

        class Plain:
            pass

        self.assertEqual(auth_status.channel_label("x", ExampleThingAgent), "Example Thing")
        self.assertEqual(auth_status.channel_label("x", Plain), "Plain")
        self.assertEqual(auth_status.channel_label("whatsapp"), "WhatsApp")


class TestAppsStatus(_StoreTestCase):
    def test_probe_subprocess_and_cache(self) -> None:
        self.write_brave_key()
        apps, checked_at = sidebar_panels.apps_status()
        self.assertGreater(checked_at, (time.time() - 60) * 1000)
        by_name = {a["name"]: a for a in apps}
        self.assertIs(by_name["brave"]["authenticated"], True)
        self.assertIs(by_name["slack"]["authenticated"], False)
        first = sidebar_panels._apps_checked_at

        # Within the TTL the cached answer is served, even though the
        # store changed.
        _brave_config().unlink()
        apps, _ = sidebar_panels.apps_status()
        self.assertEqual(sidebar_panels._apps_checked_at, first)
        self.assertIs({a["name"]: a for a in apps}["brave"]["authenticated"], True)

        # A refresh probes again and sees the removed key.
        apps, _ = sidebar_panels.apps_status(refresh=True)
        self.assertGreater(sidebar_panels._apps_checked_at, first)
        self.assertIs({a["name"]: a for a in apps}["brave"]["authenticated"], False)


@requires_unix_sockets
class TestSidebarPanelCommandsOverUds(_StoreTestCase):
    """``getCronJobs`` / ``getAppsStatus`` get direct replies over UDS.

    The VS Code extension's webviews reach the daemon through the host's
    UDS connection (FORWARDED_COMMANDS in SorcarSidebarView.ts).
    """

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(self.server._uds_handler, path=self.sock_path),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)], timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()
        super().tearDown()

    def _ask(self, *commands: dict[str, Any]) -> list[dict[str, Any]]:
        """Send *commands* on one connection; return one reply per command."""

        async def _talk() -> list[dict[str, Any]]:
            reader, writer = await asyncio.open_unix_connection(self.sock_path)
            try:
                for command in commands:
                    writer.write(json.dumps(command).encode() + b"\n")
                await writer.drain()
                events = []
                for _ in commands:
                    line = await asyncio.wait_for(reader.readline(), timeout=60)
                    events.append(json.loads(line))
                return events
            finally:
                writer.close()
                await writer.wait_closed()

        return asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(timeout=90)

    def test_get_cron_jobs(self) -> None:
        cron_agent.save_jobs([
            {"id": "j1", "name": "Nightly", "schedule": "0 2 * * *", "prompt": "p"},
        ])
        (event,) = self._ask({"type": "getCronJobs"})
        self.assertEqual(event["type"], "cronJobs")
        self.assertEqual([j["name"] for j in event["jobs"]], ["Nightly"])

    def test_get_apps_status(self) -> None:
        self.write_brave_key()
        (event,) = self._ask({"type": "getAppsStatus", "refresh": True})
        self.assertEqual(event["type"], "appsStatus")
        self.assertGreater(event["checkedAt"], 0)
        by_name = {a["name"]: a for a in event["apps"]}
        self.assertIs(by_name["brave"]["authenticated"], True)

    def test_get_spend_report(self) -> None:
        db_dir = tempfile.mkdtemp()
        saved_db = _redirect(db_dir)
        try:
            key, noon = _noon(2)
            _spent("claude-fable-5-1", 0.75, 500, noon)
            (event,) = self._ask({"type": "getSpendReport"})
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
                th._db_conn = None
            _restore(saved_db)
            shutil.rmtree(db_dir, ignore_errors=True)
        self.assertEqual(event, {
            "type": "spendReport",
            "total": {"cost": 0.75, "tokens": 500, "tasks": 1},
            "days": [{"date": key, "cost": 0.75, "tokens": 500, "tasks": 1}],
            "totalByModel": [
                {"model": "claude-fable-5-1", "cost": 0.75, "tokens": 500, "tasks": 1}
            ],
            "daysByModel": {
                key: [{"model": "claude-fable-5-1", "cost": 0.75, "tokens": 500, "tasks": 1}]
            },
        })


    def test_an_apps_probe_does_not_hold_up_later_commands(self) -> None:
        """The probe runs in the background: a command sent right after
        getAppsStatus on the same connection is answered first."""
        events = self._ask(
            {"type": "getAppsStatus", "refresh": True}, {"type": "getCronJobs"},
        )
        self.assertEqual([e["type"] for e in events], ["cronJobs", "appsStatus"])


if __name__ == "__main__":
    unittest.main()
