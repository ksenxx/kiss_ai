# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The settings panel exposes a "Working directory" option.

Frontend behaviour (real ``populateConfigForm`` / ``collectConfigForm``
from ``main.js`` replayed through a Node harness):

* the ``#cfg-work-dir`` field is populated from ``configData``'s
  ``config.work_dir``;
* in a VS Code webview (no ``remote-chat`` body class) the field is
  read-only — the work_dir is ALWAYS the workspace folder open in that
  window — and is omitted from ``saveConfig`` so one window's save can
  never overwrite the persisted work_dir;
* in the standalone web client (``remote-chat`` body class) the field
  is editable and its value is sent back via ``saveConfig``.

Backend behaviour (real ``RemoteAccessServer`` over UDS):

* ``getConfig`` reports each connection's (= each VS Code window's)
  own work_dir when nothing is persisted;
* a remote ``saveConfig`` carrying ``config.work_dir`` persists it to
  ``config.json`` and updates the daemon-wide fallback.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.server.vscode_config as vc
from kiss.server.web_server import RemoteAccessServer

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"
_MAIN_JS = _VSCODE_DIR / "media" / "main.js"


def _extract_fn_body(src: str, header: str) -> str:
    """Return the source of a top-level ``function name(...) { ... }``
    block whose header matches *header* (braces matched by counting).
    """
    start = src.index(header)
    brace = src.index("{", start)
    depth = 0
    i = brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated function body for {header}")


def _run_config_form_harness(
    body_classes: list[str],
    populate_cfg: dict[str, Any],
    edited_work_dir: str | None,
    pinned_work_dir: str | None = None,
) -> dict[str, Any]:
    """Replay the real config-form functions in Node.

    Builds a minimal DOM stub (including ``sessionStorage`` and a
    ``vscode.postMessage`` recorder), optionally pre-pins
    *pinned_work_dir* under the ``sorcar-work-dir`` sessionStorage key
    (simulating a webapp instance that already adopted a folder), runs
    ``populateConfigForm(populate_cfg)``, optionally overwrites the
    ``#cfg-work-dir`` value (simulating the user editing the field),
    runs ``collectConfigForm()`` and ``saveSettingsIfPopulated()``, and
    returns the observed state — including every message posted to the
    backend — as a dict.
    """
    src = _MAIN_JS.read_text()
    populate = _extract_fn_body(src, "function populateConfigForm(")
    collect = _extract_fn_body(src, "function collectConfigForm()")
    save = _extract_fn_body(src, "function saveSettingsIfPopulated()")
    script = f"""
'use strict';
const elements = {{}};
function getEl(id) {{
  if (!elements[id]) {{
    elements[id] = {{
      value: '', checked: false, readOnly: false, title: '',
      type: 'text', style: {{display: ''}},
    }};
  }}
  return elements[id];
}}
const bodyClasses = {json.dumps(body_classes)};
const document = {{
  getElementById: getEl,
  body: {{classList: {{contains: c => bodyClasses.indexOf(c) >= 0}}}},
}};
const _ss = {{}};
const pinned = {json.dumps(pinned_work_dir)};
if (pinned !== null) _ss['sorcar-work-dir'] = pinned;
const sessionStorage = {{
  getItem: k => (k in _ss ? _ss[k] : null),
  setItem: (k, v) => {{ _ss[k] = String(v); }},
}};
const posted = [];
const vscode = {{postMessage: m => posted.push(m)}};
let demoMode = false;
let configFormPopulated = false;
// Module-scope declaration (main.js line 165) that the extracted
// populateConfigForm/getCurrentWorkDir bodies assign to and read.
let configWorkDir = '';
{populate}
{collect}
{save}
populateConfigForm({json.dumps(populate_cfg)}, {{}});
const afterPopulate = {{
  value: getEl('cfg-work-dir').value,
  readOnly: getEl('cfg-work-dir').readOnly,
}};
const edited = {json.dumps(edited_work_dir)};
if (edited !== null) getEl('cfg-work-dir').value = edited;
const collected = collectConfigForm();
saveSettingsIfPopulated();
console.log(JSON.stringify({{
  afterPopulate,
  hasWorkDir: 'work_dir' in collected.config,
  collectedWorkDir: collected.config.work_dir,
  posted,
}}));
"""
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"node error: {result.stderr}\nstdout: {result.stdout}"
    )
    out = json.loads(result.stdout.strip())
    assert isinstance(out, dict)
    return out


class TestSettingsPanelWorkDirField(unittest.TestCase):
    """Behaviour of the ``#cfg-work-dir`` settings field."""

    def test_remote_client_field_is_editable_and_saved(self) -> None:
        """In the standalone web client the field is populated from
        ``configData``, stays editable, and the (trimmed) edited value
        is included in the ``saveConfig`` payload."""
        out = _run_config_form_harness(
            body_classes=["remote-chat"],
            populate_cfg={"work_dir": "/srv/project"},
            edited_work_dir="  /srv/elsewhere  ",
        )
        self.assertEqual(out["afterPopulate"]["value"], "/srv/project")
        self.assertFalse(out["afterPopulate"]["readOnly"])
        self.assertTrue(out["hasWorkDir"])
        self.assertEqual(out["collectedWorkDir"], "/srv/elsewhere")

    def test_vscode_webview_field_is_read_only_and_not_saved(self) -> None:
        """In a VS Code webview the field shows the window's work_dir
        but is read-only, and ``saveConfig`` omits ``work_dir`` so one
        window's save can never clobber the persisted value."""
        out = _run_config_form_harness(
            body_classes=[],
            populate_cfg={"work_dir": "/home/user/ws_a"},
            edited_work_dir=None,
        )
        self.assertEqual(out["afterPopulate"]["value"], "/home/user/ws_a")
        self.assertTrue(out["afterPopulate"]["readOnly"])
        self.assertFalse(out["hasWorkDir"])

    def test_empty_work_dir_populates_blank(self) -> None:
        """A missing ``work_dir`` in ``configData`` leaves the field
        blank (and an untouched blank remote field saves as empty)."""
        out = _run_config_form_harness(
            body_classes=["remote-chat"],
            populate_cfg={},
            edited_work_dir=None,
        )
        self.assertEqual(out["afterPopulate"]["value"], "")
        self.assertTrue(out["hasWorkDir"])
        self.assertEqual(out["collectedWorkDir"], "")

    @staticmethod
    def _set_work_dirs(out: dict[str, Any]) -> list[str]:
        """Extract the workDir of every posted ``setWorkDir`` message."""
        return [
            str(m.get("workDir", ""))
            for m in out["posted"]
            if m.get("type") == "setWorkDir"
        ]

    def test_remote_instance_prefers_pinned_work_dir(self) -> None:
        """A webapp instance with a pinned work_dir (sessionStorage
        ``sorcar-work-dir``) displays the pin, NOT the globally
        persisted ``config.work_dir`` another instance may have saved,
        and does not re-adopt the global value."""
        out = _run_config_form_harness(
            body_classes=["remote-chat"],
            populate_cfg={"work_dir": "/srv/other-instance"},
            edited_work_dir=None,
            pinned_work_dir="/srv/mine",
        )
        self.assertEqual(out["afterPopulate"]["value"], "/srv/mine")
        self.assertNotIn("/srv/other-instance", self._set_work_dirs(out))

    def test_remote_instance_adopts_global_work_dir_when_unpinned(
        self,
    ) -> None:
        """A fresh webapp instance (no pin yet) adopts the globally
        persisted work_dir as its own pin by posting ``setWorkDir``."""
        out = _run_config_form_harness(
            body_classes=["remote-chat"],
            populate_cfg={"work_dir": "/srv/project"},
            edited_work_dir=None,
            pinned_work_dir=None,
        )
        self.assertEqual(out["afterPopulate"]["value"], "/srv/project")
        adoption = out["posted"][0]
        self.assertEqual(adoption.get("type"), "setWorkDir")
        self.assertEqual(adoption.get("workDir"), "/srv/project")

    def test_remote_save_repins_edited_work_dir(self) -> None:
        """Saving an edited work_dir from the remote settings panel
        re-pins THIS instance via ``setWorkDir`` (in addition to the
        global ``saveConfig`` persistence)."""
        out = _run_config_form_harness(
            body_classes=["remote-chat"],
            populate_cfg={"work_dir": "/srv/project"},
            edited_work_dir="/srv/elsewhere",
            pinned_work_dir="/srv/project",
        )
        self.assertIn("/srv/elsewhere", self._set_work_dirs(out))

    def test_vscode_webview_never_posts_set_work_dir(self) -> None:
        """The VS Code webview must not post ``setWorkDir`` from the
        settings form — the extension itself announces the workspace
        folder on connect; the form is read-only there."""
        out = _run_config_form_harness(
            body_classes=[],
            populate_cfg={"work_dir": "/home/user/ws_a"},
            edited_work_dir=None,
        )
        self.assertEqual(self._set_work_dirs(out), [])


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestWorkDirConfigRoundTrip(IsolatedAsyncioTestCase):
    """getConfig / saveConfig handle ``work_dir`` over real UDS."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        # Isolate config.json so this test never sees (or pollutes)
        # values saved by other tests in the same process.
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_b = Path(self.tmpdir) / "ws_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: asyncio.StreamWriter, cmd: dict[str, Any],
    ) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )

    @staticmethod
    def _config_data_with_work_dir(
        work_dir: str,
    ) -> Callable[[dict[str, Any]], bool]:
        def _pred(msg: dict[str, Any]) -> bool:
            return (
                msg.get("type") == "configData"
                and msg.get("config", {}).get("work_dir") == work_dir
            )
        return _pred

    async def test_get_config_reports_each_windows_own_work_dir(
        self,
    ) -> None:
        """With no persisted work_dir, ``getConfig`` fills it from the
        requesting connection's own work_dir — so each VS Code window's
        settings panel can show its own workspace folder."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )

        await self._send(writer_a, {"type": "getConfig"})
        await self._drain_until(
            reader_a, self._config_data_with_work_dir(str(self.dir_a)),
        )

        await self._send(writer_b, {"type": "getConfig"})
        await self._drain_until(
            reader_b, self._config_data_with_work_dir(str(self.dir_b)),
        )

    async def test_get_config_prefers_connection_work_dir_over_persisted(
        self,
    ) -> None:
        """A connection that announced its own folder via ``setWorkDir``
        must see THAT folder in ``getConfig`` even when a different
        work_dir is persisted globally (e.g. saved by another webapp
        instance) — the stamped work_dir is what its commands actually
        run in.  A connection that never announced a folder still sees
        the persisted global value."""
        vc.save_config({"work_dir": str(self.dir_a)})

        reader_pinned, writer_pinned = await self._connect()
        await self._send(
            writer_pinned,
            {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        await self._send(writer_pinned, {"type": "getConfig"})
        await self._drain_until(
            reader_pinned, self._config_data_with_work_dir(str(self.dir_b)),
        )

        reader_fresh, writer_fresh = await self._connect()
        await self._send(writer_fresh, {"type": "getConfig"})
        await self._drain_until(
            reader_fresh, self._config_data_with_work_dir(str(self.dir_a)),
        )

    async def test_save_config_work_dir_persists_and_updates_fallback(
        self,
    ) -> None:
        """A ``saveConfig`` carrying ``config.work_dir`` (sent by the
        standalone web client's editable settings field) persists the
        value and moves the daemon-wide fallback work_dir."""
        reader, writer = await self._connect()
        await self._send(
            writer,
            {"type": "saveConfig", "config": {"work_dir": str(self.dir_b)}},
        )
        await self._drain_until(
            reader, self._config_data_with_work_dir(str(self.dir_b)),
        )
        self.assertEqual(
            vc.load_config().get("work_dir"), str(self.dir_b),
        )
        self.assertEqual(
            self.server._vscode_server.work_dir, str(self.dir_b),
        )

    async def test_save_config_without_work_dir_keeps_persisted_value(
        self,
    ) -> None:
        """A VS Code window's ``saveConfig`` (which omits ``work_dir``
        because its field is read-only) must not clobber a previously
        persisted work_dir."""
        vc.save_config({"work_dir": str(self.dir_a)})
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "saveConfig", "config": {"max_budget": 42}},
        )
        await self._drain_until(
            reader,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("max_budget") == 42
            ),
        )
        self.assertEqual(
            vc.load_config().get("work_dir"), str(self.dir_a),
        )
