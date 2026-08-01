# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9, review round (gpt-5.6-sol review of the findings-2 fixes).

End-to-end regression tests (no mocks/patches/fakes) for the issues the
review found in the first round of fixes:

* G2-R01 — a batch that fails every write attempt must be preserved in
  a durable sidecar journal, not silently acknowledged and lost.
* G2-R02 — concurrent post-commit hook installs must not crash on a
  shared temp file or lose the user's hook body.
* G2-R03 — a ``build_graph`` that cannot get the update lock must not
  build unserialized; it returns the existing graph or raises.
* G2-R04 — ``mcp_servers`` must work where ``fcntl`` does not exist
  (Windows), exercised in a real subprocess with ``fcntl`` blocked.
* G2-R06 — OAuth token files for names differing only by case must not
  alias on case-insensitive filesystems; reserved Windows basenames
  are prefixed.
* G2-R07 — server names containing ``#`` resolve and report correctly.
* G2-R08 — an MCP tool whose synthesized name collides with a built-in
  agent tool (e.g. ``run_parallel``) is suffixed, not fatal.
* G2-R09 — a pre-existing world-readable auth dir is tightened to 0700.
"""

from __future__ import annotations

import asyncio
import json
import os
import pty
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.sorcar.code_graph as cg
import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.code_graph import (
    _HOOK_BEGIN,
    build_graph,
    graph_dir,
    install_post_commit_hook,
    load_graph,
)
from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _connection_key,
    _key_display_name,
    make_mcp_tool_wrapper,
    make_mcp_tools,
)
from kiss.agents.sorcar.persistence import _add_task


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestFailedBatchJournal(_TempDbTestBase):
    """G2-R01: permanently unwritable batches land in the sidecar."""

    def test_exhausted_retries_preserve_events_in_sidecar(self) -> None:
        task_id, _ = _add_task("journal target")
        db = th._get_db()
        # A real schema-level rejection: every INSERT into events fails.
        db.execute(
            "CREATE TRIGGER reject_events BEFORE INSERT ON events "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )
        batch = [(task_id, json.dumps({"type": "t"}), time.time(),
                  th._current_db_path())]

        th._persist_batch_with_retry(batch)

        count = db.execute(
            "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()["n"]
        assert count == 0
        sidecar = Path(th._current_db_path() + ".failed_events.jsonl")
        assert sidecar.is_file(), "failed batch was dropped without a journal"
        rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
        assert rows == [{
            "task_id": task_id,
            "event_json": json.dumps({"type": "t"}),
            "timestamp": batch[0][2],
            "origin_db_path": th._current_db_path(),
        }]
        # Recovery: once the DB accepts writes again, new events persist.
        db.execute("DROP TRIGGER reject_events")
        th._persist_batch_with_retry(
            [(task_id, json.dumps({"type": "ok"}), time.time(),
              th._current_db_path())]
        )
        count = db.execute(
            "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()["n"]
        assert count == 1


class TestConcurrentHookInstall:
    """G2-R02: hook RMW is serialized and temp files are unique."""

    def test_two_thread_install_never_crashes_or_loses_body(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", str(repo)], check=True, capture_output=True,
        )
        hook = repo / ".git" / "hooks" / "post-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        def install(barrier: threading.Barrier) -> None:
            try:
                barrier.wait(timeout=10)
                time.sleep(0.001 * (os.getpid() % 3))
                install_post_commit_hook(str(repo))
            except BaseException as exc:  # noqa: BLE001 — recorded for assert
                errors.append(repr(exc))

        for _ in range(30):
            hook.write_text("#!/bin/sh\necho user body\n")
            hook.chmod(0o755)
            barrier = threading.Barrier(2)
            threads = [
                threading.Thread(target=install, args=(barrier,))
                for _ in range(2)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, f"concurrent install crashed: {errors}"
            text = hook.read_text()
            assert text.count(_HOOK_BEGIN) == 1, "hook section lost/duplicated"
            assert "echo user body" in text, "user hook body lost"


class TestBuildLockTimeoutIsSafe:
    """G2-R03: a busy update lock never leads to an unserialized build."""

    def setup_method(self) -> None:
        self.saved_timeout = cg._BUILD_LOCK_TIMEOUT_S
        cg._BUILD_LOCK_TIMEOUT_S = 0.3

    def teardown_method(self) -> None:
        cg._BUILD_LOCK_TIMEOUT_S = self.saved_timeout

    def _hold_lock(self, work_dir: Path) -> Path:
        storage = graph_dir(str(work_dir))
        storage.mkdir(parents=True, exist_ok=True)
        lock = storage / ".update.lock"
        # Our own live PID: _pid_is_running() is true, so the lock is
        # genuinely held from build_graph's perspective.
        lock.write_text(f"{os.getpid()}\n")
        return lock

    def test_timeout_returns_existing_graph_unmodified(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "a.py").write_text("def before():\n    pass\n")
        build_graph(str(tmp_path))
        (tmp_path / "a.py").write_text("def after():\n    pass\n")
        lock = self._hold_lock(tmp_path)
        try:
            graph = build_graph(str(tmp_path))
        finally:
            lock.unlink()
        labels = {n["label"] for n in graph.nodes.values()}
        assert "before" in labels and "after" not in labels, (
            "build proceeded (or rebuilt) despite a held update lock"
        )
        on_disk = load_graph(str(tmp_path))
        assert on_disk is not None
        disk_labels = {n["label"] for n in on_disk.nodes.values()}
        assert "after" not in disk_labels, "unserialized build overwrote cache"

    def test_timeout_without_existing_graph_raises(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "a.py").write_text("def f():\n    pass\n")
        lock = self._hold_lock(tmp_path)
        try:
            with pytest.raises(RuntimeError, match="lock"):
                build_graph(str(tmp_path))
        finally:
            lock.unlink()
        # After the lock is released the build works normally.
        graph = build_graph(str(tmp_path))
        assert any(n["label"] == "f" for n in graph.nodes.values())


class TestWindowsCompatibility:
    """G2-R04: MCP config save/load must work where fcntl is missing."""

    def test_mcp_servers_work_without_fcntl(self, tmp_path: Path) -> None:
        shim = tmp_path / "shim"
        shim.mkdir()
        # A real import of this module fails exactly as on Windows.
        (shim / "fcntl.py").write_text(
            'raise ImportError("No module named fcntl (Windows)")\n'
        )
        work_dir = tmp_path / "project"
        work_dir.mkdir()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(shim)
        env["KISS_HOME"] = str(tmp_path / ".kisshome")
        proc = subprocess.run(
            [sys.executable, "-c", (
                "from kiss.agents.sorcar.mcp_servers import ("
                "MCPServerConfig, save_mcp_server, load_mcp_servers, "
                "remove_mcp_server)\n"
                f"wd = {str(work_dir)!r}\n"
                "save_mcp_server(MCPServerConfig(name='winsrv', "
                "command='echo'), 'project', wd)\n"
                "assert 'winsrv' in load_mcp_servers(wd)\n"
                "assert remove_mcp_server('winsrv', wd)\n"
                "print('WINDOWS-OK')\n"
            )],
            capture_output=True, text=True, timeout=180, env=env,
            cwd=str(work_dir),
        )
        assert proc.returncode == 0, proc.stderr
        assert "WINDOWS-OK" in proc.stdout


class TestTokenFileCaseAndReservedNames:
    """G2-R06 / G2-R09: token files are injective under case folding."""

    def test_case_variants_get_distinct_filenames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("KISS_HOME", str(tmp_path))
        upper = FileTokenStorage("GitHub")
        lower = FileTokenStorage("github")
        assert upper.path.name.lower() != lower.path.name.lower(), (
            "case variants alias one file on case-insensitive filesystems"
        )
        assert lower.path.name == "github.json"  # historical name kept

        from mcp.shared.auth import OAuthToken

        asyncio.run(upper.set_tokens(
            OAuthToken(access_token="SECRET-UPPER", token_type="Bearer")
        ))
        assert asyncio.run(lower.get_tokens()) is None, (
            "the lowercase server read the mixed-case server's token"
        )

    def test_reserved_windows_basenames_are_prefixed(self) -> None:
        for name in ("con", "NUL", "com1", "lpt9"):
            file_name = FileTokenStorage(name).path.name
            assert file_name.split(".", 1)[0].lower() not in (
                "con", "nul", "com1", "lpt9",
            ), f"{name!r} produced reserved basename {file_name!r}"

    def test_existing_world_readable_auth_dir_is_tightened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("KISS_HOME", str(tmp_path))
        auth_dir = tmp_path / "mcp_auth"
        auth_dir.mkdir(parents=True, mode=0o755)
        from mcp.shared.auth import OAuthToken

        storage = FileTokenStorage("srv")
        asyncio.run(storage.set_tokens(
            OAuthToken(access_token="a", token_type="Bearer")
        ))
        assert stat.S_IMODE(auth_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600


class TestHashInServerNames:
    """G2-R07: names containing ``#`` parse and display correctly."""

    def test_key_display_name_round_trip(self) -> None:
        for name in ("plain", "foo#bar", "a#b#c"):
            key = _connection_key(MCPServerConfig(name=name, command="echo"))
            assert _key_display_name(key) == name
        # A bare name (no digest suffix) is returned unchanged.
        assert _key_display_name("foo#bar") == "foo#bar"

    def test_call_tool_error_reports_full_hash_name(self) -> None:
        out = MCPManager.instance().call_tool("foo#bar", "t", {})
        assert out.startswith("Error: MCP server 'foo#bar' is not connected")


_SERVER_TEMPLATE = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("{server_name}")


@mcp.tool()
def {tool_name}() -> str:
    """Return this server's identity marker."""
    return "{marker}"


if __name__ == "__main__":
    mcp.run()
'''


@pytest.fixture
def real_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Iterator[None]:
    """Give ``sys.stdin`` (and the MCP errlog) a real file descriptor.

    Same environment plumbing as in ``test_sorcar_mcp.py``: pytest's
    captured std streams have no ``fileno()``, which the MCP stdio
    transport requires to spawn the real server subprocess.
    """
    master_fd, slave_fd = pty.openpty()
    stdin_stream = os.fdopen(slave_fd, "r", closefd=True)
    errlog = (tmp_path / "mcp_errlog.txt").open("w", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin_stream)
    monkeypatch.setattr(sys, "stderr", errlog)

    from mcp.client.stdio import stdio_client

    wrapped = stdio_client.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(wrapped, "__defaults__", (errlog,))
    try:
        yield
    finally:
        errlog.close()
        stdin_stream.close()
        os.close(master_fd)


class TestBuiltinNameCollision:
    """G2-R08: MCP tools never steal a built-in agent tool's name."""

    def test_builtin_collision_gets_suffix(
        self, tmp_path: Path, real_stdin: object,
    ) -> None:
        work_dir = tmp_path / "project"
        (work_dir / ".kiss").mkdir(parents=True)
        script = tmp_path / "run_server.py"
        script.write_text(_SERVER_TEMPLATE.format(
            server_name="run", tool_name="parallel", marker="MCP-RUN",
        ), encoding="utf-8")
        config = {
            "mcpServers": {
                "run": {
                    "type": "stdio", "command": sys.executable,
                    "args": [str(script)],
                },
            },
        }
        (work_dir / ".kiss" / "mcp.json").write_text(json.dumps(config))

        tools = make_mcp_tools(str(work_dir))

        names = [t.__name__ for t in tools]
        assert "run_parallel" not in names, (
            "MCP tool shadowed the built-in run_parallel; agent tool "
            f"registration would abort: {names}"
        )
        assert "run_parallel_2" in names, names
        by_name = {t.__name__: t for t in tools}
        assert by_name["run_parallel_2"]() == "MCP-RUN"

    def test_sanitize_produces_ascii_tool_names(self) -> None:
        from types import SimpleNamespace

        tool = SimpleNamespace(
            name="frac½tool",
            description="unicode tool name",
            inputSchema={"type": "object", "properties": {}, "required": []},
        )
        wrapper = make_mcp_tool_wrapper(MCPManager.instance(), "srv½", tool)
        assert wrapper.__name__.isascii()
        assert wrapper.__name__.isidentifier()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
