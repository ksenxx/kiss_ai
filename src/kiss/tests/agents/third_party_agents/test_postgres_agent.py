# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the PostgreSQL channel agent.

Runs against a REAL PostgreSQL server in Docker — no mocks, patches,
or fakes.  The Docker-backed tests are skipped when ``docker info``
fails at collection time; the no-server tests always run.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
import shutil
import socket
import stat
import subprocess
import sys
import time

import psycopg
import pytest

import kiss.agents.third_party_agents.postgres_agent as pg_mod
from kiss.agents.third_party_agents.postgres_agent import (
    PostgresAgent,
    PostgresChannelBackend,
    _config,
)

# Nothing listens on the discard port, so this URI is always unreachable.
_UNREACHABLE_URI = "postgresql://user:x@127.0.0.1:9/db"


def _docker_available() -> bool:
    """Return True when the Docker daemon answers ``docker info``."""
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=60, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


_DOCKER_OK = _docker_available()


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted PostgreSQL config."""
    _config.clear()
    yield
    _config.clear()


def _backend(uri: str, read_only: bool) -> PostgresChannelBackend:
    """Build a backend pointed at *uri* in the given access mode."""
    b = PostgresChannelBackend()
    b._database_uri = uri
    b._read_only = read_only
    return b


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes exactly the auth trio."""
    agent = PostgresAgent()
    assert agent.name == "PostgreSQL Agent"
    assert agent._is_authenticated() is False
    names = [t.__name__ for t in agent._get_tools()]
    assert names == ["check_postgres_auth", "authenticate_postgres", "clear_postgres_auth"]


def test_check_auth_unauthenticated_message() -> None:
    """check_postgres_auth explains how to configure when unauthenticated."""
    agent = PostgresAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_postgres_auth"]()
    assert "authenticate_postgres" in msg
    assert "postgresql://" in msg


def test_authenticate_rejects_empty_uri() -> None:
    """authenticate_postgres refuses an empty or whitespace URI."""
    agent = PostgresAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cannot be empty" in tools["authenticate_postgres"]("")
    assert "cannot be empty" in tools["authenticate_postgres"]("   ")
    assert not _config.path.exists()


def test_authenticate_unreachable_saves_nothing() -> None:
    """An unreachable database yields ok:false and persists no config."""
    agent = PostgresAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_postgres"](_UNREACHABLE_URI))
    assert result["ok"] is False
    assert "Connection failed" in result["error"]
    assert not _config.path.exists()
    assert agent._is_authenticated() is False


def test_persisted_config_loaded_by_new_agent() -> None:
    """A new agent picks up a persisted config (written 0600)."""
    _config.save({"database_uri": _UNREACHABLE_URI, "read_only": "false"})
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    agent = PostgresAgent()
    assert agent._is_authenticated() is True
    assert agent._backend._database_uri == _UNREACHABLE_URI
    assert agent._backend._read_only is False
    names = {t.__name__ for t in agent._get_tools()}
    assert {
        "pg_query",
        "pg_execute",
        "pg_list_schemas",
        "pg_list_tables",
        "pg_describe_table",
        "pg_list_indexes",
        "pg_explain",
    } <= names

    tools = {t.__name__: t for t in agent._get_tools()}
    cleared = tools["clear_postgres_auth"]()
    assert "cleared" in cleared.lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert agent._backend._read_only is True
    assert len(agent._get_tools()) == 3


def test_read_only_defaults_to_true_when_key_missing() -> None:
    """A config without the read_only key loads as read-only."""
    _config.save({"database_uri": _UNREACHABLE_URI})
    agent = PostgresAgent()
    assert agent._backend._read_only is True


def test_tools_module_function() -> None:
    """Module-level tools() returns a non-empty tool list."""
    tools = pg_mod.tools()
    assert len(tools) >= 3
    assert all(callable(t) for t in tools)


def test_connect_without_config_fails() -> None:
    """connect() fails cleanly when no config is persisted."""
    b = PostgresChannelBackend()
    assert b.connect() is False
    assert "No PostgreSQL config" in b.connection_info


def test_connect_with_config_succeeds() -> None:
    """connect() loads the persisted URI and mode into the backend."""
    _config.save({"database_uri": _UNREACHABLE_URI, "read_only": "false"})
    b = PostgresChannelBackend()
    assert b.connect() is True
    assert b._database_uri == _UNREACHABLE_URI
    assert b._read_only is False
    assert "read-write" in b.connection_info


def test_poll_messages_returns_empty() -> None:
    """poll_messages returns no messages: PostgreSQL has no inbound stream."""
    b = _backend(_UNREACHABLE_URI, read_only=True)
    messages, cursor = b.poll_messages("anything", "42", limit=5)
    assert messages == []
    assert cursor == "42"


def test_pg_execute_read_only_refusal_without_connecting() -> None:
    """In read-only mode pg_execute refuses BEFORE any connection attempt.

    The URI is unreachable, so a connection attempt would produce a
    connection error — the exact 'read-only mode' message proves no
    connection was tried.
    """
    b = _backend(_UNREACHABLE_URI, read_only=True)
    start = time.monotonic()
    result = json.loads(b.pg_execute("INSERT INTO t VALUES (1)"))
    assert result == {"ok": False, "error": "Postgres agent is in read-only mode"}
    assert time.monotonic() - start < 1.0  # no connect_timeout wait


def test_pg_explain_analyze_refused_in_read_only() -> None:
    """pg_explain(analyze=True) is refused in read-only mode without connecting."""
    b = _backend(_UNREACHABLE_URI, read_only=True)
    result = json.loads(b.pg_explain("SELECT 1", analyze=True))
    assert result["ok"] is False
    assert "read-only" in result["error"]
    assert "Connection" not in result["error"]


def test_params_json_validation_errors() -> None:
    """Malformed or non-array params_json yields ok:false before connecting."""
    b = _backend(_UNREACHABLE_URI, read_only=False)
    result = json.loads(b.pg_query("SELECT %s", "{not json"))
    assert result["ok"] is False
    assert "not valid JSON" in result["error"]
    result = json.loads(b.pg_query("SELECT %s", '{"a": 1}'))
    assert result == {"ok": False, "error": "params_json must be a JSON array"}
    result = json.loads(b.pg_execute("INSERT INTO t VALUES (%s)", "not json"))
    assert result["ok"] is False
    assert "not valid JSON" in result["error"]
    result = json.loads(b.pg_execute("INSERT INTO t VALUES (%s)", "42"))
    assert result == {"ok": False, "error": "params_json must be a JSON array"}


def test_every_tool_returns_ok_false_when_unreachable() -> None:
    """Every tool returns ok:false against an unreachable server — never raises."""
    b = _backend(_UNREACHABLE_URI, read_only=False)
    for call in (
        lambda: b.pg_query("SELECT 1"),
        lambda: b.pg_execute("CREATE TABLE t (x int)"),
        lambda: b.pg_list_schemas(),
        lambda: b.pg_list_tables(),
        lambda: b.pg_describe_table("t"),
        lambda: b.pg_list_indexes("t"),
        lambda: b.pg_explain("SELECT 1"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert result["error"]


@pytest.mark.skipif(not _DOCKER_OK, reason="docker daemon not available")
class TestPostgresLive:
    """End-to-end tests against a real PostgreSQL 17 server in Docker."""

    @pytest.fixture(scope="class")
    def pg_uri(self):
        """Start a disposable PostgreSQL container; yield its connection URI."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        subprocess.run(
            ["docker", "pull", "postgres:17-alpine"],
            check=True,
            capture_output=True,
            timeout=900,
        )
        container_id = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "-e",
                "POSTGRES_PASSWORD=kisstest",
                "-p",
                f"127.0.0.1:{port}:5432",
                "postgres:17-alpine",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
        uri = f"postgresql://postgres:kisstest@127.0.0.1:{port}/postgres"
        try:
            deadline = time.monotonic() + 120
            while True:
                try:
                    with psycopg.connect(uri, connect_timeout=3) as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT 1")
                    break
                except psycopg.OperationalError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1)
            yield uri
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=60,
                check=False,
            )

    def test_authenticate_against_live_server_persists(self, pg_uri: str) -> None:
        """authenticate_postgres verifies SELECT 1 and persists the config."""
        agent = PostgresAgent()
        tools = {t.__name__: t for t in agent._get_tools()}
        result = json.loads(tools["authenticate_postgres"](pg_uri, False))
        assert result["ok"] is True
        assert "read-write" in result["message"]
        saved = json.loads(_config.path.read_text(encoding="utf-8"))
        assert saved == {"database_uri": pg_uri, "read_only": "false"}
        tools = {t.__name__: t for t in agent._get_tools()}
        checked = json.loads(tools["check_postgres_auth"]())
        assert checked == {"ok": True, "read_only": False}

    def test_authenticate_save_failure_returns_error(self, pg_uri: str) -> None:
        """A failed config save yields ok:false and leaves the agent unconfigured.

        A regular file pre-created at the config *directory* path makes
        ``_config.save``'s mkdir fail, so the never-raise contract of the
        authenticate tool is exercised end to end against the live server.
        """
        config_dir = _config.path.parent
        if config_dir.is_dir():  # earlier tests may have created it
            shutil.rmtree(config_dir)
        config_dir.parent.mkdir(parents=True, exist_ok=True)
        config_dir.write_text("blocker", encoding="utf-8")
        try:
            agent = PostgresAgent()
            tools = {t.__name__: t for t in agent._get_tools()}
            result = json.loads(tools["authenticate_postgres"](pg_uri, False))
            assert result["ok"] is False
            assert "could not save config" in result["error"]
            assert agent._backend._database_uri == ""
            assert agent._backend._read_only is True
            assert agent._is_authenticated() is False
        finally:
            config_dir.unlink()

    def test_execute_create_and_insert_with_params(self, pg_uri: str) -> None:
        """pg_execute runs DDL and parameterized inserts in read-write mode."""
        b = _backend(pg_uri, read_only=False)
        result = json.loads(b.pg_execute("CREATE TABLE live_users (id int PRIMARY KEY, name text)"))
        assert result == {"ok": True, "row_count": -1}
        result = json.loads(b.pg_execute("INSERT INTO live_users VALUES (%s, %s)", '[1, "alice"]'))
        assert result == {"ok": True, "row_count": 1}
        result = json.loads(b.pg_query("SELECT name FROM live_users WHERE id = %s", "[1]"))
        assert result["rows"] == [["alice"]]

    def test_query_columns_rows_and_max_rows_truncation(self, pg_uri: str) -> None:
        """pg_query returns columns/rows and respects max_rows truncation."""
        b = _backend(pg_uri, read_only=False)
        assert json.loads(b.pg_execute("CREATE TABLE live_nums (n int)"))["ok"] is True
        assert (
            json.loads(b.pg_execute("INSERT INTO live_nums SELECT generate_series(1, 5)"))[
                "row_count"
            ]
            == 5
        )
        result = json.loads(b.pg_query("SELECT n FROM live_nums ORDER BY n", max_rows=3))
        assert result["ok"] is True
        assert result["columns"] == ["n"]
        assert result["rows"] == [[1], [2], [3]]
        assert result["row_count"] == 3
        assert result["truncated"] is True
        result = json.loads(b.pg_query("SELECT n FROM live_nums ORDER BY n"))
        assert result["row_count"] == 5
        assert result["truncated"] is False

    def test_read_only_guc_blocks_writes_server_side(self, pg_uri: str) -> None:
        """In read-only mode the SERVER rejects an INSERT sent via pg_query."""
        rw = _backend(pg_uri, read_only=False)
        assert json.loads(rw.pg_execute("CREATE TABLE live_ro (x int)"))["ok"] is True
        ro = _backend(pg_uri, read_only=True)
        result = json.loads(ro.pg_query("INSERT INTO live_ro VALUES (1)"))
        assert result["ok"] is False
        assert "read-only transaction" in result["error"]
        # Reads still work in read-only mode.
        result = json.loads(ro.pg_query("SELECT count(*) FROM live_ro"))
        assert result["ok"] is True
        assert result["rows"] == [[0]]

    def test_read_only_multi_statement_bypass_blocked(self, pg_uri: str) -> None:
        """A multi-statement 'BEGIN READ WRITE; ...' cannot bypass read-only mode.

        The extended query protocol (prepare=True) makes the server
        reject multi-statement strings, so the write never happens.
        """
        ro = _backend(pg_uri, read_only=True)
        result = json.loads(
            ro.pg_query(
                "BEGIN READ WRITE; CREATE TABLE ro_bypass(x int); "
                "INSERT INTO ro_bypass VALUES (1); COMMIT"
            )
        )
        assert result["ok"] is False
        assert "multiple commands" in result["error"]
        # A separate read-write connection confirms the table was never created.
        rw = _backend(pg_uri, read_only=False)
        check = json.loads(rw.pg_query("SELECT to_regclass('public.ro_bypass')"))
        assert check["ok"] is True
        assert check["rows"] == [[None]]

    def test_explain_rejects_multi_statement_in_both_modes(self, pg_uri: str) -> None:
        """pg_explain never executes an appended second statement."""
        for read_only in (True, False):
            b = _backend(pg_uri, read_only=read_only)
            result = json.loads(b.pg_explain("SELECT 1; CREATE TABLE explain_rw(x int)"))
            assert result["ok"] is False, f"read_only={read_only}"
            assert "multiple commands" in result["error"]
        rw = _backend(pg_uri, read_only=False)
        check = json.loads(rw.pg_query("SELECT to_regclass('public.explain_rw')"))
        assert check["ok"] is True
        assert check["rows"] == [[None]]

    def test_max_rows_clamped_to_valid_range(self, pg_uri: str) -> None:
        """max_rows of 0/negative behaves as 1; huge values clamp to 10000."""
        b = _backend(pg_uri, read_only=False)
        assert json.loads(b.pg_execute("CREATE TABLE live_clamp (n int)"))["ok"] is True
        assert (
            json.loads(b.pg_execute("INSERT INTO live_clamp SELECT generate_series(1, 5)"))[
                "row_count"
            ]
            == 5
        )
        for bad in (0, -5):
            result = json.loads(b.pg_query("SELECT n FROM live_clamp ORDER BY n", max_rows=bad))
            assert result["ok"] is True, f"max_rows={bad}"
            assert result["rows"] == [[1]]
            assert result["row_count"] == 1
            assert result["truncated"] is True
        result = json.loads(b.pg_query("SELECT n FROM live_clamp ORDER BY n", max_rows=10**9))
        assert result["ok"] is True
        assert result["row_count"] == 5
        assert result["truncated"] is False

    def test_statement_timeout_cancels_long_query(self, pg_uri: str) -> None:
        """A lowered _statement_timeout_ms cancels a pg_sleep server-side."""
        b = _backend(pg_uri, read_only=True)
        b._statement_timeout_ms = 200
        start = time.monotonic()
        result = json.loads(b.pg_query("SELECT pg_sleep(2)"))
        assert time.monotonic() - start < 2.0  # canceled well before the sleep ends
        assert result["ok"] is False
        assert "statement timeout" in result["error"] or "canceling statement" in result["error"]

    def test_describe_table_happy_and_not_found(self, pg_uri: str) -> None:
        """pg_describe_table lists columns and reports missing tables."""
        b = _backend(pg_uri, read_only=False)
        assert (
            json.loads(
                b.pg_execute("CREATE TABLE live_desc (id serial PRIMARY KEY, note text NOT NULL)")
            )["ok"]
            is True
        )
        result = json.loads(b.pg_describe_table("live_desc"))
        assert result["ok"] is True
        by_name = {c["name"]: c for c in result["columns"]}
        assert by_name["id"]["data_type"] == "integer"
        assert by_name["id"]["is_nullable"] == "NO"
        assert "nextval" in by_name["id"]["column_default"]
        assert by_name["note"]["data_type"] == "text"
        assert by_name["note"]["column_default"] is None
        result = json.loads(b.pg_describe_table("no_such_table"))
        assert result == {"ok": False, "error": "table not found"}
        # Identifier params are bound, never interpolated: a quote is data.
        result = json.loads(b.pg_describe_table("x' OR '1'='1"))
        assert result == {"ok": False, "error": "table not found"}

    def test_list_schemas_tables_indexes(self, pg_uri: str) -> None:
        """pg_list_schemas/pg_list_tables/pg_list_indexes return catalog data."""
        b = _backend(pg_uri, read_only=False)
        assert (
            json.loads(b.pg_execute("CREATE TABLE live_idx (id int PRIMARY KEY, v text)"))["ok"]
            is True
        )
        assert json.loads(b.pg_execute("CREATE INDEX live_idx_v ON live_idx (v)"))["ok"] is True
        result = json.loads(b.pg_list_schemas())
        assert result["ok"] is True
        assert "public" in result["schemas"]
        assert "information_schema" in result["schemas"]
        result = json.loads(b.pg_list_tables())
        assert result["ok"] is True
        tables = {t["name"]: t["type"] for t in result["tables"]}
        assert tables["live_idx"] == "BASE TABLE"
        result = json.loads(b.pg_list_tables(schema="information_schema"))
        assert result["ok"] is True
        assert any(t["type"] == "VIEW" for t in result["tables"])
        result = json.loads(b.pg_list_indexes("live_idx"))
        assert result["ok"] is True
        names = {i["name"] for i in result["indexes"]}
        assert names == {"live_idx_pkey", "live_idx_v"}
        assert all("CREATE" in i["definition"] for i in result["indexes"])

    def test_explain_plain_and_analyze(self, pg_uri: str) -> None:
        """pg_explain returns a JSON plan, with and without ANALYZE."""
        b = _backend(pg_uri, read_only=False)
        result = json.loads(b.pg_explain("SELECT 1"))
        assert result["ok"] is True
        assert result["plan"][0]["Plan"]["Node Type"] == "Result"
        result = json.loads(b.pg_explain("SELECT 1", analyze=True))
        assert result["ok"] is True
        assert "Actual Rows" in result["plan"][0]["Plan"]
        # Read-only mode still allows plain EXPLAIN.
        ro = _backend(pg_uri, read_only=True)
        result = json.loads(ro.pg_explain("SELECT 1"))
        assert result["ok"] is True

    def test_query_with_no_result_set_returns_rowcount(self, pg_uri: str) -> None:
        """pg_query of a statement with no result set returns just row_count."""
        b = _backend(pg_uri, read_only=False)
        result = json.loads(b.pg_query("CREATE TABLE live_norows (x int)"))
        assert result == {"ok": True, "row_count": -1}
        result = json.loads(b.pg_query("INSERT INTO live_norows VALUES (1), (2)"))
        assert result == {"ok": True, "row_count": 2}

    def test_datetime_and_decimal_serialized_via_str(self, pg_uri: str) -> None:
        """Non-JSON-native values (timestamp, numeric) come back as strings."""
        b = _backend(pg_uri, read_only=False)
        assert (
            json.loads(b.pg_execute("CREATE TABLE live_types (ts timestamptz, amt numeric)"))["ok"]
            is True
        )
        assert (
            json.loads(
                b.pg_execute(
                    "INSERT INTO live_types VALUES (%s, %s)",
                    '["2024-06-01T12:30:00+00:00", "12.50"]',
                )
            )["ok"]
            is True
        )
        result = json.loads(b.pg_query("SELECT ts, amt FROM live_types"))
        assert result["ok"] is True
        ts_value, amt_value = result["rows"][0]
        assert isinstance(ts_value, str)
        assert ts_value.startswith("2024-06-01")
        assert isinstance(amt_value, str)
        assert amt_value == "12.50"

    def test_sql_error_returns_ok_false(self, pg_uri: str) -> None:
        """A SQL syntax/relation error yields ok:false, never an exception."""
        b = _backend(pg_uri, read_only=False)
        result = json.loads(b.pg_query("SELECT * FROM no_such_relation_xyz"))
        assert result["ok"] is False
        assert "no_such_relation_xyz" in result["error"]
        result = json.loads(b.pg_explain("SELECT * FROM no_such_relation_xyz"))
        assert result["ok"] is False
