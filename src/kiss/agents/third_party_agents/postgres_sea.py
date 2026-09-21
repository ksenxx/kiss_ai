# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""PostgreSQL Agent — channel agent for PostgreSQL databases via psycopg 3.

Provides read (and optionally write) access to a PostgreSQL database
identified by a ``postgresql://`` connection URI.  Stores config in
``~/.kiss/third_party_agents/postgres/config.json``.

Read-only enforcement is done at the root cause, entirely server-side —
no SQL parsing is involved.  When the agent is in read-only mode (the
default) every connection is opened with
``options="-c default_transaction_read_only=on"`` so the server rejects
any write statement, and ``pg_query`` executes via the extended query
protocol (``prepare=True``) so the server also rejects multi-statement
strings that could otherwise open an explicit ``READ WRITE``
transaction.  ``pg_explain`` always uses the extended protocol so
appended statements after a ``;`` are rejected in both modes.

Read paths (``pg_query``, ``pg_explain``, and the catalog tools) run
with a server-side ``statement_timeout`` (default 60s, see
``PostgresChannelBackend._statement_timeout_ms``); ``pg_execute`` has
no statement timeout because long-running DDL is legitimate.

PostgreSQL has no inbound message stream, so this adapter is
outbound-only: ``poll_messages`` always returns no messages and the
``--channel`` poll mode is disabled (``main`` passes
``make_backend=None`` to ``channel_main``).

Usage::

    agent = PostgresAgent()
    agent.run(prompt_template="List the ten most recent orders")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import psycopg

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_MAX_ROWS_CAP = 10000

_POSTGRES_DIR = Path.home() / ".kiss" / "third_party_agents" / "postgres"
_config = ChannelConfig(_POSTGRES_DIR, ("database_uri",))


def _read_only_from_config(cfg: dict[str, str]) -> bool:
    """Parse the persisted ``read_only`` flag, defaulting to True.

    Args:
        cfg: Loaded config dictionary.

    Returns:
        False only when the config explicitly stores ``"false"``.
    """
    return cfg.get("read_only", "true").strip().lower() != "false"


def _parse_params(params_json: str) -> tuple[list[Any] | None, str | None]:
    """Parse an optional JSON-array string of SQL bind parameters.

    Args:
        params_json: JSON array string (e.g. ``'["x", 1]'``) or ``""``.

    Returns:
        ``(params, None)`` on success (``params`` is ``None`` when
        *params_json* is empty), or ``(None, error_json)`` where
        ``error_json`` is an ``{"ok": false, ...}`` JSON string.
    """
    if not params_json.strip():
        return None, None
    try:
        parsed = json.loads(params_json)
    except ValueError as e:
        return None, json.dumps({"ok": False, "error": f"params_json is not valid JSON: {e}"})
    if not isinstance(parsed, list):
        return None, json.dumps({"ok": False, "error": "params_json must be a JSON array"})
    return parsed, None


class PostgresChannelBackend(ToolMethodBackend):
    """Channel backend for PostgreSQL via psycopg 3.

    Opens a fresh autocommit connection per tool call and closes it via
    a context manager.  Outbound-only: there is no inbound message
    stream, so :meth:`poll_messages` always returns no messages.
    """

    def __init__(self) -> None:
        self._database_uri: str = ""
        self._read_only: bool = True
        self._connection_info: str = ""
        self._statement_timeout_ms: int = 60000

    def connect(self) -> bool:
        """Load the PostgreSQL config from disk.

        Returns:
            True if a valid config with ``database_uri`` was loaded.
        """
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No PostgreSQL config found."
            return False
        self._database_uri = cfg["database_uri"]
        self._read_only = _read_only_from_config(cfg)
        mode = "read-only" if self._read_only else "read-write"
        self._connection_info = f"PostgreSQL configured ({mode})."
        return True

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return no messages: PostgreSQL has no inbound message stream.

        Args:
            channel_id: Ignored.
            oldest: Cursor, returned unchanged.
            limit: Ignored.

        Returns:
            ``([], oldest)``.
        """
        return [], oldest

    def _connect(self, statement_timeout: bool = True) -> psycopg.Connection[Any]:
        """Open a fresh autocommit connection honoring the read-only mode.

        When :attr:`_read_only` is set, the connection is opened with
        ``-c default_transaction_read_only=on`` so the server itself
        rejects every write statement.

        Args:
            statement_timeout: When True (the default), set a
                server-side ``statement_timeout`` of
                :attr:`_statement_timeout_ms` milliseconds so a runaway
                query cannot hang the tool.  ``pg_execute`` passes
                False because long-running DDL is legitimate.

        Returns:
            An open :class:`psycopg.Connection`.
        """
        options: list[str] = []
        if self._read_only:
            options.append("-c default_transaction_read_only=on")
        if statement_timeout:
            options.append(f"-c statement_timeout={self._statement_timeout_ms}")
        kwargs: dict[str, Any] = {"connect_timeout": _CONNECT_TIMEOUT, "autocommit": True}
        if options:
            kwargs["options"] = " ".join(options)
        return psycopg.connect(self._database_uri, **kwargs)

    def pg_query(self, sql: str, params_json: str = "", max_rows: int = 100) -> str:
        """Run a SQL query and return the resulting rows.

        Args:
            sql: SQL statement to execute, with ``%s`` placeholders for
                bind parameters.
            params_json: Optional JSON array of bind parameter values,
                e.g. ``'["alice", 42]'``.  Empty means no parameters.
            max_rows: Maximum number of rows to return (default 100,
                clamped to the range 1..10000).

        The query runs with a server-side statement timeout (default
        60s).  In read-only mode ``sql`` must be a single statement:
        the server rejects multi-statement strings.

        Returns:
            JSON string ``{"ok": true, "columns": [...], "rows": [...],
            "row_count": n, "truncated": bool}``; for statements with
            no result set, ``{"ok": true, "row_count": n}``; or
            ``{"ok": false, "error": ...}``.
        """
        try:
            max_rows = min(max(max_rows, 1), _MAX_ROWS_CAP)
            params, err = _parse_params(params_json)
            if err:
                return err
            with self._connect() as conn, conn.cursor() as cur:
                # In read-only mode, prepare=True forces the extended
                # query protocol: the server rejects multi-statement
                # strings, so 'BEGIN READ WRITE; ...' cannot bypass the
                # default_transaction_read_only GUC.
                prepare = True if self._read_only else None
                cur.execute(sql, params, prepare=prepare)  # type: ignore[arg-type]
                if cur.description is None:
                    return json.dumps({"ok": True, "row_count": cur.rowcount})
                columns = [d.name for d in cur.description]
                fetched = cur.fetchmany(max_rows + 1)
                truncated = len(fetched) > max_rows
                rows = [list(row) for row in fetched[:max_rows]]
                return json.dumps(
                    {
                        "ok": True,
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows),
                        "truncated": truncated,
                    },
                    default=str,
                )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pg_execute(self, sql: str, params_json: str = "") -> str:
        """Execute a write statement (INSERT/UPDATE/DELETE/DDL).

        Refused without even connecting when the agent is in read-only
        mode; use ``authenticate_postgres(..., read_only=False)`` to
        enable writes.  Runs without a statement timeout because
        long-running DDL is legitimate.

        Args:
            sql: SQL statement to execute, with ``%s`` placeholders for
                bind parameters.
            params_json: Optional JSON array of bind parameter values.
                Empty means no parameters.

        Returns:
            JSON string ``{"ok": true, "row_count": n}`` (``n`` is the
            affected row count, ``-1`` for DDL) or
            ``{"ok": false, "error": ...}``.
        """
        try:
            if self._read_only:
                return json.dumps({"ok": False, "error": "Postgres agent is in read-only mode"})
            params, err = _parse_params(params_json)
            if err:
                return err
            with self._connect(statement_timeout=False) as conn, conn.cursor() as cur:
                if params is None:
                    cur.execute(sql)  # type: ignore[arg-type]
                else:
                    cur.execute(sql, params)  # type: ignore[arg-type]
                return json.dumps({"ok": True, "row_count": cur.rowcount})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pg_list_schemas(self) -> str:
        """List all schemas in the database.

        Returns:
            JSON string ``{"ok": true, "schemas": [...]}`` or
            ``{"ok": false, "error": ...}``.
        """
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name"
                )
                schemas = [row[0] for row in cur.fetchall()]
                return json.dumps({"ok": True, "schemas": schemas}, default=str)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pg_list_tables(self, schema: str = "public") -> str:
        """List the tables and views in a schema.

        Args:
            schema: Schema name (default ``"public"``).

        Returns:
            JSON string ``{"ok": true, "tables": [{"name": ...,
            "type": ...}, ...]}`` or ``{"ok": false, "error": ...}``.
        """
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name, table_type FROM information_schema.tables "
                    "WHERE table_schema = %s ORDER BY table_name",
                    (schema,),
                )
                tables = [{"name": row[0], "type": row[1]} for row in cur.fetchall()]
                return json.dumps({"ok": True, "tables": tables}, default=str)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pg_describe_table(self, table: str, schema: str = "public") -> str:
        """Describe the columns of a table.

        Args:
            table: Table name.
            schema: Schema name (default ``"public"``).

        Returns:
            JSON string ``{"ok": true, "columns": [{"name": ...,
            "data_type": ..., "is_nullable": ..., "column_default":
            ...}, ...]}``; ``{"ok": false, "error": "table not found"}``
            when the table has no columns; or
            ``{"ok": false, "error": ...}``.
        """
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                    (schema, table),
                )
                columns = [
                    {
                        "name": row[0],
                        "data_type": row[1],
                        "is_nullable": row[2],
                        "column_default": row[3],
                    }
                    for row in cur.fetchall()
                ]
                if not columns:
                    return json.dumps({"ok": False, "error": "table not found"})
                return json.dumps({"ok": True, "columns": columns}, default=str)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pg_list_indexes(self, table: str, schema: str = "public") -> str:
        """List the indexes of a table.

        Args:
            table: Table name.
            schema: Schema name (default ``"public"``).

        Returns:
            JSON string ``{"ok": true, "indexes": [{"name": ...,
            "definition": ...}, ...]}`` or
            ``{"ok": false, "error": ...}``.
        """
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = %s AND tablename = %s ORDER BY indexname",
                    (schema, table),
                )
                indexes = [{"name": row[0], "definition": row[1]} for row in cur.fetchall()]
                return json.dumps({"ok": True, "indexes": indexes}, default=str)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def pg_explain(self, sql: str, analyze: bool = False) -> str:
        """Show the PostgreSQL execution plan for a SQL statement.

        Args:
            sql: A single SQL statement to explain.  Multi-statement
                strings are rejected by the server (the EXPLAIN runs
                over the extended query protocol), so nothing after a
                ``;`` can be executed.
            analyze: Run ``EXPLAIN ANALYZE`` (actually executes the
                statement).  Refused in read-only mode.

        Returns:
            JSON string ``{"ok": true, "plan": [...]}`` or
            ``{"ok": false, "error": ...}``.
        """
        try:
            if analyze and self._read_only:
                return json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "EXPLAIN ANALYZE executes the statement and is refused "
                            "in read-only mode"
                        ),
                    }
                )
            prefix = "EXPLAIN (ANALYZE, FORMAT JSON) " if analyze else "EXPLAIN (FORMAT JSON) "
            with self._connect() as conn, conn.cursor() as cur:
                # prepare=True forces the extended query protocol in
                # BOTH modes: the server rejects multi-statement input,
                # so 'SELECT 1; CREATE TABLE ...' cannot execute the
                # appended statement.
                cur.execute(prefix + sql, prepare=True)  # type: ignore[arg-type]
                row = cur.fetchone()
                plan = row[0] if row else None
                return json.dumps({"ok": True, "plan": plan}, default=str)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class PostgresAgent(BaseChannelAgent):
    """Channel agent with PostgreSQL query tools."""

    channel_system_prompt = (
        "You are operating a PostgreSQL database through psycopg. Use "
        "pg_list_schemas, pg_list_tables, pg_describe_table, and "
        "pg_list_indexes to explore the schema, pg_query to run SELECT "
        "queries (bind values via a JSON-array params_json, never by "
        "string interpolation), pg_explain to inspect query plans, and "
        "pg_execute for INSERT/UPDATE/DDL. The agent is read-only by "
        "default: the server rejects all writes, pg_query accepts only "
        "a single statement, and pg_execute refuses to run. Re-run "
        "authenticate_postgres(database_uri, read_only=False) to "
        "enable writes. pg_explain always takes a single statement. "
        "Reads run with a 60s server-side statement timeout; pg_query "
        "returns at most 10000 rows per call."
    )

    def __init__(self) -> None:
        super().__init__("PostgreSQL Agent")
        self._backend = PostgresChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._database_uri = cfg["database_uri"]
            self._backend._read_only = _read_only_from_config(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._database_uri)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_postgres_auth() -> str:
            """Check if PostgreSQL is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for PostgreSQL. Use "
                    "authenticate_postgres() to configure.\n"
                    "You need a PostgreSQL connection URI of the form "
                    "postgresql://user:password@host:port/database. Pass "
                    "read_only=False only if the agent should be allowed "
                    "to modify the database."
                )
            return json.dumps({"ok": True, "read_only": agent._backend._read_only})

        def authenticate_postgres(database_uri: str, read_only: bool = True) -> str:
            """Configure the PostgreSQL connection URI and access mode.

            Verifies connectivity with a ``SELECT 1`` before saving; an
            unreachable database is reported and nothing is persisted.

            Args:
                database_uri: Connection URI, e.g.
                    ``postgresql://user:password@host:5432/database``.
                read_only: Keep the agent read-only (default True).
                    Pass False to allow writes via pg_execute.

            Returns:
                Configuration result or error message.
            """
            uri = database_uri.strip()
            if not uri:
                return "database_uri cannot be empty."
            try:
                with psycopg.connect(uri, connect_timeout=_CONNECT_TIMEOUT) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
            except Exception as e:
                return json.dumps({"ok": False, "error": f"Connection failed: {e}"})
            try:
                _config.save(
                    {"database_uri": uri, "read_only": "true" if read_only else "false"}
                )
                agent._backend._database_uri = uri
                agent._backend._read_only = read_only
            except Exception as e:
                return json.dumps({"ok": False, "error": f"could not save config: {e}"})
            mode = "read-only" if read_only else "read-write"
            return json.dumps({"ok": True, "message": f"PostgreSQL configured ({mode})."})

        def clear_postgres_auth() -> str:
            """Clear the stored PostgreSQL configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._database_uri = ""
            agent._backend._read_only = True
            return "PostgreSQL configuration cleared."

        return [check_postgres_auth, authenticate_postgres, clear_postgres_auth]


def main() -> None:
    """Run the PostgresAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): PostgreSQL has no
    inbound message stream to poll.
    """
    channel_main(
        PostgresAgent,
        "kiss-postgres",
        channel_name="Postgres",
        make_backend=None,
    )


def tools() -> list:
    """Return the PostgreSQL channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return PostgresAgent()._get_tools()


if __name__ == "__main__":
    main()
