# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for bughunt round 9 on the CLI client.

Scope: :mod:`kiss.agents.sorcar.cli_client` and
:mod:`kiss.agents.sorcar.cli_daemon_bridge` only.  Each test spins up
(or reuses, via :class:`CliClientBase`) a REAL
:class:`RemoteAccessServer` on a temporary Unix-domain socket — no
mocks, patches, or fakes.

Issues covered:

1. Custom slash commands expanded server-side were submitted through a
   hard-coded ``_submit_task(client, prompt)`` call with DEFAULT run
   flags, silently dropping the operator's ``--no-worktree`` /
   ``--no-parallel`` / ``--auto-commit`` choices (the daemon reads
   ``useWorktree`` / ``useParallel`` / ``autoCommit`` from every
   ``run`` command) and bypassing the anchored steering submit path.
   ``_handle_client_slash`` must route the expanded prompt through the
   caller-supplied ``submit`` callable — the same one
   :func:`_run_repl_loop` uses for plain task lines.

2. The daemon's ``models`` event carries ``"selected"`` — the server's
   canonical current model — but the client dispatcher dropped it, so
   ``/model list`` never refreshed ``current_model`` even though the
   code comment claimed that side effect.

3. A daemon ``error`` event with ``text: null`` rendered the literal
   string ``None`` on the terminal: the dispatcher's error branch used
   ``event.get("text", "")`` which only falls back when the key is
   absent (the ``_render`` branches were already hardened by review
   #36; the ``error`` branch was missed).

4. ``_sock_path()`` was duplicated verbatim in ``cli_client`` and
   ``cli_daemon_bridge``; the client now re-exports the bridge's
   implementation.  The test pins the shared behaviour (env override
   honoured, both modules agree) so the refactor cannot regress.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path

from kiss.agents.sorcar import cli_client as cli_client_mod
from kiss.agents.sorcar import cli_daemon_bridge
from kiss.agents.sorcar.cli_client import (
    CliClient,
    _EventDispatcher,
    _handle_client_slash,
    _request_models,
    _run_repl_loop,
)
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.agents.sorcar.test_cli_client import CliClientBase


class TestCustomCommandUsesCallerSubmit(CliClientBase):
    """Issue 1: expanded custom commands must flow through ``submit``."""

    def _write_custom_command(self) -> None:
        cmds_dir = Path(self.harness.work_dir) / ".kiss" / "commands"
        cmds_dir.mkdir(parents=True, exist_ok=True)
        (cmds_dir / "greet.md").write_text(
            "---\ndescription: Greet someone\n---\nHello $ARGUMENTS",
        )

    def test_slash_handler_routes_expansion_through_submit(self) -> None:
        """``_handle_client_slash`` must call the supplied ``submit``.

        Pre-fix the function had no ``submit`` parameter and always
        called ``_submit_task`` with default run flags, so the CLI's
        ``--no-worktree`` / ``--no-parallel`` / ``--auto-commit``
        choices were dropped for custom commands.
        """
        self._write_custom_command()
        submitted: list[str] = []

        def recording_submit(prompt: str) -> None:
            submitted.append(prompt)

        result = _handle_client_slash(
            self.client, "/greet world", submit=recording_submit,
        )
        self.assertFalse(result)
        self.assertEqual(submitted, ["Hello world"])
        # The expanded prompt must NOT have been sent as a raw ``run``
        # with default flags behind the caller's back.
        run_cmds = [
            c for c in self.harness.received_cmds if c.get("type") == "run"
        ]
        self.assertEqual(run_cmds, [])

    def test_repl_loop_routes_custom_command_through_its_submit(self) -> None:
        """The shared REPL loop must reuse its own ``submit`` binding.

        This is the end-to-end path: a ``/greet world`` line typed at
        the prompt is expanded by the REAL daemon (``cliInfo`` /
        ``expandCommand`` round-trip over the UDS) and the expansion
        must be executed by the very same ``submit`` callable the loop
        uses for plain task lines — the one carrying the operator's
        run flags (and, in anchored mode, the anchored steering box).
        """
        self._write_custom_command()
        lines = iter(["/greet world", None])
        submitted: list[str] = []

        def read_line() -> str | None:
            return next(lines)

        def recording_submit(prompt: str) -> None:
            submitted.append(prompt)

        _run_repl_loop(self.client, read_line, recording_submit)
        self.assertEqual(submitted, ["Hello world"])
        run_cmds = [
            c for c in self.harness.received_cmds if c.get("type") == "run"
        ]
        self.assertEqual(run_cmds, [])


class TestModelsSelectedRefreshesCurrentModel(CliClientBase):
    """Issue 2: the ``models`` event's ``selected`` field must be used."""

    def test_request_models_refreshes_current_model(self) -> None:
        self.assertEqual(self.client.dispatcher.current_model, "")
        models = _request_models(self.client)
        self.assertGreater(len(models), 0)
        with self.harness.server._vscode_server._state_lock:
            server_selected = self.harness.server._vscode_server._default_model
        self.assertNotEqual(server_selected, "")
        self.assertEqual(
            self.client.dispatcher.current_model, server_selected,
        )

    def test_dispatch_ignores_non_string_selected(self) -> None:
        """A malformed ``selected`` must not corrupt ``current_model``."""
        disp = self.client.dispatcher
        disp.current_model = "keep-me"
        disp.dispatch({"type": "models", "models": [], "selected": None})
        disp.models_q.get(timeout=1)
        self.assertEqual(disp.current_model, "keep-me")
        disp.dispatch({"type": "models", "models": [], "selected": 7})
        disp.models_q.get(timeout=1)
        self.assertEqual(disp.current_model, "keep-me")


class TestErrorEventNullText(CliClientBase):
    """Issue 3: ``error`` events with ``text: null`` must not print None."""

    def test_error_event_with_null_text_over_uds(self) -> None:
        """Full e2e: the daemon broadcasts ``{"type":"error","text":null}``
        and the client terminal output must not contain ``None``."""
        out = io.StringIO()
        printer = ConsolePrinter(file=out)
        client = CliClient(
            sock_path=Path(self.harness.sock_path),
            work_dir=self.harness.work_dir,
            tab_id=uuid.uuid4().hex,
            printer=printer,
        )
        client.start(timeout=5.0)
        try:
            # ``start()`` returns once the UDS connection is open and the
            # hello commands are written, but the daemon registers the
            # connection for broadcasts (``add_uds_writer``) only when its
            # accept handler runs on the server loop — a broadcast issued
            # before that is legitimately dropped (broadcasts reach current
            # subscribers only).  A request/reply round-trip closes the
            # race deterministically: the handler registers the writer
            # BEFORE reading any command, so once the ``models`` reply
            # arrives the client is guaranteed subscribed.  The reply
            # must actually arrive (non-empty) — a timed-out round-trip
            # would leave the race unsynchronised.
            self.assertGreater(len(_request_models(client)), 0)
            self.harness.server._printer.broadcast(
                {"type": "error", "text": None},
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if "✗" in out.getvalue():
                    break
                time.sleep(0.05)
            rendered = out.getvalue()
            self.assertIn("✗", rendered)
            self.assertNotIn("None", rendered)
        finally:
            client.close()

    def test_dispatch_error_null_text_direct(self) -> None:
        """Dispatcher-level check with a real ConsolePrinter (no server
        round-trip) so the coercion is pinned even for locally injected
        events."""
        out = io.StringIO()
        disp = _EventDispatcher(ConsolePrinter(file=out))
        disp.dispatch({"type": "error", "text": None})
        rendered = out.getvalue()
        self.assertIn("✗", rendered)
        self.assertNotIn("None", rendered)


class TestSockPathSharedBehaviour(CliClientBase):
    """Issue 4: both modules must resolve the exact same socket path."""

    def test_env_override_honoured_and_modules_agree(self) -> None:
        # CliClientBase's harness exports KISS_SORCAR_SOCK.
        expected = Path(os.environ["KISS_SORCAR_SOCK"])
        self.assertEqual(cli_client_mod._sock_path(), expected)
        self.assertEqual(cli_daemon_bridge._sock_path(), expected)
        self.assertEqual(
            cli_client_mod._sock_path(), cli_daemon_bridge._sock_path(),
        )
