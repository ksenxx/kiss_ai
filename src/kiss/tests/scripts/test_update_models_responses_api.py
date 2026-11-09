# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for OpenAI v2 (Responses API) support in update_models.py.

``update_models.py`` live-probes every tested model through
``/v1/responses`` (via ``update_responses_api_support.probe_responses_support``)
and records the verdict as ``"use_responses_api": true`` on passing entries:

* new models carry the probed flag into their catalog entry, and every
  generated ``-{level}`` thinking alias inherits it;
* ``--test-existing`` re-probes and flips the stored flag in either
  direction (a falsy verdict removes the key entirely);
* ``--skip-test`` models stay unverified and therefore unflagged.

These tests run end-to-end through ``apply_updates_to_file`` / ``main``,
writing real JSON files in ``tmp_path`` and re-reading them.  Only module
path constants and network-fetch entry points are redirected.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ALL_LEVELS = ("low", "medium", "high", "xhigh")


def _read(target: Path) -> dict[str, dict]:
    return json.loads(target.read_text())  # type: ignore[no-any-return]


def _entry(use_responses_api: bool | None = None, thinking: str | None = None) -> dict:
    e: dict = {
        "context_length": 400_000,
        "input_price_per_1M": 5.0,
        "output_price_per_1M": 30.0,
        "fc": True,
        "emb": False,
        "gen": True,
    }
    if thinking is not None:
        e["thinking"] = thinking
    if use_responses_api is not None:
        e["use_responses_api"] = use_responses_api
    return e


def _new_model(name: str, use_responses_api: bool, thinking: str | None = None) -> dict:
    return {
        "name": name,
        "context_length": 400_000,
        "input_price_per_1M": 5.0,
        "output_price_per_1M": 30.0,
        "source": "openai",
        "fc": True,
        "emb": False,
        "gen": True,
        "thinking": thinking,
        "use_responses_api": use_responses_api,
        "needs_pricing": False,
    }


class TestApplyNewModels:
    """New-model entries must carry the probed flag (and aliases inherit)."""

    def test_probed_flag_is_written_and_inherited_by_aliases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A passing probe writes the flag on the base and all aliases."""
        target = tmp_path / "MODEL_INFO.json"
        target.write_text("{}\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [], [_new_model("gpt-7", True, thinking="xhigh")], [], {}, dry_run=False
        )
        data = _read(target)
        assert data["gpt-7"]["use_responses_api"] is True
        for level in ALL_LEVELS:
            assert data[f"gpt-7-{level}"]["use_responses_api"] is True
            assert data[f"gpt-7-{level}"]["alias_of"] == "gpt-7"

    def test_failed_probe_leaves_the_entry_unflagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing probe must not write the key at all (compact catalog)."""
        target = tmp_path / "MODEL_INFO.json"
        target.write_text("{}\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [], [_new_model("chat-only-model", False)], [], {}, dry_run=False
        )
        data = _read(target)
        assert "use_responses_api" not in data["chat-only-model"]


class TestApplyUpdates:
    """Flag flips arriving as ``changes`` must be applied in both directions."""

    def test_true_change_adds_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """changes={'use_responses_api': True} writes the key."""
        target = tmp_path / "MODEL_INFO.json"
        initial = {"gpt-7": _entry()}
        target.write_text(json.dumps(initial) + "\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [{"name": "gpt-7", "changes": {"use_responses_api": True}}],
            [],
            [],
            dict(initial),
            dry_run=False,
        )
        assert _read(target)["gpt-7"]["use_responses_api"] is True

    def test_false_change_removes_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """changes={'use_responses_api': False} removes the key entirely."""
        target = tmp_path / "MODEL_INFO.json"
        initial = {"gpt-7": _entry(use_responses_api=True)}
        target.write_text(json.dumps(initial) + "\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [{"name": "gpt-7", "changes": {"use_responses_api": False}}],
            [],
            [],
            dict(initial),
            dry_run=False,
        )
        assert "use_responses_api" not in _read(target)["gpt-7"]


def test_main_test_existing_flips_the_flag_from_probe_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--test-existing`` records a v2-support change and updates the entry."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-7": _entry(), "chat-only-model": _entry(use_responses_api=True)}
    target.write_text(json.dumps(initial, indent=2) + "\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    for fetcher in (
        "fetch_openrouter",
        "fetch_together",
        "fetch_anthropic",
        "fetch_gemini",
        "fetch_openai",
        "fetch_bedrock",
    ):
        monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))

    def probe(name: str, verbose: bool = False, decisions: bool = False) -> dict[str, object]:
        return {
            "gen": True,
            "emb": False,
            "fc": True,
            "thinking": None,
            "use_responses_api": name == "gpt-7",
        }

    monkeypatch.setattr(mod, "test_model_capabilities", probe)
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

    mod.main()

    output = capsys.readouterr().out
    assert "gpt-7: use_responses_api changed False -> True" in output
    assert "chat-only-model: use_responses_api changed True -> False" in output
    data = _read(target)
    assert data["gpt-7"]["use_responses_api"] is True
    assert "use_responses_api" not in data["chat-only-model"]


class TestSafeArithmetic:
    """The probe calculator must never execute model-controlled code."""

    def test_plain_arithmetic_is_evaluated(self) -> None:
        """Simple numeric expressions produce their value."""
        from kiss.scripts.update_models import safe_arithmetic

        assert safe_arithmetic("25*4") == "100"
        assert safe_arithmetic("2+3") == "5"
        assert safe_arithmetic("-3 + 5") == "2"
        assert safe_arithmetic("7 // 2") == "3"
        assert safe_arithmetic("7 % 2") == "1"
        assert safe_arithmetic("1/4") == "0.25"

    def test_executable_expressions_are_rejected(self) -> None:
        """Imports, calls, names, attributes, strings and ** all fail."""
        from kiss.scripts.update_models import safe_arithmetic

        hostile = [
            "__import__('os').system('echo pwned')",
            "().__class__.__mro__",
            "open('/etc/passwd').read()",
            "os.environ",
            "'a' * 10",
            "2**10",
            "9**9**9**9",
            "[1,2][0]",
            "(lambda: 1)()",
            "",
        ]
        for expression in hostile:
            assert safe_arithmetic(expression) == "error", expression


class TestInconclusiveVerdicts:
    """Inconclusive probe outcomes must never flip a stored catalog flag."""

    def test_apply_results_preserves_flag_on_inconclusive(self) -> None:
        """An exhausted-transient verdict keeps an existing flag (and aliases)."""
        from kiss.scripts.update_responses_api_support import (
            ProbeResult,
            _apply_results,
        )

        data = {
            "gpt-7": _entry(use_responses_api=True, thinking="high"),
            "gpt-7-low": {
                **_entry(use_responses_api=True, thinking="low"),
                "alias_of": "gpt-7",
            },
            "gpt-8": _entry(),
        }
        results = [
            ProbeResult(
                "gpt-7",
                supported=False,
                detail="generation failed: TimeoutError",
                conclusive=False,
            ),
            ProbeResult(
                "gpt-8", supported=False, detail="probe timed out", conclusive=False
            ),
        ]
        flagged, unflagged = _apply_results(data, results)
        assert (flagged, unflagged) == (0, 0)
        assert data["gpt-7"]["use_responses_api"] is True
        assert data["gpt-7-low"]["use_responses_api"] is True
        assert "use_responses_api" not in data["gpt-8"]

    def test_apply_results_acts_on_conclusive_verdicts(self) -> None:
        """Definitive verdicts still add and remove flags, aliases included."""
        from kiss.scripts.update_responses_api_support import (
            ProbeResult,
            _apply_results,
        )

        data = {
            "gpt-7": _entry(use_responses_api=True, thinking="high"),
            "gpt-7-low": {
                **_entry(use_responses_api=True, thinking="low"),
                "alias_of": "gpt-7",
            },
            "gpt-8": _entry(),
        }
        results = [
            ProbeResult("gpt-7", supported=False, detail="400 not supported"),
            ProbeResult("gpt-8", supported=True, detail="ok"),
        ]
        flagged, unflagged = _apply_results(data, results)
        assert (flagged, unflagged) == (1, 2)
        assert "use_responses_api" not in data["gpt-7"]
        assert "use_responses_api" not in data["gpt-7-low"]
        assert data["gpt-8"]["use_responses_api"] is True

    def test_main_test_existing_ignores_inconclusive_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``None`` capability verdict must not schedule a flag change."""
        target = tmp_path / "MODEL_INFO.json"
        initial = {"gpt-7": _entry(use_responses_api=True)}
        target.write_text(json.dumps(initial, indent=2) + "\n")

        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
        for fetcher in (
            "fetch_openrouter",
            "fetch_together",
            "fetch_anthropic",
            "fetch_gemini",
            "fetch_openai",
            "fetch_bedrock",
        ):
            monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
        monkeypatch.setattr(
            mod, "fetch_codex_supported_slugs", lambda verbose=False: set()
        )
        monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))
        monkeypatch.setattr(
            mod,
            "test_model_capabilities",
            lambda name, verbose=False, decisions=False: {
                "gen": True,
                "emb": False,
                "fc": True,
                "thinking": None,
                "use_responses_api": None,
            },
        )
        monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

        mod.main()

        assert _read(target)["gpt-7"]["use_responses_api"] is True

    def test_main_test_existing_keeps_capabilities_through_endpoint_trouble(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A vendor 5xx, a refused connection or a missing key changes nothing in the catalog.

        ``gpt-7`` is routed to a local endpoint that answers 503 to every
        request; ``claude-7`` to a local port nobody listens on, so the
        Anthropic SDK's own connection error (not a builtin one) is what
        the probe catches; ``glm-7`` has no ``ZAI_API_KEY``.  None of the
        failures says anything about the models, so ``fc``, ``thinking``,
        the v2 flag and every generated effort alias must survive the run.
        """
        import dataclasses

        import kiss.scripts.update_models as mod
        from kiss.core import config as config_module
        from kiss.core.models import model_info

        target = tmp_path / "MODEL_INFO.json"
        target.write_text("{}\n")
        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
        mod.apply_updates_to_file(
            [],
            [
                _new_model("gpt-7", True, thinking="xhigh"),
                _new_model("claude-7", False, thinking="high"),
                _new_model("glm-7", True),
            ],
            [],
            {},
        )
        before = _read(target)
        assert {"gpt-7", "gpt-7-low", "gpt-7-xhigh", "claude-7", "claude-7-low", "glm-7"} <= set(
            before
        )

        for fetcher in (
            "fetch_openrouter",
            "fetch_together",
            "fetch_anthropic",
            "fetch_gemini",
            "fetch_openai",
            "fetch_bedrock",
        ):
            monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
        monkeypatch.setattr(
            mod, "fetch_codex_supported_slugs", lambda verbose=False: set()
        )
        monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(before))
        monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ZAI_API_KEY", "")
        monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

        with _http_status_server(503) as endpoint, _refused_port() as refused:
            providers = tuple(
                dataclasses.replace(p, base_url=f"{endpoint}/v1") if p.name == "openai" else p
                for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
            )
            monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS", providers)
            monkeypatch.setenv("ANTHROPIC_BASE_URL", refused)
            mod.main()

        assert _read(target) == before
        out = capsys.readouterr().out
        assert "changed" not in out
        assert "gpt-7: probe inconclusive" in out
        assert "claude-7: probe inconclusive" in out
        assert "glm-7: probe inconclusive" in out


class _StatusOnlyHandler(BaseHTTPRequestHandler):
    """Answer every request with one fixed HTTP status (a vendor outage stand-in)."""

    status = 503

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": {"message": "overloaded"}}')

    do_GET = do_POST  # noqa: N815 — BaseHTTPRequestHandler API

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@contextmanager
def _refused_port() -> Iterator[str]:
    """Hold a loopback TCP port nobody listens on; yield its URL.

    The socket stays bound (so no other process can take the port) but
    never listens, so every connection to it is refused at once -- the
    endpoint of a vendor that is down, with the real SDK transport.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        sock.close()


@contextmanager
def _http_status_server(status: int) -> Iterator[str]:
    """Serve ``status`` for every request on a loopback port; yield the base URL."""
    handler = type("Handler", (_StatusOnlyHandler,), {"status": status})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _probe_everything_as_supported(
    monkeypatch: pytest.MonkeyPatch, catalog: Path, during_probe: Callable[[], None]
) -> None:
    """Route the batch's live probe to a stub that runs ``during_probe`` and passes.

    ``probe_responses_support`` is the module's single network entry point;
    the scan loads the catalog before the pool starts, so ``during_probe``
    runs between the scan's read and its publish.
    """
    import kiss.scripts.update_responses_api_support as scan

    def probe(name: str, fc: bool = True) -> scan.ProbeResult:
        during_probe()
        return scan.ProbeResult(name=name, supported=True, detail="ok")

    monkeypatch.setattr(scan, "probe_responses_support", probe)
    monkeypatch.setattr(sys, "argv", ["responses", "--model-info", str(catalog), "--workers", "1"])


def test_responses_scan_keeps_an_update_committed_during_its_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A price change and a new model committed mid-scan survive the scan's publish."""
    import kiss.scripts.update_models as mod
    import kiss.scripts.update_responses_api_support as scan

    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(json.dumps({"gpt-7": _entry()}, indent=2) + "\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")

    def ordinary_update() -> None:
        mod.apply_updates_to_file(
            [{"name": "gpt-7", "changes": {"input_price_per_1M": 9.0}, "source": "openai"}],
            [_new_model("gpt-8", use_responses_api=False)],
            [],
            {"gpt-7": _entry()},
        )

    _probe_everything_as_supported(monkeypatch, catalog, ordinary_update)
    scan.main()

    written = _read(catalog)
    assert written["gpt-7"]["input_price_per_1M"] == 9.0
    assert written["gpt-7"]["use_responses_api"] is True
    assert "gpt-8" in written and "use_responses_api" not in written["gpt-8"]


_STUCK_BATCH_CHILD = """
import signal, sys, threading
from kiss.scripts import update_responses_api_support as scan

mode, target = sys.argv[1:3]
# A test run started in the background of a non-interactive shell
# inherits SIGINT ignored, and Python then installs no handler for it;
# the Ctrl-C under test needs the one an interactive run has.
signal.signal(signal.SIGINT, signal.default_int_handler)

def interrupt_per_stdin_line():
    # Windows: the test cannot send SIGINT (Popen.send_signal refuses it,
    # and a console Ctrl-C event reaches every process of the console),
    # so each line it writes here raises the SIGINT from a thread, which
    # lands in the main thread as a console Ctrl-C does.
    for _line in sys.stdin:
        signal.raise_signal(signal.SIGINT)

if sys.platform == "win32":
    threading.Thread(target=interrupt_per_stdin_line, daemon=True).start()

def stuck(name, fc=True):
    print("PROBE_STARTED", flush=True)
    threading.Event().wait()

def interrupt_at_shutdown(frame, event, arg):
    # A real SIGINT the moment main() enters _shutdown_pool, i.e. after
    # the batch was published and before the stuck worker is dealt with.
    if frame.f_code is scan._shutdown_pool.__code__:
        sys.settrace(None)
        signal.raise_signal(signal.SIGINT)
    return None

class Candidates(list):
    # Yields its one name, then never ends: main() is still inside its
    # submission loop, the first probe already running, when the
    # interrupt lands.
    def __iter__(self):
        yield from list.__iter__(self)
        threading.Event().wait()

candidates = Candidates(["gpt-7"]) if mode == "interrupt-submit" else ["gpt-7"]
scan.probe_responses_support = stuck
scan._candidate_names = lambda data, only: (candidates, [])
scan._BATCH_WALL_CLOCK_SECONDS = 3600.0 if mode.startswith("interrupt") else 0.05
sys.argv = ["responses", "--model-info", target, "--workers", "1"]
if mode == "write-error":
    sys.argv += ["--results-json", target + "-no-such-dir/results.json"]
if mode == "shutdown-interrupt":
    sys.settrace(interrupt_at_shutdown)
scan.main()
"""


class TestStuckBatchTerminates:
    """A worker stuck in a vendor stream must never keep the process alive.

    The pool's workers are non-daemon threads, so interpreter shutdown
    would join a stuck one forever; every exit path, not just the happy
    one, has to bypass that join.  Each scenario runs the real ``main`` in
    a subprocess with a probe that blocks forever.
    """

    def _run(self, tmp_path: Path, mode: str) -> tuple[int, str, str]:
        catalog = tmp_path / f"{mode}.json"
        catalog.write_text(json.dumps({"gpt-7": _entry()}) + "\n")
        env = {**os.environ, "KISS_HOME": str(tmp_path / "home")}
        # The child's stdin, written by hand: a line on it is the Ctrl-C on
        # Windows (see the script), and ``communicate`` would close a
        # ``stdin=PIPE`` on its first call, before a second line is due.
        trigger_read, trigger_write = os.pipe()
        # No -u: stdout is a block-buffered pipe, as in a real piped run;
        # what the test reads is flushed explicitly or by the hard exits.
        child = subprocess.Popen(
            [sys.executable, "-c", _STUCK_BATCH_CHILD, mode, str(catalog)],
            stdin=trigger_read,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        os.close(trigger_read)
        try:
            assert child.stdout is not None
            while True:
                line = child.stdout.readline()
                assert line, "child exited before its probe started"
                if "PROBE_STARTED" in line:
                    break
            if mode == "broken-pipe":
                child.stdout.close()
                # ``communicate`` reads the pipes left: its Windows reader
                # thread would fail on the closed one.
                child.stdout = None
            # CPython acts on a signal only when the main thread next runs
            # bytecode; one that lands in the moment between its last
            # bytecode and the futex wait of the batch deadline is noticed
            # when that wait returns (observed in about 4% of runs under
            # load).  So a Ctrl-C the child did not act on within a second
            # is sent again, as a user would press it again.  On Windows
            # the child raises it per line on its stdin (see the script).
            interrupt = mode.startswith("interrupt")
            for _ in range(20 if interrupt else 1):
                if interrupt and sys.platform == "win32":
                    os.write(trigger_write, b"\n")
                elif interrupt:
                    child.send_signal(signal.SIGINT)
                try:
                    out, err = child.communicate(timeout=1 if interrupt else 20)
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                child.kill()
                out, err = child.communicate(timeout=10)
                pytest.fail(f"{mode}: process still alive with a stuck worker\n{out}\n{err}")
        finally:
            os.close(trigger_write)
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
        return child.returncode, out, err

    def test_deadline_exits_partial(self, tmp_path: Path) -> None:
        """Deadline reached, publish fine: status 2 and the catalog written."""
        code, out, _err = self._run(tmp_path, "control")
        assert code == 2
        assert "probe timed out (inconclusive)" in out
        assert "Written to" in out

    def test_publication_failure_after_deadline_exits(self, tmp_path: Path) -> None:
        """A failed --results-json write after the deadline still exits, traceback included."""
        code, out, err = self._run(tmp_path, "write-error")
        assert code == 1
        assert "FileNotFoundError" in err
        assert "Written to" not in out

    def test_interrupt_during_stuck_probe_exits(self, tmp_path: Path) -> None:
        """Ctrl-C while a probe is stuck ends the process with the interrupt visible."""
        code, _out, err = self._run(tmp_path, "interrupt")
        assert code == 1
        assert "KeyboardInterrupt" in err

    def test_interrupt_during_submission_exits(self, tmp_path: Path) -> None:
        """Ctrl-C while probes are still being submitted, the first already stuck, ends too.

        Submission starts workers before the batch is waited on, so an
        interrupt that lands there has a running worker to abandon as
        well.
        """
        code, _out, err = self._run(tmp_path, "interrupt-submit")
        assert code == 1
        assert "KeyboardInterrupt" in err

    def test_interrupt_after_publication_exits(self, tmp_path: Path) -> None:
        """Ctrl-C after the batch was published, as the pool is shut down, ends too.

        The child raises the SIGINT itself the moment ``_shutdown_pool`` is
        entered, so it lands deterministically after publication and
        before the stuck worker is abandoned.
        """
        code, out, err = self._run(tmp_path, "shutdown-interrupt")
        assert code == 1
        assert "KeyboardInterrupt" in err
        assert "Written to" in out

    def test_broken_stdout_pipe_still_exits(self, tmp_path: Path) -> None:
        """A stdout reader that went away must not keep the process alive.

        The publication print fails with ``BrokenPipeError`` (on Windows
        the C runtime reports the dead pipe as ``EINVAL``); the failed
        flush keeps the buffered bytes, so the error path's own
        ``sys.stdout.flush()`` raises again and must not get in the way
        of the hard exit.
        """
        code, _out, err = self._run(tmp_path, "broken-pipe")
        assert code == 1
        dead_pipe = "BrokenPipeError"
        if sys.platform == "win32":
            dead_pipe = "OSError: [Errno 22] Invalid argument"
        assert dead_pipe in err, err


class TestResponsesApiProbe:
    """test_responses_api / probe_responses_support end-to-end behavior."""

    def test_model_without_openai_compatible_endpoint_reports_false(self) -> None:
        """Anthropic/Gemini-native names have no /v1/responses to probe."""
        from kiss.scripts.update_models import test_responses_api
        from kiss.scripts.update_responses_api_support import probe_responses_support

        assert test_responses_api("claude-test-model", fc=True) is False
        result = probe_responses_support("gemini-test-model")
        assert result.supported is False
        assert result.detail == "no OpenAI-compatible endpoint"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_live_probe_matches_documented_support(self) -> None:
        """Live: a Responses-capable model passes, a chat-only model fails."""
        from kiss.scripts.update_models import test_responses_api

        assert test_responses_api("gpt-4o-mini", fc=True) is True
        assert test_responses_api("gpt-4o-mini-search-preview", fc=False) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
