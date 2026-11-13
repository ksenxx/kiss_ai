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
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
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
        the v2 flag, every generated effort alias and ``glm-7``'s ``emb``
        must survive the run.
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
                {**_new_model("glm-7", True), "emb": True},
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

    @pytest.mark.parametrize(
        ("status", "kept"), [(401, True), (402, True), (403, True), (404, False)]
    )
    def test_scan_keeps_flags_when_the_vendor_rejects_the_key(
        self, status: int, kept: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real scan clears no flag on a rejected key (401 / 403), every flag on a 404.

        Audit SC1: nor on a 402, the answer of an account out of credits (OpenRouter's
        ``payment_required``, Anthropic's ``billing_error``), which cleared them all.
        """
        import kiss.scripts.update_models as mod
        import kiss.scripts.update_responses_api_support as scan

        catalog = tmp_path / "MODEL_INFO.json"
        monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
        mod.apply_updates_to_file([], [_new_model("gpt-7", True, thinking="high")], [], {})
        before = _read(catalog)
        monkeypatch.setattr(sys, "argv", ["responses", "--model-info", str(catalog)])
        with _http_status_server(status) as endpoint:
            _use_local_openai(monkeypatch, endpoint)
            scan.main()
        flagged = [name for name, entry in _read(catalog).items() if "use_responses_api" in entry]
        assert flagged == (list(before) if kept else [])


class _StatusOnlyHandler(BaseHTTPRequestHandler):
    """Answer every request with one fixed HTTP status (a vendor outage stand-in)."""

    status = 503
    replies: dict[str, tuple[int, bytes]] = {}  # a path's own (status, body)
    seen: list[str]  # the path of every request, in arrival order

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.seen.append(self.path)
        default = (self.status, b'{"error": {"message": "overloaded"}}')
        status, body = self.replies.get(self.path, default)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

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
def _http_status_server(
    status: int, replies: dict | None = None, seen: list[str] | None = None
) -> Iterator[str]:
    """Serve ``status`` (a path in ``replies``: its entry) on a loopback port; yield the URL.

    ``seen``, when given, collects the path of every request.
    """
    fields = {"status": status, "replies": replies or {}, "seen": [] if seen is None else seen}
    handler = type("Handler", (_StatusOnlyHandler,), fields)
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _use_local_openai(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    """Route the OpenAI vendor, with a dummy key, to the local ``endpoint``.

    The registry fixes the vendor URL and OpenAI answers no 401, 403 or 503 on
    request; the SDK, adapters, probes and catalog writer stay the real ones.
    """
    import dataclasses

    from kiss.core import config as config_module
    from kiss.core.models import model_info

    providers = tuple(
        dataclasses.replace(p, base_url=f"{endpoint}/v1") if p.name == "openai" else p
        for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
    )
    monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS", providers)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENAI_API_KEY", "sk-test")


_CHAT_OK = b'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
_EMB_OK = b'{"data":[{"index":0,"embedding":[0.5]}]}'

# The real ``main()`` in a fresh interpreter, because the runtime catalog is read at import.
# It sets values only: the probes' URL and dummy key, and the URL of the three unkeyed listings.
_RETEST_CHILD = """
import dataclasses, sys
import kiss.scripts.update_models as mod
from kiss.core import config
from kiss.core.models import model_info

endpoint = sys.argv.pop()
model_info.OPENAI_COMPATIBLE_PROVIDERS = tuple(
    dataclasses.replace(p, base_url=endpoint + "/v1") if p.name == "openai" else p
    for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
)
config.DEFAULT_CONFIG.OPENAI_API_KEY = "sk-test"
mod.OPENROUTER_MODELS_URL = mod.OPENROUTER_DECISIONS_MODELS_URL = endpoint + "/listing"
mod._CODEX_MODELS_JSON_URL = endpoint + "/listing"
mod.main()
"""


def _keyless_env(home: Path, target: Path) -> dict[str, str]:
    """The environment of a ``_RETEST_CHILD`` run: no vendor key, its own home and catalog."""
    from kiss.core.user_config import API_KEY_ENV_VARS

    env = {name: value for name, value in os.environ.items() if name not in API_KEY_ENV_VARS}
    return {**env, "KISS_HOME": str(home), "KISS_MODEL_INFO_PATH": str(target)}


@pytest.mark.parametrize(
    ("chat", "embeddings", "stored", "expected", "v2"),
    [
        (404, 200, (True, False), (False, True), None),
        (200, 404, (False, True), (True, False), None),
        (503, 200, (True, False), (True, True), None),
        (200, 503, (False, True), (True, True), None),
        (200, 404, (True, False), (True, False), True),
        # Audit SC1: an account out of credits (402) is no verdict on the model either.
        (402, 200, (True, False), (True, True), None),
    ],
)
def test_main_test_existing_applies_generation_and_embedding_verdicts(
    chat: int,
    embeddings: int,
    stored: tuple[bool, bool],
    expected: tuple[bool, bool],
    v2: bool | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real retest writes each answered ``(gen, emb)`` probe; a 503 keeps the stored flag.

    The updater (``_RETEST_CHILD``) has no vendor key in its environment, so its keyed
    listings skip.  A new candidate whose embedding probe went unanswered gets false, never null.
    ``/v1/responses`` answers 404 throughout: a model stored on that route (``v2``) that Chat
    Completions still serves stays a generator and loses only its ``use_responses_api`` flag.
    Its tool-calling and thinking probes go over Chat too: the refused route gets one request.
    """
    import kiss.scripts.update_models as mod

    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-7": {**_entry(v2), "gen": stored[0], "emb": stored[1], "fc": False}}
    target.write_text(json.dumps(initial))
    env = _keyless_env(tmp_path / "home", target)
    retest = [sys.executable, "-c", _RETEST_CHILD, "--model-info", str(target), "--test-existing"]
    replies = {"/v1/chat/completions": (chat, _CHAT_OK), "/v1/embeddings": (embeddings, _EMB_OK)}
    seen: list[str] = []
    with _http_status_server(404, {**replies, "/listing": (200, b"{}")}, seen) as endpoint:
        subprocess.run([*retest, endpoint], env=env, check=True)
        if v2:
            assert seen.count("/v1/responses") == 1
        if embeddings == 503:
            _use_local_openai(monkeypatch, endpoint)
            candidates = mod._probe_new_models([_new_model("gpt-8", False)])
            assert [type(nm["emb"]) for nm in candidates] == [bool]
    written = _read(target)["gpt-7"]
    assert (written["gen"], written["emb"]) == expected
    assert "use_responses_api" not in written


def _probe_everything_as_supported(
    monkeypatch: pytest.MonkeyPatch, catalog: Path, during_probe: Callable[[], None]
) -> None:
    """Route the batch's live probe to a stub that runs ``during_probe`` and passes.

    ``probe_responses_support`` is the module's single network entry point;
    the scan loads the catalog before the pool starts, so ``during_probe``
    runs between the scan's read and its publish.  The scan skips a model
    whose vendor key is not configured, so an OpenAI key is: the batch is
    probed on a host that has none, too.
    """
    import kiss.scripts.update_responses_api_support as scan
    from kiss.core import config as config_module

    def probe(name: str, fc: bool = True, thinking: str | None = None) -> scan.ProbeResult:
        during_probe()
        return scan.ProbeResult(name=name, supported=True, detail="ok")

    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENAI_API_KEY", "sk-test")
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


_OWN_EFFORT = {"gpt-7": "low", "gpt-8": None}  # the only reasoning effort each model takes
_TEXT_ITEM = {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
_CALL_ITEM = {"type": "function_call", "call_id": "c", "name": "calculator", "arguments": "{}"}


def test_scan_probes_each_row_at_its_own_default_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The scan requests each target row's own effort (none when it has none), not the runtime's.

    The endpoint takes a model at its own effort only, as a vendor does.  The runtime catalog
    says ``high`` for both models, which a probe that is given no row still requests
    (``update_models.py`` probes the catalog it runs on).
    """
    import kiss.scripts.update_responses_api_support as scan
    from kiss.core.models import model_info
    from kiss.tests.core.models import openai_sse_harness as sse

    sent: set[tuple[str, str | None]] = set()

    def answer(request: sse.Request) -> sse.Reply:
        body = request.body
        effort = body.get("reasoning", {}).get("effort")
        sent.add((body["model"], effort))
        if effort != _OWN_EFFORT[body["model"]]:
            return sse.Reply(status=400, json_body={"error": {"message": "unsupported effort"}})
        answered = any(item.get("type") == "function_call_output" for item in body["input"])
        item = _CALL_ITEM if "tools" in body and not answered else _TEXT_ITEM
        done = sse.responses_event("response.completed", {"response": {"output": [item]}})
        return sse.Reply(sse_chunks=[done])

    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(json.dumps({"gpt-7": _entry(thinking="low"), "gpt-8": _entry()}))
    runtime_row = model_info._build_model_info_entry(_entry(thinking="high"))
    for name in _OWN_EFFORT:
        monkeypatch.setitem(model_info.MODEL_INFO, name, runtime_row)
    monkeypatch.setattr(sys, "argv", ["responses", "--model-info", str(catalog)])
    with sse.ScriptedOpenAIServer(answer) as server:
        _use_local_openai(monkeypatch, server.base_url.removesuffix("/v1"))
        scan.main()
        assert sent == set(_OWN_EFFORT.items())
        assert not scan.probe_responses_support("gpt-8").supported
        assert ("gpt-8", "high") in sent
    assert [row.get("use_responses_api") for row in _read(catalog).values()] == [True, True]
    assert f"Written to {catalog}" in capsys.readouterr().out


_STREAM_ERRORS = {  # the error each model's stream ends in
    "gpt-7": {"code": "server_error", "message": "An error occurred. You can retry your request."},
    "gpt-8": {"code": "rate_limit_exceeded", "message": "Rate limit reached for gpt-8."},
    "gpt-9": {"code": "invalid_prompt", "message": "The prompt is not valid."},
}
_STREAM_ENDINGS = ("response.failed", "error", "error-object")
"""Where a stream reports its error: in the failed response of ``response.failed``, in the
bare ``error`` event itself, or in a gateway's ``error`` object, which the SDK raises."""


def _stream_ending(shape: str, error: dict[str, str]) -> bytes:
    """Render the last event of a Responses stream that reports *error* as *shape* does."""
    from kiss.tests.core.models import openai_sse_harness as sse

    if shape == "response.failed":
        failed = {"status": "failed", "error": error, "output": []}
        return sse.responses_event(shape, {"response": failed})
    return sse.responses_event("error", error if shape == "error" else {"error": error})


@pytest.mark.parametrize("shape", _STREAM_ENDINGS)
def test_scan_keeps_the_flag_of_a_model_whose_stream_failed_on_the_server(
    shape: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stream the server fails is retried, then inconclusive; any other failure is a verdict.

    A ``server_error`` or a rate limit inside the stream cleared the flag like a rejection of
    the model, and the summary and the ``--results-json`` dump listed a kept flag as a model
    staying on Chat Completions.  Reported outside a failed response it still did.
    """
    import kiss.scripts.update_responses_api_support as scan
    from kiss.tests.core.models import openai_sse_harness as sse

    def answer(request: sse.Request) -> sse.Reply:
        return sse.Reply(sse_chunks=[_stream_ending(shape, _STREAM_ERRORS[request.body["model"]])])

    catalog = tmp_path / "MODEL_INFO.json"
    flagged_entries = {name: _entry(use_responses_api=True) for name in _STREAM_ERRORS}
    catalog.write_text(json.dumps(flagged_entries))
    dump = tmp_path / "results.json"
    monkeypatch.setattr(scan, "_RETRY_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(
        sys, "argv", ["responses", "--model-info", str(catalog), "--results-json", str(dump)]
    )
    with sse.ScriptedOpenAIServer(answer) as server:
        _use_local_openai(monkeypatch, server.base_url.removesuffix("/v1"))
        scan.main()
        asked = [request.body["model"] for request in server.requests]
    assert [asked.count(name) for name in _STREAM_ERRORS] == [3, 3, 1]
    flagged = [name for name, row in _read(catalog).items() if row.get("use_responses_api")]
    assert flagged == ["gpt-7", "gpt-8"]
    verdicts = {name: (r["supported"], r["conclusive"]) for name, r in _read(dump).items()}
    assert verdicts == {"gpt-7": (False, False), "gpt-8": (False, False), "gpt-9": (False, True)}
    out = capsys.readouterr().out
    assert "\nModels staying on Chat Completions (probe failed):\n  - gpt-9\n" in out
    assert "\nProbes inconclusive (flags unchanged):\n  - gpt-7\n  - gpt-8\n" in out


@pytest.mark.parametrize("shape", _STREAM_ENDINGS)
def test_main_test_existing_keeps_the_flags_of_a_model_whose_stream_failed_on_the_server(
    shape: str, tmp_path: Path
) -> None:
    """The retest reads a ``server_error`` inside the Responses stream as endpoint trouble.

    It took it for the route's refusal: the model generated over Chat Completions, which serves
    no tool call here, and lost ``use_responses_api`` and ``fc`` to a failure on OpenAI's side.
    """
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps({"gpt-7": _entry(True)}))
    replies = {
        "/v1/responses": (200, _stream_ending(shape, _STREAM_ERRORS["gpt-7"])),
        "/v1/chat/completions": (200, _CHAT_OK),
        "/listing": (200, b"{}"),
    }
    retest = [sys.executable, "-c", _RETEST_CHILD, "--model-info", str(target), "--test-existing"]
    with _http_status_server(404, replies) as endpoint:
        subprocess.run([*retest, endpoint], env=_keyless_env(tmp_path / "home", target), check=True)
    assert _read(target) == {"gpt-7": _entry(True)}


def test_scan_writes_no_catalog_it_did_not_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing catalog is refused, not created; a scan that flips no flag rewrites nothing."""
    import kiss.scripts.update_responses_api_support as scan

    missing = tmp_path / "typo" / "MODEL_INFO.json"
    monkeypatch.setattr(sys, "argv", ["responses", "--model-info", str(missing)])
    with pytest.raises(SystemExit, match="does not exist"):
        scan.main()
    assert not missing.parent.exists()

    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(compact := json.dumps({"claude-7": _entry()}))  # a rewrite would indent it
    monkeypatch.setattr(sys, "argv", ["responses", "--model-info", str(catalog)])
    scan.main()
    assert catalog.read_text() == compact
    assert "Written to" not in capsys.readouterr().out


def test_stale_new_candidate_keeps_the_entry_another_updater_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified entry published while a slower updater probed the model survives its candidate."""
    import kiss.scripts.update_models as mod

    catalog = tmp_path / "MODEL_INFO.json"
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    mod.apply_updates_to_file([], [_new_model("gpt-8", True, thinking="xhigh")], [], {})
    published = _read(catalog)
    stale = {**_new_model("gpt-8", False), "input_price_per_1M": 0.0, "needs_pricing": True}
    mod.apply_updates_to_file([], [stale, _new_model("gpt-9", False)], [], {})
    assert _read(catalog) == {**published, "gpt-9": {**_entry(), "comment": "NEW"}}


def test_stale_update_does_not_resurrect_a_model_another_updater_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An update computed before another updater retired the model, committed after, is dropped."""
    import kiss.scripts.update_models as mod

    catalog = tmp_path / "MODEL_INFO.json"
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    mod.apply_updates_to_file(
        [], [_new_model("gpt-7", True, thinking="xhigh"), _new_model("gpt-9", False)], [], {}
    )
    snapshot = _read(catalog)  # what the slower updater computed its update from
    mod.apply_updates_to_file([], [], [{"name": "gpt-7", "reason": "removed upstream"}], snapshot)
    retired = _read(catalog)
    assert list(retired) == ["gpt-9"]
    stale = {"name": "gpt-7", "changes": {"input_price_per_1M": 7.0, "thinking": "high"}}
    mod.apply_updates_to_file([stale], [], [], snapshot)
    assert _read(catalog) == retired


# One writer of the catalog ``target``.  "hold" is a read-modify-write cycle under the catalog
# lock that pauses after its read until a line arrives on stdin; "add" is an ordinary update.
_CATALOG_WRITER_CHILD = """
import faulthandler
import sys
from pathlib import Path

faulthandler.dump_traceback_later(300, exit=True)  # bounds the test's reads of this child's pipe
import kiss.scripts.update_models as mod

mode, target = sys.argv[1:]
mod.MODEL_INFO_PATH = Path(target)
entry = {"context_length": 1, "input_price_per_1M": 1.0, "output_price_per_1M": 1.0}
if mode == "hold":
    with mod.catalog_lock(mod.MODEL_INFO_PATH):
        data = mod._read_model_info_json(mod.MODEL_INFO_PATH)
        print("READ", flush=True)
        sys.stdin.readline()
        mod._write_model_info_json(mod.MODEL_INFO_PATH, {**data, "gpt-A": entry})
else:
    print("START", flush=True)
    mod.apply_updates_to_file([], [{"name": "gpt-B", **entry}], [], {})
"""


def test_writers_under_different_homes_share_the_catalog_lock(tmp_path: Path) -> None:
    """A writer with another ``KISS_HOME`` sits out the holder's cycle: both models survive."""
    target = tmp_path / "MODEL_INFO.json"
    target.write_text("{}")
    command = [sys.executable, "-c", _CATALOG_WRITER_CHILD]
    env_a, env_b = ({**os.environ, "KISS_HOME": str(tmp_path / home)} for home in "ab")
    holder = subprocess.Popen(
        [*command, "hold", str(target)],
        env=env_a,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    writer = None
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "READ"
        writer = subprocess.Popen(
            [*command, "add", str(target)], env=env_b, stdout=subprocess.PIPE, text=True
        )
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == "START"
        with suppress(subprocess.TimeoutExpired):
            writer.wait(timeout=2)  # blocked on the lock the holder took under another home
        holder.communicate("\n", timeout=60)
        writer.communicate(timeout=60)
        assert (holder.returncode, writer.returncode) == (0, 0)
    finally:
        for child in (holder, writer):
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    assert set(_read(target)) == {"gpt-A", "gpt-B"}


class _EndlessHandler(BaseHTTPRequestHandler):
    """A peer that never finishes: listing body bytes, header bytes or SSE deltas until released."""

    release: threading.Event

    def _send_until_released(self, piece: bytes) -> None:
        with suppress(OSError):  # the client hung up
            while not self.release.wait(0.01):
                self.wfile.write(piece)
                self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/slow-headers":
            self.wfile.write(b"HTTP/1.0 200 OK\r\nX-Pad: ")
            self._send_until_released(b"a")
            return
        self.send_response(200)
        self.end_headers()
        if self.path == "/listing":
            self.wfile.write(b"{}")
        else:
            self._send_until_released(b" ")

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._send_until_released(b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n')

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@contextmanager
def _endless_server() -> Iterator[str]:
    """Serve :class:`_EndlessHandler` on a loopback port for the block; yield its URL."""
    release = threading.Event()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), type("Handler", (_EndlessHandler,), {"release": release})
    )
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


_LISTING_CHILD = """
import sys
import time
from urllib.request import Request

import kiss.scripts.update_models as mod

mod._LISTING_DEADLINE_SECONDS = 0.5
url = sys.argv[1]
try:
    if url.endswith("/reader"):
        mod._fetch_json(Request(url), time.monotonic() + 0.5)
    else:
        mod.api_get(url)
except TimeoutError:
    print("LISTING CUT OFF")
"""


@pytest.mark.parametrize("path", ["/slow-body", "/slow-headers", "/reader"])
def test_listing_that_never_ends_is_cut_off(path: str, tmp_path: Path) -> None:
    """A listing peer trickling body or header bytes costs three bounded attempts, not forever.

    ``/reader`` is the body reader without its caller: left behind, it stops at the deadline too.
    """
    env = {**os.environ, "KISS_HOME": str(tmp_path / "home")}
    with _endless_server() as endpoint:
        child = subprocess.run(
            [sys.executable, "-c", _LISTING_CHILD, endpoint + path],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert (child.returncode, child.stdout.strip()) == (0, "LISTING CUT OFF"), child.stderr


def test_main_test_existing_survives_a_probe_stream_that_never_ends(tmp_path: Path) -> None:
    """The real updater against a vendor streaming deltas forever: cut off, capabilities kept."""
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps({"gpt-7": _entry()}))
    child = _RETEST_CHILD.replace("mod.main()", "mod._CAPABILITY_DEADLINE_SECONDS = 1\nmod.main()")
    with _endless_server() as endpoint:
        done = subprocess.run(
            [sys.executable, "-c", child, "--model-info", str(target), "--test-existing", endpoint],
            env=_keyless_env(tmp_path / "home", target),
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert done.returncode == 0, done.stderr
    assert "gpt-7: probe inconclusive, stored capabilities kept" in done.stdout
    assert _read(target) == {"gpt-7": _entry()}


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

def stuck(name, fc=True, thinking=None):
    print("PROBE_STARTED", flush=True)
    if mode == "dump-error":
        return scan.ProbeResult(name, True, "ok")
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
scan._BATCH_WALL_CLOCK_SECONDS = 3600.0 if mode.startswith(("interrupt", "dump")) else 0.05
sys.argv = ["responses", "--model-info", target, "--workers", "1"]
if mode in ("write-error", "dump-error"):
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
        """Deadline reached, publish fine: status 2 after the report (no verdict, no write)."""
        code, out, _err = self._run(tmp_path, "control")
        assert code == 2
        assert "probe timed out (inconclusive)" in out
        assert "Entries flagged: 0, flags removed: 0" in out

    def test_publication_failure_after_deadline_exits(self, tmp_path: Path) -> None:
        """A failed --results-json write after the deadline still exits, traceback included."""
        code, out, err = self._run(tmp_path, "write-error")
        assert code == 1
        assert "FileNotFoundError" in err
        assert "Written to" not in out

    def test_failed_results_dump_keeps_the_published_verdict(self, tmp_path: Path) -> None:
        """A ``--results-json`` that cannot be written costs the dump, not the catalog update."""
        code, _out, err = self._run(tmp_path, "dump-error")
        assert code == 1
        assert "FileNotFoundError" in err
        assert _read(tmp_path / "dump-error.json")["gpt-7"]["use_responses_api"] is True

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
        assert "Entries flagged: 0, flags removed: 0" in out

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

    @pytest.mark.live_api
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
