# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Moonshot/Kimi reasoning-effort alias generation.

Kimi K3 accepts a vendor-specific ``reasoning_effort`` scale — ``low`` /
``high`` / ``max`` (vendor default ``max``; ``medium`` and ``xhigh`` are
rejected; thinking cannot be disabled).  ``update_models.py`` used to
hard-gate thinking-level detection to OpenAI-family models and assume the
OpenAI ladder (``low``/``medium``/``high``/``xhigh``), so ``kimi-k3``
never received ``-low`` / ``-high`` / ``-max`` catalog aliases.

The fixed script must:

* select the Moonshot scale for ``kimi-*`` / ``moonshot-*`` /
  ``moonshotai/*`` / ``openrouter/moonshotai/*`` names,
* probe those models over the wire in descending scale order
  (``max`` → ``high`` → ``low``),
* materialize ``-low`` / ``-high`` / ``-max`` aliases (never ``-medium``
  or ``-xhigh``) around a base entry capped at ``thinking="high"``.

These tests run end-to-end through ``apply_updates_to_file`` / ``main`` /
``detect_thinking_level``, writing real JSON files in ``tmp_path`` and —
for the wire test — speaking real HTTP to an in-process Moonshot
endpoint emulator.  No mocks or fakes wrap the code under test; only
module path constants and network-fetch entry points are redirected.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MOONSHOT_LEVELS = ("low", "high", "max")
OPENAI_LEVELS = ("low", "medium", "high", "xhigh")


def _redirect_model_info(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Point ``update_models.MODEL_INFO_PATH`` at ``target`` for the test."""
    import kiss.scripts.update_models as mod

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("{}\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)


def _read(target: Path) -> dict[str, dict]:
    return json.loads(target.read_text())  # type: ignore[no-any-return]


def _entry(
    thinking: str | None = None,
    alias_of: str | None = None,
    ctx: int = 500_000,
    inp: float = 3.0,
    out: float = 15.0,
) -> dict:
    e: dict = {
        "context_length": ctx,
        "input_price_per_1M": inp,
        "output_price_per_1M": out,
        "fc": True,
        "emb": False,
        "gen": True,
    }
    if thinking is not None:
        e["thinking"] = thinking
    if alias_of is not None:
        e["alias_of"] = alias_of
    return e


def _new_model(name: str, thinking: str | None) -> dict:
    return {
        "name": name,
        "context_length": 500_000,
        "input_price_per_1M": 3.0,
        "output_price_per_1M": 15.0,
        "source": "moonshot",
        "fc": True,
        "emb": False,
        "gen": True,
        "thinking": thinking,
        "needs_pricing": False,
    }


class TestThinkingScaleSelection:
    """The vendor scale must be chosen from the model-name prefix."""

    def test_moonshot_prefixes_use_low_high_max(self) -> None:
        import kiss.scripts.update_models as mod

        for name in (
            "kimi-k3",
            "kimi-k2-0905-preview",
            "moonshot-v1-128k",
            "moonshotai/Kimi-K3",
            "openrouter/moonshotai/kimi-k3",
            "openrouter/~moonshotai/kimi-k3",
        ):
            assert mod._thinking_scale_for(name) == MOONSHOT_LEVELS, name

    def test_other_models_keep_openai_ladder(self) -> None:
        import kiss.scripts.update_models as mod

        for name in (
            "gpt-5.6-sol",
            "o4-mini",
            "openrouter/openai/gpt-5.5",
            "claude-opus-4-7",
            "gemini-3.6-flash",
            "glm-4.6",
        ):
            assert mod._thinking_scale_for(name) == OPENAI_LEVELS, name


def test_new_kimi_k3_with_max_emits_low_high_max_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary regression: kimi-k3 with thinking=max must be expanded.

    Before the fix this exact flow wrote a single bare ``kimi-k3`` entry
    (with a raw ``thinking="max"``) and no aliases at all.
    """
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("kimi-k3", "max")], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k3", "kimi-k3-low", "kimi-k3-high", "kimi-k3-max"}
    assert data["kimi-k3"]["thinking"] == "high", "Base must be capped at high"
    assert "alias_of" not in data["kimi-k3"]
    for level in MOONSHOT_LEVELS:
        alias = data[f"kimi-k3-{level}"]
        assert alias["thinking"] == level
        assert alias["alias_of"] == "kimi-k3"
        for field in (
            "context_length",
            "input_price_per_1M",
            "output_price_per_1M",
            "fc",
            "emb",
            "gen",
            "comment",
        ):
            assert alias[field] == data["kimi-k3"][field], (
                f"-{level} alias must inherit {field}"
            )
    assert "kimi-k3-medium" not in data, "medium is not on the Moonshot scale"
    assert "kimi-k3-xhigh" not in data, "xhigh is not on the Moonshot scale"


def test_new_kimi_with_high_emits_low_high_aliases_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Kimi model whose probed max is ``high`` must not get ``-max``."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("kimi-k2-thinking", "high")], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k2-thinking", "kimi-k2-thinking-low", "kimi-k2-thinking-high"}
    assert data["kimi-k2-thinking"]["thinking"] == "high"
    for level in ("low", "high"):
        assert data[f"kimi-k2-thinking-{level}"]["thinking"] == level
        assert data[f"kimi-k2-thinking-{level}"]["alias_of"] == "kimi-k2-thinking"


def test_kimi_with_off_scale_level_writes_plainly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A level outside the Moonshot scale (e.g. medium) must not split."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("kimi-k3", "medium")], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k3"}
    assert "kimi-k3-medium" not in data


def test_openrouter_and_together_kimi_expand_on_moonshot_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-routed Kimi K3 rows must expand on the same scale."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file(
        [],
        [
            _new_model("openrouter/moonshotai/kimi-k3", "max"),
            _new_model("moonshotai/Kimi-K3", "max"),
        ],
        [],
        {},
        dry_run=False,
    )

    data = _read(target)
    for base in ("openrouter/moonshotai/kimi-k3", "moonshotai/Kimi-K3"):
        assert data[base]["thinking"] == "high"
        for level in MOONSHOT_LEVELS:
            assert data[f"{base}-{level}"]["thinking"] == level
            assert data[f"{base}-{level}"]["alias_of"] == base
        assert f"{base}-medium" not in data
        assert f"{base}-xhigh" not in data


def test_retest_downgrade_from_max_drops_only_max_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retest max→high must drop ``-max`` but keep ``-low``/``-high``."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"kimi-k3": _entry(thinking="high")}
    initial.update(
        {f"kimi-k3-{lvl}": _entry(thinking=lvl, alias_of="kimi-k3") for lvl in MOONSHOT_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [{"name": "kimi-k3", "changes": {"thinking": "high"}, "source": "retest"}]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k3", "kimi-k3-low", "kimi-k3-high"}
    assert data["kimi-k3"]["thinking"] == "high"


def test_retest_thinking_none_removes_all_moonshot_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retest reporting no thinking support must remove ``-max`` too."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"kimi-k3": _entry(thinking="high")}
    initial.update(
        {f"kimi-k3-{lvl}": _entry(thinking=lvl, alias_of="kimi-k3") for lvl in MOONSHOT_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [{"name": "kimi-k3", "changes": {"thinking": None}, "source": "retest"}]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k3"}
    assert "thinking" not in data["kimi-k3"]


def test_routine_update_trusts_existing_max_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pricing-only update must keep ``-max`` and sync fields everywhere."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"kimi-k3": _entry(thinking="high")}
    initial.update(
        {f"kimi-k3-{lvl}": _entry(thinking=lvl, alias_of="kimi-k3") for lvl in MOONSHOT_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [
        {
            "name": "kimi-k3",
            "changes": {"input_price_per_1M": 2.5, "context_length": 400_000},
            "source": "openrouter",
        }
    ]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k3"} | {f"kimi-k3-{lvl}" for lvl in MOONSHOT_LEVELS}, (
        "A routine update is no evidence of lost max support"
    )
    for name in data:
        assert data[name]["input_price_per_1M"] == 2.5
        assert data[name]["context_length"] == 400_000
    assert data["kimi-k3-max"]["thinking"] == "max"


def test_deprecating_kimi_base_removes_max_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deprecation must sweep the ``-max`` alias with its base."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"kimi-k3": _entry(thinking="high")}
    initial.update(
        {f"kimi-k3-{lvl}": _entry(thinking=lvl, alias_of="kimi-k3") for lvl in MOONSHOT_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    deprecated = [{"name": "kimi-k3", "reason": "removed upstream"}]
    mod.apply_updates_to_file([], [], deprecated, dict(initial), dry_run=False)

    assert _read(target) == {}


def test_moonshot_alias_generation_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running normalization on an expanded Kimi catalog changes nothing."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("kimi-k3", "max")], [], {}, dry_run=False)
    first = _read(target)
    assert not mod._has_thinking_normalization_changes(first), (
        "Freshly written catalog must already be normalized"
    )
    mod.apply_updates_to_file([], [], [], {}, dry_run=False)
    second = _read(target)
    assert second == first
    assert "kimi-k3-max-max" not in second
    assert "kimi-k3-high-high" not in second


def test_normalization_expands_base_with_max_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """base(high) + marked ``-max`` sibling must regenerate ``-low``/``-high``."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "kimi-k3": _entry(thinking="high"),
        "kimi-k3-max": _entry(thinking="max", alias_of="kimi-k3"),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"kimi-k3"} | {f"kimi-k3-{lvl}" for lvl in MOONSHOT_LEVELS}
    for level in MOONSHOT_LEVELS:
        assert data[f"kimi-k3-{level}"]["thinking"] == level
        assert data[f"kimi-k3-{level}"]["alias_of"] == "kimi-k3"


def test_find_deprecated_models_skips_marked_moonshot_aliases() -> None:
    """Generated Kimi aliases must never be probed as upstream names."""
    import kiss.scripts.update_models as mod

    current: dict[str, dict] = {
        "openrouter/moonshotai/kimi-k3": {"source": "openrouter", "alias_of": None},
        "openrouter/moonshotai/kimi-k3-low": {
            "source": "openrouter",
            "alias_of": "openrouter/moonshotai/kimi-k3",
        },
        "openrouter/moonshotai/kimi-k3-max": {
            "source": "openrouter",
            "alias_of": "openrouter/moonshotai/kimi-k3",
        },
    }
    openrouter = {"openrouter/moonshotai/kimi-k3": {"source": "openrouter"}}

    deprecated = mod.find_deprecated_models(current, openrouter, {}, {}, {})

    assert deprecated == [], (
        "Generated aliases are managed with their base entries and must not "
        "be checked as independent upstream model names"
    )


class _MoonshotHandler(BaseHTTPRequestHandler):
    """Emulates api.moonshot.ai/v1/chat/completions for kimi-k3.

    Accepts ``reasoning_effort`` values in ``accepted_efforts`` (and
    requests without the field), rejects every other value with HTTP 400
    — exactly Moonshot's documented K3 behavior. Tests may shrink
    ``accepted_efforts`` to exercise the probe's descending fallback.
    """

    captured_efforts: list[object] = []
    captured_bodies: list[dict] = []
    accepted_efforts: tuple[str, ...] = ("low", "high", "max")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        effort = body.get("reasoning_effort")
        self.__class__.captured_efforts.append(effort)
        self.__class__.captured_bodies.append(body)
        if effort is not None and effort not in self.__class__.accepted_efforts:
            payload = json.dumps(
                {
                    "error": {
                        "message": f"Invalid value for reasoning_effort: {effort}",
                        "type": "invalid_request_error",
                    }
                }
            ).encode()
            self.send_response(400)
        elif body.get("stream"):
            chunk = {
                "id": "cmpl-k3",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "kimi-k3",
                "choices": [
                    {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                ],
            }
            done = {
                "id": "cmpl-k3",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
            payload = (
                f"data: {json.dumps(chunk)}\n\n"
                f"data: {json.dumps(done)}\n\n"
                "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        else:
            payload = json.dumps(
                {
                    "id": "cmpl-k3",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "kimi-k3",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def moonshot_wire(monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    """Route the registered moonshot vendor at an in-process K3 emulator."""
    import dataclasses

    from kiss.core import config as config_module
    from kiss.core.models import model_info

    _MoonshotHandler.captured_efforts = []
    _MoonshotHandler.captured_bodies = []
    _MoonshotHandler.accepted_efforts = ("low", "high", "max")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MoonshotHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    providers = tuple(
        dataclasses.replace(p, base_url=base_url) if p.name == "moonshot" else p
        for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
    )
    monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS", providers)
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "MOONSHOT_API_KEY", "test-key", raising=False
    )
    try:
        yield base_url
    finally:
        server.shutdown()


class TestDetectThinkingLevelMoonshot:
    """detect_thinking_level must probe Moonshot models on the K3 scale."""

    def test_kimi_k3_detects_max_over_the_wire(self, moonshot_wire: str) -> None:
        """Against a faithful K3 endpoint the detected level must be max.

        Before the fix ``detect_thinking_level("kimi-k3")`` returned None
        without a single HTTP request (OpenAI-only gate), and even without
        the gate it would have reported ``high`` because ``max`` was not
        in the probe ladder.
        """
        import kiss.scripts.update_models as mod

        level = mod.detect_thinking_level("kimi-k3")

        assert level == "max"
        assert _MoonshotHandler.captured_efforts == ["max"], (
            "The probe must try the vendor's top level first and stop on success"
        )

    def test_probe_falls_back_to_high_when_max_rejected(self, moonshot_wire: str) -> None:
        """A K3 endpoint rejecting ``max`` must yield ``high`` on retry."""
        import kiss.scripts.update_models as mod

        _MoonshotHandler.accepted_efforts = ("low", "high")

        level = mod.detect_thinking_level("kimi-k3")

        assert level == "high"
        assert _MoonshotHandler.captured_efforts == ["max", "high"], (
            "The probe must walk the vendor scale in descending order"
        )

    def test_probe_returns_none_when_every_level_rejected(self, moonshot_wire: str) -> None:
        """An endpoint rejecting the whole scale must yield None."""
        import kiss.scripts.update_models as mod

        _MoonshotHandler.accepted_efforts = ()

        level = mod.detect_thinking_level("kimi-k3")

        assert level is None
        assert _MoonshotHandler.captured_efforts == ["max", "high", "low"], (
            "Every level of the Moonshot scale (and only those) must be tried"
        )

    def test_non_k3_moonshot_models_are_gated_out_without_http(
        self, moonshot_wire: str
    ) -> None:
        """K2.x / moonshot-v1 models must be skipped with zero requests.

        ``reasoning_effort`` is a K3-introduced API surface; older Moonshot
        models control thinking differently (or not at all), and probing
        them through a gateway that drops unknown params would fabricate
        alias levels.
        """
        import kiss.scripts.update_models as mod

        for name in (
            "kimi-k2-0905-preview",
            "kimi-k2.6",
            "moonshot-v1-128k",
            "moonshotai/Kimi-K2-Thinking",
            "openrouter/moonshotai/kimi-k2-thinking",
        ):
            assert mod.detect_thinking_level(name) is None, name
        assert _MoonshotHandler.captured_efforts == []

    def test_non_reasoning_gates_still_skip_without_http(self) -> None:
        """Gated-out families must return None without any network call."""
        import kiss.scripts.update_models as mod

        _MoonshotHandler.captured_efforts = []
        for name in ("claude-opus-4-7", "gemini-3.6-flash", "codex/gpt-5.5", "glm-4.6"):
            assert mod.detect_thinking_level(name) is None, name
        assert _MoonshotHandler.captured_efforts == []


def test_main_discovers_new_kimi_and_writes_moonshot_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full main() flow: a newly-fetched Kimi model probed at thinking=max
    must land in MODEL_INFO.json with ``-low``/``-high``/``-max`` aliases.
    """
    target = tmp_path / "MODEL_INFO.json"
    target.write_text("{}\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    openrouter = {
        "openrouter/moonshotai/kimi-k4": {
            "context_length": 500_000,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 15.0,
            "source": "openrouter",
        }
    }
    monkeypatch.setattr(mod, "fetch_openrouter", lambda verbose=False: dict(openrouter))
    monkeypatch.setattr(mod, "fetch_together", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_anthropic", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_gemini", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_openai", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: {})

    def probe(name: str, verbose: bool = False) -> dict[str, object]:
        return {"gen": True, "emb": False, "fc": True, "thinking": "max"}

    monkeypatch.setattr(mod, "test_model_capabilities", probe)
    monkeypatch.setattr(sys, "argv", ["update_models.py"])

    mod.main()

    data = _read(target)
    base = "openrouter/moonshotai/kimi-k4"
    assert data[base]["thinking"] == "high"
    for level in MOONSHOT_LEVELS:
        alias = data[f"{base}-{level}"]
        assert alias["thinking"] == level
        assert alias["alias_of"] == base
    assert f"{base}-medium" not in data
    assert f"{base}-xhigh" not in data


def test_main_test_existing_promotes_stored_max_from_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--test-existing`` must treat base(high)+``-max`` as stored max.

    When the probe re-confirms ``max``, no ``thinking changed`` update may
    be emitted (the on-disk state already encodes max support).
    """
    target = tmp_path / "MODEL_INFO.json"
    initial = {"kimi-k3": _entry(thinking="high")}
    initial.update(
        {f"kimi-k3-{lvl}": _entry(thinking=lvl, alias_of="kimi-k3") for lvl in MOONSHOT_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    monkeypatch.setattr(mod, "fetch_openrouter", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_together", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_anthropic", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_gemini", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_openai", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))
    calls: list[str] = []

    def record_probe(name: str, verbose: bool = False) -> dict[str, object]:
        calls.append(name)
        return {"gen": True, "emb": False, "fc": True, "thinking": "max"}

    monkeypatch.setattr(mod, "test_model_capabilities", record_probe)
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

    mod.main()

    output = capsys.readouterr().out
    assert calls == ["kimi-k3"], f"Only the base must be probed, got {calls}"
    assert "thinking changed" not in output
    data = _read(target)
    assert set(data) == set(initial)
    for level in MOONSHOT_LEVELS:
        assert data[f"kimi-k3-{level}"]["thinking"] == level


def test_bundled_catalog_ships_kimi_k3_level_aliases() -> None:
    """The shipped MODEL_INFO.json must expose the K3 per-level aliases.

    This pins the user-visible outcome: ``kimi-k3-low`` / ``kimi-k3-high``
    / ``kimi-k3-max`` are selectable model names in the bundled catalog,
    and no OpenAI-only levels leak onto the Moonshot rows.
    """
    import kiss.scripts.update_models as mod

    data = json.loads(
        (Path(mod.PROJECT_ROOT) / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json")
        .read_text(encoding="utf-8")
    )
    for base in ("kimi-k3", "openrouter/moonshotai/kimi-k3"):
        assert data[base]["thinking"] == "high", base
        for level in MOONSHOT_LEVELS:
            name = f"{base}-{level}"
            assert name in data, f"{name} missing from bundled catalog"
            assert data[name]["thinking"] == level
            assert data[name]["alias_of"] == base
        assert f"{base}-medium" not in data
        assert f"{base}-xhigh" not in data
    assert not mod._has_thinking_normalization_changes(data), (
        "Bundled catalog must be fully normalized"
    )


class TestRuntimeAliasResolution:
    """Kimi K3 aliases must resolve and route like every other alias."""

    def test_strip_thinking_alias_maps_max_to_base(self) -> None:
        from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias

        for level in MOONSHOT_LEVELS:
            name = f"kimi-k3-{level}"
            assert name in MODEL_INFO, f"{name} missing from loaded MODEL_INFO"
            assert _strip_thinking_alias(name) == "kimi-k3"

    def test_provider_model_name_and_effort_for_max_alias(self) -> None:
        from kiss.core.models.openai_compatible_model import (
            _model_thinking_level,
            _provider_model_name,
        )

        assert _provider_model_name("kimi-k3-max") == "kimi-k3"
        assert _model_thinking_level("kimi-k3-max") == "max"
        assert (
            _provider_model_name("openrouter/moonshotai/kimi-k3-max")
            == "moonshotai/kimi-k3"
        )
        assert _model_thinking_level("openrouter/moonshotai/kimi-k3-max") == "max"

    def test_unmarked_max_suffix_is_not_stripped(self) -> None:
        """A name ending in -max with no catalog alias marker stays intact."""
        from kiss.core.models.model_info import _strip_thinking_alias

        assert _strip_thinking_alias("acme/custom-max") == "acme/custom-max"

    def test_factory_builds_kimi_max_alias_with_effort(self, moonshot_wire: str) -> None:
        """model('kimi-k3-max') must send model=kimi-k3, effort=max."""
        from kiss.core.models.model_info import model as create_model

        m = create_model("kimi-k3-max")
        assert m.model_config.get("reasoning_effort") == "max"
        assert getattr(m, "_api_model_name", None) == "kimi-k3"

    def test_alias_generate_sends_base_model_and_max_on_the_wire(
        self, moonshot_wire: str
    ) -> None:
        """A real generate() through the alias must carry both wire fields."""
        from kiss.core.models.model_info import model as create_model

        m = create_model("kimi-k3-max")
        m.initialize("Say hello in one word.")
        text, _ = m.generate()

        assert text.strip() == "hello"
        assert len(_MoonshotHandler.captured_bodies) == 1
        body = _MoonshotHandler.captured_bodies[0]
        assert body["model"] == "kimi-k3", "The wire id must be the base model"
        assert body["reasoning_effort"] == "max"

    def test_kimi_aliases_cost_the_same_as_base(self) -> None:
        from kiss.core.models.model_info import calculate_cost

        base_cost = calculate_cost("kimi-k3", 1000, 500, 200, 100)
        for level in MOONSHOT_LEVELS:
            assert calculate_cost(f"kimi-k3-{level}", 1000, 500, 200, 100) == base_cost


ALL_KNOWN_LEVELS = ("low", "medium", "high", "xhigh", "max")


def test_every_bundled_alias_thinking_is_a_known_level() -> None:
    """Every marked alias in the bundled catalog uses a known level name."""
    from kiss.core.models.model_info import MODEL_INFO

    for name, info in MODEL_INFO.items():
        if info.alias_of:
            assert info.thinking in ALL_KNOWN_LEVELS, (name, info.thinking)
