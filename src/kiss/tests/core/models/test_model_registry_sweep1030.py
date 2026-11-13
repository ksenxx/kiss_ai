# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: hand-written model registries cannot break model lookup.

Each test runs a fresh Python process with its own ``KISS_HOME`` (or
``KISS_MODEL_INFO_PATH``), because the catalog and ``MY_MODELS.json``
are merged when :mod:`kiss.core.models.model_info` is imported.

* A ``MY_MODELS.json`` entry (or a user ``MODEL_INFO.json`` entry) whose
  price is a string, ``null`` or a boolean used to pass the schema check
  and then raise ``TypeError`` in the import-time cache-pricing loop,
  failing every KISS process until the file was fixed by hand; one added
  after start raised the same error out of every cost lookup.  So did a
  ``-{level}`` entry whose ``alias_of`` is not a string (``AttributeError``
  in the same loop, which reads it as a model name).
* An entry whose ``fallback`` or ``thinking`` is not a string was loaded:
  the fallback switch then asked the factory for the model ``123``
  (``AttributeError``), and the Anthropic adapter's lookup of the effort
  level ``["high"]`` raised ``TypeError`` on every request.
* :func:`model` stripped ``openai/`` / ``anthropic/`` / ``google/`` from a
  registered custom model's own name, so the endpoint received another id
  and every cost lookup for the renamed model failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_GOOD = {"context_length": 8192, "input_price_per_1M": 1.5, "output_price_per_1M": 3}
_ENDPOINT = "http://127.0.0.1:9/v1"


def _run(code: str, home: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run *code* in a fresh interpreter with ``KISS_HOME=home``; return its JSON output."""
    env = {name: value for name, value in os.environ.items() if name != "KISS_MODEL_INFO_PATH"}
    env.update({"KISS_HOME": str(home), **(extra_env or {})})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def _home_with_registry(tmp_path: Path, registry: dict[str, Any]) -> Path:
    """Create a ``KISS_HOME`` whose ``MY_MODELS.json`` holds *registry*."""
    home = tmp_path / "kiss_home"
    home.mkdir()
    (home / "MY_MODELS.json").write_text(json.dumps(registry), encoding="utf-8")
    return home


_BAD_PRICES = {
    "claude-my-str": {
        "context_length": 200000, "input_price_per_1M": "3", "output_price_per_1M": 15
    },
    "gpt-my-null": {"context_length": 200000, "input_price_per_1M": None, "output_price_per_1M": 1},
    "gemini-my-bool": {
        "context_length": 1000, "input_price_per_1M": True, "output_price_per_1M": 1
    },
    "my-local-str-out": {
        "context_length": 1000, "input_price_per_1M": 1, "output_price_per_1M": "2"
    },
    "my-local-str-ctx": {
        "context_length": "8192", "input_price_per_1M": 1, "output_price_per_1M": 2
    },
    "claude-my-cache": {**_GOOD, "cache_read_price_per_1M": "0.1"},
    "gpt-my-alias-int-high": {**_GOOD, "alias_of": 123},
    "gpt-my-alias-list-high": {**_GOOD, "alias_of": ["gpt-4"]},
    "gpt-my-alias-dict-high": {**_GOOD, "alias_of": {"base": "gpt-4"}},
    "my-fallback-int": {**_GOOD, "fallback": 123},
    "my-fallback-list": {**_GOOD, "fallback": ["gpt-4"]},
    "claude-my-thinking-bool": {**_GOOD, "thinking": True},
    "claude-my-thinking-list": {**_GOOD, "thinking": ["high"]},
}

_LOOKUPS = """
import json
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info as mi

def cost(name):
    try:
        return mi.calculate_cost(name, 1_000_000, 0)
    except KISSError:
        return "unknown"

names = %r
result = {name: [mi.is_known_model(name), cost(name)] for name in names}
"""


def test_import_skips_registry_entries_with_non_numeric_prices(tmp_path: Path) -> None:
    home = _home_with_registry(tmp_path, {**_BAD_PRICES, "my-good": _GOOD})
    code = _LOOKUPS % [*_BAD_PRICES, "my-good"] + "print(json.dumps(result))\n"
    result = _run(code, home)
    assert result.pop("my-good") == [True, 1.5]
    assert result == {name: [False, "unknown"] for name in _BAD_PRICES}


def test_entries_added_after_start_with_bad_prices_are_unknown(tmp_path: Path) -> None:
    home = _home_with_registry(tmp_path, {})
    late = json.dumps({**_BAD_PRICES, "my-good": _GOOD})
    code = (
        _LOOKUPS % [*_BAD_PRICES, "my-good"]
        + f"mi.USER_MY_MODELS_PATH.write_text({late!r}, encoding='utf-8')\n"
        + "result = {name: [mi.is_known_model(name), cost(name)] for name in names}\n"
        + "print(json.dumps(result))\n"
    )
    result = _run(code, home)
    assert result.pop("my-good") == [True, 1.5]
    assert result == {name: [False, "unknown"] for name in _BAD_PRICES}


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("claude-only-here", {"input_price_per_1M": "3"}),
        ("gpt-only-here-high", {"alias_of": 123}),
        ("gpt-only-here-high", {"alias_of": ["gpt-4"]}),
        ("gpt-only-here-high", {"alias_of": {"base": "gpt-4"}}),
        ("claude-only-here", {"fallback": 123}),
        ("claude-only-here", {"thinking": ["high"]}),
    ],
)
def test_a_user_catalog_with_a_string_price_falls_back_to_the_bundled_one(
    tmp_path: Path, name: str, bad: dict[str, Any]
) -> None:
    home = _home_with_registry(tmp_path, {})
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(json.dumps({name: {**_GOOD, **bad}}), encoding="utf-8")
    code = (
        "import json\n"
        "from kiss.core.models import model_info as mi\n"
        f"print(json.dumps([{name!r} in mi.MODEL_INFO, len(mi.MODEL_INFO) > 100]))\n"
    )
    assert _run(code, home, {"KISS_MODEL_INFO_PATH": str(catalog)}) == [False, True]


_CUSTOM_MODEL_RUN = """
import json
from kiss.core.models import model_info as mi

def run(name):
    built = mi.model(name, model_config=mi.custom_model_config(name))
    return [built.model_name, mi.calculate_cost(built.model_name, 1_000_000, 0)]

result = {"merged": run("openai/gpt-5-custom")}
mi.save_custom_model("anthropic/claude-custom-late", endpoint=%r)
result["hot_added"] = run("anthropic/claude-custom-late")
bundled = next(n for n in sorted(mi.MODEL_INFO) if n.startswith("gpt-"))
alias = mi.model("openai/" + bundled, model_config={"base_url": %r, "api_key": "x"})
result["harbor_alias"] = [alias.model_name == bundled]
print(json.dumps(result))
"""


def test_model_keeps_a_registered_custom_name(tmp_path: Path) -> None:
    home = _home_with_registry(
        tmp_path, {"openai/gpt-5-custom": {**_GOOD, "endpoint": _ENDPOINT, "api_key": "k"}}
    )
    result = _run(_CUSTOM_MODEL_RUN % (_ENDPOINT, _ENDPOINT), home)
    assert result["merged"] == ["openai/gpt-5-custom", 1.5]
    assert result["hot_added"] == ["anthropic/claude-custom-late", 0.0]
    assert result["harbor_alias"] == [True]


# One bundled model per capability a settings-panel save must keep: the
# Responses transport (its thinking aliases mirror the base's flag), a
# decisions-only model, an embedding model, and a model with a fallback
# and the Anthropic thinking overrides.
_FLAGGED = (
    "gpt-6-astra",
    "openrouter/~typesafe/jev-latest",
    "text-embedding-3-small",
    "claude-fable-5",
)
_FLAGS = (
    "is_function_calling_supported",
    "is_embedding_supported",
    "is_generation_supported",
    "is_decisions_supported",
    "use_responses_api",
    "fallback",
    "extended_thinking",
    "adaptive_thinking",
)

_FLAG_SNAPSHOT = """
import json
from kiss.core.models import model_info as mi

names, flags, save = %r, %r, %r
related = [n for n in sorted(mi.MODEL_INFO) if n in names or mi.MODEL_INFO[n].alias_of in names]
result = {n: {f: getattr(mi.MODEL_INFO[n], f) for f in flags} for n in related}
if save:
    for n in names:
        assert mi.save_custom_model(n, endpoint=%r, api_key="k") is None
else:
    name = "openrouter/~typesafe/jev-latest"
    built = mi.model(name, model_config=mi.custom_model_config(name))
    result["decisions_class"] = type(built).__name__
print(json.dumps(result))
"""


def test_a_saved_credential_keeps_the_catalog_capability_flags(tmp_path: Path) -> None:
    """A settings-panel save of a bundled model keeps that model's flags.

    The seeded entry copied the context length, thinking level and
    prices only, and the loader replaces the bundled entry with it on
    the next start: a decisions-only model came back as a generative
    tool-calling one (built by the chat adapter instead of the decisions
    one), an embedding model entered the generation picker, and the
    Responses transport flag, the fallback model and the Anthropic
    thinking overrides were gone — the transport flag from every
    thinking alias of the base as well, since the aliases mirror it.
    """
    home = _home_with_registry(tmp_path, {})
    before = _run(_FLAG_SNAPSHOT % (_FLAGGED, _FLAGS, True, _ENDPOINT), home)
    assert before["gpt-6-astra"]["use_responses_api"] is True
    assert before["gpt-6-astra-high"]["use_responses_api"] is True
    assert before["openrouter/~typesafe/jev-latest"]["is_decisions_supported"] is True
    assert before["text-embedding-3-small"]["is_embedding_supported"] is True
    assert before["claude-fable-5"]["fallback"]
    assert before["claude-fable-5"]["adaptive_thinking"] is True
    after = _run(_FLAG_SNAPSHOT % (_FLAGGED, _FLAGS, False, _ENDPOINT), home)
    assert after.pop("decisions_class") == "DecisionsModel"
    assert after == before
