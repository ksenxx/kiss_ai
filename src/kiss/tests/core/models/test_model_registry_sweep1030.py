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
* A custom model the registry held at start stayed in the loaded table
  after the settings panel deleted or renamed it: still offered, still
  admitted, and run against the vendor its name routes to.
* Reading the registry live for every lookup then left a model that was
  running without its prices as soon as its entry was deleted or renamed,
  or the file could not be read: the answer the provider had already
  charged for ended the task in ``Cannot calculate budget for unknown
  model``, its tokens and cost unrecorded.
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


_BEDROCK = "bedrock/global.anthropic.claude-opus-4-6-v1"
_TWIN = "openrouter/acme/fx19-twin"
_REGISTERED_AT_START = {
    **{
        name: {**_GOOD, "endpoint": _ENDPOINT}
        for name in ("gpt-fx19-deleted", "gpt-fx19-renamed", "gpt-fx19-kept", "gpt-fx19-bundled")
    },
    _BEDROCK: _GOOD,
    _TWIN: _GOOD,
}

_DELETE_AND_RENAME = """
import json
import os
from pathlib import Path
from kiss.agents.sorcar.sea_lint import lint_all
from kiss.core.models import model_info as mi
from kiss.server.autocomplete import ranked_function_calling_models

SEA = '''
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return "a test SEA"

    def settings(self, settings):
        return settings | {"model": %%r}
'''

def sea_naming(model, number):
    script = Path(os.environ["KISS_HOME"]) / f"lint{number}" / f"lint{number}_sea.py"
    script.parent.mkdir()
    script.write_text(SEA %% model, encoding="utf-8")
    return script

def state():
    return {
        "available": [n for n in gpt if n in mi.get_available_models()],
        "picker": [n for n in gpt if n in ranked_function_calling_models()],
        "known": [n for n in names if mi.is_known_model(n)],
        "lints_clean": [n for n, script in zip(gpt, seas) if not lint_all([script])],
        "bedrock": mi.registered_bedrock_model(cheapest=True),
        "twin": mi.openrouter_twin("fx19-twin"),
    }

names = %r
gpt = [n for n in names if n.startswith("gpt-")]
seas = [sea_naming(n, number) for number, n in enumerate(gpt)]
result = {"loaded": state()}
for name in ("gpt-fx19-deleted", "gpt-fx19-bundled", %r, %r):
    assert mi.delete_custom_model(name) is None
assert mi.save_custom_model("gpt-fx19-renamed-to", %r, original_name="gpt-fx19-renamed") is None
result["edited"] = state()
print(json.dumps(result))
"""


def test_a_custom_model_deleted_or_renamed_after_start_is_gone_at_once(tmp_path: Path) -> None:
    """A custom model loaded at start counts only while the registry holds its name.

    The loaded table kept the entry until the next start, so the deleted
    model was still offered by the picker and admitted with the vendor key
    of its name (a private ``gpt-*`` gateway model then ran against
    OpenAI), still the default Bedrock pick and still a fallback twin.
    ``sea lint`` read the loaded table too: a SEA naming the deleted model
    linted clean and one naming the new name was an ``unknown-model``.
    The catalog's own ``gpt-fx19-bundled`` keeps its loaded entry when its
    override goes.
    """
    home = _home_with_registry(tmp_path, _REGISTERED_AT_START)
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(json.dumps({"gpt-fx19-bundled": _GOOD}), encoding="utf-8")
    names = [*_REGISTERED_AT_START, "gpt-fx19-renamed-to"]
    code = _DELETE_AND_RENAME % (names, _BEDROCK, _TWIN, _ENDPOINT)
    result = _run(code, home, {"KISS_MODEL_INFO_PATH": str(catalog), "OPENAI_API_KEY": "k"})
    gpt = [name for name in _REGISTERED_AT_START if name.startswith("gpt-")]
    assert result["loaded"] == {
        "available": gpt,
        "picker": gpt,
        "known": list(_REGISTERED_AT_START),
        "lints_clean": gpt,
        "bedrock": _BEDROCK,
        "twin": _TWIN,
    }
    assert result["edited"] == {
        "available": ["gpt-fx19-kept", "gpt-fx19-bundled"],
        "picker": ["gpt-fx19-kept", "gpt-fx19-bundled"],
        "known": ["gpt-fx19-kept", "gpt-fx19-bundled", "gpt-fx19-renamed-to"],
        "lints_clean": ["gpt-fx19-kept", "gpt-fx19-bundled", "gpt-fx19-renamed-to"],
        "bedrock": None,
        "twin": None,
    }


_RUNNING = "gpt-fx19c-running"
_GATEWAY = "openai/gpt-fx19c-listed"

_ENDPOINT_SERVER = """
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

name = %r
registry = Path(os.environ["KISS_HOME"]) / "MY_MODELS.json"
sent = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        sent.append(json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"])
        if len(sent) == 2:
            edit_registry()
        body = json.dumps({
            "id": "chatcmpl-fx19c", "object": "chat.completion", "created": 1, "model": name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
endpoint = f"http://127.0.0.1:{server.server_port}/v1"
entry = json.dumps({name: {**%r, "endpoint": endpoint}})


def price(model):
    try:
        return mi.calculate_cost(model, 1_000_000, 0)
    except KISSError:
        return "unknown"


def run(model_config):
    try:
        answer = agent.run(
            name, "Say done", is_agentic=False, model_config=model_config,
            verbose=False, print_prompts=False, compaction=False,
        )
    except KISSError as error:
        answer = str(error)
    return [answer, agent.total_tokens_used, agent.budget_used]
"""

_RUN_THROUGH_AN_EDIT = _ENDPOINT_SERVER + """
loaded, edit = %r, %r
if loaded:
    registry.write_text(entry, encoding="utf-8")

import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info as mi

if not loaded:
    registry.write_text(entry, encoding="utf-8")


def edit_registry():
    # The reused model's request is in flight: the registry changes under it.
    if edit == "delete":
        assert mi.delete_custom_model(name) is None
    elif edit == "rename":
        assert mi.save_custom_model(name + "-to", endpoint, original_name=name) is None
    elif edit == "corrupt":
        registry.write_text("{", encoding="utf-8")
    else:
        registry.unlink()
        registry.mkdir()


def registration():
    return [
        mi.is_known_model(name),
        name in mi.get_available_models(),
        mi.custom_model_config(name) is not None,
    ]


def task(model_config):
    runs.append(run(model_config))
    models.append(agent.model)


agent = KISSAgent("fx19c tab")  # As a tab keeps its agent, and with it the model, between tasks.
config = mi.custom_model_config(name)
result = {"registered": registration()}
runs, models = [], []
# A task of the daemon runs on a thread of its own.  The second one names no endpoint, so the
# agent reuses the model the first one built.
for model_config in (config, None):
    thread = threading.Thread(target=task, args=(model_config,))
    thread.start()
    thread.join()
result.update(
    runs=runs,
    reused=models[0] is models[1],
    edited=registration(),
    name=price(name),
    running=[
        price(agent.model),
        mi.get_max_context_length(agent.model),
        mi.get_cache_discount(agent.model),
        mi.get_cache_write_ratio(agent.model),
    ],
    saved=yaml.safe_load(agent.get_trajectory_path().read_text(encoding="utf-8"))["max_tokens"],
    built_after=price(mi.model(name, model_config=config)),
)
if edit in ("corrupt", "unreadable"):
    if edit == "unreadable":
        registry.rmdir()
    registry.write_text(entry, encoding="utf-8")
    result["readable_again"] = [*registration(), price(name)]
server.shutdown()
print(json.dumps(result))
"""


@pytest.mark.parametrize("loaded", [True, False], ids=["loaded-at-start", "added-later"])
@pytest.mark.parametrize("edit", ["delete", "rename", "corrupt", "unreadable"])
def test_a_model_that_is_running_keeps_its_prices_when_its_registration_goes(
    tmp_path: Path, edit: str, loaded: bool
) -> None:
    """A registry edit is not a Stop: the running model's answers are still billed.

    One real :class:`KISSAgent` runs twice against a local endpoint, each
    run on a thread of its own, the second reusing the model the first
    built.  While the second request is in flight the model's registry
    entry is deleted or renamed, or the registry is left half-written or
    unreadable.  The answer is returned and its 15 tokens are billed at
    the prices the model was built with, whichever thread built it; its
    context window (the one the run's saved trajectory records too) and
    cache price ratios stay known.  New work sees the
    edit at once: the name is neither known nor available, has no
    endpoint and no price, although the model is still referenced, and a
    model built then is one of an unknown name.
    """
    home = tmp_path / "kiss_home"
    home.mkdir()
    entry = {**_GOOD, "cache_read_price_per_1M": 0.3, "cache_write_price_per_1M": 3}
    code = _RUN_THROUGH_AN_EDIT % (_RUNNING, entry, loaded, edit)
    result = _run(code, home, {"OPENAI_API_KEY": "k"})
    billed = ["done", 15, pytest.approx(3e-05)]
    # Only a model loaded at start is in the table the picker walks.
    assert result.pop("registered") == [True, loaded, True]
    if edit in ("corrupt", "unreadable"):
        assert result.pop("readable_again") == [True, loaded, True, 1.5]
    assert result == {
        "runs": [billed, billed],
        "reused": True,
        "edited": [False, False, False],
        "name": "unknown",
        "running": [1.5, 8192, pytest.approx(0.2), pytest.approx(10)],
        "saved": 8192,
        "built_after": "unknown",
    }


_DELETE_AS_THE_MODEL_IS_BUILT = _ENDPOINT_SERVER + """
registry.write_text(entry, encoding="utf-8")

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info as mi
from kiss.core.models.model import Model

step = %s
deleted = []


def delete_after_the_step(frame, event, arg):
    # Schedules only: the settings panel's delete, on its own thread, lands as the step of
    # the factory returns.
    if frame.f_code is not step.__code__:
        return None
    if event == "return":
        sys.settrace(None)
        editor = threading.Thread(target=mi.delete_custom_model, args=(name,))
        editor.start()
        editor.join()
        deleted.append(mi.custom_model_config(name) is None)
    return delete_after_the_step


agent = KISSAgent("fx19c gateway task")
config = mi.custom_model_config(name)
registered = mi.calculate_cost(name, 10, 5)
sys.settrace(delete_after_the_step)
try:
    ran = run(config)
finally:
    sys.settrace(None)
server.shutdown()
print(json.dumps({
    "deleted_as_built": deleted,
    "sent": sent,
    "run": ran,
    "registered": registered,
    "name_now": mi.calculate_cost(name, 10, 5),
}))
"""


@pytest.mark.parametrize(
    "step", ["mi._read_my_models", "Model.__init__"], ids=["registry-read", "adapter-built"]
)
def test_a_model_is_billed_by_the_entry_its_route_was_chosen_by(
    tmp_path: Path, step: str
) -> None:
    """The entry that routes a model to its gateway is the one its answers are billed by.

    ``openai/gpt-fx19c-listed`` is a gateway's id, registered with the
    gateway's prices; the catalog lists ``gpt-fx19c-listed``, so once the
    entry is deleted the name reads as the harbor form of the listed
    model.  The delete lands while the factory builds the model, as soon
    as it has read the registry or once the adapter is constructed: the
    request still goes to the gateway under the gateway's id, and is
    billed at the gateway's prices, not at the listed model's.
    """
    home = tmp_path / "kiss_home"
    home.mkdir()
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(json.dumps({_GATEWAY.removeprefix("openai/"): _GOOD}), encoding="utf-8")
    entry = {**_GOOD, "input_price_per_1M": 99, "output_price_per_1M": 99}
    code = _DELETE_AS_THE_MODEL_IS_BUILT % (_GATEWAY, entry, step)
    result = _run(code, home, {"KISS_MODEL_INFO_PATH": str(catalog), "OPENAI_API_KEY": "k"})
    assert result == {
        "deleted_as_built": [True],
        "sent": [_GATEWAY],
        "run": ["done", 15, pytest.approx(0.001485)],
        "registered": pytest.approx(0.001485),
        "name_now": pytest.approx(3e-05),
    }


_PRICE_EDIT = """
import json
from kiss.core.models import model_info as mi

name, config = "fx19c-edited", {"base_url": %r}


def register(price):
    entry = {**%r, "input_price_per_1M": price, "endpoint": config["base_url"]}
    mi.USER_MY_MODELS_PATH.write_text(json.dumps({name: entry}), encoding="utf-8")


def read(model):
    # By keyword, as callers of the name-only readers may have written them.
    return [
        mi.calculate_cost(model_name=model, num_input_tokens=1_000_000, num_output_tokens=0),
        mi.get_max_context_length(model_name=model),
        mi.get_cache_discount(model_name=model),
        mi.get_cache_write_ratio(model_name=model),
    ]


register(1.5)
older = mi.model(name, model_config=config)
register(2.5)
newer = mi.model(name, model_config=config)
print(json.dumps([read(older), read(newer), read(name)]))
"""


def test_a_price_edit_reaches_the_models_built_after_it(tmp_path: Path) -> None:
    """A model keeps the entry it was built with; the name and later models follow the edit.

    The model is one added after start: an entry loaded at start is the
    one in force until the next start, edited or not, as before.
    """
    home = _home_with_registry(tmp_path, {})
    entry = {**_GOOD, "cache_read_price_per_1M": 0.3, "cache_write_price_per_1M": 3}
    rest = [8192, pytest.approx(0.3 / 1.5), pytest.approx(10)]
    edited = [2.5, 8192, pytest.approx(0.3 / 2.5), pytest.approx(10)]
    assert _run(_PRICE_EDIT % (_ENDPOINT, entry), home) == [[1.5, *rest], edited, edited]


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
