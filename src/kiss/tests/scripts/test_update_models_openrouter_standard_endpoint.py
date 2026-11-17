# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a regional-surcharge headline does not price a closed-weight model.

OpenRouter's ``/api/v1/models`` shows, per model, the price of one of its
endpoints.  For a closed-weight model that can be a regional endpoint's
(``azure/us``, ``google-vertex/us``: the list price plus a premium), which
requests to ``https://openrouter.ai`` are never routed to while a
standard-priced endpoint exists.  ``openai/gpt-5.6-sol`` was listed at
$4.40/$22 on some days and at OpenAI's $4/$20 on others (2026-11-12 to
11-15) while every request was billed $4/$20, so the daily
``update_models.py`` run flipped the catalog entry back and forth;
``google/gemini-3.8-flash`` stayed listed and catalogued at
``google-vertex/us``'s price.  ``prefer_standard_endpoint_prices`` prices
such a model by its cheapest serving standard-tier endpoint.  These tests
drive the real ``fetch_openrouter`` -> ``prefer_standard_endpoint_prices``
-> ``compute_changes`` path, and ``main()``, against an in-process
OpenRouter (the listing plus ``/api/v1/models/<id>/endpoints``) and a temp
catalog.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

import kiss.scripts.update_models as mod
from kiss.core.models import model_info

SOL_ID = "openai/gpt-5.6-sol"
SOL = f"openrouter/{SOL_ID}"
LATEST_ID = "~openai/gpt-sol-latest"
LATEST = f"openrouter/{LATEST_ID}"
GEM_ID = "google/gemini-3.8-flash"
GEM = f"openrouter/{GEM_ID}"
GLM_ID = "z-ai/glm-4.5-air"
GLM = f"openrouter/{GLM_ID}"
UNSET: Any = object()  # a listing or endpoint record that does not carry the field at all


def _usd(per_million: float) -> str:
    return f"{per_million / 1e6:.12f}"


def _sol_pricing(scale: float) -> dict[str, Any]:
    """gpt-5.6-sol's OpenRouter price sheet times *scale* (1.0 the list price, 1.1 regional)."""
    return {
        "prompt": _usd(4 * scale),
        "completion": _usd(20 * scale),
        "web_search": "0.01",
        "input_cache_read": _usd(0.4 * scale),
        "input_cache_write": _usd(5 * scale),
        "discount": 0,
        "overrides": [
            {
                "min_prompt_tokens": 272000,
                "prompt": _usd(8 * scale),
                "completion": _usd(30 * scale),
                "input_cache_read": _usd(0.8 * scale),
                "input_cache_write": _usd(10 * scale),
            }
        ],
    }


def _sol_fields(scale: float) -> dict[str, Any]:
    """The catalog price fields of ``_sol_pricing(scale)``."""

    def per_m(per_million: float) -> float:
        return round(per_million * scale, 6)

    return {
        "input_price_per_1M": per_m(4),
        "output_price_per_1M": per_m(20),
        "cache_read_price_per_1M": per_m(0.4),
        "cache_write_price_per_1M": per_m(5),
        "prompt_tiers": [
            {
                "min_prompt_tokens": 272000,
                "input_price_per_1M": per_m(8),
                "output_price_per_1M": per_m(30),
                "cache_read_price_per_1M": per_m(0.8),
                "cache_write_price_per_1M": per_m(10),
            }
        ],
    }


def _flat(prompt: float, completion: float, **extra: float) -> dict[str, Any]:
    """A ``pricing`` object: input, output and any *extra* field, in USD per 1M tokens."""
    sheet = {"prompt": _usd(prompt), "completion": _usd(completion)}
    return {**sheet, **{field: _usd(price) for field, price in extra.items()}}


def _listed(
    model_id: str, pricing: dict[str, Any], hf: Any = None, alias_of: str | None = None
) -> dict[str, Any]:
    """A listing record; *hf* is its ``hugging_face_id`` (``None``: a closed-weight model)."""
    record = {"id": model_id, "context_length": 32000, "pricing": pricing, "expiration_date": None}
    if hf is not UNSET:
        record["hugging_face_id"] = hf
    if alias_of is not None:
        record["alias_target"] = {"name": alias_of, "slug": alias_of}
    return record


def _endpoint(tag: str, pricing: Any, status: Any = 0, tier: Any = None) -> dict[str, Any]:
    endpoint = {
        "provider_name": tag.split("/")[0],
        "tag": tag,
        "status": status,
        "pricing": pricing,
    }
    if tier is not UNSET:
        endpoint["service_tier"] = tier
    return endpoint


def _endpoints(model_id: str, *endpoints: Any) -> dict[str, Any]:
    return {"data": {"id": model_id, "endpoints": list(endpoints)}}


def _sol_endpoints(standard_status: int = 0) -> dict[str, Any]:
    """gpt-5.6-sol's endpoints as OpenRouter listed them on 2026-11-16."""
    return _endpoints(
        SOL_ID,
        _endpoint("openai/flex", _sol_pricing(0.5), tier="flex"),
        _endpoint("amazon-bedrock/global", _sol_pricing(1.0), status=-5),
        _endpoint("azure/us", _sol_pricing(1.1)),
        _endpoint("azure", _sol_pricing(1.0), status=standard_status),
        _endpoint("openai", _sol_pricing(1.0), status=standard_status),
        _endpoint("amazon-bedrock/us-east-1", _sol_pricing(1.1)),
        _endpoint("azure/eu", _sol_pricing(1.1)),
        _endpoint("openai/fast", _sol_pricing(2.0), tier="priority"),
    )


def _entry(**prices: Any) -> dict[str, Any]:
    return {"context_length": 32000, "fc": True, "emb": False, "gen": True, **prices}


class _OpenRouterHandler(BaseHTTPRequestHandler):
    """Serve the models listing, its decisions variant and ``/<id>/endpoints``."""

    listing: dict[str, Any] = {"data": []}
    endpoints: dict[str, Any] = {}  # model id -> document, or an int HTTP status
    lookups: list[str] = []  # the model ids whose endpoints were requested, in order

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        handler = _OpenRouterHandler
        document: Any
        if self.path.endswith("/endpoints"):
            model_id = self.path.split("/api/v1/models/", 1)[1].removesuffix("/endpoints")
            handler.lookups.append(model_id)
            document = handler.endpoints.get(model_id, _endpoints(model_id))
        elif "output_modalities=decisions" in self.path:
            document = {"data": []}
        else:
            document = handler.listing
        if isinstance(document, int):
            self.send_response(document)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _openrouter(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, Any]], endpoints: dict[str, Any]
) -> Iterator[list[str]]:
    """Point the updater at a local OpenRouter; yield the endpoint lookups it receives."""
    _OpenRouterHandler.listing = {"data": listing}
    _OpenRouterHandler.endpoints = endpoints
    _OpenRouterHandler.lookups = []
    server = HTTPServer(("127.0.0.1", 0), _OpenRouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/v1/models"
    monkeypatch.setattr(mod, "OPENROUTER_MODELS_URL", url)
    monkeypatch.setattr(
        mod, "OPENROUTER_DECISIONS_MODELS_URL", f"{url}?output_modalities=decisions"
    )
    try:
        yield _OpenRouterHandler.lookups
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _refresh(
    monkeypatch: pytest.MonkeyPatch,
    current: dict[str, dict],
    listing: list[dict[str, Any]],
    endpoints: dict[str, Any],
    guard: bool = True,
) -> tuple[list[str], dict[str, dict], list[str], dict[str, dict], list[dict]]:
    """Fetch, settle (unless ``guard`` is off) and diff.

    Returns (replaced, fetched, endpoint lookups, changes by name, new models).
    """
    with _openrouter(monkeypatch, listing, endpoints) as lookups:
        fetched = mod.fetch_openrouter()
        replaced = mod.prefer_standard_endpoint_prices(fetched) if guard else []
    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {}, {})
    return replaced, fetched, list(lookups), {u["name"]: u["changes"] for u in updates}, new_models


def test_flipping_headline_no_longer_reprices_the_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 11-15 flip: listed at $4.40/$22 while the catalog holds the billed $4/$20."""
    current = {SOL: _entry(**_sol_fields(1.0))}
    listing = [_listed(SOL_ID, _sol_pricing(1.1))]
    endpoints = {SOL_ID: _sol_endpoints()}
    *_, flipped, _new = _refresh(monkeypatch, current, listing, endpoints, guard=False)
    assert flipped == {SOL: _sol_fields(1.1)}  # what the headline alone did every other day

    replaced, fetched, lookups, changes, _new = _refresh(monkeypatch, current, listing, endpoints)
    assert replaced == [SOL]
    assert lookups == [SOL_ID]
    assert {field: fetched[SOL][field] for field in _sol_fields(1.0)} == _sol_fields(1.0)
    assert changes == {}


# The sheets of google/gemini-3.8-flash on 2026-11-16: google-vertex/us (the listing's headline,
# 10% above the list price, and the only one publishing an image-output price) and AI Studio.
_GEM_REGIONAL = _flat(
    0.825, 4.125, input_cache_read=0.0825, audio=0.825, audio_output=8.25, image_output=33.0
)
_GEM_STANDARD = _flat(0.75, 3.75, input_cache_read=0.075, audio=1.5, audio_output=7.5)
_GEM_STANDARD_FIELDS = {
    "input_price_per_1M": 0.75,
    "output_price_per_1M": 3.75,
    "cache_read_price_per_1M": 0.075,
    "audio_input_price_per_1M": 1.5,
    "audio_output_price_per_1M": 7.5,
}


def _gem_endpoints() -> dict[str, Any]:
    return _endpoints(
        GEM_ID,
        _endpoint("google-ai-studio/flex", _flat(0.375, 1.875), tier="flex"),
        _endpoint("google-vertex/us", _GEM_REGIONAL),
        _endpoint("google-ai-studio", _GEM_STANDARD),
        _endpoint("google-vertex/global", _flat(0.75, 3.75, input_cache_read=0.07)),
        _endpoint("google-ai-studio/priority", _flat(1.35, 6.75), tier="priority"),
    )


def test_entry_catalogued_at_the_surcharge_is_corrected(monkeypatch: pytest.MonkeyPatch) -> None:
    """gemini-3.8-flash: catalog == headline == google-vertex/us, so the headline moves nothing."""
    at_surcharge = {
        "input_price_per_1M": 0.825,
        "output_price_per_1M": 4.125,
        "cache_read_price_per_1M": 0.0825,
        "audio_input_price_per_1M": 0.825,
        "audio_output_price_per_1M": 8.25,
        "image_output_price_per_1M": 33.0,
    }
    current = {GEM: _entry(**at_surcharge)}
    listing = [_listed(GEM_ID, _GEM_REGIONAL, hf="")]
    endpoints = {GEM_ID: _gem_endpoints()}
    *_, stuck, _new = _refresh(monkeypatch, current, listing, endpoints, guard=False)
    assert stuck == {}

    replaced, fetched, _lookups, changes, _new = _refresh(monkeypatch, current, listing, endpoints)
    assert replaced == [GEM]
    # The whole sheet of the first-listed cheapest endpoint: its cache and audio prices replace
    # the regional ones, and the image price only the regional endpoint publishes is gone.
    assert {f: fetched[GEM][f] for f in fetched[GEM] if "price" in f} == _GEM_STANDARD_FIELDS
    assert changes == {GEM: {**_GEM_STANDARD_FIELDS, "image_output_price_per_1M": None}}


def test_new_model_is_added_at_the_list_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model the catalog lacks is not born at the surcharge either."""
    listing = [_listed(SOL_ID, _sol_pricing(1.1))]
    replaced, _fetched, _lookups, changes, new = _refresh(
        monkeypatch, {}, listing, {SOL_ID: _sol_endpoints()}
    )
    assert (replaced, changes) == ([SOL], {})
    assert [m["name"] for m in new] == [SOL]
    assert {field: new[0][field] for field in _sol_fields(1.0)} == _sol_fields(1.0)


def test_alias_is_priced_through_its_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``~`` alias lists no endpoint of its own: its target's standard endpoint prices both."""
    current = {LATEST: _entry(**_sol_fields(1.0)), SOL: _entry(**_sol_fields(1.0))}
    listing = [
        _listed(LATEST_ID, _sol_pricing(1.1), alias_of=SOL_ID),
        _listed(SOL_ID, _sol_pricing(1.1), hf=""),
    ]
    replaced, fetched, lookups, changes, _new = _refresh(
        monkeypatch, current, listing, {SOL_ID: _sol_endpoints()}
    )
    assert replaced == [LATEST, SOL]
    assert lookups == [SOL_ID]  # once for the two of them, and none for the alias's own id
    assert changes == {}
    assert (
        fetched[LATEST]["prompt_tiers"]
        == fetched[SOL]["prompt_tiers"]
        == [_sol_fields(1.0)["prompt_tiers"][0]]
    )
    assert fetched[LATEST]["prompt_tiers"][0] is not fetched[SOL]["prompt_tiers"][0]


_OPEN_GLM = _listed(GLM_ID, _flat(0.13, 0.85), hf="zai-org/GLM-4.5-Air")


@pytest.mark.parametrize(
    ("listing", "closed_weight"),
    [
        # open-weight: its hosts, regional ones included, compete on price
        ([_OPEN_GLM], []),
        ([_listed(SOL_ID, _sol_pricing(1.1), hf=UNSET)], []),  # the listing does not say
        ([_listed("~z-ai/glm-latest", _flat(0.13, 0.85), alias_of=GLM_ID), _OPEN_GLM], []),
        ([_listed(LATEST_ID, _sol_pricing(1.1), alias_of="openai/unlisted")], []),
        # closed-weight, but with no list price to fall back to
        ([_listed("openai/gpt-gratis", _flat(0.0, 0.0))], ["openai/gpt-gratis"]),
        ([_listed("acme/router", {"prompt": "-1", "completion": "-1"})], ["acme/router"]),
        # compute_changes skips a :variant
        ([_listed("openai/gpt-4o:extended", _flat(6.6, 19.8))], ["openai/gpt-4o:extended"]),
    ],
)
def test_other_models_are_not_looked_up(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, Any]], closed_weight: list[str]
) -> None:
    """Only a paid, plain model the listing calls closed-weight costs an endpoint request."""
    headlines = {m["id"]: mod._openrouter_price_fields(m["pricing"]) for m in listing}
    # Any lookup would find a cheaper standard endpoint, and so replace the headline.
    cheaper = {m["id"]: _endpoints(m["id"], _endpoint("acme", _flat(0.12, 0.8))) for m in listing}
    replaced, fetched, lookups, _changes, _new = _refresh(monkeypatch, {}, listing, cheaper)
    assert (replaced, lookups) == ([], [])
    marked = {
        name: record["endpoints_of"] for name, record in fetched.items() if "endpoints_of" in record
    }
    assert marked == {f"openrouter/{model_id}": model_id for model_id in closed_weight}
    for model_id, headline in headlines.items():
        assert {f: fetched[f"openrouter/{model_id}"][f] for f in headline} == headline


def test_unmarked_record_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record ``fetch_openrouter`` did not mark is skipped whatever it holds."""
    bare = {"openrouter/acme/bare": {"context_length": 1}}
    with _openrouter(monkeypatch, [], {}) as lookups:
        assert mod.prefer_standard_endpoint_prices(bare) == []
    assert (lookups, bare) == ([], {"openrouter/acme/bare": {"context_length": 1}})


@pytest.mark.parametrize(
    "document",
    [
        _sol_endpoints(standard_status=-2),  # only the regional endpoints are serving
        _endpoints(SOL_ID, _endpoint("openai", _sol_pricing(1.0), status=None)),
        _endpoints(SOL_ID, _endpoint("openai", _sol_pricing(1.0), tier=UNSET)),  # does not say
        _endpoints(SOL_ID, _endpoint("openai/flex", _sol_pricing(1.0), tier="flex")),
        _endpoints(SOL_ID, _endpoint("openai", _sol_pricing(0.7))),  # 1.57x below: no surcharge
        _endpoints(SOL_ID, _endpoint("openai", _flat(4.0, 23.0))),  # dearer on output
        _endpoints(SOL_ID, _endpoint("openai", _flat(4.3999, 22.0))),  # the same price, to 0.1%
        _endpoints(SOL_ID, _endpoint("openai", _flat(0.0, 0.0)), _endpoint("azure/us", None)),
        _endpoints(SOL_ID, "junk", _endpoint("azure/us", _sol_pricing(1.1))),
        _endpoints(SOL_ID),
        {"data": {"id": SOL_ID, "endpoints": "junk"}},
        {"data": [_listed(SOL_ID, _sol_pricing(1.0))]},  # a listing, not an endpoint document
        [],
    ],
)
def test_headline_stands_without_a_cheaper_standard_endpoint(
    monkeypatch: pytest.MonkeyPatch, document: Any
) -> None:
    """Nothing serving at the standard tier undercuts the headline by a regional premium."""
    current = {SOL: _entry(**_sol_fields(1.0))}
    replaced, _fetched, lookups, changes, _new = _refresh(
        monkeypatch, current, [_listed(SOL_ID, _sol_pricing(1.1))], {SOL_ID: document}
    )
    assert (replaced, lookups) == ([], [SOL_ID])
    assert changes == {SOL: _sol_fields(1.1)}


@pytest.mark.parametrize(
    ("headline", "is_surcharge"),
    [
        ((4.4, 22.0), True),
        ((4.4, 20.0), True),  # a premium on the input price alone
        ((4.0, 22.0), True),  # ... on the output price alone
        ((4.0045, 20.0), True),  # 0.11% above: past the updater's 0.1% tolerance
        ((4.0035, 20.0), False),  # 0.09% above: the same price
        ((4.0, 20.0), False),
        ((6.0, 30.0), True),  # the largest premium taken for a regional one: 1.5x
        ((6.1, 30.0), False),
        ((6.0, 30.1), False),
        ((3.9, 22.0), False),  # the headline is cheaper on input: not this endpoint's premium
    ],
)
def test_premium_on_either_price_is_removed(
    monkeypatch: pytest.MonkeyPatch, headline: tuple[float, float], is_surcharge: bool
) -> None:
    """The headline is replaced exactly when it is the standard price plus a regional premium."""
    endpoints = _endpoints(
        SOL_ID,
        _endpoint("acme/free", _flat(0.0, 0.0)),  # a free endpoint prices nothing
        _endpoint("azure/eu", _flat(4.2, 21.0, input_cache_read=0.42)),
        _endpoint("azure/zdr", _flat(4.0, 21.0, input_cache_read=0.43)),
        _endpoint("azure", _flat(4.0, 20.0, input_cache_read=0.41)),
        _endpoint("openai", _flat(4.0, 20.0, input_cache_read=0.39)),
    )
    listing = [_listed(SOL_ID, _flat(*headline, input_cache_read=0.5))]
    replaced, fetched, _lookups, _changes, _new = _refresh(
        monkeypatch, {}, listing, {SOL_ID: endpoints}
    )
    assert replaced == ([SOL] if is_surcharge else [])
    prices = [fetched[SOL][f"{kind}_price_per_1M"] for kind in ("input", "output", "cache_read")]
    # lowest input price, then lowest output price, then the first listed: "azure"
    assert prices == ([4.0, 20.0, 0.41] if is_surcharge else [*headline, 0.5])


_UNPARSABLE = _endpoints(
    GEM_ID, _endpoint("google-ai-studio", {"prompt": "n/a", "completion": "1"})
)


@pytest.mark.parametrize("failure", [500, _UNPARSABLE])
def test_unreadable_endpoint_listing_keeps_headlines_and_ends_the_lookups(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Any
) -> None:
    """An HTTP failure or an unparsable price: one warning, no later model looked up."""
    luna = "openai/gpt-6-luna"
    listing = [
        _listed(SOL_ID, _sol_pricing(1.1)),
        _listed(GEM_ID, _GEM_REGIONAL),
        _listed(luna, _flat(0.11, 0.55)),
    ]
    endpoints = {
        SOL_ID: _sol_endpoints(),
        GEM_ID: failure,
        luna: _endpoints(luna, _endpoint("openai", _flat(0.1, 0.5))),
    }
    replaced, fetched, lookups, _changes, _new = _refresh(monkeypatch, {}, listing, endpoints)
    assert replaced == [SOL]  # settled before the failure, and kept
    assert lookups == [SOL_ID] + [GEM_ID] * (3 if failure == 500 else 1)  # api_get's attempts
    assert fetched[GEM]["input_price_per_1M"] == 0.825
    assert fetched[f"openrouter/{luna}"]["input_price_per_1M"] == 0.11
    warning = capsys.readouterr().out
    assert warning.count("WARNING: OpenRouter endpoint listing failed") == 1
    assert f"failed for {GEM_ID} " in warning


_GEM_AT_SURCHARGE = {
    "input_price_per_1M": 0.825,
    "output_price_per_1M": 4.125,
    "cache_read_price_per_1M": 0.0825,
    "audio_input_price_per_1M": 0.825,
    "audio_output_price_per_1M": 8.25,
    "image_output_price_per_1M": 33.0,
}


def _run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sol_endpoints: dict[str, Any]
) -> tuple[dict[str, Any], str, list[str]]:
    """Run ``update_models.py --skip-test`` on a temp catalog; return (catalog, path, lookups)."""
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(
        json.dumps(
            {
                SOL: _entry(**_sol_fields(1.0)),
                GEM: _entry(**_GEM_AT_SURCHARGE),
                GLM: _entry(input_price_per_1M=0.2, output_price_per_1M=1.1),
            }
        )
    )
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    for fetcher in ("fetch_together", "fetch_anthropic", "fetch_gemini", "fetch_openai"):
        monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_bedrock", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--skip-test"])
    listing = [_listed(SOL_ID, _sol_pricing(1.1)), _listed(GEM_ID, _GEM_REGIONAL, hf=""), _OPEN_GLM]
    endpoints = {SOL_ID: sol_endpoints, GEM_ID: _gem_endpoints()}
    with _openrouter(monkeypatch, listing, endpoints) as lookups:
        mod.main()
    return json.loads(catalog.read_text()), str(catalog), list(lookups)


def _prices(entry: dict[str, Any]) -> dict[str, Any]:
    return {field: value for field, value in entry.items() if "price" in field or "tiers" in field}


def test_main_writes_list_prices_and_requests_cost_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole run: no flip, the stuck surcharge corrected, a market price still followed."""
    written, path, lookups = _run_main(tmp_path, monkeypatch, _sol_endpoints())
    assert lookups == [SOL_ID, GEM_ID]  # none for the open-weight model
    assert _prices(written[SOL]) == _sol_fields(1.0)
    assert _prices(written[GEM]) == _GEM_STANDARD_FIELDS
    assert _prices(written[GLM]) == {"input_price_per_1M": 0.13, "output_price_per_1M": 0.85}
    output = capsys.readouterr().out
    for name in (SOL, GEM):
        assert f"{name}: listed at a regional surcharge, priced by its standard endpoint" in output
    assert f"{SOL}: input_price_per_1M" not in output
    assert f"{GEM}: input_price_per_1M: 0.825 -> 0.75" in output
    assert "applied 2 updates" in output

    # Requests are costed at the list prices, below and past gpt-5.6-sol's 272K long-prompt tier.
    monkeypatch.setenv("KISS_MODEL_INFO_PATH", path)
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    monkeypatch.setattr(model_info, "MODEL_INFO", model_info._load_model_info())
    short = model_info.calculate_cost(SOL, 100_000, 1_000, 50_000, 2_000)
    assert short == pytest.approx((100_000 * 4 + 1_000 * 20 + 50_000 * 0.4 + 2_000 * 5) / 1e6)
    long = model_info.calculate_cost(SOL, 200_000, 1_000, 80_000, 2_000)
    assert long == pytest.approx((200_000 * 8 + 1_000 * 30 + 80_000 * 0.8 + 2_000 * 10) / 1e6)
    flash = model_info.calculate_cost(GEM, 1_000_000, 100_000, 200_000)
    assert flash == pytest.approx((1_000_000 * 0.75 + 100_000 * 3.75 + 200_000 * 0.075) / 1e6)


def test_main_takes_the_headline_while_no_standard_endpoint_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same run with OpenAI's and Azure's global endpoints down writes the listed $4.40/$22."""
    written, _path, lookups = _run_main(tmp_path, monkeypatch, _sol_endpoints(standard_status=-2))
    assert lookups == [SOL_ID, GEM_ID]
    assert _prices(written[SOL]) == _sol_fields(1.1)
    assert _prices(written[GEM]) == _GEM_STANDARD_FIELDS
    output = capsys.readouterr().out
    assert f"{SOL}: listed at a regional surcharge" not in output
    assert f"{GEM}: listed at a regional surcharge, priced by its standard endpoint" in output
    assert "applied 3 updates" in output
