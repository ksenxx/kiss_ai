"""End-to-end tests that pin cost parity between ``-xhigh`` aliases and bases.

A KISS catalog entry whose name ends in ``-xhigh`` is a synthetic alias
that, at runtime, routes to the same provider model id as its base entry
(see ``OpenAICompatibleModel._provider_model_name``). Pricing rules are
defined by the upstream provider for the base model only; the alias must
therefore yield identical per-token cost in every dimension (input,
output, cache-read, cache-write, 5-minute and 1-hour cache, and
long-context tiers).

These tests are pure cost-math assertions over the loaded ``MODEL_INFO``
catalog and ``calculate_cost``. They cover every ``-xhigh`` alias
currently shipped in ``MODEL_INFO.json``.
"""

from __future__ import annotations

import pytest

from kiss.core.models.model_info import (
    MODEL_INFO,
    calculate_cost,
)

_XHIGH_SUFFIX = "-xhigh"


def _xhigh_pairs() -> list[tuple[str, str]]:
    """Return every ``(base_name, xhigh_alias_name)`` pair in MODEL_INFO."""
    pairs: list[tuple[str, str]] = []
    for name in MODEL_INFO:
        if name.endswith(_XHIGH_SUFFIX):
            base = name.removesuffix(_XHIGH_SUFFIX)
            assert base in MODEL_INFO, (
                f"xhigh alias {name!r} has no base entry {base!r} in MODEL_INFO"
            )
            pairs.append((base, name))
    return pairs


_TOKEN_PROFILES: tuple[tuple[int, int, int, int, int], ...] = (
    # (input, output, cache_read, cache_write, cache_write_1h)
    (0, 0, 0, 0, 0),
    (1_000, 500, 0, 0, 0),
    (1_000, 1_000, 1_000, 0, 0),
    (1_000_000, 0, 0, 0, 0),
    (0, 1_000_000, 0, 0, 0),
    (0, 0, 1_000_000, 0, 0),
    (0, 0, 0, 1_000_000, 0),
    (0, 0, 0, 0, 1_000_000),
    # Above the 200k OpenAI long-context tier, in every dimension at once.
    (250_000, 250_000, 250_000, 0, 0),
    (1_000_000, 1_000_000, 1_000_000, 0, 0),
)


def test_xhigh_alias_catalog_is_non_empty() -> None:
    """Sanity guard: the parity tests below need at least one alias to run."""
    pairs = _xhigh_pairs()
    assert pairs, "MODEL_INFO has no -xhigh aliases to validate"


@pytest.mark.parametrize(("base", "alias"), _xhigh_pairs())
def test_xhigh_alias_uses_identical_unit_prices(base: str, alias: str) -> None:
    """Every per-1M price field must match between an alias and its base."""
    b = MODEL_INFO[base]
    a = MODEL_INFO[alias]
    assert a.input_price_per_1M == b.input_price_per_1M
    assert a.output_price_per_1M == b.output_price_per_1M
    assert a.cache_read_price_per_1M == b.cache_read_price_per_1M
    assert a.cache_write_price_per_1M == b.cache_write_price_per_1M
    assert a.cache_write_1h_price_per_1M == b.cache_write_1h_price_per_1M


@pytest.mark.parametrize(("base", "alias"), _xhigh_pairs())
@pytest.mark.parametrize("profile", _TOKEN_PROFILES)
def test_xhigh_alias_calculate_cost_matches_base(
    base: str,
    alias: str,
    profile: tuple[int, int, int, int, int],
) -> None:
    """``calculate_cost`` must agree on every alias/base pair and profile.

    This includes long-context tier thresholds: at >200k tokens, an
    OpenAI base like ``gpt-5.5`` switches from the standard tier to the
    long-context tier (10.00/45.00/1.00). The alias must do the same.
    """
    in_t, out_t, cr_t, cw_t, cw1h_t = profile
    base_cost = calculate_cost(base, in_t, out_t, cr_t, cw_t, cw1h_t)
    alias_cost = calculate_cost(alias, in_t, out_t, cr_t, cw_t, cw1h_t)
    assert alias_cost == pytest.approx(base_cost), (
        f"cost mismatch for alias={alias!r} base={base!r} profile={profile}: "
        f"alias={alias_cost} base={base_cost}"
    )


def test_gpt_latest_xhigh_cache_read_uses_openai_gpt5_discount() -> None:
    """Regression: ``openrouter/~openai/gpt-latest-xhigh`` is a GPT-5.x alias.

    The OpenAI cache-read multiplier table treats the bare names
    ``gpt-latest`` and ``gpt-mini-latest`` as 0.10x (GPT-5.x). The
    ``-xhigh`` alias must inherit the same 0.10x discount; previously
    it fell through to the default 0.50x, billing 5x too much for cache
    reads.
    """
    alias = "openrouter/~openai/gpt-latest-xhigh"
    if alias not in MODEL_INFO:
        pytest.skip(f"{alias} not present in MODEL_INFO")
    info = MODEL_INFO[alias]
    assert info.cache_read_price_per_1M == pytest.approx(
        info.input_price_per_1M * 0.10
    )
