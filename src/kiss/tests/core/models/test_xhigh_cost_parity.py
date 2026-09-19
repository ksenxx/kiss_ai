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
    # Above the 272k OpenAI long-context threshold, in every dimension at once.
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

    This includes long-context tier thresholds: at >272k prompt tokens,
    an OpenAI base like ``gpt-5.5`` switches from the standard tier to
    the long-context tier (2x input/cache, 1.5x output). The alias must
    do the same.
    """
    in_t, out_t, cr_t, cw_t, cw1h_t = profile
    base_cost = calculate_cost(base, in_t, out_t, cr_t, cw_t, cw1h_t)
    alias_cost = calculate_cost(alias, in_t, out_t, cr_t, cw_t, cw1h_t)
    assert alias_cost == pytest.approx(base_cost), (
        f"cost mismatch for alias={alias!r} base={base!r} profile={profile}: "
        f"alias={alias_cost} base={base_cost}"
    )


# The rolling OpenRouter ``~openai`` latest aliases track GPT-5.6/GPT-6
# snapshots (astra -> gpt-6-astra, luna/sol/terra -> gpt-5.6-*) and must
# therefore share the snapshot families' absolute pricing rules: the 0.10x
# cache-read discount, the 1.25x billed cache writes, and the 2x/1.5x
# long-context uplift above 272k prompt tokens.  Alias/base parity tests
# above cannot catch an absolute error when base and alias are BOTH wrong,
# so these checks pin the absolute rules per family.
_ROLLING_LATEST_BASES = (
    "openrouter/~openai/gpt-astra-latest",
    "openrouter/~openai/gpt-luna-latest",
    "openrouter/~openai/gpt-sol-latest",
    "openrouter/~openai/gpt-terra-latest",
)


@pytest.mark.parametrize("base", _ROLLING_LATEST_BASES)
def test_rolling_latest_cache_read_uses_openai_gpt5_discount(base: str) -> None:
    """Rolling latest entries and their -xhigh aliases read cache at 0.10x.

    Regression: before the September 2026 catalog rename these fell
    through to the default 0.50x multiplier, billing 5x too much for
    cache reads.
    """
    for name in (base, f"{base}-xhigh"):
        assert name in MODEL_INFO, f"{name} missing from MODEL_INFO"
        info = MODEL_INFO[name]
        assert info.cache_read_price_per_1M == pytest.approx(
            info.input_price_per_1M * 0.10
        ), name


@pytest.mark.parametrize("base", _ROLLING_LATEST_BASES)
def test_rolling_latest_bills_cache_writes(base: str) -> None:
    """Rolling latest entries bill cache writes at 1.25x like their snapshots.

    Regression: they previously fell into the free-writes rule that only
    applies before the GPT-5.6 family.
    """
    info = MODEL_INFO[base]
    assert info.cache_write_price_per_1M == pytest.approx(
        info.input_price_per_1M * 1.25
    ), base


@pytest.mark.parametrize("base", _ROLLING_LATEST_BASES)
def test_rolling_latest_gets_long_context_uplift(base: str) -> None:
    """Above 272k prompt tokens the 2x input / 1.5x output tier applies.

    Regression: the uplift previously matched only the versioned family
    names, so rolling aliases billed long prompts at short-context rates.
    """
    info = MODEL_INFO[base]
    in_tokens, out_tokens = 272_001, 10_000
    expected = (
        in_tokens / 1e6 * info.input_price_per_1M * 2.0
        + out_tokens / 1e6 * info.output_price_per_1M * 1.5
    )
    assert calculate_cost(base, in_tokens, out_tokens) == pytest.approx(expected)
    # Below the threshold the standard rates apply unchanged.
    small_in, small_out = 1_000, 1_000
    small_expected = (
        small_in / 1e6 * info.input_price_per_1M
        + small_out / 1e6 * info.output_price_per_1M
    )
    assert calculate_cost(base, small_in, small_out) == pytest.approx(
        small_expected
    )


def test_gpt_mini_latest_keeps_free_writes_and_no_uplift() -> None:
    """``gpt-mini-latest`` tracks gpt-5.4-mini: 0.10x reads, no published
    write price (OpenRouter never reports cache writes for it), and NO
    long-context tier."""
    name = "openrouter/~openai/gpt-mini-latest"
    assert name in MODEL_INFO, f"{name} missing from MODEL_INFO"
    info = MODEL_INFO[name]
    assert info.cache_read_price_per_1M == pytest.approx(
        info.input_price_per_1M * 0.10
    )
    assert info.cache_write_price_per_1M is None
    in_tokens, out_tokens = 272_001, 10_000
    linear = (
        in_tokens / 1e6 * info.input_price_per_1M
        + out_tokens / 1e6 * info.output_price_per_1M
    )
    assert calculate_cost(name, in_tokens, out_tokens) == pytest.approx(linear)
