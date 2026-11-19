# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""GPT-6.2 Astra and Sol are priced as OpenAI's pricing page prices them.

The catalog entry of a direct OpenAI model holds its input and output
price only: cached input, cache writes and the long-context tier come from
name rules in ``model_info``, the processing-tier factor from
``openai_compatible_model``.  Those rules named the GPT-6.1 family, so on
its release day (2026-11-17) GPT-6.2 would have billed cached input at 10%
of the input rate instead of 5%, a prompt over 272K tokens at the
short-context rates, and Ultrafast at six times Standard instead of the
promotional three.

Every expected figure is a cell of
https://developers.openai.com/api/docs/pricing as fetched on 2026-11-18
(Standard, Fast and "Ultrafast promotion" tables).  The models are priced
through the catalog's own entry once ``update_models.py`` has added it,
and until then through an entry as the updater writes one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kiss.core.models import model_info
from kiss.core.models.model_info import (
    MODEL_INFO,
    _long_context_uplift,
    _openai_cache_read_multiplier,
    calculate_cost,
)
from kiss.core.models.openai_compatible_model import (
    _OPENAI_ULTRAFAST_PROMO_END,
    OpenAICompatibleModel,
    _openai_ultrafast_multiplier,
)
from kiss.tests.core.models.test_openai_service_tier_cost import _run_agent

# model -> (input, output, cached input, 30m cache write, 12h cache write) per 1M
# tokens, short context (at most 272K prompt tokens) and long context
_PAGE = {
    "gpt-6.2-astra": ((8.0, 40.0, 0.40, 10.0, 20.0), (16.0, 60.0, 0.80, 20.0, 40.0)),
    "gpt-6.2-sol": ((2.0, 10.0, 0.10, 2.50, 5.0), (4.0, 15.0, 0.20, 5.0, 10.0)),
}
_KINDS = ("in", "out", "cr", "cw", "cw12h")
_BEFORE_END = _OPENAI_ULTRAFAST_PROMO_END - timedelta(seconds=1)


@pytest.fixture(autouse=True)
def gpt_6_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make both models known: the catalog's entry, else one as the updater adds it.

    ``update_models.py --dry-run`` of 2026-11-18 lists the entries to add as
    ``gpt-6.2-astra (ctx=500000 $8.0/40.0) [openai]`` and ``gpt-6.2-sol
    (ctx=500000 $2.0/10.0) [openai]``: no cache price, no ``prompt_tiers``.
    """
    for name, (short, _long) in _PAGE.items():
        if name in MODEL_INFO:
            continue
        info = model_info._build_model_info_entry(
            {
                "context_length": 500_000,
                "input_price_per_1M": short[0],
                "output_price_per_1M": short[1],
                "fc": True,
                "emb": False,
                "gen": True,
                "thinking": "high",
                "use_responses_api": True,
            }
        )
        model_info._normalize_entry(name, info)
        monkeypatch.setitem(MODEL_INFO, name, info)


def _rate(name: str, kind: str, prompt: int) -> float:
    """USD per 1M tokens of *kind* in one request whose prompt holds *prompt* tokens."""
    in_rate = calculate_cost(name, prompt, 0) / prompt * 1e6
    if kind == "in":
        return in_rate
    if kind == "out":
        return calculate_cost(name, prompt, 1_000_000) - calculate_cost(name, prompt, 0)
    half = prompt // 2
    mixed = calculate_cost(
        name,
        prompt - half,
        0,
        num_cache_read_tokens=half if kind == "cr" else 0,
        num_cache_write_tokens=half if kind == "cw" else 0,
        num_cache_write_1h_tokens=half if kind == "cw12h" else 0,
    )
    return (mixed - in_rate * (prompt - half) / 1e6) / half * 1e6


class TestStandardRatesMatchThePricingPage:
    @pytest.mark.parametrize("name", sorted(_PAGE))
    def test_short_context(self, name: str) -> None:
        for kind, want in zip(_KINDS, _PAGE[name][0], strict=True):
            assert _rate(name, kind, 100_000) == pytest.approx(want), (name, kind)

    @pytest.mark.parametrize("name", sorted(_PAGE))
    def test_long_context(self, name: str) -> None:
        for kind, want in zip(_KINDS, _PAGE[name][1], strict=True):
            assert _rate(name, kind, 300_000) == pytest.approx(want), (name, kind)

    @pytest.mark.parametrize("name", sorted(_PAGE))
    def test_long_context_starts_above_272k_prompt_tokens(self, name: str) -> None:
        short, long = _PAGE[name]
        assert _rate(name, "in", 272_000) == pytest.approx(short[0])
        assert _rate(name, "in", 272_001) == pytest.approx(long[0])
        # the whole request moves to the long-context rates, cached tokens included
        assert calculate_cost(name, 1, 1_000, num_cache_read_tokens=272_000) == pytest.approx(
            (long[0] + 272_000 * long[2] + 1_000 * long[1]) / 1e6
        )

    def test_one_agent_step_of_a_cached_long_prompt(self) -> None:
        """300K cached + 20K fresh prompt tokens, 2K output on gpt-6.2-astra: $0.68."""
        got = calculate_cost("gpt-6.2-astra", 20_000, 2_000, num_cache_read_tokens=300_000)
        assert got == pytest.approx((20_000 * 16.0 + 300_000 * 0.80 + 2_000 * 60.0) / 1e6)
        assert got == pytest.approx(0.68)


class TestNameRules:
    @pytest.mark.parametrize(
        "name",
        [
            "gpt-6.2-astra",
            "gpt-6.2-sol",
            "gpt-6.2-astra-xhigh",
            "gpt-6.2-sol-2026-11-17",
            "openai/gpt-6.2-astra",
            "openrouter/openai/gpt-6.2-sol",
            "openrouter/openai/gpt-6.2-astra-high",
        ],
    )
    def test_every_spelling_of_a_gpt_6_2_model_has_the_long_context_tier(self, name: str) -> None:
        assert _long_context_uplift(name) == (272_000, 2.0, 1.5)

    def test_cached_input_is_five_percent_of_the_input_rate(self) -> None:
        assert _openai_cache_read_multiplier("gpt-6.2-astra") == 0.05
        assert _openai_cache_read_multiplier("gpt-6.2-sol") == 0.05

    def test_the_families_published_before_keep_their_rules(self) -> None:
        assert _openai_cache_read_multiplier("gpt-6.1-sol") == 0.05
        assert _openai_cache_read_multiplier("gpt-6.1-terra") == 0.05
        for name in ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.5"):
            assert _openai_cache_read_multiplier(name) == 0.10, name
            assert _long_context_uplift(name) == (272_000, 2.0, 1.5), name
        assert _long_context_uplift("gpt-6.1-terra") == (272_000, 2.0, 1.5)
        assert _long_context_uplift("gpt-5.4-mini") is None
        # gpt-6-astra: $10 / $1.00 cached, long context $20 / $2.00 / $75
        assert _rate("gpt-6-astra", "cr", 100_000) == pytest.approx(1.0)
        assert _rate("gpt-6-astra", "cr", 300_000) == pytest.approx(2.0)
        assert _rate("gpt-6-astra", "out", 300_000) == pytest.approx(75.0)

    def test_rolling_aliases_follow_the_snapshot_they_track(self) -> None:
        """OpenRouter listing of 2026-11-18: astra -> gpt-6.2-astra, sol -> gpt-6.2-sol,
        terra -> gpt-6.1-terra (cached input 5%), luna -> gpt-6-luna, mini -> gpt-5.4-mini (10%)."""
        for alias in ("gpt-astra-latest", "gpt-sol-latest", "gpt-terra-latest"):
            assert _openai_cache_read_multiplier(alias) == 0.05, alias
        for alias in ("gpt-luna-latest", "gpt-mini-latest", "gpt-latest"):
            assert _openai_cache_read_multiplier(alias) == 0.10, alias

    def test_a_later_point_release_is_priced_like_the_published_ones(self) -> None:
        """Both published GPT-6 point releases (6.1, 6.2) share the rule, so the next one
        starts on it instead of on the gpt-6-* cache rate and no long-context tier."""
        assert _openai_cache_read_multiplier("gpt-6.3-sol") == 0.05
        assert _long_context_uplift("gpt-6.3-sol-high") == (272_000, 2.0, 1.5)
        assert _long_context_uplift("gpt-60-sol") is None
        assert _openai_cache_read_multiplier("gpt-7.1-sol") == 0.50


class TestUltrafastPromotion:
    """Ultrafast is 3x Standard for GPT-6.2 through February 28, 2027 (Pacific time), then 6x."""

    @pytest.mark.parametrize(
        "name",
        [
            "gpt-6.2-astra",
            "gpt-6.2-sol",
            "gpt-6.2-sol-xhigh",
            "openai/gpt-6.2-astra",
            "openrouter/openai/gpt-6.2-astra",
        ],
    )
    def test_three_times_until_the_promotion_ends(self, name: str) -> None:
        assert _openai_ultrafast_multiplier(name, datetime(2026, 11, 18, tzinfo=UTC)) == 3.0
        assert _openai_ultrafast_multiplier(name, _BEFORE_END) == 3.0
        assert _openai_ultrafast_multiplier(name, _OPENAI_ULTRAFAST_PROMO_END) == 6.0
        assert _openai_ultrafast_multiplier(name, datetime(2027, 6, 1, tzinfo=UTC)) == 6.0

    def test_the_promotion_ends_at_midnight_pacific_on_march_1_2027(self) -> None:
        from zoneinfo import ZoneInfo

        pacific = ZoneInfo("America/Los_Angeles")
        assert _OPENAI_ULTRAFAST_PROMO_END == datetime(2027, 3, 1, tzinfo=pacific)
        last_day = datetime(2027, 2, 28, 23, 59, 59, tzinfo=pacific)
        assert _openai_ultrafast_multiplier("gpt-6.2-astra", last_day) == 3.0
        first_day = datetime(2027, 3, 1, 0, 0, 0, tzinfo=pacific)
        assert _openai_ultrafast_multiplier("gpt-6.2-astra", first_day) == 6.0

    @pytest.mark.parametrize("name", ["gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-sol-xhigh", "gpt-5.5"])
    def test_other_models_stay_at_six_times(self, name: str) -> None:
        assert _openai_ultrafast_multiplier(name, datetime(2026, 11, 18, tzinfo=UTC)) == 6.0
        assert _openai_ultrafast_multiplier(name) == 6.0

    def test_promotion_rates_match_the_pricing_page(self) -> None:
        """Ultrafast promotion table: gpt-6.2-astra $24 / $120 / $1.20, gpt-6.2-sol
        $6 / $30 / $0.30; long context $48 / $180 / $2.40 and $12 / $45 / $0.60."""
        page = {
            "gpt-6.2-astra": ((24.0, 120.0, 1.20), (48.0, 180.0, 2.40)),
            "gpt-6.2-sol": ((6.0, 30.0, 0.30), (12.0, 45.0, 0.60)),
        }
        for name, tables in page.items():
            factor = _openai_ultrafast_multiplier(name, _BEFORE_END)
            for prompt, table in zip((100_000, 300_000), tables, strict=True):
                for kind, want in zip(("in", "out", "cr"), table, strict=True):
                    assert _rate(name, kind, prompt) * factor == pytest.approx(want), (name, kind)

    def test_the_adapter_bills_the_served_tier_of_a_gpt_6_2_call(self) -> None:
        """One real agent turn per tier: Standard $8 / $40, Fast 2x ($16 / $80), Ultrafast
        at the factor of the day (3x during the promotion)."""
        standard = (1_000 * 8.0 + 100 * 40.0) / 1e6
        assert _run_agent(None, "gpt-6.2-astra").budget_used == pytest.approx(standard)
        assert _run_agent("fast", "gpt-6.2-astra").budget_used == pytest.approx(2 * standard)
        factor = _openai_ultrafast_multiplier("gpt-6.2-astra")
        assert factor == (3.0 if datetime.now(UTC) < _OPENAI_ULTRAFAST_PROMO_END else 6.0)
        agent = _run_agent("ultrafast", "gpt-6.2-astra")
        assert agent.budget_used == pytest.approx(factor * standard)
        model = OpenAICompatibleModel(
            "gpt-6.2-sol-xhigh", base_url="http://127.0.0.1:9", api_key="k"
        )
        assert model.cost_multiplier_for_response({"service_tier": "ultrafast"}) == factor
        # The page of 2026-11-18 has no Flex row for GPT-6.2 (its Batch row is half of
        # Standard): a response that reports ``flex`` bills the tier's general half rate.
        assert model.cost_multiplier_for_response({"service_tier": "flex"}) == 0.5
