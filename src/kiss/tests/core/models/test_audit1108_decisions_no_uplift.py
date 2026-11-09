# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A decisions-only catalog entry is priced flat, without the generation model's long-prompt uplift.

OpenRouter lists ``openai/gpt-6-luna-decisions`` at one input price for
every prompt length (no ``overrides``), but its name matched the
``gpt-6-`` rule of ``_long_context_uplift``, so ``calculate_cost``
doubled the input price above 272K prompt tokens.  The uplift still
applies to the generation model of the same name.
"""

from __future__ import annotations

import pytest

from kiss.core.models.model_info import MODEL_INFO, _long_context_uplift, calculate_cost

_DECISIONS = "openrouter/openai/gpt-6-luna-decisions"
_GENERATION = "openrouter/openai/gpt-6-luna"


def _per_token(name: str) -> float:
    return MODEL_INFO[name].input_price_per_1M / 1_000_000


@pytest.mark.parametrize("prompt", [1_000, 272_000, 272_001, 500_000, 1_000_000])
def test_decisions_entry_is_priced_flat(prompt: int) -> None:
    info = MODEL_INFO[_DECISIONS]
    assert info.is_decisions_supported and not info.is_generation_supported
    assert calculate_cost(_DECISIONS, prompt, 0) == pytest.approx(prompt * _per_token(_DECISIONS))


def test_generation_entry_keeps_the_uplift() -> None:
    assert MODEL_INFO[_GENERATION].is_generation_supported
    assert _long_context_uplift(_GENERATION) == (272_000, 2.0, 1.5)
    short = calculate_cost(_GENERATION, 272_000, 0)
    long = calculate_cost(_GENERATION, 272_001, 0)
    assert short == pytest.approx(272_000 * _per_token(_GENERATION))
    assert long == pytest.approx(272_001 * 2 * _per_token(_GENERATION))


def test_no_decisions_only_entry_gets_an_uplift() -> None:
    for name, info in MODEL_INFO.items():
        if info.is_decisions_supported and not info.is_generation_supported:
            flat = calculate_cost(name, 10, 0) * 100_000
            assert calculate_cost(name, 1_000_000, 0) == pytest.approx(flat), name
