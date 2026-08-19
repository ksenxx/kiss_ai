# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end offline regression tests for the core models simplification.

Pins the observable behavior of ``kiss.core.models.model``,
``kiss.core.models.model_info`` and ``kiss.core.models.__init__`` before and
after simplification.  No mocks, no network: every test calls the real code
with real values.
"""

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import (
    MODEL_INFO,
    calculate_cost,
    get_max_context_length,
    model,
)
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401


def test_model_unknown_name_raises() -> None:
    """Unrecognized model names raise KISSError."""
    with pytest.raises(KISSError, match="Unknown model name"):
        model("totally-unknown-model-xyz")


def test_calculate_cost_basic_and_unknown() -> None:
    """Cost math is (tokens * price) / 1M; unknown models allow only zero usage."""
    name = next(
        n for n, i in MODEL_INFO.items()
        if n.startswith("claude-") and i.is_generation_supported
    )
    info = MODEL_INFO[name]
    expected = (1000 * info.input_price_per_1M + 500 * info.output_price_per_1M) / 1e6
    assert calculate_cost(name, 1000, 500) == pytest.approx(expected)
    assert calculate_cost(f"anthropic/{name}", 1000, 500) == pytest.approx(expected)
    assert calculate_cost("no-such-model", 0, 0) == 0.0
    with pytest.raises(KISSError, match="unknown model"):
        calculate_cost("no-such-model", 1, 0)


def test_get_max_context_length() -> None:
    """Context lengths resolve directly and via provider-prefix stripping."""
    name = next(n for n in MODEL_INFO if n.startswith("gpt-"))
    assert get_max_context_length(name) == MODEL_INFO[name].context_length
    assert get_max_context_length(f"openai/{name}") == MODEL_INFO[name].context_length
    with pytest.raises(KISSError, match="not found in MODEL_INFO"):
        get_max_context_length("no-such-model")
