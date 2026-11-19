# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Moonshot/Kimi reasoning-effort alias generation.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.scripts.test_update_models_moonshot_thinking``; the non-core tests remain there.
"""

from __future__ import annotations

MOONSHOT_LEVELS = ("low", "high", "max")


ALL_KNOWN_LEVELS = ("low", "medium", "high", "xhigh", "max")


class TestRuntimeAliasResolution:
    """Kimi K3 aliases must resolve and route like every other alias."""

    def test_strip_thinking_alias_maps_max_to_base(self) -> None:
        from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias

        for level in MOONSHOT_LEVELS:
            name = f"openrouter/moonshotai/kimi-k3-{level}"
            assert name in MODEL_INFO, f"{name} missing from loaded MODEL_INFO"
            assert _strip_thinking_alias(name) == "openrouter/moonshotai/kimi-k3"

    def test_provider_model_name_and_effort_for_max_alias(self) -> None:
        from kiss.core.models.openai_compatible_model import (
            _model_thinking_level,
            _provider_model_name,
        )

        assert (
            _provider_model_name("openrouter/moonshotai/kimi-k3-max")
            == "moonshotai/kimi-k3"
        )
        assert _model_thinking_level("openrouter/moonshotai/kimi-k3-max") == "max"

    def test_unmarked_max_suffix_is_not_stripped(self) -> None:
        """A name ending in -max with no catalog alias marker stays intact."""
        from kiss.core.models.model_info import _strip_thinking_alias

        assert _strip_thinking_alias("acme/custom-max") == "acme/custom-max"

    def test_kimi_aliases_cost_the_same_as_base(self) -> None:
        from kiss.core.models.model_info import calculate_cost

        base = "openrouter/moonshotai/kimi-k3"
        base_cost = calculate_cost(base, 1000, 500, 200, 100)
        for level in MOONSHOT_LEVELS:
            assert calculate_cost(f"{base}-{level}", 1000, 500, 200, 100) == base_cost


def test_every_bundled_alias_thinking_is_a_known_level() -> None:
    """Every marked alias in the bundled catalog uses a known level name."""
    from kiss.core.models.model_info import MODEL_INFO

    for name, info in MODEL_INFO.items():
        if info.alias_of:
            assert info.thinking in ALL_KNOWN_LEVELS, (name, info.thinking)
