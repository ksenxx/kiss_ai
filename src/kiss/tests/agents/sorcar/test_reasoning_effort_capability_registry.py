# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SorcarAgent's endpoint carry-over set derives from the vendor registry.

The registry itself (``model_info.OPENAI_COMPATIBLE_PROVIDERS``) and the
wire behavior of the capability probe are covered by
``kiss.tests.core.models.test_reasoning_effort_capability_registry``;
this file pins the sorcar-side contract: ``set_model``'s
``_FACTORY_DEFAULT_BASE_URLS`` cannot drift from the registry.
"""

from __future__ import annotations

from kiss.core.models.model_info import OPENAI_COMPATIBLE_PROVIDERS


class TestSorcarEndpointCarryOverDerivesFromRegistry:
    """SorcarAgent and the registry are a single source of truth."""

    def test_sorcar_default_base_urls_derive_from_registry(self) -> None:
        """set_model's endpoint carry-over set covers every registered vendor."""
        from kiss.agents.sorcar.sorcar_agent import _FACTORY_DEFAULT_BASE_URLS

        assert _FACTORY_DEFAULT_BASE_URLS == frozenset(
            p.base_url.rstrip("/") for p in OPENAI_COMPATIBLE_PROVIDERS
        )
