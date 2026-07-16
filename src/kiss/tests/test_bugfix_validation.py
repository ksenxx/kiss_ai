# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests validating that bugs.md fixes are correct.

Each test verifies the fix for a specific bug by exercising real code paths —
no mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations


class TestB4DeepseekTputInModelInfo:
    def test_tput_model_in_model_info(self) -> None:
        """The -tput variant has a pricing entry so calculate_cost works."""
        from kiss.core.models.model_info import MODEL_INFO

        assert "deepseek-ai/DeepSeek-R1-0528-tput" in MODEL_INFO

    def test_tput_model_has_nonzero_pricing(self) -> None:
        """Pricing is non-zero (not free)."""
        from kiss.core.models.model_info import MODEL_INFO

        info = MODEL_INFO["deepseek-ai/DeepSeek-R1-0528-tput"]
        assert info.input_price_per_1M > 0
        assert info.output_price_per_1M > 0
