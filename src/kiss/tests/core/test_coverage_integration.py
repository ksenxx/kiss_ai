# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests targeting uncovered branches in core/.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

from unittest import TestCase

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
)
from kiss.core.utils import (
    finish as utils_finish,
)


class TestKISSAgentErrors(TestCase):
    def test_agent_budget_exceeded(self) -> None:
        agent = KISSAgent("test-agent-budget")
        agent.budget_used = 10.0
        agent.max_budget = 5.0
        agent.max_steps = 100
        agent.step_count = 0
        with self.assertRaises(KISSError) as ctx:
            agent._check_limits()
        assert "budget exceeded" in str(ctx.exception)


class TestFinalizeStreamResponseErrors(TestCase):
    def test_raises_on_empty(self) -> None:
        with pytest.raises(KISSError, match="empty"):
            OpenAICompatibleModel._finalize_stream_response(None, None)


class TestOpenAICompatibleModelEmbedding:
    def test_get_embedding_failure(self) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel(
            "fake-model", base_url="http://localhost:1", api_key="test"
        )
        m.initialize("test")
        with pytest.raises(KISSError, match="Embedding generation failed"):
            m.get_embedding("Hello world")


class TestUtilsFunctionsExtra:
    def test_utils_finish(self) -> None:
        result = utils_finish(success=True, summary_in_html="42")
        payload = yaml.safe_load(result)
        assert payload["success"] is True
        assert payload["is_continue"] is False
        assert payload["summary"] == "<p>42</p>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
