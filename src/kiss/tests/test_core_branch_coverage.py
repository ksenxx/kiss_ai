"""Test suite for increasing branch coverage of KISS core components.

These tests target specific branches and edge cases in:
- base.py: Base class for agents
- utils.py: Utility functions
- model_info.py: Model information and lookup
"""

import pytest

from kiss.core.base import Base
from kiss.core.utils import (
    read_project_file,
    read_project_file_from_package,
)

# ---------------------------------------------------------------------------
# kiss/core/base.py — Base
# ---------------------------------------------------------------------------

class TestBaseClass:
    @pytest.fixture(autouse=True)
    def base_state(self):
        original_counter = Base.agent_counter
        original_budget = Base.global_budget_used
        yield
        Base.agent_counter = original_counter
        Base.global_budget_used = original_budget

    def test_build_state_dict_unknown_model(self):
        import time

        agent = Base("test")
        agent.model_name = "unknown-model-xyz"
        agent.function_map = []
        agent.messages = []
        agent.step_count = 0
        agent.total_tokens_used = 0
        agent.budget_used = 0.0
        agent.run_start_timestamp = int(time.time())
        state = agent._build_state_dict()
        assert state["max_tokens"] is None


# ---------------------------------------------------------------------------
# kiss/core/utils.py — read_project_file, read_project_file_from_package
# ---------------------------------------------------------------------------

class TestUtils:
    def test_read_project_file_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent/path/to/file.txt")

    def test_read_project_file_from_package_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file_from_package("nonexistent_file.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
