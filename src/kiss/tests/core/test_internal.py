"""Tests for internal KISS components: utils, Base class.

Merged from: test_internal, test_core_branch_coverage.
"""

import os
import tempfile
import time
import unittest
from collections.abc import Generator

import pytest

from kiss.core.base import Base
from kiss.core.kiss_error import KISSError
from kiss.core.utils import (
    add_prefix_to_each_line,
    fc,
    read_project_file,
    read_project_file_from_package,
)


class TestUtilsFunctions(unittest.TestCase):
    def test_add_prefix_to_each_line(self) -> None:
        self.assertEqual(
            add_prefix_to_each_line("line1\nline2\nline3", "> "),
            "> line1\n> line2\n> line3",
        )
        self.assertEqual(add_prefix_to_each_line("single line", ">> "), ">> single line")

    def test_fc_reads_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for fc function")
            temp_path = f.name
        try:
            self.assertEqual(fc(temp_path), "Test content for fc function")
        finally:
            os.unlink(temp_path)


class TestReadProjectFile:
    def test_read_project_file_not_found(self) -> None:
        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent/path/to/file.txt")

    def test_read_project_file_from_package_not_found(self) -> None:
        with pytest.raises(KISSError, match="Could not find"):
            read_project_file_from_package("nonexistent_file.txt")


class TestBaseClass:
    @pytest.fixture(autouse=True)
    def base_state(self) -> Generator[None]:
        original_counter = Base.agent_counter
        original_budget = Base.global_budget_used
        yield
        Base.agent_counter = original_counter
        Base.global_budget_used = original_budget

    def test_build_state_dict_unknown_model(self) -> None:
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
