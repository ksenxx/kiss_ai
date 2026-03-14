"""Test suite for increasing branch coverage of KISS core components.

These tests target specific branches and edge cases in:
- base.py: Base class for agents
- utils.py: Utility functions
- model_info.py: Model information and lookup
"""

import pytest

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.utils import (
    read_project_file,
    read_project_file_from_package,
)


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

    def test_save_skips_when_save_trajectory_disabled(self, tmp_path, monkeypatch):
        agent = Base("no-save")
        agent.save_trajectory = False
        monkeypatch.setattr(config_module.DEFAULT_CONFIG.agent, "artifact_dir", str(tmp_path))
        agent._save()
        assert not (tmp_path / "trajectories").exists()

    def test_save_writes_trajectory_by_default(self, tmp_path, monkeypatch):
        import time

        agent = Base("save-me")
        agent.model_name = "gpt-4o-mini"
        agent.function_map = []
        agent.messages = []
        agent.step_count = 0
        agent.total_tokens_used = 0
        agent.budget_used = 0.0
        agent.run_start_timestamp = int(time.time())
        monkeypatch.setattr(config_module.DEFAULT_CONFIG.agent, "artifact_dir", str(tmp_path))
        agent._save()
        files = list((tmp_path / "trajectories").glob("trajectory_save-me_*.yaml"))
        assert len(files) == 1


class TestUtils:
    def test_read_project_file_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent/path/to/file.txt")

    def test_read_project_file_from_package_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file_from_package("nonexistent_file.txt")

    def test_read_project_file_accepts_backslash_separators(self):
        content = read_project_file(r"kiss\__init__.py")
        assert "__version__" in content


class TestModelHelpers:
    def _create_model(self):
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt, attachments=None):
                pass

            def generate(self):
                return "", None

            def generate_and_process_with_tools(self, function_map):
                return [], "", None

            def add_function_results_to_conversation_and_return(self, function_results):
                pass

            def add_message_to_conversation(self, role, content):
                pass

            def extract_input_output_token_counts_from_response(self, response):
                return 0, 0, 0, 0

            def get_embedding(self, text, embedding_model=None):
                return []

        return ConcreteModel("test_model")

    def test_type_to_json_schema_all_types(self):
        m = self._create_model()
        type_map = [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
        ]
        for py_type, expected in type_map:
            result = m._python_type_to_json_schema(py_type)
            assert result["type"] == expected

        result = m._python_type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"


class TestModelInfoEdgeCases:
    def test_unknown_model_raises_error(self):
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import model

        with pytest.raises(KISSError, match="Unknown model name"):
            model("nonexistent-model-xyz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
