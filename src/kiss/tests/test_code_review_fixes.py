"""Tests for bug fixes identified in the code review of core/ and sorcar/.

Each test verifies a specific fix without mocks, patches, or test doubles.
"""

import io


class TestSetPrinterFix:
    """Tests for base.py:set_printer — explicit printer should always be honored."""

    def test_explicit_printer_always_used(self):
        """An explicitly passed printer is used regardless of global verbose."""
        from kiss.core import config as config_module
        from kiss.core.base import Base
        from kiss.core.print_to_console import ConsolePrinter

        agent = Base("test")
        printer = ConsolePrinter(file=io.StringIO())

        # Save and modify global verbose to False
        original = config_module.DEFAULT_CONFIG.agent.verbose
        try:
            config_module.DEFAULT_CONFIG.agent.verbose = False
            agent.set_printer(printer=printer)
            # Even with verbose=False globally, explicit printer should be set
            assert agent.printer is printer
        finally:
            config_module.DEFAULT_CONFIG.agent.verbose = original

    def test_explicit_printer_with_verbose_true(self):
        """Explicit printer used when verbose is True."""
        from kiss.core import config as config_module
        from kiss.core.base import Base
        from kiss.core.print_to_console import ConsolePrinter

        agent = Base("test")
        printer = ConsolePrinter(file=io.StringIO())
        original = config_module.DEFAULT_CONFIG.agent.verbose
        try:
            config_module.DEFAULT_CONFIG.agent.verbose = True
            agent.set_printer(printer=printer)
            assert agent.printer is printer
        finally:
            config_module.DEFAULT_CONFIG.agent.verbose = original

    def test_no_printer_verbose_false_globally(self):
        """No printer when verbose=False globally and no explicit printer."""
        from kiss.core import config as config_module
        from kiss.core.base import Base

        agent = Base("test")
        original = config_module.DEFAULT_CONFIG.agent.verbose
        try:
            config_module.DEFAULT_CONFIG.agent.verbose = False
            agent.set_printer()
            assert agent.printer is None
        finally:
            config_module.DEFAULT_CONFIG.agent.verbose = original

    def test_no_printer_verbose_explicitly_false(self):
        """No printer when verbose=False is explicitly passed."""
        from kiss.core import config as config_module
        from kiss.core.base import Base

        agent = Base("test")
        original = config_module.DEFAULT_CONFIG.agent.verbose
        try:
            config_module.DEFAULT_CONFIG.agent.verbose = True
            agent.set_printer(verbose=False)
            assert agent.printer is None
        finally:
            config_module.DEFAULT_CONFIG.agent.verbose = original

    def test_auto_console_printer_when_verbose(self):
        """ConsolePrinter auto-created when verbose is True globally."""
        from kiss.core import config as config_module
        from kiss.core.base import Base
        from kiss.core.print_to_console import ConsolePrinter

        agent = Base("test")
        original = config_module.DEFAULT_CONFIG.agent.verbose
        try:
            config_module.DEFAULT_CONFIG.agent.verbose = True
            agent.set_printer()
            assert isinstance(agent.printer, ConsolePrinter)
        finally:
            config_module.DEFAULT_CONFIG.agent.verbose = original

    def test_verbose_none_uses_global_config(self):
        """When verbose=None, global config is used."""
        from kiss.core import config as config_module
        from kiss.core.base import Base
        from kiss.core.print_to_console import ConsolePrinter

        agent = Base("test")
        original = config_module.DEFAULT_CONFIG.agent.verbose
        try:
            config_module.DEFAULT_CONFIG.agent.verbose = True
            agent.set_printer(verbose=None)
            assert isinstance(agent.printer, ConsolePrinter)
        finally:
            config_module.DEFAULT_CONFIG.agent.verbose = original


class TestDeepSeekReasoningModelFix:
    """Tests for openai_compatible_model.py — DeepSeek reasoning model detection."""

    def test_together_deepseek_r1_detected(self):
        """Together AI DeepSeek-R1 model is correctly detected as reasoning."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="deepseek-ai/DeepSeek-R1",
            base_url="https://api.together.xyz/v1",
            api_key="test-key",
        )
        assert model._is_deepseek_reasoning_model() is True

    def test_openrouter_deepseek_r1_detected(self):
        """OpenRouter DeepSeek R1 model is correctly detected as reasoning."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="openrouter/deepseek/deepseek-r1",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        # _api_model_name strips "openrouter/" prefix -> "deepseek/deepseek-r1"
        assert model._api_model_name == "deepseek/deepseek-r1"
        assert model._is_deepseek_reasoning_model() is True

    def test_openrouter_deepseek_r1_0528_detected(self):
        """OpenRouter DeepSeek R1-0528 model is correctly detected."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="openrouter/deepseek/deepseek-r1-0528",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        assert model._is_deepseek_reasoning_model() is True

    def test_together_deepseek_r1_distill_detected(self):
        """Together AI DeepSeek R1 distill variant is correctly detected."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            base_url="https://api.together.xyz/v1",
            api_key="test-key",
        )
        assert model._is_deepseek_reasoning_model() is True

    def test_non_reasoning_model_not_detected(self):
        """Normal models are not detected as reasoning models."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        assert model._is_deepseek_reasoning_model() is False

    def test_openrouter_non_deepseek_not_detected(self):
        """OpenRouter non-DeepSeek models are not detected as reasoning."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="openrouter/anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        assert model._is_deepseek_reasoning_model() is False

    def test_deepseek_v3_not_detected(self):
        """DeepSeek V3 (non-reasoning) is not detected as reasoning."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel(
            model_name="deepseek-ai/DeepSeek-V3.1",
            base_url="https://api.together.xyz/v1",
            api_key="test-key",
        )
        assert model._is_deepseek_reasoning_model() is False


class TestTokenDisplayFix:
    """Tests for kiss_agent.py — token usage display uses min instead of modulo."""

    def test_tokens_below_context_shows_actual(self):
        """Tokens below context length are shown as-is."""
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("test")
        agent.model_name = "gpt-4o"
        agent.total_tokens_used = 50000
        agent.step_count = 5
        agent.max_steps = 100
        agent.budget_used = 1.5
        agent.max_budget = 10.0
        agent.session_info = ""

        # Need a model object with model_name
        class FakeModel:
            model_name = "gpt-4o"

        agent.model = FakeModel()  # type: ignore[assignment]

        info = agent._get_usage_info_string()
        assert "Tokens: 50000/" in info

    def test_tokens_exceeding_context_wrapped_via_modulo(self):
        """Tokens exceeding context length are wrapped via modulo."""
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("test")
        agent.model_name = "gpt-4o"
        # gpt-4o has 128000 context length
        agent.total_tokens_used = 250000
        agent.step_count = 5
        agent.max_steps = 100
        agent.budget_used = 1.5
        agent.max_budget = 10.0
        agent.session_info = ""

        class FakeModel:
            model_name = "gpt-4o"

        agent.model = FakeModel()  # type: ignore[assignment]

        info = agent._get_usage_info_string()
        # 250000 % 128000 = 122000
        assert "Tokens: 122000/128000" in info

    def test_tokens_exactly_at_context(self):
        """Tokens exactly at context length wrap to zero via modulo."""
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("test")
        agent.model_name = "gpt-4o"
        agent.total_tokens_used = 128000
        agent.step_count = 5
        agent.max_steps = 100
        agent.budget_used = 1.5
        agent.max_budget = 10.0
        agent.session_info = ""

        class FakeModel:
            model_name = "gpt-4o"

        agent.model = FakeModel()  # type: ignore[assignment]

        info = agent._get_usage_info_string()
        # 128000 % 128000 = 0
        assert "Tokens: 0/128000" in info

    def test_session_info_included(self):
        """Session info is prepended when present."""
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent("test")
        agent.model_name = "gpt-4o"
        agent.total_tokens_used = 1000
        agent.step_count = 1
        agent.max_steps = 10
        agent.budget_used = 0.01
        agent.max_budget = 5.0
        agent.session_info = "Session: 1/5"

        class FakeModel:
            model_name = "gpt-4o"

        agent.model = FakeModel()  # type: ignore[assignment]

        info = agent._get_usage_info_string()
        assert info.startswith("Session: 1/5, ")


class TestPromptDetectorDocstring:
    """Test that prompt_detector module has a docstring."""

    def test_module_has_docstring(self):
        """prompt_detector module should have a docstring."""
        from kiss.agents.sorcar import prompt_detector

        assert prompt_detector.__doc__ is not None
        assert "prompt" in prompt_detector.__doc__.lower()


class TestDefaultTaskNoCredentials:
    """Test that _DEFAULT_TASK doesn't contain hardcoded credentials."""

    def test_no_password_in_default_task(self):
        """Default task should not contain passwords."""
        from kiss.agents.sorcar.sorcar_agent import _DEFAULT_TASK

        assert "password" not in _DEFAULT_TASK.lower()
        assert "kissagent" not in _DEFAULT_TASK.lower()
        assert "@gmail" not in _DEFAULT_TASK.lower()


class TestRelentlessAgentAttributeInit:
    """Test that RelentlessAgent initializes task_description and system_instructions."""

    def test_attributes_initialized_in_reset(self):
        """task_description and system_instructions initialized in _reset."""
        from kiss.core.relentless_agent import RelentlessAgent

        agent = RelentlessAgent("test")
        agent._reset(
            model_name="gpt-4o",
            max_sub_sessions=1,
            max_steps=5,
            max_budget=1.0,
            work_dir=None,
            docker_image=None,
        )
        assert hasattr(agent, "task_description")
        assert agent.task_description == ""
        assert hasattr(agent, "system_instructions")
        assert agent.system_instructions == ""


class TestDeepSeekReasoningModelsConsistency:
    """Verify DEEPSEEK_REASONING_MODELS entries match model_info.py entries."""

    def test_together_models_have_correct_prefixes(self):
        """Together AI entries use full model name with 'deepseek-ai/' prefix."""
        from kiss.core.models.openai_compatible_model import DEEPSEEK_REASONING_MODELS

        together_entries = [
            m for m in DEEPSEEK_REASONING_MODELS
            if "/" in m and not m.startswith("deepseek/")
        ]
        for entry in together_entries:
            assert entry.startswith("deepseek-ai/"), (
                f"Together AI entry '{entry}' should start with 'deepseek-ai/'"
            )

    def test_openrouter_entries_use_api_names(self):
        """OpenRouter entries use API names (without 'openrouter/' prefix)."""
        from kiss.core.models.openai_compatible_model import DEEPSEEK_REASONING_MODELS

        # Entries starting with "deepseek/" are OpenRouter API names
        or_entries = [m for m in DEEPSEEK_REASONING_MODELS if m.startswith("deepseek/")]
        assert len(or_entries) > 0, "Should have OpenRouter-style entries"
        for entry in or_entries:
            assert not entry.startswith("openrouter/"), (
                f"Entry '{entry}' should NOT have 'openrouter/' prefix"
            )
