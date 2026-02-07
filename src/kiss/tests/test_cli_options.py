# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for verifying command line options work correctly.

These tests verify that CLI arguments defined in various config.py files
are properly parsed and applied to the configuration system.

Note: These tests define config classes locally to avoid import issues with
optional dependencies (like google.adk). This tests the config_builder
mechanism works correctly with the actual field types and names used
in the real config files.
"""

import sys
import unittest

from pydantic import BaseModel, Field

from kiss.core import config as config_module
from kiss.core.config_builder import add_config


class CLITestBase(unittest.TestCase):
    """Base class for CLI tests that handles setup and teardown of config state."""

    def setUp(self):
        """Save original config and argv before each test."""
        self.original_config = config_module.DEFAULT_CONFIG
        self.original_argv = sys.argv

    def tearDown(self):
        """Restore original config and argv after each test."""
        sys.argv = self.original_argv
        config_module.DEFAULT_CONFIG = self.original_config


class TestCoreConfigCLI(CLITestBase):
    """Tests for core config CLI options from kiss.core.config."""

    def test_agent_max_steps(self):
        """Test --agent.max-steps CLI option."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--agent.max-steps", "200"]
        add_config("dummy", DummyConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.agent.max_steps, 200)

    def test_agent_max_steps_underscore(self):
        """Test --agent.max_steps CLI option with underscore style."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--agent.max_steps", "150"]
        add_config("dummy", DummyConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.agent.max_steps, 150)

    def test_agent_verbose_flag(self):
        """Test --agent.verbose and --no-agent.verbose flags."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        # Test disabling verbose
        sys.argv = ["test", "--no-agent.verbose"]
        add_config("dummy", DummyConfig)
        self.assertFalse(config_module.DEFAULT_CONFIG.agent.verbose)

        # Reset and test enabling
        config_module.DEFAULT_CONFIG = self.original_config
        sys.argv = ["test", "--agent.verbose"]
        add_config("dummy", DummyConfig)
        self.assertTrue(config_module.DEFAULT_CONFIG.agent.verbose)

    def test_agent_debug_flag(self):
        """Test --agent.debug and --no-agent.debug flags."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--agent.debug"]
        add_config("dummy", DummyConfig)
        self.assertTrue(config_module.DEFAULT_CONFIG.agent.debug)

    def test_agent_max_budget(self):
        """Test --agent.max-agent-budget CLI option."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--agent.max-agent-budget", "25.5"]
        add_config("dummy", DummyConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.agent.max_agent_budget, 25.5)

    def test_agent_global_max_budget(self):
        """Test --agent.global-max-budget CLI option."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--agent.global-max-budget", "500.0"]
        add_config("dummy", DummyConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.agent.global_max_budget, 500.0)

    def test_agent_use_web_flag(self):
        """Test --agent.use-web and --no-agent.use-web flags."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--no-agent.use-web"]
        add_config("dummy", DummyConfig)
        self.assertFalse(config_module.DEFAULT_CONFIG.agent.use_web)

    def test_nested_relentless_coding_agent_config(self):
        """Test nested relentless_coding_agent config options."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = [
            "test",
            "--agent.relentless-coding-agent.orchestrator-model-name",
            "gpt-4",
            "--agent.relentless-coding-agent.max-steps",
            "300",
            "--agent.relentless-coding-agent.max-budget",
            "50.0",
            "--agent.relentless-coding-agent.trials",
            "100",
        ]
        add_config("dummy", DummyConfig)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.relentless_coding_agent.orchestrator_model_name,
            "gpt-4",
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.relentless_coding_agent.max_steps, 300
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.relentless_coding_agent.max_budget, 50.0
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.relentless_coding_agent.trials, 100
        )

    def test_nested_kiss_coding_agent_config(self):
        """Test nested kiss_coding_agent config options."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = [
            "test",
            "--agent.kiss-coding-agent.orchestrator-model-name",
            "claude-opus-4-6",
            "--agent.kiss-coding-agent.subtasker-model-name",
            "claude-sonnet-4-5",
            "--agent.kiss-coding-agent.refiner-model-name",
            "gemini-3-flash",
            "--agent.kiss-coding-agent.max-steps",
            "250",
            "--agent.kiss-coding-agent.max-budget",
            "75.0",
        ]
        add_config("dummy", DummyConfig)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.kiss_coding_agent.orchestrator_model_name,
            "claude-opus-4-6",
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.kiss_coding_agent.subtasker_model_name,
            "claude-sonnet-4-5",
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.kiss_coding_agent.refiner_model_name,
            "gemini-3-flash",
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.kiss_coding_agent.max_steps, 250
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.agent.kiss_coding_agent.max_budget, 75.0
        )

    def test_docker_config(self):
        """Test Docker config CLI options."""
        from kiss.core.config_builder import add_config

        class DummyConfig(BaseModel):
            dummy: str = Field(default="test")

        sys.argv = ["test", "--docker.client-shared-path", "/custom/path"]
        add_config("dummy", DummyConfig)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.docker.client_shared_path, "/custom/path"
        )


class TestAlgoTuneConfigCLI(CLITestBase):
    """Tests for AlgoTune config CLI options.

    Uses a local config class matching the real AlgoTuneConfig to avoid
    import issues with optional dependencies.
    """

    def _get_algotune_config(self):
        """Create a config class matching AlgoTuneConfig."""

        class AlgoTuneConfig(BaseModel):
            task: str = Field(default="matrix_multiplication")
            all_tasks: bool = Field(default=False)
            algotune_path: str = Field(default="/tmp/AlgoTune")
            algotune_repo_url: str = Field(
                default="https://github.com/oripress/AlgoTune.git"
            )
            num_test_problems: int = Field(default=3)
            problem_size: int = Field(default=100)
            num_timing_runs: int = Field(default=5)
            random_seed: int = Field(default=42)
            model: str = Field(default="gemini-3-flash-preview")

        return AlgoTuneConfig

    def test_algotune_task(self):
        """Test --algotune.task CLI option."""
        algotune_config = self._get_algotune_config()
        sys.argv = ["test", "--algotune.task", "sorting"]
        add_config("algotune", algotune_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.algotune.task, "sorting")

    def test_algotune_all_tasks_flag(self):
        """Test --algotune.all-tasks flag."""
        algotune_config = self._get_algotune_config()
        sys.argv = ["test", "--algotune.all-tasks"]
        add_config("algotune", algotune_config)
        self.assertTrue(config_module.DEFAULT_CONFIG.algotune.all_tasks)

    def test_algotune_num_test_problems(self):
        """Test --algotune.num-test-problems CLI option."""
        algotune_config = self._get_algotune_config()
        sys.argv = ["test", "--algotune.num-test-problems", "10"]
        add_config("algotune", algotune_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.algotune.num_test_problems, 10)

    def test_algotune_problem_size(self):
        """Test --algotune.problem-size CLI option."""
        algotune_config = self._get_algotune_config()
        sys.argv = ["test", "--algotune.problem-size", "500"]
        add_config("algotune", algotune_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.algotune.problem_size, 500)

    def test_algotune_model(self):
        """Test --algotune.model CLI option."""
        algotune_config = self._get_algotune_config()
        sys.argv = ["test", "--algotune.model", "gpt-4-turbo"]
        add_config("algotune", algotune_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.algotune.model, "gpt-4-turbo")

    def test_algotune_random_seed(self):
        """Test --algotune.random-seed CLI option."""
        algotune_config = self._get_algotune_config()
        sys.argv = ["test", "--algotune.random-seed", "123"]
        add_config("algotune", algotune_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.algotune.random_seed, 123)


class TestSWEBenchConfigCLI(CLITestBase):
    """Tests for SWE-bench Verified config CLI options.

    Uses a local config class matching the real SWEBenchVerifiedConfig to avoid
    import issues with optional dependencies.
    """

    def _get_swebench_config(self):
        """Create a config class matching SWEBenchVerifiedConfig."""

        class SWEBenchVerifiedConfig(BaseModel):
            dataset_name: str = Field(default="princeton-nlp/SWE-bench_Verified")
            split: str = Field(default="test")
            instance_id: str = Field(default="")
            instance_ids: list[str] = Field(default_factory=list)
            max_instances: int = Field(default=0)
            docker_image_base: str = Field(
                default="slimshetty/swebench-verified:sweb.eval.x86_64."
            )
            workdir: str = Field(default="/testbed")
            model: str = Field(default="gemini-3-pro-preview")
            max_steps: int = Field(default=100)
            max_budget: float = Field(default=5.0)
            num_samples: int = Field(default=1)
            run_evaluation: bool = Field(default=True)
            max_workers: int = Field(default=8)
            run_id: str = Field(default="kiss_swebench_verified")
            save_patches: bool = Field(default=True)
            save_trajectories: bool = Field(default=True)

        return SWEBenchVerifiedConfig

    def test_swebench_dataset_name(self):
        """Test --swebench-verified.dataset-name CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.dataset-name", "custom/dataset"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.swebench_verified.dataset_name, "custom/dataset"
        )

    def test_swebench_instance_id(self):
        """Test --swebench-verified.instance-id CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.instance-id", "django__django-12345"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.swebench_verified.instance_id, "django__django-12345"
        )

    def test_swebench_max_instances(self):
        """Test --swebench-verified.max-instances CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.max-instances", "50"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.max_instances, 50)

    def test_swebench_model(self):
        """Test --swebench-verified.model CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.model", "claude-opus-4-6"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.model, "claude-opus-4-6")

    def test_swebench_max_steps(self):
        """Test --swebench-verified.max-steps CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.max-steps", "200"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.max_steps, 200)

    def test_swebench_max_budget(self):
        """Test --swebench-verified.max-budget CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.max-budget", "10.0"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.max_budget, 10.0)

    def test_swebench_num_samples(self):
        """Test --swebench-verified.num-samples CLI option."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--swebench-verified.num-samples", "5"]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.num_samples, 5)

    def test_swebench_run_evaluation_flag(self):
        """Test --swebench-verified.run-evaluation flag."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--no-swebench-verified.run-evaluation"]
        add_config("swebench_verified", swebench_config)
        self.assertFalse(config_module.DEFAULT_CONFIG.swebench_verified.run_evaluation)

    def test_swebench_save_patches_flag(self):
        """Test --swebench-verified.save-patches flag."""
        swebench_config = self._get_swebench_config()
        sys.argv = ["test", "--no-swebench-verified.save-patches"]
        add_config("swebench_verified", swebench_config)
        self.assertFalse(config_module.DEFAULT_CONFIG.swebench_verified.save_patches)


class TestKISSEvolveConfigCLI(CLITestBase):
    """Tests for KISSEvolve config CLI options.

    Uses a local config class matching the real KISSEvolveConfig to avoid
    import issues with optional dependencies.
    """

    def _get_kiss_evolve_config(self):
        """Create a config class matching KISSEvolveConfig."""

        class KISSEvolveConfig(BaseModel):
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)
            mutation_rate: float = Field(default=0.7)
            elite_size: int = Field(default=2)
            num_islands: int = Field(default=2)
            migration_frequency: int = Field(default=5)
            migration_size: int = Field(default=1)
            migration_topology: str = Field(default="ring")
            enable_novelty_rejection: bool = Field(default=False)
            novelty_threshold: float = Field(default=0.95)
            max_rejection_attempts: int = Field(default=5)
            parent_sampling_method: str = Field(default="power_law")
            power_law_alpha: float = Field(default=1.0)
            performance_novelty_lambda: float = Field(default=1.0)

        return KISSEvolveConfig

    def test_kiss_evolve_max_generations(self):
        """Test --kiss-evolve.max-generations CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.max-generations", "20"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.max_generations, 20)

    def test_kiss_evolve_population_size(self):
        """Test --kiss-evolve.population-size CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.population-size", "16"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.population_size, 16)

    def test_kiss_evolve_mutation_rate(self):
        """Test --kiss-evolve.mutation-rate CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.mutation-rate", "0.5"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.mutation_rate, 0.5)

    def test_kiss_evolve_elite_size(self):
        """Test --kiss-evolve.elite-size CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.elite-size", "4"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.elite_size, 4)

    def test_kiss_evolve_num_islands(self):
        """Test --kiss-evolve.num-islands CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.num-islands", "4"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.num_islands, 4)

    def test_kiss_evolve_migration_topology(self):
        """Test --kiss-evolve.migration-topology CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.migration-topology", "fully_connected"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.kiss_evolve.migration_topology, "fully_connected"
        )

    def test_kiss_evolve_enable_novelty_rejection(self):
        """Test --kiss-evolve.enable-novelty-rejection flag."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.enable-novelty-rejection"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertTrue(config_module.DEFAULT_CONFIG.kiss_evolve.enable_novelty_rejection)

    def test_kiss_evolve_novelty_threshold(self):
        """Test --kiss-evolve.novelty-threshold CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.novelty-threshold", "0.8"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.novelty_threshold, 0.8)

    def test_kiss_evolve_parent_sampling_method(self):
        """Test --kiss-evolve.parent-sampling-method CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.parent-sampling-method", "tournament"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.kiss_evolve.parent_sampling_method, "tournament"
        )

    def test_kiss_evolve_power_law_alpha(self):
        """Test --kiss-evolve.power-law-alpha CLI option."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = ["test", "--kiss-evolve.power-law-alpha", "2.0"]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.power_law_alpha, 2.0)


class TestSelfEvolvingMultiAgentConfigCLI(CLITestBase):
    """Tests for Self Evolving Multi Agent config CLI options.

    Uses a local config class matching the real SelfEvolvingMultiAgentConfig to avoid
    import issues with optional dependencies.
    """

    def _get_self_evolving_config(self):
        """Create a config class matching SelfEvolvingMultiAgentConfig."""

        class SelfEvolvingMultiAgentConfig(BaseModel):
            model: str = Field(default="gemini-3-flash-preview")
            sub_agent_model: str = Field(default="gemini-3-flash-preview")
            evolver_model: str = Field(default="gemini-3-flash-preview")
            max_steps: int = Field(default=100)
            max_budget: float = Field(default=10.0)
            max_retries: int = Field(default=3)
            sub_agent_max_steps: int = Field(default=50)
            sub_agent_max_budget: float = Field(default=2.0)
            docker_image: str = Field(default="python:3.12-slim")
            workdir: str = Field(default="/workspace")

        return SelfEvolvingMultiAgentConfig

    def test_self_evolving_model(self):
        """Test --self-evolving-multi-agent.model CLI option."""
        self_evolving_config = self._get_self_evolving_config()
        sys.argv = ["test", "--self-evolving-multi-agent.model", "gpt-4-turbo"]
        add_config("self_evolving_multi_agent", self_evolving_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.self_evolving_multi_agent.model, "gpt-4-turbo"
        )

    def test_self_evolving_sub_agent_model(self):
        """Test --self-evolving-multi-agent.sub-agent-model CLI option."""
        self_evolving_config = self._get_self_evolving_config()
        sys.argv = ["test", "--self-evolving-multi-agent.sub-agent-model", "claude-sonnet-4-5"]
        add_config("self_evolving_multi_agent", self_evolving_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.self_evolving_multi_agent.sub_agent_model,
            "claude-sonnet-4-5",
        )

    def test_self_evolving_max_steps(self):
        """Test --self-evolving-multi-agent.max-steps CLI option."""
        self_evolving_config = self._get_self_evolving_config()
        sys.argv = ["test", "--self-evolving-multi-agent.max-steps", "150"]
        add_config("self_evolving_multi_agent", self_evolving_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.self_evolving_multi_agent.max_steps, 150)

    def test_self_evolving_max_budget(self):
        """Test --self-evolving-multi-agent.max-budget CLI option."""
        self_evolving_config = self._get_self_evolving_config()
        sys.argv = ["test", "--self-evolving-multi-agent.max-budget", "15.0"]
        add_config("self_evolving_multi_agent", self_evolving_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.self_evolving_multi_agent.max_budget, 15.0)

    def test_self_evolving_docker_image(self):
        """Test --self-evolving-multi-agent.docker-image CLI option."""
        self_evolving_config = self._get_self_evolving_config()
        sys.argv = ["test", "--self-evolving-multi-agent.docker-image", "python:3.11-slim"]
        add_config("self_evolving_multi_agent", self_evolving_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.self_evolving_multi_agent.docker_image, "python:3.11-slim"
        )

    def test_self_evolving_workdir(self):
        """Test --self-evolving-multi-agent.workdir CLI option."""
        self_evolving_config = self._get_self_evolving_config()
        sys.argv = ["test", "--self-evolving-multi-agent.workdir", "/app"]
        add_config("self_evolving_multi_agent", self_evolving_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.self_evolving_multi_agent.workdir, "/app")


class TestAgentCreatorConfigCLI(CLITestBase):
    """Tests for Agent Creator (create_and_optimize_agent) config CLI options.

    Uses local config classes matching the real AgentCreatorConfig to avoid
    import issues with optional dependencies.
    """

    def _get_agent_creator_config(self):
        """Create config classes matching AgentCreatorConfig."""

        class ImproverConfig(BaseModel):
            model_name: str = Field(default="claude-sonnet-4-5")
            max_steps: int = Field(default=100)
            max_budget: float = Field(default=20.0)

        class EvolverConfig(BaseModel):
            model_name: str = Field(default="claude-sonnet-4-5")
            max_generations: int = Field(default=10)
            initial_frontier_size: int = Field(default=4)
            max_frontier_size: int = Field(default=6)
            mutation_probability: float = Field(default=0.8)
            initial_agent_max_steps: int = Field(default=50)
            initial_agent_max_budget: float = Field(default=50.0)
            evolve_to_solve_task: bool = Field(default=False)

        class AgentCreatorConfig(BaseModel):
            improver: ImproverConfig = Field(default_factory=ImproverConfig)
            evolver: EvolverConfig = Field(default_factory=EvolverConfig)

        return AgentCreatorConfig

    def test_improver_model_name(self):
        """Test --create-and-optimize-agent.improver.model-name CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.improver.model-name", "gpt-4"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.improver.model_name, "gpt-4"
        )

    def test_improver_max_steps(self):
        """Test --create-and-optimize-agent.improver.max-steps CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.improver.max-steps", "200"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.improver.max_steps, 200
        )

    def test_improver_max_budget(self):
        """Test --create-and-optimize-agent.improver.max-budget CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.improver.max-budget", "30.0"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.improver.max_budget, 30.0
        )

    def test_evolver_model_name(self):
        """Test --create-and-optimize-agent.evolver.model-name CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.evolver.model-name", "claude-opus-4-6"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.evolver.model_name,
            "claude-opus-4-6",
        )

    def test_evolver_max_generations(self):
        """Test --create-and-optimize-agent.evolver.max-generations CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.evolver.max-generations", "20"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.evolver.max_generations, 20
        )

    def test_evolver_initial_frontier_size(self):
        """Test --create-and-optimize-agent.evolver.initial-frontier-size CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.evolver.initial-frontier-size", "8"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.evolver.initial_frontier_size, 8
        )

    def test_evolver_mutation_probability(self):
        """Test --create-and-optimize-agent.evolver.mutation-probability CLI option."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.evolver.mutation-probability", "0.9"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertEqual(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.evolver.mutation_probability, 0.9
        )

    def test_evolver_evolve_to_solve_task(self):
        """Test --create-and-optimize-agent.evolver.evolve-to-solve-task flag."""
        agent_creator_config = self._get_agent_creator_config()
        sys.argv = ["test", "--create-and-optimize-agent.evolver.evolve-to-solve-task"]
        add_config("create_and_optimize_agent", agent_creator_config)
        self.assertTrue(
            config_module.DEFAULT_CONFIG.create_and_optimize_agent.evolver.evolve_to_solve_task
        )


class TestGEPAConfigCLI(CLITestBase):
    """Tests for GEPA config CLI options.

    Uses a local config class matching the real GEPAConfig to avoid
    import issues with optional dependencies.
    """

    def _get_gepa_config(self):
        """Create a config class matching GEPAConfig."""

        class GEPAConfig(BaseModel):
            reflection_model: str = Field(default="gemini-3-flash-preview")
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)
            pareto_size: int = Field(default=4)
            mutation_rate: float = Field(default=0.5)

        return GEPAConfig

    def test_gepa_reflection_model(self):
        """Test --gepa.reflection-model CLI option."""
        gepa_config = self._get_gepa_config()
        sys.argv = ["test", "--gepa.reflection-model", "gpt-4-turbo"]
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.reflection_model, "gpt-4-turbo")

    def test_gepa_max_generations(self):
        """Test --gepa.max-generations CLI option."""
        gepa_config = self._get_gepa_config()
        sys.argv = ["test", "--gepa.max-generations", "15"]
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 15)

    def test_gepa_population_size(self):
        """Test --gepa.population-size CLI option."""
        gepa_config = self._get_gepa_config()
        sys.argv = ["test", "--gepa.population-size", "12"]
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.population_size, 12)

    def test_gepa_pareto_size(self):
        """Test --gepa.pareto-size CLI option."""
        gepa_config = self._get_gepa_config()
        sys.argv = ["test", "--gepa.pareto-size", "6"]
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.pareto_size, 6)

    def test_gepa_mutation_rate(self):
        """Test --gepa.mutation-rate CLI option."""
        gepa_config = self._get_gepa_config()
        sys.argv = ["test", "--gepa.mutation-rate", "0.6"]
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.mutation_rate, 0.6)


class TestMultipleCLIOptions(CLITestBase):
    """Tests for combining multiple CLI options at once."""

    def _get_kiss_evolve_config(self):
        """Create a config class matching KISSEvolveConfig."""

        class KISSEvolveConfig(BaseModel):
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)
            mutation_rate: float = Field(default=0.7)
            elite_size: int = Field(default=2)
            num_islands: int = Field(default=2)

        return KISSEvolveConfig

    def _get_gepa_config(self):
        """Create a config class matching GEPAConfig."""

        class GEPAConfig(BaseModel):
            reflection_model: str = Field(default="gemini-3-flash-preview")
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)

        return GEPAConfig

    def test_multiple_options_same_config(self):
        """Test passing multiple options from the same config."""
        kiss_evolve_config = self._get_kiss_evolve_config()
        sys.argv = [
            "test",
            "--kiss-evolve.max-generations",
            "25",
            "--kiss-evolve.population-size",
            "32",
            "--kiss-evolve.mutation-rate",
            "0.9",
            "--kiss-evolve.elite-size",
            "5",
            "--kiss-evolve.num-islands",
            "8",
        ]
        add_config("kiss_evolve", kiss_evolve_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.max_generations, 25)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.population_size, 32)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.mutation_rate, 0.9)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.elite_size, 5)
        self.assertEqual(config_module.DEFAULT_CONFIG.kiss_evolve.num_islands, 8)

    def test_multiple_options_different_configs(self):
        """Test passing options from multiple different configs."""
        gepa_config = self._get_gepa_config()
        sys.argv = [
            "test",
            "--agent.max-steps",
            "75",
            "--agent.debug",
            "--gepa.max-generations",
            "12",
            "--gepa.population-size",
            "10",
        ]
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.agent.max_steps, 75)
        self.assertTrue(config_module.DEFAULT_CONFIG.agent.debug)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 12)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.population_size, 10)


class TestCLIEdgeCases(CLITestBase):
    """Tests for edge cases in CLI argument parsing."""

    def _get_gepa_config(self):
        """Create a config class matching GEPAConfig."""

        class GEPAConfig(BaseModel):
            reflection_model: str = Field(default="gemini-3-flash-preview")
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)
            mutation_rate: float = Field(default=0.5)

        return GEPAConfig

    def _get_swebench_config(self):
        """Create a config class matching SWEBenchVerifiedConfig."""

        class SWEBenchVerifiedConfig(BaseModel):
            max_steps: int = Field(default=100)
            max_budget: float = Field(default=5.0)

        return SWEBenchVerifiedConfig

    def test_empty_argv(self):
        """Test that defaults are used when no CLI args are provided."""
        gepa_config = self._get_gepa_config()
        sys.argv = ["test"]  # No additional args
        add_config("gepa", gepa_config)
        # Should use defaults
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 10)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.population_size, 8)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.mutation_rate, 0.5)

    def test_unknown_args_ignored(self):
        """Test that unknown CLI args are ignored (for pytest compatibility)."""
        gepa_config = self._get_gepa_config()
        sys.argv = [
            "test",
            "--gepa.max-generations",
            "15",
            "--unknown-arg",
            "value",
            "-v",
            "--pytest-arg",
        ]
        # Should not raise an error
        add_config("gepa", gepa_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 15)

    def test_mixed_dash_underscore_styles(self):
        """Test mixing dash and underscore styles in same command."""
        swebench_config = self._get_swebench_config()
        sys.argv = [
            "test",
            "--swebench-verified.max-steps",
            "100",  # dash style
            "--swebench_verified.max_budget",
            "7.5",  # underscore style
        ]
        add_config("swebench_verified", swebench_config)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.max_steps, 100)
        self.assertEqual(config_module.DEFAULT_CONFIG.swebench_verified.max_budget, 7.5)


if __name__ == "__main__":
    unittest.main()
