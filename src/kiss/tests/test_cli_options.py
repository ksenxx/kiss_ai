"""Test suite for verifying command line options work correctly.

These tests verify that CLI arguments defined in various config.py files
are properly parsed and applied to the configuration system.
"""

import sys
import unittest

from pydantic import BaseModel, Field

from kiss.core import config as config_module
from kiss.core.config_builder import add_config


class CLITestBase(unittest.TestCase):
    def setUp(self):
        self.original_config = config_module.DEFAULT_CONFIG
        self.original_argv = sys.argv

    def tearDown(self):
        sys.argv = self.original_argv
        config_module.DEFAULT_CONFIG = self.original_config

    def _get_attr(self, root, path: str):
        value = root
        for part in path.split("."):
            value = getattr(value, part)
        return value

    def _assert_cli_value(self, args, config_name, config_class, attr_path, expected):
        sys.argv = ["test"] + args
        add_config(config_name, config_class)
        actual = self._get_attr(config_module.DEFAULT_CONFIG, attr_path)
        self.assertEqual(actual, expected)
        config_module.DEFAULT_CONFIG = self.original_config
        sys.argv = self.original_argv

    def _assert_cli_values(self, args, config_name, config_class, expected_map):
        sys.argv = ["test"] + args
        add_config(config_name, config_class)
        for attr_path, expected in expected_map.items():
            actual = self._get_attr(config_module.DEFAULT_CONFIG, attr_path)
            self.assertEqual(actual, expected)
        config_module.DEFAULT_CONFIG = self.original_config
        sys.argv = self.original_argv


class TestSWEBenchConfigCLI(CLITestBase):
    def _get_swebench_config(self):
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

    def test_swebench_options(self):
        cases = [
            (
                ["--swebench-verified.dataset-name", "custom/dataset"],
                "swebench_verified.dataset_name",
                "custom/dataset",
            ),
            (
                ["--swebench-verified.instance-id", "django__django-12345"],
                "swebench_verified.instance_id",
                "django__django-12345",
            ),
            (
                ["--swebench-verified.max-instances", "50"],
                "swebench_verified.max_instances",
                50,
            ),
            (
                ["--swebench-verified.model", "claude-opus-4-6"],
                "swebench_verified.model",
                "claude-opus-4-6",
            ),
            (
                ["--swebench-verified.max-steps", "200"],
                "swebench_verified.max_steps",
                200,
            ),
            (
                ["--swebench-verified.max-budget", "10.0"],
                "swebench_verified.max_budget",
                10.0,
            ),
            (
                ["--swebench-verified.num-samples", "5"],
                "swebench_verified.num_samples",
                5,
            ),
            (
                ["--no-swebench-verified.run-evaluation"],
                "swebench_verified.run_evaluation",
                False,
            ),
            (
                ["--no-swebench-verified.save-patches"],
                "swebench_verified.save_patches",
                False,
            ),
        ]
        for args, attr_path, expected in cases:
            with self.subTest(args=args):
                self._assert_cli_value(
                    args,
                    "swebench_verified",
                    self._get_swebench_config(),
                    attr_path,
                    expected,
                )


class TestSelfEvolvingMultiAgentConfigCLI(CLITestBase):
    def _get_self_evolving_config(self):
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

    def test_self_evolving_options(self):
        cases = [
            (
                ["--self-evolving-multi-agent.model", "gpt-4-turbo"],
                "self_evolving_multi_agent.model",
                "gpt-4-turbo",
            ),
            (
                ["--self-evolving-multi-agent.sub-agent-model", "claude-sonnet-4-5"],
                "self_evolving_multi_agent.sub_agent_model",
                "claude-sonnet-4-5",
            ),
            (
                ["--self-evolving-multi-agent.max-steps", "150"],
                "self_evolving_multi_agent.max_steps",
                150,
            ),
            (
                ["--self-evolving-multi-agent.max-budget", "15.0"],
                "self_evolving_multi_agent.max_budget",
                15.0,
            ),
            (
                ["--self-evolving-multi-agent.docker-image", "python:3.11-slim"],
                "self_evolving_multi_agent.docker_image",
                "python:3.11-slim",
            ),
            (
                ["--self-evolving-multi-agent.workdir", "/app"],
                "self_evolving_multi_agent.workdir",
                "/app",
            ),
        ]
        for args, attr_path, expected in cases:
            with self.subTest(args=args):
                self._assert_cli_value(
                    args,
                    "self_evolving_multi_agent",
                    self._get_self_evolving_config(),
                    attr_path,
                    expected,
                )


class TestAgentCreatorConfigCLI(CLITestBase):
    def _get_agent_creator_config(self):
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

    def test_evolver_options(self):
        cases = [
            (
                ["--create-and-optimize-agent.evolver.model-name", "claude-opus-4-6"],
                "create_and_optimize_agent.evolver.model_name",
                "claude-opus-4-6",
            ),
            (
                ["--create-and-optimize-agent.evolver.max-generations", "20"],
                "create_and_optimize_agent.evolver.max_generations",
                20,
            ),
            (
                ["--create-and-optimize-agent.evolver.initial-frontier-size", "8"],
                "create_and_optimize_agent.evolver.initial_frontier_size",
                8,
            ),
            (
                ["--create-and-optimize-agent.evolver.mutation-probability", "0.9"],
                "create_and_optimize_agent.evolver.mutation_probability",
                0.9,
            ),
            (
                ["--create-and-optimize-agent.evolver.evolve-to-solve-task"],
                "create_and_optimize_agent.evolver.evolve_to_solve_task",
                True,
            ),
        ]
        for args, attr_path, expected in cases:
            with self.subTest(args=args):
                self._assert_cli_value(
                    args,
                    "create_and_optimize_agent",
                    self._get_agent_creator_config(),
                    attr_path,
                    expected,
                )


class TestMultipleCLIOptions(CLITestBase):
    def test_multiple_options_different_configs(self):
        class GEPAConfig(BaseModel):
            reflection_model: str = Field(default="gemini-3-flash-preview")
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)

        sys.argv = [
            "test",
            "--agent.max-steps", "75",
            "--agent.debug",
            "--gepa.max-generations", "12",
            "--gepa.population-size", "10",
        ]
        add_config("gepa", GEPAConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.agent.max_steps, 75)
        self.assertTrue(config_module.DEFAULT_CONFIG.agent.debug)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 12)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.population_size, 10)


class TestCLIEdgeCases(CLITestBase):
    def test_empty_argv(self):
        class GEPAConfig(BaseModel):
            max_generations: int = Field(default=10)
            population_size: int = Field(default=8)
            mutation_rate: float = Field(default=0.5)

        sys.argv = ["test"]
        add_config("gepa", GEPAConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 10)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.population_size, 8)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.mutation_rate, 0.5)

    def test_unknown_args_ignored(self):
        class GEPAConfig(BaseModel):
            max_generations: int = Field(default=10)

        sys.argv = ["test", "--gepa.max-generations", "15", "--unknown-arg", "value", "-v"]
        add_config("gepa", GEPAConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.gepa.max_generations, 15)


if __name__ == "__main__":
    unittest.main()
