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


if __name__ == "__main__":
    unittest.main()
