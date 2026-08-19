# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the channel-agent CLI helpers: arg parsing, task
resolution, and the default-task credential guard.

The bash-streaming test (pure kiss.agents.sorcar + kiss.server
closure) moved to ``kiss.tests.server.test_sorcar_agent``."""

from __future__ import annotations

from pathlib import Path

from kiss.agents.third_party_agents._channel_cli import (
    _build_arg_parser,
    _resolve_task,
)


class TestResolveTask:
    def test_resolve_task_default(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        result = _resolve_task(args)
        assert "weather" in result.lower()

    def test_resolve_task_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "task.txt"
        f.write_text("File task content")
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", str(f)])
        result = _resolve_task(args)
        assert result == "File task content"


class TestDefaultTaskNoCredentials:
    def test_no_password_in_default_task(self) -> None:
        from kiss.agents.third_party_agents._channel_cli import _DEFAULT_TASK

        assert "password" not in _DEFAULT_TASK.lower()
        assert "kissagent" not in _DEFAULT_TASK.lower()
        assert "@gmail" not in _DEFAULT_TASK.lower()


class TestBuildArgParser:
    def test_custom_args(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--model_name", "gpt-4",
            "--max_budget", "1.5",
            "--work_dir", "/tmp/test",
            "--task", "hello world",
        ])
        assert args.model_name == "gpt-4"
        assert args.max_budget == 1.5
        assert args.work_dir == "/tmp/test"
        assert args.task == "hello world"
