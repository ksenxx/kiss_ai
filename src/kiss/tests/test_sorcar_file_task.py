"""Integration tests for the -f file task option in sorcar_agent main()."""

from __future__ import annotations

import os
import tempfile

import pytest

from kiss.agents.sorcar.sorcar_agent import (
    _DEFAULT_TASK,
    _build_arg_parser,
    _resolve_task,
)


class TestBuildArgParser:
    """Tests for _build_arg_parser()."""

    def test_parser_has_f_argument(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", "/tmp/task.txt"])
        assert args.f == "/tmp/task.txt"

    def test_f_default_is_none(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.f is None

    def test_task_argument_still_works(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--task", "do something"])
        assert args.task == "do something"

    def test_both_f_and_task(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", "/tmp/task.txt", "--task", "ignored"])
        assert args.f == "/tmp/task.txt"
        assert args.task == "ignored"

    def test_all_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.model_name == "claude-opus-4-6"
        assert args.max_steps == 30
        assert args.max_budget == 5.0
        assert args.work_dir is None
        assert args.headless is False
        assert args.verbose is True
        assert args.task is None
        assert args.f is None


class TestResolveTask:
    """Tests for _resolve_task() with all three branches."""

    def test_file_option_reads_file_contents(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("task from file")
            f.flush()
            path = f.name
        try:
            parser = _build_arg_parser()
            args = parser.parse_args(["-f", path])
            result = _resolve_task(args)
            assert result == "task from file"
        finally:
            os.unlink(path)

    def test_file_option_reads_multiline_content(self) -> None:
        content = "line 1\nline 2\nline 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            parser = _build_arg_parser()
            args = parser.parse_args(["-f", path])
            result = _resolve_task(args)
            assert result == content
        finally:
            os.unlink(path)

    def test_file_option_reads_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            parser = _build_arg_parser()
            args = parser.parse_args(["-f", path])
            result = _resolve_task(args)
            assert result == ""
        finally:
            os.unlink(path)

    def test_file_option_reads_unicode_content(self) -> None:
        content = "タスク: テストを書く 🎉\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            parser = _build_arg_parser()
            args = parser.parse_args(["-f", path])
            result = _resolve_task(args)
            assert result == content
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", "/nonexistent/path/task.txt"])
        with pytest.raises(FileNotFoundError):
            _resolve_task(args)

    def test_task_option_returns_task_string(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--task", "my custom task"])
        result = _resolve_task(args)
        assert result == "my custom task"

    def test_neither_returns_default(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        result = _resolve_task(args)
        assert result == _DEFAULT_TASK

    def test_file_takes_priority_over_task(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("from file")
            f.flush()
            path = f.name
        try:
            parser = _build_arg_parser()
            args = parser.parse_args(["-f", path, "--task", "from flag"])
            result = _resolve_task(args)
            assert result == "from file"
        finally:
            os.unlink(path)
