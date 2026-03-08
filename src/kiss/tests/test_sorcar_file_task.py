"""Integration tests for the -f file task option in sorcar_agent main()."""

from __future__ import annotations

import os
import tempfile

from kiss.agents.sorcar.sorcar_agent import (
    _DEFAULT_TASK,
    _build_arg_parser,
    _resolve_task,
)


class TestResolveTask:
    """Tests for _resolve_task() with all three branches."""

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
