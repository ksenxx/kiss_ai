# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the channel-agent CLI helpers in ``_channel_cli``: argument
parsing, ``run`` kwargs and task resolution.

``KISS_WORKDIR`` defaulting lives in ``test_cli_launch_work_dir``; the
``-m``/``-b`` omission sentinels in ``test_hermes_runner``.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.third_party_agents._channel_cli import (
    _build_arg_parser,
    _build_run_kwargs,
    _resolve_task,
)


def test_build_run_kwargs(tmp_path: Path) -> None:
    """``_build_run_kwargs`` forwards task, work dir, endpoint and the ``--no-web`` flag."""
    parser = _build_arg_parser()
    args = parser.parse_args(
        ["-t", "do something", "-w", str(tmp_path), "-e", "http://localhost:1234"]
    )
    kwargs = _build_run_kwargs(args)
    assert kwargs["prompt_template"] == "do something"
    assert kwargs["work_dir"] == str(tmp_path)
    assert kwargs["model_config"]["base_url"] == "http://localhost:1234"
    assert kwargs["web_tools"] is None  # not given; ``--no-web`` makes it False
    args = parser.parse_args(["-t", "x", "-w", str(tmp_path), "--no-web"])
    assert _build_run_kwargs(args)["web_tools"] is False


def test_long_options(tmp_path: Path) -> None:
    """The long spellings of ``-m``/``-b``/``-w``/``-t`` parse to the same attributes."""
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--model_name", "gpt-4",
            "--max_budget", "1.5",
            "--work_dir", str(tmp_path),
            "--task", "hello world",
        ]
    )
    assert args.model_name == "gpt-4"
    assert args.max_budget == 1.5
    assert args.work_dir == str(tmp_path)
    assert args.task == "hello world"
    kwargs = _build_run_kwargs(args)
    assert kwargs["model_name"] == "gpt-4" and kwargs["max_budget"] == 1.5
    assert kwargs["work_dir"] == str(tmp_path)
    assert kwargs["prompt_template"] == "hello world"


def test_build_run_kwargs_headers() -> None:
    """Repeated ``--header Key:Value`` become ``model_config["extra_headers"]``."""
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "-t",
            "x",
            "-e",
            "http://localhost:8080/v1",
            "--header",
            "X-Custom:value1",
            "--header",
            "Authorization:Bearer tok",
        ]
    )
    kwargs = _build_run_kwargs(args)
    assert kwargs["model_config"]["base_url"] == "http://localhost:8080/v1"
    assert kwargs["model_config"]["extra_headers"] == {
        "X-Custom": "value1",
        "Authorization": "Bearer tok",
    }
    # Nothing given: no override at all, so the daemon keeps the
    # endpoint registered for the model.
    assert _build_run_kwargs(parser.parse_args(["-t", "x"]))["model_config"] is None


def test_resolve_task_default() -> None:
    """Without ``-t``/``-f`` the built-in demo task is used."""
    args = _build_arg_parser().parse_args([])
    assert "weather" in _resolve_task(args).lower()


def test_resolve_task_from_file(tmp_path: Path) -> None:
    """``-f`` reads the task text from the named file."""
    f = tmp_path / "task.txt"
    f.write_text("File task content")
    args = _build_arg_parser().parse_args(["-f", str(f)])
    assert _resolve_task(args) == "File task content"
