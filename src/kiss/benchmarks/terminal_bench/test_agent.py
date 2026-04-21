# Author: Koushik Sen (ksen@berkeley.edu)

"""Tests for the terminal bench harbor agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from kiss._version import __version__
from kiss.benchmarks.terminal_bench.agent import (
    _SKIP_PHRASES,
    SorcarHarborAgent,
)


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass
class FakeEnvironment:
    """Minimal stand-in for BaseEnvironment.

    By default every exec call succeeds.  Override ``fail_commands``
    with substrings to make matching exec calls return non-zero.
    """

    exec_calls: list[str] = field(default_factory=list)
    uploaded_files: list[tuple[str, str]] = field(default_factory=list)
    fail_commands: set[str] = field(default_factory=set)

    async def exec(
        self,
        command: str,
        **kwargs: object,
    ) -> FakeExecResult:
        self.exec_calls.append(command)
        for pat in self.fail_commands:
            if pat in command:
                return FakeExecResult(
                    stderr=f"simulated failure for {pat}",
                    return_code=1,
                )
        return FakeExecResult()

    async def upload_file(
        self,
        source_path: object,
        target_path: str,
    ) -> None:
        self.uploaded_files.append((str(source_path), target_path))


@dataclass
class FakeContext:
    """Minimal stand-in for AgentContext."""

    metadata: dict[str, object] | None = None

    def is_empty(self) -> bool:
        return self.metadata is None


def _make_agent() -> SorcarHarborAgent:
    import tempfile
    from pathlib import Path

    return SorcarHarborAgent(
        logs_dir=Path(tempfile.mkdtemp()),
        model_name="claude-opus-4-6",
    )


class TestSkipPhrases:
    """Verify _SKIP_PHRASES is a non-empty tuple of strings."""

    def test_skip_phrases_non_empty(self) -> None:
        assert len(_SKIP_PHRASES) > 0

    def test_skip_phrases_are_strings(self) -> None:
        for phrase in _SKIP_PHRASES:
            assert isinstance(phrase, str)
            assert len(phrase) > 0


class TestAgentIdentity:
    """Agent name and version."""

    def test_name(self) -> None:
        assert SorcarHarborAgent.name() == "sorcar"

    def test_version_matches_package(self) -> None:
        agent = _make_agent()
        assert agent.version() == __version__


class TestRunSkipsImpossibleTasks:
    """Verify that run() returns immediately for impossible tasks."""

    def test_skip_compcert(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment()
        ctx = FakeContext()
        instruction = (
            "Under /tmp/CompCert/, build the CompCert C verified compiler "
            "(version 3.13.1) from source."
        )
        asyncio.run(agent.run(instruction, env, ctx))  # type: ignore[arg-type]
        assert ctx.metadata is not None
        assert ctx.metadata["skipped"] is True
        assert ctx.metadata["reason"] == "CompCert C verified compiler"
        assert len(env.exec_calls) == 0

    def test_skip_windows_311(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment()
        ctx = FakeContext()
        asyncio.run(
            agent.run("Run Windows 3.11 for Workgroups", env, ctx),  # type: ignore[arg-type]
        )
        assert ctx.metadata is not None
        assert ctx.metadata["skipped"] is True

    def test_skip_ocaml_gc(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment()
        ctx = FakeContext()
        asyncio.run(
            agent.run("Fix the OCaml garbage collector issue", env, ctx),  # type: ignore[arg-type]
        )
        assert ctx.metadata is not None
        assert ctx.metadata["skipped"] is True
        assert ctx.metadata["reason"] == "OCaml garbage collector"

    def test_non_skip_task_runs_normally(self) -> None:
        """A normal task runs which-check, sorcar, then verifies."""
        agent = _make_agent()
        env = FakeEnvironment()
        ctx = FakeContext()
        asyncio.run(
            agent.run("Fix the bug in /app/main.py", env, ctx),  # type: ignore[arg-type]
        )
        # 1: which sorcar, 2: sorcar -t ..., 3: bash test.sh
        assert len(env.exec_calls) == 3
        assert "which sorcar" in env.exec_calls[0]
        assert "sorcar" in env.exec_calls[1]
        assert "test.sh" in env.exec_calls[2]
        assert ctx.metadata is not None
        assert "skipped" not in ctx.metadata
        assert ctx.metadata["return_code"] == 0


class TestSetup:
    """Verify setup runs the expected installation steps."""

    def test_setup_two_steps(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment()
        asyncio.run(agent.setup(env))  # type: ignore[arg-type]
        assert len(env.exec_calls) == 2
        assert "curl" in env.exec_calls[0]
        # Step 2: installs from uploaded local wheel, not PyPI
        assert "uv tool install --python 3.13" in env.exec_calls[1]
        assert "/tmp/kiss_agent_framework-" in env.exec_calls[1]
        assert len(env.uploaded_files) == 1
        src, dst = env.uploaded_files[0]
        assert src.endswith(".whl")
        assert dst.startswith("/tmp/kiss_agent_framework-")
        # No SYSTEM.md or extra prompt is written into the container.
        for call in env.exec_calls:
            assert "SYSTEM.md" not in call

    def test_setup_aborts_on_uv_failure(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment(fail_commands={"curl"})
        asyncio.run(agent.setup(env))  # type: ignore[arg-type]
        # Only the first step should have been attempted
        assert len(env.exec_calls) == 1
        assert len(env.uploaded_files) == 0

    def test_setup_aborts_on_pip_failure(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment(fail_commands={"uv tool install"})
        asyncio.run(agent.setup(env))  # type: ignore[arg-type]
        # First step succeeds, second fails; no further steps.
        assert len(env.exec_calls) == 2
        # Wheel was uploaded before the install command ran
        assert len(env.uploaded_files) == 1


class TestRunSorcarNotFound:
    """When sorcar is not installed, run returns early with an error."""

    def test_sorcar_missing(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment(fail_commands={"which sorcar"})
        ctx = FakeContext()
        asyncio.run(
            agent.run("Fix the bug in /app/main.py", env, ctx),  # type: ignore[arg-type]
        )
        assert len(env.exec_calls) == 1  # only the which-check
        assert ctx.metadata is not None
        assert ctx.metadata["error"] == "sorcar not installed"
