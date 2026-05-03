# Author: Koushik Sen (ksen@berkeley.edu)

"""Tests for the terminal bench harbor agent."""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from kiss._version import __version__
from kiss.benchmarks.terminal_bench.agent import (
    _SKIP_PHRASES,
    SorcarHarborAgent,
    _get_wheel,
)
from kiss.benchmarks.terminal_bench.run import build_package


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
        """A normal task runs which-check then sorcar exactly once."""
        agent = _make_agent()
        env = FakeEnvironment()
        ctx = FakeContext()
        asyncio.run(
            agent.run("Fix the bug in /app/main.py", env, ctx),  # type: ignore[arg-type]
        )
        assert len(env.exec_calls) == 2
        assert "which sorcar" in env.exec_calls[0]
        assert "sorcar -t" in env.exec_calls[1]
        for call in env.exec_calls:
            assert "test.sh" not in call
            assert "/tests" not in call
        assert ctx.metadata is not None
        assert "skipped" not in ctx.metadata
        assert ctx.metadata["return_code"] == 0
        assert "tests_passed" not in ctx.metadata
        assert "tests_total" not in ctx.metadata
        assert "partial_score" not in ctx.metadata


class TestSetup:
    """Verify setup runs the expected installation steps."""

    def test_setup_two_steps(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment()
        asyncio.run(agent.setup(env))  # type: ignore[arg-type]
        assert len(env.exec_calls) == 2
        assert "curl" in env.exec_calls[0]
        assert "-o /tmp/install-uv.sh" in env.exec_calls[0]
        assert "sh /tmp/install-uv.sh" in env.exec_calls[0]
        assert "test -x /root/.local/bin/uv" in env.exec_calls[0]
        assert "| sh" not in env.exec_calls[0]
        assert "uv tool install --python 3.13" in env.exec_calls[1]
        assert "/tmp/kiss_agent_framework-" in env.exec_calls[1]
        assert len(env.uploaded_files) == 1
        src, dst = env.uploaded_files[0]
        assert src.endswith(".whl")
        assert dst.startswith("/tmp/kiss_agent_framework-")
        for call in env.exec_calls:
            assert "SYSTEM.md" not in call

    def test_setup_aborts_on_uv_failure(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment(fail_commands={"curl"})
        asyncio.run(agent.setup(env))  # type: ignore[arg-type]
        assert len(env.exec_calls) == 1
        assert len(env.uploaded_files) == 0

    def test_setup_aborts_on_pip_failure(self) -> None:
        agent = _make_agent()
        env = FakeEnvironment(fail_commands={"uv tool install"})
        asyncio.run(agent.setup(env))  # type: ignore[arg-type]
        assert len(env.exec_calls) == 2
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
        assert len(env.exec_calls) == 1
        assert ctx.metadata is not None
        assert ctx.metadata["error"] == "sorcar not installed"


class TestBuildPackage:
    """Verify build_package() produces a fresh wheel."""

    def test_build_package_returns_wheel(self) -> None:
        wheel = build_package()
        assert wheel.exists()
        assert wheel.suffix == ".whl"
        assert "kiss_agent_framework" in wheel.name

    def test_build_package_cleans_old_wheels(self) -> None:
        """build_package removes stale wheels before building."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        dist = project_root / "dist"
        dist.mkdir(exist_ok=True)
        stale = dist / "kiss_agent_framework-0.0.0-py3-none-any.whl"
        stale.write_bytes(b"stale")
        wheel = build_package()
        assert not stale.exists(), "stale wheel should have been removed"
        assert wheel.exists()


class TestGetWheelRespectsEnvVar:
    """Verify _get_wheel() uses KISS_WHEEL_PATH when set."""

    def test_get_wheel_uses_env_var(self) -> None:
        import kiss.benchmarks.terminal_bench.agent as agent_mod

        with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as f:
            f.write(b"dummy wheel")
            tmp_wheel = Path(f.name)
        try:
            old = agent_mod._wheel_path
            agent_mod._wheel_path = None
            old_env = os.environ.get("KISS_WHEEL_PATH")
            os.environ["KISS_WHEEL_PATH"] = str(tmp_wheel)
            try:
                result = _get_wheel()
                assert result == tmp_wheel
            finally:
                agent_mod._wheel_path = old
                if old_env is None:
                    os.environ.pop("KISS_WHEEL_PATH", None)
                else:
                    os.environ["KISS_WHEEL_PATH"] = old_env
        finally:
            tmp_wheel.unlink(missing_ok=True)

    def test_get_wheel_falls_back_to_build(self) -> None:
        """Without KISS_WHEEL_PATH, _get_wheel builds from source."""
        import kiss.benchmarks.terminal_bench.agent as agent_mod

        old = agent_mod._wheel_path
        agent_mod._wheel_path = None
        old_env = os.environ.pop("KISS_WHEEL_PATH", None)
        try:
            result = _get_wheel()
            assert result.exists()
            assert "kiss_agent_framework" in result.name
        finally:
            agent_mod._wheel_path = old
            if old_env is not None:
                os.environ["KISS_WHEEL_PATH"] = old_env


class TestRunTerminalBenchBuildsFirst:
    """Verify run_terminal_bench() builds the package before harbor."""

    def test_run_terminal_bench_calls_build_package(self) -> None:
        """Structural: run_terminal_bench source contains build_package call."""
        from kiss.benchmarks.terminal_bench.run import run_terminal_bench

        src = inspect.getsource(run_terminal_bench)
        assert "build_package()" in src
        assert 'KISS_WHEEL_PATH' in src

    def test_build_package_source_structure(self) -> None:
        """Structural: build_package cleans old wheels and builds fresh."""
        src = inspect.getsource(build_package)
        assert "uv" in src
        assert "build" in src
        assert "unlink" in src
        assert "kiss_agent_framework" in src
