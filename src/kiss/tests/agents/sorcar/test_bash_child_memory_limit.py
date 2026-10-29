# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for the per-child ``RLIMIT_DATA`` cap of the Bash tool.

In a container with a memory limit (an s10s pod) the kernel's
``memory.oom.group`` kills every process of the container when one of
them pushes it over the limit, so a script loading a dataset whole used
to take the agent down with it.  ``_popen_kwargs`` now gives every Bash
child (sync, ``run_commands_parallel`` and background jobs) a
``preexec_fn`` that caps ``RLIMIT_DATA`` at ``child_memory_limit()``:
``KISS_BASH_MEMORY_LIMIT`` when set, else 3/4 of the cgroup limit, else
nothing.  The child fails with ``MemoryError`` and the tool output says
why.
"""

from __future__ import annotations

import logging
import re
import resource
import shlex
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    child_memory_limit,
    format_memory_size,
    parse_memory_size,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX")
# macOS accepts RLIMIT_DATA but does not enforce it on malloc/mmap, so only
# Linux can show an allocation actually failing under the cap.
_enforced_on_linux = pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_DATA is enforced on Linux only"
)

_JOB_ID_RE = re.compile(r"Started background job (j[0-9a-f]{8})")
_PRINT_RLIMIT = "python3 -c 'import resource; print(resource.getrlimit(resource.RLIMIT_DATA))'"
_ALLOC_400M = "python3 -c 'b = bytearray(400 * 1024 * 1024); print(len(b))'"
_ALLOC_50M = "python3 -c 'b = bytearray(50 * 1024 * 1024); print(len(b))'"
_NOTE = "Note: each process started by this tool is capped at"
_200M = 200 * 1024 * 1024


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8G", 8 << 30),
        ("512m", 512 << 20),
        ("1024", 1024),
        (" 2KB ", 2048),
        ("1T", 1 << 40),
        ("1tb", 1 << 40),
        ("abc", None),
        ("1.5G", None),
        ("", None),
        ("-1", None),
    ],
)
def test_parse_memory_size(text: str, expected: int | None) -> None:
    assert parse_memory_size(text) == expected


def test_format_memory_size() -> None:
    assert format_memory_size(9663676416) == "9.0 GiB"
    assert format_memory_size(1 << 30) == "1.0 GiB"
    assert format_memory_size(_200M) == "200 MiB"


class TestChildMemoryLimit:
    def test_env_override_wins_over_cgroup(self, tmp_path: Path, monkeypatch) -> None:
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("1000\n")
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "8G")
        assert child_memory_limit((str(cgroup),)) == 8 << 30

    @pytest.mark.parametrize("value", ["0", "off", "none", "OFF"])
    def test_env_disables_the_cap(self, tmp_path: Path, monkeypatch, value: str) -> None:
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("1000\n")
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", value)
        assert child_memory_limit((str(cgroup),)) is None

    @pytest.mark.parametrize("value", ["lots", "99999999T"])
    def test_unusable_env_is_ignored_with_a_warning(
        self, tmp_path: Path, monkeypatch, caplog, value: str
    ) -> None:
        # ``99999999T`` parses but exceeds what setrlimit accepts; applying
        # it would make every Bash call fail in its preexec_fn.
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("1000\n")
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", value)
        with caplog.at_level(logging.WARNING, logger="kiss.agents.sorcar.useful_tools"):
            assert child_memory_limit((str(cgroup),)) == 750
        assert f"KISS_BASH_MEMORY_LIMIT={value!r}" in caplog.text

    def test_oversized_env_does_not_break_bash(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "99999999T")
        tools = UsefulTools(work_dir=str(tmp_path))
        assert tools.Bash("echo hi", "still runs") == "hi\n"

    def test_cgroup_v2_limit_gives_three_quarters(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("KISS_BASH_MEMORY_LIMIT", raising=False)
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("12884901888\n")
        assert child_memory_limit((str(cgroup),)) == 9663676416

    def test_cgroup_v2_max_means_no_cap(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("KISS_BASH_MEMORY_LIMIT", raising=False)
        cgroup = tmp_path / "memory.max"
        cgroup.write_text("max\n")
        v1 = tmp_path / "memory.limit_in_bytes"
        v1.write_text("1000\n")
        # The first existing file decides; a later one is not consulted.
        assert child_memory_limit((str(cgroup), str(v1))) is None

    def test_cgroup_v1_unlimited_means_no_cap(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("KISS_BASH_MEMORY_LIMIT", raising=False)
        v1 = tmp_path / "memory.limit_in_bytes"
        v1.write_text("9223372036854771712\n")
        assert child_memory_limit((str(tmp_path / "missing"), str(v1))) is None

    def test_missing_v2_falls_back_to_v1(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("KISS_BASH_MEMORY_LIMIT", raising=False)
        v1 = tmp_path / "memory.limit_in_bytes"
        v1.write_text("4294967296\n")
        assert child_memory_limit((str(tmp_path / "missing"), str(v1))) == 3221225472

    def test_no_cgroup_file_means_no_cap(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("KISS_BASH_MEMORY_LIMIT", raising=False)
        assert child_memory_limit((str(tmp_path / "missing"),)) is None


class TestBashChildren:
    def test_child_and_grandchild_get_the_rlimit(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        assert tools.Bash(_PRINT_RLIMIT, "child").strip() == f"({_200M}, {_200M})"
        # The shell's children inherit the cap: ``xargs -P 4 ./run_export.sh``
        # style fan-outs are capped per process too.
        nested = tools.Bash(f"sh -c {shlex.quote(_PRINT_RLIMIT)}", "grandchild")
        assert nested.strip() == f"({_200M}, {_200M})"

    @_enforced_on_linux
    def test_oversized_allocation_fails_alone_with_a_hint(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        output = tools.Bash(_ALLOC_400M, "too big")
        assert output.startswith("Error (exit code 1):")
        assert "MemoryError" in output
        assert f"{_NOTE} 200 MiB of memory (RLIMIT_DATA, KISS_BASH_MEMORY_LIMIT)" in output
        # A reasonable allocation under the cap still works.
        assert tools.Bash(_ALLOC_50M, "fits").strip() == str(50 * 1024 * 1024)

    def test_hint_only_for_memory_failures(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        output = tools.Bash("echo plain failure; exit 2", "other failure")
        assert output == "Error (exit code 2):\nplain failure\n"
        # A memory failure that is not ours (cap disabled) gets no hint.
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "off")
        output = tools.Bash("echo MemoryError; exit 1", "uncapped")
        assert output == "Error (exit code 1):\nMemoryError\n"

    def test_cap_disabled_leaves_the_inherited_rlimit(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "off")
        tools = UsefulTools(work_dir=str(tmp_path))
        inherited = resource.getrlimit(resource.RLIMIT_DATA)
        assert tools.Bash(_PRINT_RLIMIT, "uncapped").strip() == str(inherited)

    def test_cap_never_exceeds_the_inherited_hard_limit(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The outer Bash child gets a 200M hard limit; inside it a second
        # UsefulTools asks for 500M, which setrlimit would refuse, so the
        # cap is clamped to the hard limit it inherited.
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        inner = (
            "from kiss.agents.sorcar.useful_tools import UsefulTools; "
            f"print(UsefulTools(work_dir={str(tmp_path)!r}).Bash({_PRINT_RLIMIT!r}, 'x'), end='')"
        )
        command = (
            f"KISS_BASH_MEMORY_LIMIT=500M {shlex.quote(sys.executable)} -c {shlex.quote(inner)}"
        )
        output = tools.Bash(command, "nested tools", timeout_seconds=120)
        assert output.strip() == f"({_200M}, {_200M})"

    @_enforced_on_linux
    def test_run_commands_parallel_children_are_capped(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        report = tools.run_commands_parallel(
            f'["{_ALLOC_400M}", "{_PRINT_RLIMIT}"]', description="both"
        )
        assert "MemoryError" in report
        assert f"({_200M}, {_200M})" in report


class TestBackgroundJobs:
    @_enforced_on_linux
    def test_background_job_is_capped_and_report_explains(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        started = tools.Bash(_ALLOC_400M, "bg too big", background=True)
        match = _JOB_ID_RE.search(started)
        assert match, started
        report = tools.bash_job(match.group(1), action="wait", timeout_seconds=30)
        assert "exited with code 1" in report
        assert "MemoryError" in report
        assert f"{_NOTE} 200 MiB" in report

    def test_successful_job_gets_no_hint(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        started = tools.Bash("echo MemoryError mentioned in a log", "bg ok", background=True)
        match = _JOB_ID_RE.search(started)
        assert match, started
        report = tools.bash_job(match.group(1), action="wait", timeout_seconds=30)
        assert "exited with code 0" in report
        assert _NOTE not in report

    def test_running_job_gets_no_hint(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("KISS_BASH_MEMORY_LIMIT", "200M")
        tools = UsefulTools(work_dir=str(tmp_path))
        started = tools.Bash("echo MemoryError; exec sleep 3", "bg running", background=True)
        match = _JOB_ID_RE.search(started)
        assert match, started
        job_id = match.group(1)
        try:
            report = tools.bash_job(job_id, action="tail")
            assert "running (pid" in report
            assert _NOTE not in report
        finally:
            tools.bash_job(job_id, action="kill")
            tools.bash_job(job_id, action="wait", timeout_seconds=10)
