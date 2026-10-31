# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Running the tests through a package or build tool is a check, not a file change.

:data:`kiss.agents.sorcar.evidence_gate.WRITING_COMMAND` used to match the
bare tool names ``uv``, ``npm``, ``cargo``, ``make``..., so the verification
run itself (``uv run pytest -q``) marked the work unchecked, and the gate's
one refusal was spent on a false "files were changed after the last shell
command" while a later finish that really hedged passed.
"""

from __future__ import annotations

from kiss.agents.sorcar import evidence_gate as eg

CHECKS = (
    "uv run pytest -q", "uv run check --full", "npm test", "npm run test", "yarn test",
    "pnpm test", "cargo test", "cargo clippy", "make test", "make check", "ninja test",
    "python -m pytest -q",
)
WRITES = (
    "pip install x", "pip3 uninstall -y x", "uv pip install -e .", "uv add httpx", "uv sync",
    "uv lock", "uv build", "npm install", "npm ci", "npm i lodash", "npm run build",
    "yarn add left-pad", "pnpm remove x", "cargo build --release", "cargo install ripgrep",
    "cargo fmt", "apt-get install -y curl", "apt remove vim", "make", "make -j8",
    "make test && make install", "cmake -S . -B build", "ninja -C build",
)


def test_package_and_build_tools_write_only_through_their_mutating_subcommands() -> None:
    for command in CHECKS:
        assert not eg.writes_files({"command": command}), command
    for command in WRITES:
        assert eg.writes_files({"command": command}), command


def test_a_finish_after_the_test_run_passes_and_the_one_refusal_is_kept() -> None:
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"})
    gate.observe("Bash", {"command": "uv run pytest -q"})
    assert not gate.unchecked_change
    assert gate.verdict({"success": True, "summary_in_html": "All 12 tests pass."}) is None
    assert not gate.used
    verdict = gate.verdict({"success": True, "summary_in_html": "not tested, educated guess"})
    assert verdict is not None and '"educated guess"' in verdict
    assert "files were changed" not in verdict
