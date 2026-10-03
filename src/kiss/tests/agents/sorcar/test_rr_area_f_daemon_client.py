# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the client-side agent-script path resolver.

``resolve_agent_path`` is the only Python-file resolver of the daemon
client (the former ``resolve_tools_file`` went with the tools-file
contract).  These tests pin its public contract — including the exact
error messages: it accepts only ``str`` values and maps ``""`` and
``None`` to ``""``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar import daemon_client
from kiss.agents.sorcar.daemon_client import resolve_agent_path


def test_tools_file_resolver_is_gone() -> None:
    assert not hasattr(daemon_client, "resolve_tools_file")
    assert not hasattr(daemon_client, "_resolve_py_file")


class TestResolveAgentPath:
    def test_none_and_empty_map_to_empty(self) -> None:
        assert resolve_agent_path(None) == ""
        assert resolve_agent_path("") == ""

    def test_str_resolves_absolutely(self, tmp_path: Path) -> None:
        script = tmp_path / "agent.py"
        script.write_text("def model():\n    return 'm'\n")
        assert resolve_agent_path(str(script)) == str(script.resolve())

    def test_path_object_rejected(self, tmp_path: Path) -> None:
        script = tmp_path / "agent.py"
        script.write_text("def model():\n    return 'm'\n")
        with pytest.raises(
            ValueError,
            match=r"agent_path must be a string path to a Python file, "
                  rf"got {type(script).__name__}",
        ):
            resolve_agent_path(script)  # type: ignore[arg-type]

    def test_wrong_type_message(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"agent_path must be a string path to a Python file, "
                  r"got int: 42",
        ):
            resolve_agent_path(42)  # type: ignore[arg-type]

    def test_non_py_suffix_message(self, tmp_path: Path) -> None:
        other = tmp_path / "agent.sh"
        other.write_text("x")
        with pytest.raises(ValueError, match=r"is not a Python \(\.py\) file") as exc:
            resolve_agent_path(str(other))
        assert str(exc.value).startswith("agent script ")

    def test_missing_file_message(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"does not exist") as exc:
            resolve_agent_path(str(tmp_path / "absent.py"))
        assert str(exc.value).startswith("agent script ")
