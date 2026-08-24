# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the deduplicated path resolvers (F-R5).

``resolve_tools_file`` and ``resolve_agent_path`` were copy-paste
near-duplicates; they now share ``_resolve_py_file``.  These tests pin
the public contracts — including the exact error messages and the
deliberate asymmetries: ``resolve_tools_file`` accepts ``pathlib.Path``
values and rejects ``""`` (it resolves to the working directory, not a
``.py`` file), while ``resolve_agent_path`` accepts only ``str`` and
maps ``""`` to ``""``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.daemon_client import (
    resolve_agent_path,
    resolve_tools_file,
)


class TestResolveToolsFile:
    def test_none_maps_to_empty(self) -> None:
        assert resolve_tools_file(None) == ""

    def test_str_and_path_resolve_absolutely(self, tmp_path: Path) -> None:
        script = tmp_path / "tools.py"
        script.write_text("def get_tools():\n    return []\n")
        assert resolve_tools_file(str(script)) == str(script.resolve())
        assert resolve_tools_file(script) == str(script.resolve())

    def test_wrong_type_message(self) -> None:
        with pytest.raises(ValueError, match=r"tools must be a path to a Python file, got int: 42"):
            resolve_tools_file(42)  # type: ignore[arg-type]

    def test_non_py_suffix_message(self, tmp_path: Path) -> None:
        other = tmp_path / "tools.txt"
        other.write_text("x")
        with pytest.raises(ValueError, match=r"is not a Python \(\.py\) file") as exc:
            resolve_tools_file(str(other))
        assert str(exc.value).startswith("tools file ")

    def test_missing_file_message(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"does not exist") as exc:
            resolve_tools_file(str(tmp_path / "absent.py"))
        assert str(exc.value).startswith("tools file ")

    def test_empty_string_still_rejected(self) -> None:
        # "" resolves to the working directory — never a .py file.
        with pytest.raises(ValueError, match=r"is not a Python \(\.py\) file"):
            resolve_tools_file("")


class TestResolveAgentPath:
    def test_none_and_empty_map_to_empty(self) -> None:
        assert resolve_agent_path(None) == ""
        assert resolve_agent_path("") == ""

    def test_str_resolves_absolutely(self, tmp_path: Path) -> None:
        script = tmp_path / "agent.py"
        script.write_text("def get_model():\n    return 'm'\n")
        assert resolve_agent_path(str(script)) == str(script.resolve())

    def test_path_object_rejected(self, tmp_path: Path) -> None:
        script = tmp_path / "agent.py"
        script.write_text("def get_model():\n    return 'm'\n")
        with pytest.raises(
            ValueError,
            match=r"agent_path must be a string path to a Python file, "
                  r"got PosixPath",
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
