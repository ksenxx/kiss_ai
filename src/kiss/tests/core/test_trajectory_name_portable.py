"""The trajectory file name is valid on every platform whatever the agent is called.

``run_parallel`` names each worker ``Parallel-<task[:40]>``, so an agent's
name can carry any text the user typed: Windows paths (``C:\\Users\\...``),
``*``, ``?``, quotes.  ``Base.get_trajectory_path`` used to copy such a
name into the file name after replacing only spaces and ``/``, which made
``_save`` raise ``OSError: [WinError 123]`` on Windows (2026-10-04 full
suite, ``test_run_parallel_integration``).  Real agents, real saves.
"""

from __future__ import annotations

import re

from kiss.core.base import Base

_WINDOWS_RESERVED = ':\\*?"<>|'


def test_hostile_agent_name_saves_a_trajectory_with_a_portable_file_name() -> None:
    """Every character Windows rejects is replaced; the file is written."""
    agent = Base('Parallel-Read the file C:\\Users\\me\\a*b?"c"<d>|e/f')
    agent._add_message("user", "hello")
    agent._save()

    path = agent.get_trajectory_path()
    assert path.is_file()
    assert not any(ch in path.name for ch in _WINDOWS_RESERVED), path.name
    assert " " not in path.name and "/" not in path.name
    assert path.name.startswith("trajectory_Parallel-Read_the_file_C__Users_me_a_b__c__d__e_f_")
    assert re.fullmatch(r"trajectory_[\w.-]+\.yaml", path.name), path.name
    assert "hello" in path.read_text(encoding="utf-8")


def test_plain_names_keep_their_spelling() -> None:
    """Letters, digits, dots, dashes and underscores pass through unchanged."""
    agent = Base("talk-speech_synthesis.v2")
    assert agent.get_trajectory_path().name.startswith(
        f"trajectory_talk-speech_synthesis.v2_{agent.id}_"
    )
