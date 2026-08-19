# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Vendor-display test for Z.AI + Moonshot extracted from
``kiss.tests.agents.vscode.test_zai_moonshot_keys``.

Moved here because ``kiss.server.helpers.model_vendor`` is the only
production dependency; the settings-panel media checks stayed behind
in tests/agents/vscode.
"""

from __future__ import annotations

from kiss.server import helpers


def test_model_vendor_zai_and_moonshot() -> None:
    """`model_vendor` routes glm-* to Z.AI and kimi-*/moonshot-* to Moonshot."""
    assert helpers.model_vendor("glm-4.6")[0] == "Z.AI"
    assert helpers.model_vendor("kimi-k2.6")[0] == "Moonshot"
    assert helpers.model_vendor("moonshot-v1-32k")[0] == "Moonshot"
    assert helpers.model_vendor("minimax-m2.5")[0] != "MiniMax"
