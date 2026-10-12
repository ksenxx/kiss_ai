# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Every channel backend uses the ToolMethodBackend mixin.

The backends are discovered from every ``*_sea`` module under
``kiss.agents.third_party_agents`` (classes named ``*ChannelBackend``), so a
new channel is covered without editing this file.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import kiss.agents.third_party_agents as _agents_pkg
from kiss.agents.third_party_agents._channel_agent_utils import ToolMethodBackend


def _all_channel_backends() -> list[type]:
    """Return every ``*ChannelBackend`` class defined in a ``*_sea`` module."""
    classes: list[type] = []
    for info in pkgutil.walk_packages(_agents_pkg.__path__, _agents_pkg.__name__ + "."):
        if not info.name.endswith("_sea"):
            continue
        module = importlib.import_module(info.name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if name.endswith("ChannelBackend") and cls.__module__ == module.__name__:
                classes.append(cls)
    return sorted(classes, key=lambda c: c.__name__)


ALL_BACKENDS = _all_channel_backends()


def test_backends_were_discovered() -> None:
    """Discovery finds the known channel backends (guards against a silent empty list)."""
    names = {cls.__name__ for cls in ALL_BACKENDS}
    assert {"SlackChannelBackend", "TelegramChannelBackend", "DiscordChannelBackend"} <= names
    assert len(ALL_BACKENDS) >= 40


@pytest.mark.parametrize("cls", ALL_BACKENDS, ids=lambda c: c.__name__)
def test_backend_inherits_tool_method_backend(cls: type) -> None:
    """Every channel backend class inherits from ToolMethodBackend."""
    assert issubclass(cls, ToolMethodBackend)


@pytest.mark.parametrize("cls", ALL_BACKENDS, ids=lambda c: c.__name__)
def test_no_inline_get_tool_methods(cls: type) -> None:
    """Backend class does not define its own get_tool_methods (uses mixin)."""
    assert "get_tool_methods" not in cls.__dict__
