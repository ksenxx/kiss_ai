"""Pre-configured AppAgent factory for named apps.

Provides a convenience function and class to create an AppAgent for any
app name (e.g. 'slack', 'github', 'spotify') without manually wiring
ChannelAgent + AppAgent.
"""

from __future__ import annotations

from typing import Any

from kiss.channels.channel_agent import AppAgent, ChannelAgent


def create_app_agent(app_name: str, **kwargs: Any) -> AppAgent:
    """Create an AppAgent pre-configured for the named app.

    Args:
        app_name: App identifier (e.g. 'slack', 'github', 'spotify').
        **kwargs: Passed through to AppAgent (e.g. wait_for_user_callback).

    Returns:
        An AppAgent instance with a ChannelAgent for the named app.
    """
    return AppAgent(ChannelAgent(app_name), **kwargs)
