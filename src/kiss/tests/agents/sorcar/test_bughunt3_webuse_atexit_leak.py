# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: ``WebUseTool`` leaks one atexit registration per instance.

``__init__`` calls ``atexit.register(self.close)`` but ``close()`` never
unregisters it, so every agent run (``SorcarAgent.run`` constructs a
fresh ``WebUseTool`` and closes it in its ``finally``) permanently
retains the closed tool object in the atexit table — unbounded growth
over a long VS Code session.  After ``close()`` the registration must be
gone; ``_ensure_browser`` re-registers when the tool is revived.
"""

import atexit

from kiss.agents.sorcar.web_use_tool import WebUseTool


def test_close_unregisters_atexit_hook() -> None:
    """Creating then closing a WebUseTool must not grow the atexit table."""
    before = atexit._ncallbacks()
    tool = WebUseTool(user_data_dir=None)
    assert atexit._ncallbacks() == before + 1
    tool.close()
    assert atexit._ncallbacks() == before


def test_repeated_create_close_does_not_accumulate() -> None:
    before = atexit._ncallbacks()
    for _ in range(5):
        WebUseTool(user_data_dir=None).close()
    assert atexit._ncallbacks() == before
