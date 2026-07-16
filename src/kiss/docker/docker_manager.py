# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Back-compat alias — the implementation lives in ``kiss.core.docker_manager``.

``DockerManager`` moved into ``kiss.core`` so that core code (e.g.
``kiss.core.relentless_agent``) never depends on code outside
``kiss/core/``.  Aliasing the module object in ``sys.modules`` keeps
every historical import path AND monkeypatch target
(``kiss.docker.docker_manager.X``) pointing at the one real module.
"""

import sys

# Static re-exports so type checkers resolve historical from-imports
# against this path; at runtime the ``sys.modules`` alias below makes
# this module *be* ``kiss.core.docker_manager``.
from kiss.core import docker_manager as _docker_manager
from kiss.core.docker_manager import *  # noqa: F401,F403 — re-exported

sys.modules[__name__] = _docker_manager
