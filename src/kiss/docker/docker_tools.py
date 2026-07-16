# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Back-compat alias — the implementation lives in ``kiss.core.docker_tools``.

``DockerTools`` moved into ``kiss.core`` so that core and sorcar code
never depend on code outside their allowed layers.  Aliasing the module
object in ``sys.modules`` keeps every historical import path AND
monkeypatch target (``kiss.docker.docker_tools.X``) pointing at the one
real module.
"""

import sys

# Static re-exports so type checkers resolve historical from-imports
# against this path; at runtime the ``sys.modules`` alias below makes
# this module *be* ``kiss.core.docker_tools``.
from kiss.core import docker_tools as _docker_tools
from kiss.core.docker_tools import *  # noqa: F401,F403 — re-exported

sys.modules[__name__] = _docker_tools
