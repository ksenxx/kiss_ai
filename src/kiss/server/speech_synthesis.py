# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Back-compat alias — the implementation lives in ``kiss.core.speech_synthesis``.

The ``talk`` speech synthesis moved into ``kiss.core`` so that sorcar
code (``kiss.agents.sorcar.sorcar_agent``) never depends on the server
layer.  Aliasing the module object in ``sys.modules`` keeps every
historical import path AND monkeypatch target
(``kiss.server.speech_synthesis.X``) pointing at the one real module.
"""

import sys

# Static re-exports so type checkers resolve historical from-imports
# against this path; at runtime the ``sys.modules`` alias below makes
# this module *be* ``kiss.core.speech_synthesis``.
from kiss.core import speech_synthesis as _speech_synthesis
from kiss.core.speech_synthesis import *  # noqa: F401,F403 — re-exported

sys.modules[__name__] = _speech_synthesis
