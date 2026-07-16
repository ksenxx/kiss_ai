# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Back-compat alias — the implementation lives in ``kiss.core.useful_tools``.

``UsefulTools`` moved into ``kiss.core`` so that core code (e.g. the
``RelentlessAgent`` trajectory summarizer) never depends on sorcar.
Importing :mod:`kiss.agents.sorcar.code_graph` first registers the
code-graph grep-hint provider with the core module (the dependency
inversion that replaced the old direct import).  Aliasing the module
object in ``sys.modules`` keeps every historical import path AND
monkeypatch target (``kiss.agents.sorcar.useful_tools.X``) pointing at
the one real module.
"""

import sys

from kiss.agents.sorcar import (
    code_graph as _code_graph,  # noqa: F401 — registers grep-hint provider
)

# Static re-exports so type checkers resolve historical from-imports
# against this path; at runtime the ``sys.modules`` alias below makes
# this module *be* ``kiss.core.useful_tools``.
from kiss.core import useful_tools as _useful_tools
from kiss.core.useful_tools import *  # noqa: F401,F403 — re-exported
from kiss.core.useful_tools import (  # noqa: F401 — private names used by importers
    _MAX_BINARY_READ_BYTES,
    _absolutize,
    _active_worktree_remap,
    _bash_parent_repo_guard,
    _clean_env,
    _kill_process_group,
    _stale_worktree_fallback,
    _stop_monitor,
    _truncate_output,
)

sys.modules[__name__] = _useful_tools
