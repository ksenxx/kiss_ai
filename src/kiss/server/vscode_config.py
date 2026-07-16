# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Back-compat alias — the implementation lives in ``kiss.core.vscode_config``.

The config store moved into ``kiss.core`` so that sorcar code
(``kiss.agents.sorcar.persistence`` / ``skills``) never depends on the
server layer.  Aliasing the module object in ``sys.modules`` keeps every
historical import path AND monkeypatch target
(``kiss.server.vscode_config.X``) pointing at the one real module.
"""

import sys

# Static re-exports so type checkers resolve historical from-imports
# against this path; at runtime the ``sys.modules`` alias below makes
# this module *be* ``kiss.core.vscode_config``.
from kiss.core import vscode_config as _vscode_config
from kiss.core.vscode_config import *  # noqa: F401,F403 — re-exported
from kiss.core.vscode_config import (  # noqa: F401 — private names used by importers
    _atomic_write_text_secure,
    _config_dir,
    _config_path,
    _get_user_shell,
    _refresh_config,
    _resolve_shell_path,
    _shell_rc_path,
)

sys.modules[__name__] = _vscode_config
