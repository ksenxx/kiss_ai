"""Make the ``attacks`` and ``guardians`` packages importable from pytest.

We do not want to require a ``pip install -e .`` step just to run the
break-suite, so we add the repository directory itself to ``sys.path``.
The guardians package must already be installed in the active venv
(``uv pip install -e meijer_break/guardians``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent   # meijer_break/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
