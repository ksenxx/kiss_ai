# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Importing the ``kiss`` / ``kiss.core`` packages does not build the config.

``kiss/__init__.py`` imports ``kiss.core._version``, which runs
``kiss/core/__init__.py``.  That file used to import the ``Config`` and
``KISSError`` it re-exports eagerly, so every ``kiss.*`` import pulled in
pydantic and constructed ``DEFAULT_CONFIG`` as a side effect, reading the
API-key environment variables at that moment.  It now imports them on
first use, and ``from kiss.core import Config, KISSError`` still works.
"""

from __future__ import annotations

import os
import subprocess
import sys

import kiss.core
from kiss.core import config, kiss_error

_PROBE = """
import os, sys
import kiss, kiss.core, kiss.core._version
loaded = sorted(m for m in ("kiss.core.config", "pydantic") if m in sys.modules)
os.environ["OPENAI_API_KEY"] = "sk-set-after-import-kiss"
from kiss.core.config import DEFAULT_CONFIG
print(loaded, DEFAULT_CONFIG.OPENAI_API_KEY)
"""


def test_importing_the_packages_leaves_the_config_unbuilt() -> None:
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    ).stdout
    assert out.strip() == "[] sk-set-after-import-kiss"


def test_the_package_still_exports_its_public_names() -> None:
    from kiss.core import Config, KISSError, utils

    assert Config is config.Config
    assert KISSError is kiss_error.KISSError
    assert utils.__name__ == "kiss.core.utils", "submodules still import from the package"
    assert not hasattr(kiss.core, "no_such_name")
