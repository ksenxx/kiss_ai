# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``kiss.core.DEFAULT_CONFIG`` must track config_builder rebinds.

``config_builder.add_config``/``build_config`` rebind
``kiss.core.config.DEFAULT_CONFIG`` to a brand-new object at runtime.
A static ``from kiss.core.config import DEFAULT_CONFIG`` inside
``kiss/core/__init__.py`` would freeze the package attribute at import time.
The PEP 562 export keeps repeated ``kiss.core.DEFAULT_CONFIG`` attribute
lookups live.  As with every Python ``from module import name`` statement,
a caller's local binding is still a snapshot; production callers therefore
use module/attribute lookup rather than importing this mutable binding by
value.
"""

import sys

from pydantic import BaseModel

import kiss.core
from kiss.core import config as config_module
from kiss.core.config_builder import add_config


class _Wave2TestSection(BaseModel):
    knob: int = 7


def test_package_export_tracks_add_config_rebind() -> None:
    """After add_config rebinds, kiss.core.DEFAULT_CONFIG is the new object."""
    orig = config_module.DEFAULT_CONFIG
    saved_argv = sys.argv
    sys.argv = ["prog"]
    try:
        add_config("wave2test", _Wave2TestSection)
        rebound = config_module.DEFAULT_CONFIG
        assert rebound is not orig
        assert kiss.core.DEFAULT_CONFIG is rebound
        assert getattr(kiss.core, "DEFAULT_CONFIG") is rebound
        assert kiss.core.DEFAULT_CONFIG.wave2test.knob == 7  # type: ignore[attr-defined]
    finally:
        sys.argv = saved_argv
        config_module.DEFAULT_CONFIG = orig
    # Restoring the binding must also be observed live.
    assert kiss.core.DEFAULT_CONFIG is orig


def test_package_getattr_rejects_unknown_names() -> None:
    """Unknown attributes still raise AttributeError, not silently None."""
    try:
        kiss.core.NO_SUCH_ATTRIBUTE_WAVE2  # type: ignore[attr-defined]  # noqa: B018
    except AttributeError as err:
        assert "NO_SUCH_ATTRIBUTE_WAVE2" in str(err)
    else:  # pragma: no cover - fails the test if no exception raised
        raise AssertionError("expected AttributeError")
