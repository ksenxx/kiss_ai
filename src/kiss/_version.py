# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Back-compat alias — the version is single-sourced in ``kiss.core._version``.

The real module lives in ``kiss/core/_version.py`` so that code inside
``kiss.core`` (which must not depend on anything outside ``kiss.core``)
can import the version without reaching upward in the package tree.
"""

from kiss.core._version import __version__

__all__ = ["__version__"]
