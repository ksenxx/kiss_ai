# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Top-level Kiss module for the project."""

from kiss.core._version import __version__
from kiss.core.thread_join import install_bounded_join

install_bounded_join()

__all__ = ["__version__"]
