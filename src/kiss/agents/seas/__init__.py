# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bundled Sorcar Extension Agents (SEAs) that extend Sorcar itself.

Unlike :mod:`kiss.agents.third_party_agents`, which wrap external
services, the SEAs here drive Sorcar's own workflows — for example
:mod:`kiss.agents.seas.merge_sea`, the merge-conflict resolver the
auto-commit worktree merge runs when a squash merge conflicts, or
:mod:`kiss.agents.seas.sh_sea`, which runs the shell command in its
prompt with the ``Bash`` tool alone and returns the output.
Every ``*_sea.py`` module in this package is also exposed as a chat
slash command (``/merge``, ``/sh``, ...) by
:mod:`kiss.agents.sorcar.sea_commands`.
"""
