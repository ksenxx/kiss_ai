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
prompt with the ``Bash`` tool alone and returns the output, or
:mod:`kiss.agents.seas.autoroute_sea`, which finishes a task at the lowest
cost per accepted result by routing each unit of work to the cheapest
model tier that passes its acceptance check, or
:mod:`kiss.agents.seas.skillopt_sea`, which optimizes the prompt text of
a skill or of another SEA against an eval set (SkillOpt), or
:mod:`kiss.agents.seas.write_paper_sea`, which writes or revises a
research paper under the rules of ``templates/write_paper_prompt.md``
with tools for the AI-slop gates and the LaTeX build, or
:mod:`kiss.agents.seas.review_paper_sea`, which reviews a paper for any
venue with tools that read the PDF page by page and check the review's
structure and AI-slop gates.
Every ``*_sea.py`` module in this package is also exposed as a chat
slash command (``/merge``, ``/sh``, ``/autoroute``, ``/skillopt``, ``/write_paper``,
``/review_paper``, ...) by
:mod:`kiss.agents.sorcar.sea_commands`.
"""
