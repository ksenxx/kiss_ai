# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Interactive terminal (CLI) front end for the sorcar agents.

This package hosts the sorcar CLI's interactive code — the daemon
client REPL, steering input box, prompt/panel rendering, voice and
talk playback, console printers, and the ``sorcar mcp`` management
subcommand.  It sits ABOVE the agents: modules here may import from
:mod:`kiss.agents.sorcar`, but the agent modules never import this
package at module level (the ``sorcar`` entry point wires the two
together lazily inside ``main``).
"""
