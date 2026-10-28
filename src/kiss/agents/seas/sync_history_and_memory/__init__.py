# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``/sync_history_and_memory`` Sorcar Extension Agent and the engine behind it.

``sync_history_and_memory_sea.py`` is the SEA; ``peer_sync.py`` brings the
task history and the memory of this machine and of every registered machine
to the same state; ``memory_delta.py`` is the stdlib-only half that also
runs on the remote machines; ``ssh_peer.py`` is one multiplexed ssh
connection per machine.
"""
