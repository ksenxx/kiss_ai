"""A stand-in ``signal-cli`` program shared by the Signal backend tests.

A Python program (not a shell script) so the same stand-in runs on
Windows, where ``install_cli_script`` adds the ``.cmd`` shim.  ``receive``
prints three envelopes from two senders; ``send`` to ``+FAIL`` exits 1
with an error on stderr and ``send`` to ``+WARN`` exits 0 with a
warning mentioning "error" on stderr (the stderr heuristic case).
"""

FAKE_SIGNAL_CLI = """#!/usr/bin/env python3
import json
import sys
args = sys.argv[1:]
if args[:1] == ["-u"]:
    args = args[2:]
cmd = args[0] if args else ""
if cmd == "receive":
    for source, ts, text in (
        ("+1AAA", 111, "hello A1"),
        ("+1BBB", 112, "hello B"),
        ("+1AAA", 113, "hello A2"),
    ):
        print(json.dumps({"envelope": {
            "source": source, "timestamp": ts, "dataMessage": {"message": text},
        }}))
    sys.exit(0)
if cmd == "send":
    if args[-1] == "+FAIL":
        print("Failed to send message: ERROR unregistered recipient", file=sys.stderr)
        sys.exit(1)
    if args[-1] == "+WARN":
        print("WARNING: error while updating profile", file=sys.stderr)
sys.exit(0)
"""
