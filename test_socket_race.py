"""Reproduce the socket race condition."""

import json
import socket
import threading
import time
from pathlib import Path
import tempfile
import sys
import textwrap

from kiss.agents.third_party_agents.cron_manager_daemon import CronDaemon


def test_socket_race_condition():
    """Demonstrate the race condition between socket file existence and listen()."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sock_path = tmp_path / "test.sock"
        pid_path = tmp_path / "test.pid"
        
        # Create a fake crontab script
        tab = tmp_path / "tab"
        tab.write_text("")
        script = tmp_path / "crontab"
        script.write_text(
            textwrap.dedent(f"""\
            #!{sys.executable}
            import sys, pathlib
            TAB = pathlib.Path({str(tab)!r})
            if sys.argv[1:] == ["-l"]:
                content = TAB.read_text() if TAB.exists() else ""
                if content:
                    sys.stdout.write(content)
                    sys.exit(0)
                sys.stderr.write("no crontab for test\\n")
                sys.exit(1)
            elif sys.argv[1:] == ["-"]:
                TAB.write_text(sys.stdin.read())
                sys.exit(0)
            """)
        )
        script.chmod(0o755)
        
        # Start daemon
        d = CronDaemon(sock_path=sock_path, pid_path=pid_path, crontab_cmd=str(script))
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        
        # Wait for socket file to exist (current behavior)
        for _ in range(100):
            if sock_path.exists():
                break
            time.sleep(0.05)
        
        # Try to connect multiple times without retry (simulating the test)
        for attempt in range(5):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.settimeout(1.0)
                s.connect(str(sock_path))
                print(f"Attempt {attempt}: Connected successfully")
                s.close()
                break
            except ConnectionRefusedError as e:
                print(f"Attempt {attempt}: ConnectionRefusedError - {e}")
                s.close()
                time.sleep(0.05)
        
        # Cleanup
        d._stop_event.set()
        t.join(timeout=5)


if __name__ == "__main__":
    test_socket_race_condition()
