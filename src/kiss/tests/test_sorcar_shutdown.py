"""Integration test: sorcar process exits when all SSE clients disconnect."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


def test_process_exits_when_sse_client_disconnects():
    """Start a sorcar server, connect SSE, disconnect, verify process exits."""
    server_script = str(
        Path(__file__).parent / "_sorcar_test_server.py"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmpdir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmpdir, capture_output=True,
        )

        port_file = os.path.join(tmpdir, "port.txt")

        proc = subprocess.Popen(
            [sys.executable, server_script, port_file, tmpdir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # Wait for the server to write the port file (up to 20s)
            port = None
            for _ in range(200):
                if proc.poll() is not None:
                    out = proc.stdout.read().decode() if proc.stdout else ""
                    err = proc.stderr.read().decode() if proc.stderr else ""
                    raise AssertionError(
                        f"Server exited early (rc={proc.returncode})\n"
                        f"stdout: {out[:500]}\nstderr: {err[:500]}"
                    )
                if os.path.exists(port_file):
                    content = open(port_file).read().strip()
                    if content:
                        port = int(content)
                        break
                time.sleep(0.1)
            assert port is not None, "Server did not write port file in time"

            base_url = f"http://127.0.0.1:{port}"

            # Wait for HTTP to be ready
            for _ in range(50):
                try:
                    requests.get(base_url, timeout=1)
                    break
                except requests.ConnectionError:
                    time.sleep(0.1)

            # Connect to SSE endpoint (simulates browser opening)
            session = requests.Session()
            sse_resp = session.get(f"{base_url}/events", stream=True, timeout=5)
            assert sse_resp.status_code == 200

            time.sleep(0.5)

            # Disconnect (simulates closing the browser window)
            sse_resp.close()
            session.close()

            # The process should exit within ~10 seconds
            # (1s disconnect detection + 5s shutdown timer + margin)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise AssertionError(
                    "Server process did not exit after SSE client disconnected"
                )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
