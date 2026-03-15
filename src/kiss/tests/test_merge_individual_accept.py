"""Test: merge panel closes after resolving all diffs one-by-one."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import requests


def _wait_for_port(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = Path(port_file).read_text().strip()
            if text:
                return int(text)
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.3)
    raise TimeoutError("Server did not write port file")


class TestMergeIndividualAccept:
    """When all diffs are accepted one-by-one, merge_ended must be broadcast."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        tmpdir = tmp_path_factory.mktemp("merge_indiv")
        work_dir = str(tmpdir / "work")
        os.makedirs(work_dir)

        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=work_dir, capture_output=True,
        )
        Path(work_dir, "file.txt").write_text("line1\nline2\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True
        )

        port_file = str(tmpdir / "port")
        cov_file = str(tmpdir / ".coverage.merge")

        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "_sorcar_merge_test_server.py"),
                port_file,
                work_dir,
                cov_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        keepalive = None
        try:
            port = _wait_for_port(port_file)
            base_url = f"http://127.0.0.1:{port}"

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    if requests.get(base_url, timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    time.sleep(0.3)
            else:
                raise TimeoutError("Server not responsive")

            keepalive = requests.get(f"{base_url}/events", stream=True, timeout=300)
            yield base_url, work_dir, proc, str(tmpdir)
        finally:
            if keepalive is not None:
                keepalive.close()
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            try:
                import coverage as cov_mod

                main_cov = os.path.join(os.getcwd(), ".coverage")
                cov = cov_mod.Coverage(data_file=main_cov)
                cov.combine(data_paths=[cov_file], keep=True)
                cov.save()
            except Exception:
                pass

    def test_individual_accepts_close_merge(self, server) -> None:
        base_url, work_dir, _, tmpdir = server

        # Collect SSE events in background
        events: list[str] = []
        stop = threading.Event()

        def collect_events() -> None:
            try:
                resp = requests.get(
                    f"{base_url}/events", stream=True, timeout=60
                )
                for line in resp.iter_lines(decode_unicode=True):
                    if stop.is_set():
                        break
                    if line and line.startswith("data:"):
                        events.append(line[5:].strip())
            except Exception:
                pass

        t = threading.Thread(target=collect_events, daemon=True)
        t.start()
        time.sleep(0.5)  # let SSE connect

        # Submit a task — the agent modifies file.txt
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "modify file", "model": "gemini-2.0-flash"},
            timeout=30,
        )
        assert resp.status_code == 200

        # Wait for merge_started event (agent finishes → merge view opens)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if any("merge_started" in e for e in events):
                break
            time.sleep(0.3)
        assert any("merge_started" in e for e in events), (
            f"merge_started not received; events={events}"
        )

        # Read the pending-merge.json to find hunk count
        cs_data_dir = None
        for entry in Path(tmpdir).rglob("pending-merge.json"):
            cs_data_dir = str(entry.parent)
            break

        # Also check cs data dir from work_dir hash
        if cs_data_dir is None:
            import hashlib

            wd_hash = hashlib.md5(work_dir.encode()).hexdigest()[:8]
            kiss_dir = Path.home() / ".kiss"
            cs_dir = kiss_dir / f"cs-{wd_hash}"
            pm = cs_dir / "pending-merge.json"
            if pm.exists():
                cs_data_dir = str(cs_dir)

        # The pending-merge.json may have been consumed by extension already.
        # Instead, count hunks from the diff programmatically.
        from kiss.agents.sorcar.code_server import _parse_diff_hunks

        post_hunks = _parse_diff_hunks(work_dir)
        total_hunks = sum(len(hunks) for hunks in post_hunks.values())
        # The agent changed "line1" to "MODIFIED line1" and added "new line3"
        # This should give us at least 1 hunk
        assert total_hunks >= 1, f"Expected hunks, got {total_hunks}"

        # Send individual "accept" actions one by one
        events_before = len(events)
        for i in range(total_hunks):
            resp = requests.post(
                f"{base_url}/merge-action",
                json={"action": "accept"},
                timeout=5,
            )
            assert resp.status_code == 200

        # Wait for merge_ended event
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if any("merge_ended" in e for e in events[events_before:]):
                break
            time.sleep(0.3)

        stop.set()
        t.join(timeout=5)

        merge_ended_found = any("merge_ended" in e for e in events[events_before:])
        assert merge_ended_found, (
            f"merge_ended not received after {total_hunks} individual accepts; "
            f"events={events[events_before:]}"
        )
