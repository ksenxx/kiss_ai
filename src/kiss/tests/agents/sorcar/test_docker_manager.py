# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test suite for DockerManager without mocking."""

import socket
import time
import unittest

import docker
import pytest
import requests

from kiss.agents.sorcar.docker_manager import DockerManager


def is_docker_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.mark.slow
@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManager(unittest.TestCase):
    def test_port_mapping(self) -> None:
        def find_free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port: int = s.getsockname()[1]
                return port

        host_port = find_free_port()

        with DockerManager("python:3.11-slim", ports={8000: host_port}) as env:
            env.Bash("echo 'Hello from Docker!' > /tmp/index.html", "Create test file")
            # The server must not keep the exec's stdout/stderr: the daemon
            # closes those streams once the exec's shell exits, and the
            # server's request log would then die on a broken pipe and drop
            # the connection without a response.
            env.Bash(
                "cd /tmp && python -m http.server 8000 > /tmp/server.log 2>&1 &",
                "Start HTTP server",
            )

            self.assertEqual(env.get_host_port(8000), host_port)

            deadline = time.monotonic() + 30
            while True:
                try:
                    response = requests.get(
                        f"http://localhost:{host_port}/index.html", timeout=5
                    )
                    break
                except requests.exceptions.ConnectionError:
                    if time.monotonic() >= deadline:
                        self.fail(f"Could not connect to HTTP server on port {host_port}")
                    time.sleep(0.2)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Hello from Docker!", response.text)


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManagerStreaming(unittest.TestCase):

    def test_streaming_error_exit_code(self) -> None:
        streamed: list[str] = []
        with DockerManager("python:3.11-slim") as env:
            env.stream_callback = streamed.append
            result = env.Bash("echo before_fail && false", "Error stream test")
        assert "[exit code:" in result

    def test_streaming_stderr(self) -> None:
        streamed: list[str] = []
        with DockerManager("python:3.11-slim") as env:
            env.stream_callback = streamed.append
            env.Bash("echo stdout_msg && echo stderr_msg >&2", "Stderr stream")
        joined = "".join(streamed)
        assert "stdout_msg" in joined
        assert "stderr_msg" in joined


if __name__ == "__main__":
    unittest.main()
