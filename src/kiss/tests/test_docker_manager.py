"""Test suite for DockerManager without mocking."""

import socket
import unittest

import docker
import requests

from kiss.docker.docker_manager import DockerManager


def is_docker_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


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
            env.Bash(
                "echo 'Hello from Docker!' > /tmp/index.html", "Create test file"
            )
            env.Bash("cd /tmp && python -m http.server 8000 &", "Start HTTP server")

            import time

            time.sleep(2)

            self.assertEqual(env.get_host_port(8000), host_port)

            try:
                response = requests.get(f"http://localhost:{host_port}/index.html", timeout=5)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Hello from Docker!", response.text)
            except requests.exceptions.ConnectionError:
                self.fail(f"Could not connect to HTTP server on port {host_port}")


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManagerStreaming(unittest.TestCase):
    def test_streaming_captures_output(self) -> None:
        streamed: list[str] = []
        with DockerManager("python:3.11-slim") as env:
            env.stream_callback = streamed.append
            result = env.Bash("echo line1 && echo line2 && echo line3", "Stream test")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        joined = "".join(streamed)
        assert "line1" in joined
        assert "line2" in joined
        assert "line3" in joined

    def test_streaming_error_exit_code(self) -> None:
        streamed: list[str] = []
        with DockerManager("python:3.11-slim") as env:
            env.stream_callback = streamed.append
            result = env.Bash("echo before_fail && false", "Error stream test")
        assert "[exit code:" in result

    def test_no_streaming_without_callback(self) -> None:
        with DockerManager("python:3.11-slim") as env:
            assert env.stream_callback is None
            result = env.Bash("echo normal_output", "No stream test")
        assert "normal_output" in result

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
