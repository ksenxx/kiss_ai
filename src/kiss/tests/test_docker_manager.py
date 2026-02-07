"""Test suite for DockerManager without mocking."""

import os
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
    def test_actual_no_mock(self) -> None:
        with DockerManager("ubuntu:latest") as env:
            output = env.run_bash_command('echo "Hello, World!"', "Echo command")
            self.assertIn("Hello, World!", output)

    def test_host_to_container_shared_volume(self) -> None:
        with DockerManager("ubuntu:latest") as env:
            assert env.host_shared_path is not None
            host_file_path = os.path.join(env.host_shared_path, "testfile.txt")
            test_content = "Data written from host for Docker shared path test."
            with open(host_file_path, "w", encoding="utf-8") as f:
                f.write(test_content)

            client_file_path = os.path.join(env.client_shared_path, "testfile.txt")
            output = env.run_bash_command(
                f'cat "{client_file_path}"', "Read file written from host in container"
            )
            self.assertEqual(test_content, output.strip())

    def test_port_mapping(self) -> None:
        def find_free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port: int = s.getsockname()[1]
                return port

        host_port = find_free_port()

        with DockerManager("python:3.11-slim", ports={8000: host_port}) as env:
            env.run_bash_command(
                "echo 'Hello from Docker!' > /tmp/index.html", "Create test file"
            )
            env.run_bash_command("cd /tmp && python -m http.server 8000 &", "Start HTTP server")

            import time

            time.sleep(2)

            self.assertEqual(env.get_host_port(8000), host_port)

            try:
                response = requests.get(f"http://localhost:{host_port}/index.html", timeout=5)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Hello from Docker!", response.text)
            except requests.exceptions.ConnectionError:
                self.fail(f"Could not connect to HTTP server on port {host_port}")


if __name__ == "__main__":
    unittest.main()
