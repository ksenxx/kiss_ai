# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for DockerManager without mocking."""

import os
import socket
import unittest

import docker
import requests

from kiss.docker.docker_manager import DockerManager


def is_docker_available() -> bool:
    """Check if Docker daemon is running and accessible.

    Returns:
        True if Docker daemon is running and accessible, False otherwise.
    """
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManager(unittest.TestCase):
    """Test suite for DockerManager."""

    def test_actual_no_mock(self) -> None:
        """Test the actual workflow without mocking.

        Verifies that DockerManager can start a container and execute
        a basic bash command, returning the expected output.

        Returns:
            None
        """
        with DockerManager("ubuntu:latest") as env:
            output = env.run_bash_command('echo "Hello, World!"', "Echo command")
            self.assertIn("Hello, World!", output)

    def test_host_to_container_shared_volume(self) -> None:
        """Test writing a file on host_shared_path and verifying its existence and contents.

        Verifies that files written on the host's shared path are accessible
        inside the container via the client_shared_path.

        Returns:
            None
        """

        with DockerManager("ubuntu:latest") as env:
            # Write a file on the host shared path
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
        """Test port mapping from container to host.

        Verifies that ports inside the container are correctly mapped to
        host ports and that services running in the container are accessible
        from the host via the mapped ports.

        Returns:
            None
        """

        def find_free_port() -> int:
            """Find a free port on localhost.

            Returns:
                An available port number on localhost.
            """
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port: int = s.getsockname()[1]
                return port

        host_port = find_free_port()

        # Use Python image and start a simple HTTP server on port 8000 inside container
        with DockerManager("python:3.11-slim", ports={8000: host_port}) as env:
            # Start a simple HTTP server in the background
            env.run_bash_command(
                "echo 'Hello from Docker!' > /tmp/index.html",
                "Create test file for HTTP server",
            )
            env.run_bash_command(
                "cd /tmp && python -m http.server 8000 &",
                "Start HTTP server in background",
            )

            # Give the server a moment to start
            import time

            time.sleep(2)

            # Verify the port is mapped correctly
            mapped_port = env.get_host_port(8000)
            self.assertEqual(mapped_port, host_port)

            # Test that we can actually connect to the HTTP server via the mapped port
            try:
                response = requests.get(f"http://localhost:{host_port}/index.html", timeout=5)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Hello from Docker!", response.text)
            except requests.exceptions.ConnectionError:
                self.fail(f"Could not connect to HTTP server on port {host_port}")


if __name__ == "__main__":
    unittest.main()
