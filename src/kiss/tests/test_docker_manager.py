# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for DockerManager without mocking."""

import os
import unittest

import docker

from kiss.docker.docker_manager import DockerManager


def is_docker_available() -> bool:
    """Check if Docker daemon is running and accessible."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManager(unittest.TestCase):
    """Test suite for DockerManager."""

    def test_actual_no_mock(self):
        """Test the actual workflow without mocking."""
        with DockerManager("ubuntu:latest") as env:
            output = env.run_bash_command('echo "Hello, World!"', "Echo command")
            self.assertIn("Hello, World!", output)
            self.assertIn("EXIT_CODE", output)
            self.assertIn("0", output)

    def test_host_to_container_shared_volume(self):
        """Test writing a file on host_shared_path and verifying its existence and contents."""

        with DockerManager("ubuntu:latest") as env:
            # Write a file on the host shared path
            host_file_path = os.path.join(env.host_shared_path, "testfile.txt")
            test_content = "Data written from host for Docker shared path test."
            with open(host_file_path, "w", encoding="utf-8") as f:
                f.write(test_content)

            client_file_path = os.path.join(env.client_shared_path, "testfile.txt")
            output = env.run_bash_command(
                f'cat "{client_file_path}"', "Read file written from host in container"
            )
            print(output)
            start_tag = "----STDOUT-----\n"
            end_tag = "\n----STDERR-----"
            start_idx = output.find(start_tag)
            end_idx = output.find(end_tag)
            if start_idx != -1 and end_idx != -1:
                file_contents_from_container = output[start_idx + len(start_tag) : end_idx]
            else:
                file_contents_from_container = output

            self.assertEqual(test_content, file_contents_from_container.strip())


if __name__ == "__main__":
    unittest.main()
