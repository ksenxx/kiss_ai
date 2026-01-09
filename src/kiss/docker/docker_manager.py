# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Docker library for managing Docker containers and executing commands."""

import os
import shlex
import shutil
import tempfile
from typing import Any

import docker
from docker.models.containers import Container  # type: ignore[assignment]

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.simple_formatter import SimpleFormatter


class DockerManager:
    """Manages Docker container lifecycle and command execution."""

    def __init__(
        self,
        image_name: str,
        tag: str = "latest",
        workdir: str = "/",
        mount_shared_volume: bool = True,
    ) -> None:
        """Initialize the Docker client.

        Args:
            image_name: The name of the Docker image (e.g., 'ubuntu', 'python')
            tag: The tag/version of the image (default: 'latest')
            workdir: The working directory inside the container
            mount_shared_volume: Whether to mount a shared volume. Set to False
                for images that already have content in the workdir (e.g., SWE-bench).
        """
        self.client = docker.from_env()
        self.container: Container | None = None
        self.formatter = SimpleFormatter()
        self.workdir = workdir
        self.mount_shared_volume = mount_shared_volume
        self.client_shared_path = DEFAULT_CONFIG.docker.client_shared_path
        self.host_shared_path = tempfile.mkdtemp() if mount_shared_volume else None

        if ":" in image_name:
            self.image, self.tag = image_name.rsplit(":", 1)
        else:
            self.image = image_name
            self.tag = tag

    def open(self) -> None:
        """
        Pull and load a Docker image, then create and start a container.

        Args:
            image_name: The name of the Docker image (e.g., 'ubuntu', 'python')
            tag: The tag/version of the image (default: 'latest')
        """
        image = self.image
        tag = self.tag
        full_image_name = f"{image}:{tag}"
        # Pull the image if it doesn't exist locally
        self.formatter.print_status(f"Pulling Docker image: {full_image_name}")
        try:
            self.client.images.get(full_image_name)
        except docker.errors.ImageNotFound:
            self.client.images.pull(image, tag=tag)
        # Create and start a container
        self.formatter.print_status(f"Creating and starting container from {full_image_name}")
        container_kwargs: dict[str, Any] = {
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "command": "/bin/bash",
        }
        if self.mount_shared_volume and self.host_shared_path:
            container_kwargs["volumes"] = {
                self.host_shared_path: {"bind": self.client_shared_path, "mode": "rw"}
            }
        self.container = self.client.containers.run(full_image_name, **container_kwargs)
        container_id = self.container.id[:12] if self.container.id else "unknown"
        self.formatter.print_status(f"Container {container_id} is now running")

    def run_bash_command(self, command: str, description: str) -> str:
        """
        Execute a bash command in the running Docker container.

        Args:
            command: The bash command to execute
            description: A short description of the command in natural language
        Returns:
            The output of the command, including stdout, stderr, and exit code
        """
        if self.container is None:
            raise KISSError("No container is open. Please call open() first.")

        self.formatter.print_status(f"{description}")
        exec_result = self.container.exec_run(
            f"/bin/bash -c {shlex.quote(command)}",
            stdout=True,
            stderr=True,
            demux=True,
            workdir=self.workdir,
        )

        stdout_bytes, stderr_bytes = exec_result.output
        stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""
        exit_code = exec_result.exit_code
        return (
            f"----STDOUT-----\n{stdout}\n----STDERR-----\n{stderr}\n"
            f"----EXIT_CODE-----\n{exit_code}\n"
        )

    def close(self) -> None:
        """Stop and remove the Docker container."""
        if self.container is None:
            print("No container to close.")
            return

        container_id = self.container.id[:12] if self.container.id else "unknown"
        self.formatter.print_status(f"Stopping container {container_id}")
        self.container.stop()

        self.formatter.print_status(f"Removing container {container_id}")
        self.container.remove()

        self.container = None

        # Clean up temporary directory
        if self.host_shared_path and os.path.exists(self.host_shared_path):
            shutil.rmtree(self.host_shared_path)

        print("Container closed successfully")

    def __enter__(self) -> "DockerManager":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()
