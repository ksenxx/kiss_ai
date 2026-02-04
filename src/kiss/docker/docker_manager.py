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
        ports: dict[int, int] | None = None,
    ) -> None:
        """Initialize the Docker client.

        Args:
            image_name: The name of the Docker image (e.g., 'ubuntu', 'python')
            tag: The tag/version of the image (default: 'latest')
            workdir: The working directory inside the container
            mount_shared_volume: Whether to mount a shared volume. Set to False
                for images that already have content in the workdir (e.g., SWE-bench).
            ports: Port mapping from container port to host port.
                Example: {8080: 8080} maps container port 8080 to host port 8080.
                Example: {80: 8000, 443: 8443} maps multiple ports.
        """
        self.client = docker.from_env()
        self.container: Container | None = None
        self.formatter = SimpleFormatter()
        self.workdir = workdir
        self.mount_shared_volume = mount_shared_volume
        self.ports = ports
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
        except docker.errors.ImageNotFound:  # type: ignore[attr-defined]
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
        if self.ports:
            container_kwargs["ports"] = {f"{cp}/tcp": hp for cp, hp in self.ports.items()}
        self.container = self.client.containers.run(full_image_name, **container_kwargs)
        assert self.container is not None
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
        return stdout + "\n" + stderr

    def get_host_port(self, container_port: int) -> int | None:
        """Get the host port mapped to a container port.

        Args:
            container_port: The container port to look up.

        Returns:
            The host port mapped to the container port, or None if not mapped.
        """
        if self.container is None:
            raise KISSError("No container is open. Please call open() first.")

        self.container.reload()
        port_bindings = self.container.attrs.get("NetworkSettings", {}).get("Ports", {})
        port_key = f"{container_port}/tcp"
        if port_key in port_bindings and port_bindings[port_key]:
            return int(port_bindings[port_key][0]["HostPort"])
        return None

    def close(self) -> None:
        """Stop and remove the Docker container.

        Handles cleanup of both the container and any temporary directories
        created for shared volumes.
        """
        if self.container is None:
            print("No container to close.")
            return

        container_id = self.container.id[:12] if self.container.id else "unknown"
        try:
            self.formatter.print_status(f"Stopping container {container_id}")
            self.container.stop()
        except Exception as e:
            self.formatter.print_error(f"Failed to stop container {container_id}: {e}")

        try:
            self.formatter.print_status(f"Removing container {container_id}")
            self.container.remove()
        except Exception as e:
            self.formatter.print_error(f"Failed to remove container {container_id}: {e}")

        self.container = None

        # Clean up temporary directory
        if self.host_shared_path and os.path.exists(self.host_shared_path):
            try:
                shutil.rmtree(self.host_shared_path)
            except Exception as e:
                self.formatter.print_error(f"Failed to clean up temp directory: {e}")

        print("Container closed successfully")

    def __enter__(self) -> "DockerManager":
        """Context manager entry point.

        Returns:
            DockerManager: The initialized DockerManager instance with running container.
        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit point.

        Args:
            exc_type: The exception type if an exception was raised.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.
        """
        self.close()
