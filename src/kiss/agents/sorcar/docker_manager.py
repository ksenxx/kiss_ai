# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Docker library for managing Docker containers and executing commands."""

import codecs
import logging
import os
import queue
import shlex
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import docker
from docker.models.containers import Container  # type: ignore[assignment]

from kiss.agents.sorcar.useful_tools import _truncate_output
from kiss.core.kiss_error import KISSError

logger = logging.getLogger(__name__)

#: Default cap on the characters a single command may return, matching
#: ``UsefulTools.Bash``.  Without it an unbounded ``pip install`` log goes
#: straight into the conversation and blows the model's context window.
MAX_OUTPUT_CHARS = 50000

#: Environment variable used to tag a streaming exec — and every process
#: it spawns — so a timed-out command can be killed by matching
#: ``/proc/<pid>/environ`` inside the container's own pid namespace.
_EXEC_TOKEN_VAR = "KISS_EXEC_TOKEN"


def _new_utf8_decoder() -> Any:
    """Return an incremental UTF-8 decoder that never raises.

    Docker delivers exec output as byte frames split at arbitrary
    boundaries, so a single multi-byte character (an accented letter, an
    emoji, a progress-bar glyph) is routinely delivered as two frames.
    Decoding each frame on its own would raise ``UnicodeDecodeError``
    nondeterministically, and strict decoding would also lose the entire
    output of a command that merely printed a stray binary byte.  An
    incremental decoder in ``replace`` mode carries the partial sequence
    over to the next frame and substitutes U+FFFD for genuinely invalid
    bytes — the same guarantee ``UsefulTools._spawn`` gives with
    ``errors="replace"``.

    Returns:
        A fresh ``codecs`` incremental decoder; feed it with
        ``decoder.decode(chunk)`` and flush with ``decoder.decode(b"", True)``.
    """
    return codecs.getincrementaldecoder("utf-8")("replace")


def _drain_exec_stream(
    output_gen: Iterator[Any],
    out_queue: "queue.Queue[tuple[bool, str] | None]",
) -> None:
    """Decode a docker exec stream onto *out_queue* until it ends.

    Runs on a reader thread so the caller can enforce a timeout on a
    generator that otherwise blocks forever.  stdout and stderr each get
    their own incremental decoder because their frames interleave.

    Args:
        output_gen: The demuxed generator from ``exec_start``.
        out_queue: Receives ``(is_stderr, text)`` items and a final
            ``None`` sentinel marking end of stream.
    """
    decoders = {False: _new_utf8_decoder(), True: _new_utf8_decoder()}
    try:
        for chunk in output_gen:
            if isinstance(chunk, tuple):  # pragma: no branch
                stdout_chunk, stderr_chunk = chunk
            else:
                stdout_chunk, stderr_chunk = chunk, None
            for is_stderr, raw in ((False, stdout_chunk), (True, stderr_chunk)):
                if not raw:
                    continue
                text = decoders[is_stderr].decode(raw)
                if text:
                    out_queue.put((is_stderr, text))
    except Exception:  # pragma: no cover — docker socket error mid-stream
        logger.debug("docker exec stream failed", exc_info=True)
    finally:
        for is_stderr, decoder in decoders.items():
            trailing = decoder.decode(b"", True)
            if trailing:
                out_queue.put((is_stderr, trailing))
        out_queue.put(None)


def _with_exit_code(output: str, exit_code: int) -> str:
    """Append the ``[exit code: N]`` marker for a failed command.

    Args:
        output: The command's combined output.
        exit_code: The command's exit status.

    Returns:
        *output* unchanged on success, else *output* plus the marker.
    """
    if exit_code == 0:
        return output
    suffix = f"[exit code: {exit_code}]"
    return f"{output}\n{suffix}" if output else suffix

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

        self.workdir = workdir
        self.mount_shared_volume = mount_shared_volume
        self.ports = ports
        self.client_shared_path = "/testbed"
        self.host_shared_path: str | None = None
        self.stream_callback: Callable[[str], None] | None = None

        if ":" in image_name:  # pragma: no branch
            self.image, self.tag = image_name.rsplit(":", 1)
        else:
            self.image = image_name
            self.tag = tag

    def open(self) -> None:
        """Pull and load a Docker image, then create and start a container."""
        image = self.image
        tag = self.tag
        full_image_name = f"{image}:{tag}"
        print(f"Pulling Docker image: {full_image_name}")
        try:
            self.client.images.get(full_image_name)
        except docker.errors.ImageNotFound:  # type: ignore[attr-defined]
            logger.debug("Exception caught", exc_info=True)
            self.client.images.pull(image, tag=tag)
        print(f"Creating and starting container from {full_image_name}")
        container_kwargs: dict[str, Any] = {
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "command": "/bin/bash",
        }
        if self.mount_shared_volume:  # pragma: no branch
            self.host_shared_path = tempfile.mkdtemp()
        if self.mount_shared_volume and self.host_shared_path:  # pragma: no branch
            container_kwargs["volumes"] = {
                self.host_shared_path: {"bind": self.client_shared_path, "mode": "rw"}
            }
        if self.ports:
            container_kwargs["ports"] = {f"{cp}/tcp": hp for cp, hp in self.ports.items()}
        self.container = self.client.containers.run(full_image_name, **container_kwargs)
        assert self.container is not None
        container_id = self.container.id[:12] if self.container.id else "unknown"
        print(f"Container {container_id} is now running")

    def Bash(  # noqa: N802
        self,
        command: str,
        description: str,
        timeout_seconds: int = 30,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> str:  # noqa: N802
        """
        Execute a bash command in the running Docker container.

        Args:
            command: The bash command to execute
            description: A short description of the command in natural language
            timeout_seconds: Maximum time to wait before treating the command as hung.
            max_output_chars: Maximum characters in output before truncation.

        Returns:
            The output of the command, including stdout, stderr, and exit code
        """
        if self.container is None:  # pragma: no branch
            raise KISSError("No container is open. Please call open() first.")

        print(f"{description}")

        if self.stream_callback:
            return self._bash_streaming(command, timeout_seconds, max_output_chars)

        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        container = self.container
        assert container is not None

        def run_exec() -> None:
            try:
                result_holder["result"] = container.exec_run(
                    f"/bin/bash -c {shlex.quote(command)}",
                    stdout=True,
                    stderr=True,
                    demux=True,
                    workdir=self.workdir,
                )
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=run_exec, daemon=True)
        thread.start()
        thread.join(timeout_seconds)
        if thread.is_alive():  # pragma: no branch
            return f"Error: command timed out after {timeout_seconds}s"
        if error_holder:  # pragma: no branch
            raise error_holder["error"]

        exec_result = result_holder["result"]
        output_payload = exec_result.output
        if output_payload:  # pragma: no branch
            stdout_bytes, stderr_bytes = output_payload
        else:
            stdout_bytes, stderr_bytes = None, None
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        output_parts = [part for part in (stdout, stderr) if part]
        output = "\n".join(output_parts)
        return _truncate_output(
            _with_exit_code(output, exec_result.exit_code), max_output_chars,
        )

    def _bash_streaming(
        self, command: str, timeout_seconds: float, max_output_chars: int,
    ) -> str:
        """Run *command*, streaming its output, and return the full result.

        The docker exec stream is drained on a reader thread so this
        thread can enforce *timeout_seconds*; the callback is invoked
        here (not on the reader) because printers attribute output to a
        task via thread-local state.

        Args:
            command: The bash command to execute.
            timeout_seconds: Maximum time to wait before treating the
                command as hung; the container-side process is killed.
            max_output_chars: Maximum characters in output before truncation.

        Returns:
            The command's output, or the timeout error.
        """
        assert self.container is not None
        assert self.stream_callback is not None
        token = uuid.uuid4().hex
        exec_resp = self.client.api.exec_create(
            self.container.id,
            f"/bin/bash -c {shlex.quote(command)}",
            stdout=True,
            stderr=True,
            workdir=self.workdir,
            environment={_EXEC_TOKEN_VAR: token},
        )
        exec_id = exec_resp["Id"]
        output_gen = self.client.api.exec_start(exec_id, stream=True, demux=True)
        out_queue: queue.Queue[tuple[bool, str] | None] = queue.Queue()
        threading.Thread(
            target=_drain_exec_stream, args=(output_gen, out_queue), daemon=True,
        ).start()

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        deadline = time.monotonic() + timeout_seconds
        eof = False
        while True:
            try:
                item = out_queue.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                break
            if item is None:
                eof = True
                break
            is_stderr, text = item
            (stderr_parts if is_stderr else stdout_parts).append(text)
            self.stream_callback(text)

        if not eof:
            self._kill_exec(token)
            return f"Error: command timed out after {timeout_seconds}s"

        exit_code = self.client.api.exec_inspect(exec_id).get("ExitCode", 0)
        output = "\n".join(
            part for part in ("".join(stdout_parts), "".join(stderr_parts)) if part
        )
        return _truncate_output(_with_exit_code(output, exit_code), max_output_chars)

    def _kill_exec(self, token: str) -> None:
        """Kill the container-side processes of a timed-out exec.

        Without this the hung command keeps running (and holding the
        stream open) for the rest of the container's life.  The exec is
        tagged with a unique environment variable, which every child
        inherits, so matching on ``/proc/<pid>/environ`` kills the whole
        tree.  ``exec_inspect``'s ``Pid`` is deliberately not used: it is
        a *host*-namespace pid and means nothing inside the container.

        Args:
            token: The unique tag given to the exec's environment.
        """
        assert self.container is not None
        script = (
            "for d in /proc/[0-9]*; do\n"
            '  env=$(tr "\\0" "\\n" < "$d/environ" 2>/dev/null)\n'
            f'  case "$env" in *"{_EXEC_TOKEN_VAR}={token}"*)\n'
            '    kill -9 "${d#/proc/}" 2>/dev/null;;\n'
            "  esac\n"
            "done"
        )
        try:
            self.container.exec_run(["/bin/sh", "-c", script])
        except Exception:  # pragma: no cover — container already gone
            logger.debug("could not kill timed-out exec", exc_info=True)

    def get_host_port(self, container_port: int) -> int | None:
        """Get the host port mapped to a container port.

        Args:
            container_port: The container port to look up.

        Returns:
            The host port mapped to the container port, or None if not mapped.
        """
        if self.container is None:  # pragma: no branch
            raise KISSError("No container is open. Please call open() first.")

        self.container.reload()
        port_bindings = self.container.attrs.get("NetworkSettings", {}).get("Ports", {})
        port_key = f"{container_port}/tcp"
        if port_key in port_bindings and port_bindings[port_key]:  # pragma: no branch
            return int(port_bindings[port_key][0]["HostPort"])
        return None

    def close(self) -> None:
        """Stop and remove the Docker container.

        Handles cleanup of both the container and any temporary directories
        created for shared volumes.
        """
        if self.container is None:  # pragma: no branch
            print("No container to close.")
            return

        container_id = self.container.id[:12] if self.container.id else "unknown"
        try:
            print(f"Stopping container {container_id}")
            self.container.stop()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            print(f"Failed to stop container {container_id}: {e}")

        try:
            print(f"Removing container {container_id}")
            self.container.remove()
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            print(f"Failed to remove container {container_id}: {e}")

        self.container = None

        if self.host_shared_path and os.path.exists(self.host_shared_path):  # pragma: no branch
            try:
                shutil.rmtree(self.host_shared_path)
            except Exception as e:
                logger.debug("Exception caught", exc_info=True)
                print(f"Failed to clean up temp directory: {e}")

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
