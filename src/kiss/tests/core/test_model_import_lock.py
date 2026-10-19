# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Two threads importing two provider model modules at once must both succeed.

The ``openai`` package's import graph is cyclic (``openai.resources.chat``
and ``openai.resources.responses`` reach each other through
``openai.types``).  Python's per-module import locks resolve such a cycle
only when one thread walks it: a second thread entering it from the other
module trips the import system's deadlock detector, and its import fails
with ``_DeadlockError``.  This is exactly what happened to the first task
of a freshly started daemon, whose thread built its ``OpenAICompatibleModel``
while the server's prewarm thread was importing ``openai_compatible_model2``.

The same cycle is reached from ``transcribe_audio`` (``client.audio``
imports ``openai.resources.audio``), which an Anthropic task thread calls
for an audio attachment.

Each scenario runs in a fresh interpreter (the modules must not be imported
yet) with a barrier releasing both threads together; without the lock in
``kiss.core.models.model`` the deadlock is reproduced on every run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

CHILD = textwrap.dedent(
    """
    import threading
    import kiss.core.models as models
    from kiss.core.models.model import transcribe_audio
    from kiss.server.server import _prewarm_task_dependencies

    def transcribe():
        try:
            transcribe_audio(b"RIFF", "audio/wav", api_key="dummy")
        except RuntimeError:
            pass  # the request to the unroutable OPENAI_BASE_URL failed, as meant

    SCENARIOS = {
        "two_classes": (
            lambda: models.OpenAICompatibleModel,
            lambda: models.OpenAICompatibleModel2,
        ),
        "prewarm_vs_task": (
            _prewarm_task_dependencies,
            lambda: models.OpenAICompatibleModel2,
        ),
        "adapter_vs_transcription": (
            lambda: models.OpenAICompatibleModel,
            transcribe,
        ),
    }
    import sys
    first, second = SCENARIOS[sys.argv[1]]
    barrier = threading.Barrier(2)
    errors = []

    def run(step):
        try:
            barrier.wait()
            step()
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run, args=(step,)) for step in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    print("|".join(errors) or "ok")
    """
)


@pytest.mark.parametrize(
    "scenario", ["two_classes", "prewarm_vs_task", "adapter_vs_transcription"],
)
def test_concurrent_provider_imports_do_not_deadlock(scenario: str) -> None:
    # No network: the transcription's request fails at once on a closed port.
    env = {**os.environ, "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, scenario],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok", proc.stdout + proc.stderr


if __name__ == "__main__":
    pytest.main([__file__])
