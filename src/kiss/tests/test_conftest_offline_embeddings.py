# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Only a live OpenAI test sees the developer's ``OPENAI_API_KEY``.

An agent run with memory on embeds its task text through OpenAI
whenever that key is exported, so ``src/kiss/tests/conftest.py`` hides
it from every test that is neither marked ``live_api`` nor gated by
``requires_openai_api_key``.  The fixture used to cover the SEA tests'
folder only.

The ``test_probe_*`` tests only run inside the nested pytest process
:func:`test_only_a_live_openai_test_sees_the_key` starts with a dummy
key exported (selected by ``OFFLINE_EMBEDDINGS_PROBE``); each appends
what it sees to the file ``PROBE_OUT`` names.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.core import config, user_config
from kiss.core.memoryfield.index import default_embedder
from kiss.tests.conftest import requires_openai_api_key

_PROBE_ENV = "OFFLINE_EMBEDDINGS_PROBE"
_DUMMY_KEY = "sk-offline-embeddings-probe"

probe = pytest.mark.skipif(
    os.environ.get(_PROBE_ENV) != "1", reason="probe for the nested run below"
)


def _report(name: str) -> None:
    """Append what the probe *name* sees: the variable, the embedder, the configured key."""
    embedder = default_embedder()
    seen = (
        f"{name}: env={os.environ.get('OPENAI_API_KEY', '')!r}"
        f" embedder={getattr(embedder, '__name__', type(embedder).__name__)}"
        f" config={config.DEFAULT_CONFIG.OPENAI_API_KEY!r}\n"
    )
    with open(os.environ["PROBE_OUT"], "a", encoding="utf-8") as out:
        out.write(seen)


@probe
def test_probe_plain() -> None:
    """An ordinary test; it then re-reads the keys as a daemon does when it starts."""
    _report("plain")
    user_config.load_api_keys_readonly()
    _report("plain after a daemon start")


@probe
@pytest.mark.live_api
def test_probe_live() -> None:
    """A test marked ``live_api``, after the plain one blanked the configured key."""
    _report("live")


@probe
@requires_openai_api_key
def test_probe_gated() -> None:
    """A test gated by the conftest's own ``requires_openai_api_key``."""
    _report("gated")


def test_only_a_live_openai_test_sees_the_key(tmp_path: Path) -> None:
    """With the key exported, a plain test embeds offline and a live one keeps the key."""
    probe_out = tmp_path / "probes.txt"
    tmp = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "test_probe",
         __file__],
        env=dict(
            os.environ, TMPDIR=tmp, TMP=tmp, TEMP=tmp, OPENAI_API_KEY=_DUMMY_KEY,
            PROBE_OUT=str(probe_out), **{_PROBE_ENV: "1"},
        ),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert probe_out.read_text(encoding="utf-8").splitlines() == [
        f"plain: env='' embedder=hashed_embedding config={_DUMMY_KEY!r}",
        "plain after a daemon start: env='' embedder=hashed_embedding config=''",
        f"live: env={_DUMMY_KEY!r} embedder=ModelEmbedder config={_DUMMY_KEY!r}",
        f"gated: env={_DUMMY_KEY!r} embedder=ModelEmbedder config={_DUMMY_KEY!r}",
    ]
