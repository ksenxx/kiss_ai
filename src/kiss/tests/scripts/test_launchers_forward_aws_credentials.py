# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``rsorcar`` and ``sorcar-docker`` launchers carry Amazon Bedrock credentials.

``rsorcar`` copies this machine's API keys to a remote only when a shell rc
file holds a key, and skips the copy when the remote already counts one
(``scripts/count-api-keys.sh`` runs the same patterns there).  The Bedrock
credentials follow AWS's naming (``AWS_BEARER_TOKEN_BEDROCK``,
``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY``) rather than the
``*_API_KEY`` / ``*_TOKEN`` convention, so both patterns list them by name.
``sorcar-docker`` forwards an explicit allowlist of variables into the
container; the five ``AWS_*`` settings are on it.

Each test executes the launcher's own bash block (cut out of the real file)
against temporary homes and environments.

Run with::

    uv run pytest src/kiss/tests/scripts/test_launchers_forward_aws_credentials.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Both launchers are POSIX shell scripts run from a Unix shell; Windows has
# no bash on PATH and never runs them.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX launchers")

REPO = Path(__file__).resolve().parents[4]
RSORCAR = REPO / "rsorcar"
SORCAR_DOCKER = REPO / "sorcar-docker"
COUNT_SCRIPT = REPO / "scripts" / "count-api-keys.sh"

_AWS_ENV = {
    "AWS_BEARER_TOKEN_BEDROCK": "test-bearer",
    "AWS_ACCESS_KEY_ID": "AKIATEST",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
    "AWS_SESSION_TOKEN": "test-session",
    "AWS_REGION": "eu-west-1",
}
_OLD_KEYS = (
    "GH_TOKEN",
    "KISS_WEB_PASSWORD",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_WORKSPACE_ID",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
)


def _rsorcar_key_block() -> str:
    """Return the key-pattern and ``find_rc_file`` block of ``rsorcar``."""
    source = RSORCAR.read_text(encoding="utf-8")
    start = source.index("KEY_RE='")
    end = source.index("# Distil this machine", start)
    return source[start:end]


def _rc_probe(tmp_path: Path, rc_text: str) -> tuple[str, int]:
    """Run ``find_rc_file`` and the remote key count on a home whose ``.bashrc`` is *rc_text*."""
    home = tmp_path / "home"
    (home / ".kiss").mkdir(parents=True)
    (home / ".bashrc").write_text(rc_text, encoding="utf-8")
    (home / ".kiss" / "api_keys.env").write_text(rc_text, encoding="utf-8")
    script = (
        _rsorcar_key_block()
        + '\nprintf "%s\\n" "$(find_rc_file)"\n'
        + 'export KEY_RE="$KEY_LINE_RE" FISH_RE="$FISH_KEY_LINE_RE"\n'
        + f'bash "{COUNT_SCRIPT}"\n'
    )
    env = {**os.environ, "HOME": str(home)}
    result = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True, check=True
    )
    rc_file, count = result.stdout.splitlines()  # an empty first line: no rc file found
    return rc_file, int(count)


@pytest.mark.parametrize(
    "rc_text",
    [
        "export AWS_BEARER_TOKEN_BEDROCK=test-bearer\nexport AWS_REGION=us-east-1\n",
        "export AWS_ACCESS_KEY_ID=AKIATEST\nexport AWS_SECRET_ACCESS_KEY='test-secret'\n",
        "export ANTHROPIC_API_KEY=sk-test\n",
    ],
)
def test_rsorcar_recognises_bedrock_credentials(tmp_path: Path, rc_text: str) -> None:
    """An rc file holding only Bedrock credentials is found and counted like one with an API key."""
    rc_file, count = _rc_probe(tmp_path, rc_text)
    assert rc_file == str(tmp_path / "home" / ".bashrc")
    assert count >= 2  # api_keys.env and .bashrc both hold the credential


@pytest.mark.parametrize(
    "rc_text",
    [
        "export AWS_REGION=us-east-1\n",  # a region alone is not a credential
        "export AWS_BEARER_TOKEN_BEDROCK=\nexport AWS_ACCESS_KEY_ID=''\n",  # emptied values
        "# export AWS_SECRET_ACCESS_KEY=commented-out\n",
    ],
)
def test_rsorcar_ignores_non_credentials(tmp_path: Path, rc_text: str) -> None:
    """A region, an emptied assignment or a comment is not a key."""
    rc_file, count = _rc_probe(tmp_path, rc_text)
    assert rc_file == ""
    assert count == 0


def test_rsorcar_fish_pattern_recognises_bedrock_credentials(tmp_path: Path) -> None:
    """fish's ``set -gx AWS_SECRET_ACCESS_KEY value`` counts on the remote too."""
    home = tmp_path / "home"
    (home / ".config" / "fish").mkdir(parents=True)
    (home / ".config" / "fish" / "config.fish").write_text(
        "set -gx AWS_ACCESS_KEY_ID AKIATEST\nset -gx AWS_SECRET_ACCESS_KEY test-secret\n"
        "set -gx AWS_REGION us-east-1\n",
        encoding="utf-8",
    )
    script = (
        _rsorcar_key_block()
        + 'export KEY_RE="$KEY_LINE_RE" FISH_RE="$FISH_KEY_LINE_RE"\n'
        + f'bash "{COUNT_SCRIPT}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "HOME": str(home)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "2"


def _docker_env_args(env: dict[str, str]) -> list[str]:
    """Run ``sorcar-docker``'s environment-collection block and return its ``docker run`` args."""
    source = SORCAR_DOCKER.read_text(encoding="utf-8")
    block = source[source.index("ENV_ARGS=()") : source.index("# 6. Start container")]
    result = subprocess.run(
        ["bash", "-c", block + '\nprintf "%s\\n" "${ENV_ARGS[@]}"'],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_sorcar_docker_forwards_aws_credentials() -> None:
    """Every AWS variable set in the launching shell reaches the container; unset ones do not."""
    base = {k: v for k, v in os.environ.items() if k not in _OLD_KEYS and k not in _AWS_ENV}
    args = _docker_env_args({**base, **_AWS_ENV})
    assert args == [
        item for key in _AWS_ENV for item in ("-e", f"{key}={_AWS_ENV[key]}")
    ]
    partial = {**base, "AWS_ACCESS_KEY_ID": "AKIATEST", "AWS_SECRET_ACCESS_KEY": "s"}
    assert _docker_env_args(partial) == [
        "-e", "AWS_ACCESS_KEY_ID=AKIATEST", "-e", "AWS_SECRET_ACCESS_KEY=s"
    ]
    assert _docker_env_args(base) == []
