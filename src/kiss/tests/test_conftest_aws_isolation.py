# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The suite hides a developer's exported AWS key pair from the config singleton too.

The autouse ``no_ambient_aws_credentials`` fixture (``conftest.py``) unset
the pair in ``os.environ``, but ``DEFAULT_CONFIG`` had copied it when it
was built, before any test.  ``bedrock_credentials_configured()`` reads
that copy first, so on a machine with ``AWS_ACCESS_KEY_ID`` and
``AWS_SECRET_ACCESS_KEY`` exported, Bedrock stayed "configured" in every
test until some test of the same process reloaded the config from the
scrubbed environment: a test that expects no provider
(``test_same_family_dispatch_2026_10_31.py``'s refusal that has no other
family to name) failed when it ran alone and passed behind such a test.

No assertion here has a credential of the machine among its operands:
a failure must not print one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core.models.model_info import bedrock_credentials_configured
from kiss.tests.conftest import hide_ambient_aws_credentials

_PAIR = {"AWS_ACCESS_KEY_ID": "kiss-test-key-id", "AWS_SECRET_ACCESS_KEY": "kiss-test-secret"}
_BEARER_TOKEN = "AWS_BEARER_TOKEN_BEDROCK"


def test_an_exported_key_pair_is_hidden_from_the_config_singleton(tmp_path: Path) -> None:
    """The pair a developer exported, as the config copied it at import, configures Bedrock;
    under the fixture's hiding it does not, and it is back once the test is over."""
    keys = config_module.DEFAULT_CONFIG
    with pytest.MonkeyPatch.context() as developer:
        # A bearer token is a credential of its own, which the fixture leaves alone: without
        # one, the pair is all that configures Bedrock here.
        developer.delenv(_BEARER_TOKEN, raising=False)
        developer.setattr(keys, _BEARER_TOKEN, "")
        for name, value in _PAIR.items():
            developer.setenv(name, value)
            developer.setattr(keys, name, value)
        assert bedrock_credentials_configured()
        with pytest.MonkeyPatch.context() as test:
            hide_ambient_aws_credentials(test, tmp_path)
            assert not bedrock_credentials_configured()
        assert bedrock_credentials_configured()
        assert [name for name, value in _PAIR.items() if getattr(keys, name) != value] == []


def test_every_test_starts_without_the_pair_in_the_config_singleton() -> None:
    """The autouse fixture has run for this test: whatever the machine exports, the config
    singleton holds no key pair and no session token of it."""
    keys = config_module.DEFAULT_CONFIG
    fields = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
    assert [name for name in fields if getattr(keys, name)] == []
