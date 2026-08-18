# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test that reproduces bug I6 listed in bugs.md.

Moved from ``kiss.tests.agents.third_party_agents.test_bugs_reproduction`` because this test
depends only on ``kiss.agents.sorcar`` (the Discord bug I2 test, which
depends on ``kiss.agents.third_party_agents``, remains in the original
module).  The test demonstrates the buggy behavior and should FAIL
until the bug is fixed.  No mocks, patches, fakes, or test doubles are
used.
"""

import inspect


class TestI6DocstringReferencesNonExistentParams:
    def test_open_docstring_does_not_reference_args(self) -> None:
        """open() takes no parameters, so its docstring shouldn't list Args."""
        from kiss.agents.sorcar.docker_manager import DockerManager

        doc = inspect.getdoc(DockerManager.open) or ""
        sig = inspect.signature(DockerManager.open)
        params = [p for p in sig.parameters if p != "self"]

        assert "image_name" not in doc or params, (
            "open() docstring references 'image_name' parameter but "
            f"open() takes no arguments (params={params})"
        )
