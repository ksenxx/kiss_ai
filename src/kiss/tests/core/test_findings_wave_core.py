# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test pinning the ``kiss.core.utils.finish`` result contract (no mocks).

``kiss.core.utils.finish`` must emit the success/is_continue/summary
contract that ``kiss.core.printer.parse_result_yaml`` recognizes.  The
sorcar-side halves of the findings wave (summarizer prompt/tool contract,
canonical finish reuse) live in
``kiss.tests.agents.sorcar.test_findings_wave_core``.
"""

import unittest

from kiss.core.printer import parse_result_yaml
from kiss.core.utils import finish as utils_finish


class UtilsFinishContract(unittest.TestCase):
    def test_utils_finish_recognized_by_parse_result_yaml(self) -> None:
        raw = utils_finish(True, False, "the final code")
        parsed = parse_result_yaml(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed,
            {"success": True, "is_continue": False, "summary": "<p>the final code</p>"},
        )


if __name__ == "__main__":
    unittest.main()
