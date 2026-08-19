# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``kiss.core.printer.extract_extras`` must not truncate argument values."""

import unittest


class TestExtractExtrasNoTruncation(unittest.TestCase):
    """Verify extract_extras does not truncate long argument values."""

    def test_long_value_not_truncated(self):
        from kiss.core.printer import extract_extras
        long_val = "x" * 500
        result = extract_extras({"custom_arg": long_val})
        assert result == {"custom_arg": long_val}
        assert "..." not in result["custom_arg"]

    def test_known_keys_excluded(self):
        from kiss.core.printer import extract_extras
        result = extract_extras({
            "file_path": "/a/b.py", "command": "ls", "extra": "val",
        })
        assert result == {"extra": "val"}


if __name__ == "__main__":
    unittest.main()
