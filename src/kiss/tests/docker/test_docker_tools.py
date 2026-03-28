"""Tests for DockerTools — Read, Write, Edit inside Docker containers."""

import unittest

import docker

from kiss.docker.docker_manager import DockerManager
from kiss.docker.docker_tools import DockerTools


def is_docker_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerTools(unittest.TestCase):
    """Integration tests for DockerTools using a real Docker container."""

    env: DockerManager
    tools: DockerTools

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()
        cls.tools = DockerTools(cls.env.Bash)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    # ── Write tests ──────────────────────────────────────────────

    def test_write_and_read_basic(self) -> None:
        self.tools.Write("/tmp/test_basic.txt", "hello world\n")
        result = self.tools.Read("/tmp/test_basic.txt")
        self.assertIn("hello world", result)

    def test_write_creates_parent_dirs(self) -> None:
        result = self.tools.Write("/tmp/a/b/c/deep.txt", "deep content")
        self.assertIn("Successfully wrote", result)
        read = self.tools.Read("/tmp/a/b/c/deep.txt")
        self.assertIn("deep content", read)

    def test_write_special_characters(self) -> None:
        content = "quotes: 'single' \"double\"\nbackslash: \\\ndollar: $HOME\nstar: *\n"
        self.tools.Write("/tmp/test_special.txt", content)
        result = self.tools.Read("/tmp/test_special.txt")
        self.assertIn("'single'", result)
        self.assertIn('"double"', result)
        self.assertIn("\\", result)
        self.assertIn("$HOME", result)

    def test_write_unicode(self) -> None:
        content = "日本語テスト 🎉 émojis café\n"
        self.tools.Write("/tmp/test_unicode.txt", content)
        result = self.tools.Read("/tmp/test_unicode.txt")
        self.assertIn("日本語テスト", result)
        self.assertIn("🎉", result)
        self.assertIn("café", result)

    def test_write_empty_file(self) -> None:
        result = self.tools.Write("/tmp/test_empty.txt", "")
        self.assertIn("Successfully wrote 0 characters", result)

    def test_write_overwrite(self) -> None:
        self.tools.Write("/tmp/test_overwrite.txt", "original")
        self.tools.Write("/tmp/test_overwrite.txt", "replaced")
        result = self.tools.Read("/tmp/test_overwrite.txt")
        self.assertIn("replaced", result)
        self.assertNotIn("original", result)

    def test_write_multiline(self) -> None:
        content = "line1\nline2\nline3\n"
        self.tools.Write("/tmp/test_multiline.txt", content)
        result = self.tools.Read("/tmp/test_multiline.txt")
        self.assertIn("line1", result)
        self.assertIn("line2", result)
        self.assertIn("line3", result)

    # ── Read tests ───────────────────────────────────────────────

    def test_read_file_not_found(self) -> None:
        result = self.tools.Read("/tmp/nonexistent_file_xyz.txt")
        self.assertIn("Error", result)
        self.assertIn("File not found", result)

    def test_read_truncation(self) -> None:
        # Create a file with 10 lines, read only 3
        content = "\n".join(f"line{i}" for i in range(10)) + "\n"
        self.tools.Write("/tmp/test_truncate.txt", content)
        result = self.tools.Read("/tmp/test_truncate.txt", max_lines=3)
        self.assertIn("line0", result)
        self.assertIn("truncated", result)

    def test_read_no_truncation(self) -> None:
        content = "short\nfile\n"
        self.tools.Write("/tmp/test_short.txt", content)
        result = self.tools.Read("/tmp/test_short.txt", max_lines=100)
        self.assertIn("short", result)
        self.assertNotIn("truncated", result)

    # ── Edit tests ───────────────────────────────────────────────

    def test_edit_basic(self) -> None:
        self.tools.Write("/tmp/test_edit.txt", "hello world")
        result = self.tools.Edit("/tmp/test_edit.txt", "hello", "goodbye")
        self.assertIn("Successfully replaced 1 occurrence", result)
        content = self.tools.Read("/tmp/test_edit.txt")
        self.assertIn("goodbye world", content)

    def test_edit_file_not_found(self) -> None:
        result = self.tools.Edit("/tmp/nonexistent_edit.txt", "a", "b")
        self.assertIn("Error", result)
        self.assertIn("File not found", result)

    def test_edit_string_not_found(self) -> None:
        self.tools.Write("/tmp/test_edit_nf.txt", "hello world")
        result = self.tools.Edit("/tmp/test_edit_nf.txt", "xyz", "abc")
        self.assertIn("Error", result)
        self.assertIn("String not found", result)

    def test_edit_same_string(self) -> None:
        self.tools.Write("/tmp/test_edit_same.txt", "hello")
        result = self.tools.Edit("/tmp/test_edit_same.txt", "hello", "hello")
        self.assertIn("Error", result)
        self.assertIn("must be different", result)

    def test_edit_non_unique_without_replace_all(self) -> None:
        self.tools.Write("/tmp/test_edit_dup.txt", "aaa bbb aaa")
        result = self.tools.Edit("/tmp/test_edit_dup.txt", "aaa", "ccc")
        self.assertIn("Error", result)
        self.assertIn("2 times", result)

    def test_edit_replace_all(self) -> None:
        self.tools.Write("/tmp/test_edit_all.txt", "aaa bbb aaa")
        result = self.tools.Edit(
            "/tmp/test_edit_all.txt", "aaa", "ccc", replace_all=True
        )
        self.assertIn("Successfully replaced 2 occurrence", result)
        content = self.tools.Read("/tmp/test_edit_all.txt")
        self.assertIn("ccc bbb ccc", content)

    def test_edit_multiline(self) -> None:
        self.tools.Write("/tmp/test_edit_ml.txt", "line1\nline2\nline3\n")
        result = self.tools.Edit("/tmp/test_edit_ml.txt", "line1\nline2", "replaced")
        self.assertIn("Successfully replaced", result)
        content = self.tools.Read("/tmp/test_edit_ml.txt")
        self.assertIn("replaced", content)
        self.assertIn("line3", content)
        self.assertNotIn("line1", content)

    def test_edit_special_characters(self) -> None:
        original = "price is $100 (USD) & tax = 10%\n"
        self.tools.Write("/tmp/test_edit_spec.txt", original)
        result = self.tools.Edit(
            "/tmp/test_edit_spec.txt",
            "$100 (USD) & tax = 10%",
            "$200 (EUR) & tax = 20%",
        )
        self.assertIn("Successfully replaced", result)
        content = self.tools.Read("/tmp/test_edit_spec.txt")
        self.assertIn("$200 (EUR) & tax = 20%", content)

    def test_write_error(self) -> None:
        # /dev/null/subdir is guaranteed to fail (not a directory)
        result = self.tools.Write("/dev/null/impossible/test.txt", "fail")
        self.assertIn("exit code:", result)

    def test_edit_with_quotes(self) -> None:
        self.tools.Write("/tmp/test_edit_q.txt", "say 'hello' and \"bye\"")
        result = self.tools.Edit(
            "/tmp/test_edit_q.txt", "'hello'", "'world'"
        )
        self.assertIn("Successfully replaced", result)
        content = self.tools.Read("/tmp/test_edit_q.txt")
        self.assertIn("'world'", content)
        self.assertIn('"bye"', content)


if __name__ == "__main__":
    unittest.main()
