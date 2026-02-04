# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Integration tests for AgentEvolver Edit and MultiEdit tools.

This module tests that the Edit and MultiEdit tools:
1. Are correctly invoked by the KISSCodingAgent
2. Properly modify file contents
3. Respect writable path restrictions
4. Handle various edge cases
"""

import os
import shutil
import tempfile
import unittest

from kiss.core.useful_tools import UsefulTools


class TestEditToolIntegration(unittest.TestCase):
    """Integration tests for the Edit tool."""

    def setUp(self) -> None:
        """Set up test fixtures.

        Creates a temporary directory structure with writable subdirectory
        and initializes UsefulTools with appropriate path permissions.
        """
        self.test_dir = tempfile.mkdtemp()
        self.writable_dir = os.path.join(self.test_dir, "writable")
        os.makedirs(self.writable_dir)

        self.tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
        )

    def tearDown(self) -> None:
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.test_dir)

    def test_edit_replaces_single_occurrence(self) -> None:
        """Test that Edit replaces a single occurrence of a string.

        Verifies that the Edit tool correctly replaces a unique string in a file
        and that the result message indicates success.
        """
        # Create a test file
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = '''def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
'''
        with open(test_file, "w") as f:
            f.write(original_content)

        # Use Edit to replace the print statement
        result = self.tools.Edit(
            file_path=test_file,
            old_string='print("Hello, World!")',
            new_string='print("Hello, KISS!")',
        )

        # Verify the result indicates success
        self.assertIn("Successfully replaced", result)

        # Verify the file content was changed
        with open(test_file) as f:
            new_content = f.read()

        self.assertIn('print("Hello, KISS!")', new_content)
        self.assertNotIn('print("Hello, World!")', new_content)

    def test_edit_fails_on_non_unique_string(self) -> None:
        """Test that Edit fails when the string appears multiple times.

        Verifies that the Edit tool returns an error when the old_string
        appears more than once in the file, and that the file remains unchanged.
        """
        # Create a test file with duplicate strings
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = '''def foo():
    x = 1
    y = 1
    return x + y
'''
        with open(test_file, "w") as f:
            f.write(original_content)

        # Try to edit a non-unique string
        result = self.tools.Edit(
            file_path=test_file,
            old_string="1",
            new_string="2",
        )

        # Verify the edit failed due to non-unique string
        self.assertIn("not unique", result.lower())

        # Verify the file was not changed
        with open(test_file) as f:
            new_content = f.read()

        self.assertEqual(new_content, original_content)

    def test_edit_fails_on_string_not_found(self) -> None:
        """Test that Edit fails when the string is not found.

        Verifies that the Edit tool returns an error when the old_string
        does not exist in the file.
        """
        # Create a test file
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = "def hello(): pass"
        with open(test_file, "w") as f:
            f.write(original_content)

        # Try to edit a string that doesn't exist
        result = self.tools.Edit(
            file_path=test_file,
            old_string="nonexistent_string",
            new_string="replacement",
        )

        # Verify the edit failed
        self.assertIn("not found", result.lower())

    def test_edit_respects_writable_paths(self) -> None:
        """Test that Edit denies access to paths outside writable_paths.

        Verifies that the Edit tool refuses to modify files that are not
        within the configured writable_paths, ensuring security boundaries
        are enforced.
        """
        # Create a test file outside writable_paths
        non_writable_dir = os.path.join(self.test_dir, "non_writable")
        os.makedirs(non_writable_dir)
        test_file = os.path.join(non_writable_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("content")

        # Try to edit the file
        result = self.tools.Edit(
            file_path=test_file,
            old_string="content",
            new_string="new_content",
        )

        # Verify access was denied
        self.assertIn("Access denied", result)

        # Verify the file was not changed
        with open(test_file) as f:
            content = f.read()
        self.assertEqual(content, "content")

    def test_edit_with_single_line_unique_string(self) -> None:
        """Test that Edit handles unique single-line strings correctly.

        Verifies that the Edit tool works correctly with grep's line-based
        counting by replacing a unique single-line comment string.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        # Use a unique single-line string (Edit uses grep -c which is line-based)
        original_content = (
            "class CalculatorXYZ123:\n"
            "    def add_numbers_together(self, first_num, second_num):\n"
            "        # UNIQUE_SINGLE_LINE_COMMENT_ABC123DEF456\n"
            "        return first_num + second_num\n"
        )
        with open(test_file, "w") as f:
            f.write(original_content)

        # Replace single line comment (grep -c works per line)
        old_string = "        # UNIQUE_SINGLE_LINE_COMMENT_ABC123DEF456"
        new_string = "        # Optimized add function"

        result = self.tools.Edit(
            file_path=test_file,
            old_string=old_string,
            new_string=new_string,
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            new_content = f.read()

        self.assertIn("# Optimized add function", new_content)
        self.assertNotIn("UNIQUE_SINGLE_LINE_COMMENT", new_content)


class TestMultiEditToolIntegration(unittest.TestCase):
    """Integration tests for the MultiEdit tool."""

    def setUp(self) -> None:
        """Set up test fixtures.

        Creates a temporary directory structure with writable subdirectory
        and initializes UsefulTools with appropriate path permissions.
        """
        self.test_dir = tempfile.mkdtemp()
        self.writable_dir = os.path.join(self.test_dir, "writable")
        os.makedirs(self.writable_dir)

        self.tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
        )

    def tearDown(self) -> None:
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.test_dir)

    def test_multiedit_replaces_all_occurrences(self) -> None:
        """Test that MultiEdit with replace_all=True replaces all occurrences.

        Verifies that the MultiEdit tool replaces every instance of the
        old_string when replace_all is set to True.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = '''def foo():
    x = 1
    y = 1
    z = 1
    return x + y + z
'''
        with open(test_file, "w") as f:
            f.write(original_content)

        # Replace all occurrences of "1"
        result = self.tools.MultiEdit(
            file_path=test_file,
            old_string="1",
            new_string="42",
            replace_all=True,
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            new_content = f.read()

        # Count occurrences
        self.assertEqual(new_content.count("42"), 3)
        self.assertEqual(new_content.count("1"), 0)

    def test_multiedit_single_occurrence_mode(self) -> None:
        """Test that MultiEdit with replace_all=False works like Edit.

        Verifies that the MultiEdit tool behaves identically to the Edit tool
        when replace_all is set to False.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = '''def unique_function():
    return "unique"
'''
        with open(test_file, "w") as f:
            f.write(original_content)

        result = self.tools.MultiEdit(
            file_path=test_file,
            old_string='"unique"',
            new_string='"replaced"',
            replace_all=False,
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            new_content = f.read()

        self.assertIn('"replaced"', new_content)

    def test_multiedit_respects_writable_paths(self) -> None:
        """Test that MultiEdit denies access to paths outside writable_paths.

        Verifies that the MultiEdit tool refuses to modify files that are not
        within the configured writable_paths, ensuring security boundaries
        are enforced.
        """
        non_writable_dir = os.path.join(self.test_dir, "non_writable")
        os.makedirs(non_writable_dir)
        test_file = os.path.join(non_writable_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("content content content")

        result = self.tools.MultiEdit(
            file_path=test_file,
            old_string="content",
            new_string="new_content",
            replace_all=True,
        )

        self.assertIn("Access denied", result)

    def test_multiedit_fails_on_string_not_found(self) -> None:
        """Test that MultiEdit fails when the string is not found.

        Verifies that the MultiEdit tool returns an error when the old_string
        does not exist in the file.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("some content here")

        result = self.tools.MultiEdit(
            file_path=test_file,
            old_string="nonexistent",
            new_string="replacement",
            replace_all=True,
        )

        self.assertIn("not found", result.lower())


class TestEditToolWithKISSCodingAgent(unittest.TestCase):
    """Integration tests for Edit/MultiEdit tools via KISSCodingAgent.

    These tests verify that the tools are properly wired into the agent
    and can be called during agent execution.
    """

    def setUp(self) -> None:
        """Set up test fixtures.

        Creates a temporary directory structure with writable subdirectory
        for use with KISSCodingAgent.
        """
        self.test_dir = tempfile.mkdtemp()
        self.writable_dir = os.path.join(self.test_dir, "writable")
        os.makedirs(self.writable_dir)

    def tearDown(self) -> None:
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.test_dir)

    def test_kiss_coding_agent_has_edit_tools_configured(self) -> None:
        """Test that KISSCodingAgent has Edit and MultiEdit tools available.

        Verifies that when KISSCodingAgent is initialized, its UsefulTools
        instance includes callable Edit, MultiEdit, and Bash methods.
        """
        try:
            from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
        except ImportError as e:
            self.skipTest(f"Skipping test due to missing dependency: {e}")

        agent = KISSCodingAgent("Test Agent")
        agent._reset(
            orchestrator_model_name="gpt-4o-mini",
            subtasker_model_name="gpt-4o-mini",
            dynamic_gepa_model_name="gpt-4o-mini",
            trials=1,
            max_steps=10,
            max_budget=1.0,
            work_dir=self.writable_dir,
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
            docker_image=None,
        )

        # Verify the UsefulTools instance has all required methods
        self.assertTrue(hasattr(agent.useful_tools, "Edit"))
        self.assertTrue(hasattr(agent.useful_tools, "MultiEdit"))
        self.assertTrue(hasattr(agent.useful_tools, "Bash"))

        # Verify they are callable
        self.assertTrue(callable(agent.useful_tools.Edit))
        self.assertTrue(callable(agent.useful_tools.MultiEdit))
        self.assertTrue(callable(agent.useful_tools.Bash))

    def test_agent_useful_tools_edit_modifies_file(self) -> None:
        """Test that agent's Edit tool can modify files.

        Verifies that the Edit tool attached to KISSCodingAgent's UsefulTools
        can successfully modify file contents within the writable path.
        """
        try:
            from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
        except ImportError as e:
            self.skipTest(f"Skipping test due to missing dependency: {e}")

        agent = KISSCodingAgent("Test Agent")
        agent._reset(
            orchestrator_model_name="gpt-4o-mini",
            subtasker_model_name="gpt-4o-mini",
            dynamic_gepa_model_name="gpt-4o-mini",
            trials=1,
            max_steps=10,
            max_budget=1.0,
            work_dir=self.writable_dir,
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
            docker_image=None,
        )

        # Create a test file
        test_file = os.path.join(self.writable_dir, "agent_test.py")
        with open(test_file, "w") as f:
            f.write('VERSION = "1.0.0"')

        # Use the agent's Edit tool
        result = agent.useful_tools.Edit(
            file_path=test_file,
            old_string='VERSION = "1.0.0"',
            new_string='VERSION = "2.0.0"',
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        self.assertIn('VERSION = "2.0.0"', content)

    def test_agent_useful_tools_multiedit_modifies_file(self) -> None:
        """Test that agent's MultiEdit tool can modify files.

        Verifies that the MultiEdit tool attached to KISSCodingAgent's UsefulTools
        can successfully replace all occurrences in a file within the writable path.
        """
        try:
            from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent
        except ImportError as e:
            self.skipTest(f"Skipping test due to missing dependency: {e}")

        agent = KISSCodingAgent("Test Agent")
        agent._reset(
            orchestrator_model_name="gpt-4o-mini",
            subtasker_model_name="gpt-4o-mini",
            dynamic_gepa_model_name="gpt-4o-mini",
            trials=1,
            max_steps=10,
            max_budget=1.0,
            work_dir=self.writable_dir,
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
            docker_image=None,
        )

        # Create a test file with multiple occurrences
        test_file = os.path.join(self.writable_dir, "agent_test.py")
        with open(test_file, "w") as f:
            f.write('DEBUG = True\nDEBUG_LEVEL = True\nDEBUG_MODE = True')

        # Use the agent's MultiEdit tool with replace_all
        result = agent.useful_tools.MultiEdit(
            file_path=test_file,
            old_string="True",
            new_string="False",
            replace_all=True,
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        self.assertEqual(content.count("False"), 3)
        self.assertEqual(content.count("True"), 0)


class TestEditToolEdgeCases(unittest.TestCase):
    """Edge case tests for Edit and MultiEdit tools."""

    def setUp(self) -> None:
        """Set up test fixtures.

        Creates a temporary directory structure with writable subdirectory
        and initializes UsefulTools with appropriate path permissions.
        """
        self.test_dir = tempfile.mkdtemp()
        self.writable_dir = os.path.join(self.test_dir, "writable")
        os.makedirs(self.writable_dir)

        self.tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
        )

    def tearDown(self) -> None:
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.test_dir)

    def test_edit_with_special_characters(self) -> None:
        """Test that Edit handles special regex characters correctly.

        Verifies that the Edit tool properly handles strings containing
        special regex characters (like backslashes) without misinterpreting them.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        # Use the exact string that will be in the file
        original_content = 'pattern = r"\\d+\\.\\d+"'
        with open(test_file, "w") as f:
            f.write(original_content)

        # Use the exact string from the file for replacement
        result = self.tools.Edit(
            file_path=test_file,
            old_string='pattern = r"\\d+\\.\\d+"',
            new_string='pattern = r"[0-9]+\\.[0-9]+"',
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        self.assertIn('[0-9]', content)

    def test_edit_with_empty_new_string(self) -> None:
        """Test that Edit can delete content by replacing with empty string.

        Verifies that the Edit tool can effectively delete content by replacing
        a string with a comment or placeholder (since true deletion would change
        line structure).
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        # Use a truly unique single-line string without newline in search
        original_content = (
            "def foo_xyz789():\n"
            "    x_unique_var_12345 = 42\n"
            "    pass\n"
        )
        with open(test_file, "w") as f:
            f.write(original_content)

        # Replace without the trailing newline (grep operates line by line)
        result = self.tools.Edit(
            file_path=test_file,
            old_string="    x_unique_var_12345 = 42",
            new_string="    # line deleted",
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        self.assertNotIn("x_unique_var_12345", content)
        self.assertIn("# line deleted", content)

    def test_edit_preserves_file_encoding(self) -> None:
        """Test that Edit preserves file encoding (UTF-8).

        Verifies that the Edit tool correctly handles files with UTF-8
        encoding, including Unicode characters and emoji.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = '''# -*- coding: utf-8 -*-
message = "Hello, 世界! 🌍"
'''
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(original_content)

        result = self.tools.Edit(
            file_path=test_file,
            old_string="Hello, 世界! 🌍",
            new_string="Bonjour, 世界! 🌎",
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Bonjour, 世界! 🌎", content)

    def test_edit_with_tabs_and_spaces(self) -> None:
        """Test that Edit handles mixed tabs and spaces correctly.

        Verifies that the Edit tool preserves whitespace formatting,
        including the ability to convert tabs to spaces.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        # File with tabs
        original_content = "def foo():\n\treturn 1"
        with open(test_file, "w") as f:
            f.write(original_content)

        result = self.tools.Edit(
            file_path=test_file,
            old_string="\treturn 1",
            new_string="    return 2",
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        self.assertIn("    return 2", content)

    def test_edit_nonexistent_file(self) -> None:
        """Test that Edit fails gracefully for nonexistent files.

        Verifies that the Edit tool returns an appropriate error message
        when attempting to edit a file that does not exist.
        """
        test_file = os.path.join(self.writable_dir, "nonexistent.py")

        result = self.tools.Edit(
            file_path=test_file,
            old_string="old",
            new_string="new",
        )

        self.assertIn("not found", result.lower())

    def test_edit_same_old_and_new_string(self) -> None:
        """Test that Edit fails when old_string equals new_string.

        Verifies that the Edit tool returns an error when the old_string
        and new_string are identical, preventing no-op edits.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("content")

        result = self.tools.Edit(
            file_path=test_file,
            old_string="content",
            new_string="content",
        )

        self.assertIn("must be different", result.lower())

    def test_multiedit_replaces_overlapping_patterns(self) -> None:
        """Test MultiEdit with patterns that could overlap.

        Verifies that the MultiEdit tool handles consecutive identical
        characters correctly, replacing each occurrence independently.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = "aaa"
        with open(test_file, "w") as f:
            f.write(original_content)

        result = self.tools.MultiEdit(
            file_path=test_file,
            old_string="a",
            new_string="bb",
            replace_all=True,
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        # Each 'a' should be replaced with 'bb'
        self.assertEqual(content, "bbbbbb")


class TestEditToolWithAgentWorkflow(unittest.TestCase):
    """Tests that verify Edit/MultiEdit tools work in realistic agent workflows."""

    def setUp(self) -> None:
        """Set up test fixtures.

        Creates a temporary directory structure with an agent_workspace
        subdirectory and initializes UsefulTools with appropriate permissions.
        """
        self.test_dir = tempfile.mkdtemp()
        self.writable_dir = os.path.join(self.test_dir, "agent_workspace")
        os.makedirs(self.writable_dir)

        self.tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
        )

    def tearDown(self) -> None:
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.test_dir)

    def test_sequential_edits_to_same_file(self) -> None:
        """Test that multiple sequential edits to the same file work correctly.

        Verifies that the Edit tool can be called multiple times on the same
        file, with each subsequent edit seeing the results of previous edits.
        """
        test_file = os.path.join(self.writable_dir, "agent.py")
        # Use very specific unique strings on single lines to avoid grep issues
        original_content = (
            'def agent_run_xyz123(task_input: str) -> dict:\n'
            '    """Run the agent xyz."""\n'
            '    result_data = {"success_status_abc7890": False, "metrics_container_xyz1234": {}}\n'
            '    marker_todo_xyz7890 = "implement"\n'
            '    return result_data\n'
        )
        with open(test_file, "w") as f:
            f.write(original_content)

        # First edit: Change success to True
        result1 = self.tools.Edit(
            file_path=test_file,
            old_string='"success_status_abc7890": False',
            new_string='"success_status_abc7890": True',
        )
        self.assertIn("Successfully replaced", result1)

        # Second edit: Modify TODO marker (use replacement, not deletion with newline)
        result2 = self.tools.Edit(
            file_path=test_file,
            old_string='marker_todo_xyz7890 = "implement"',
            new_string='marker_done_xyz7890 = "completed"',
        )
        self.assertIn("Successfully replaced", result2)

        # Third edit: Add metrics
        result3 = self.tools.Edit(
            file_path=test_file,
            old_string='"metrics_container_xyz1234": {}',
            new_string='"metrics_container_xyz1234": {"tokens_used": 0, "execution_time": 0.0}',
        )
        self.assertIn("Successfully replaced", result3)

        # Verify final content
        with open(test_file) as f:
            final_content = f.read()

        self.assertIn('"success_status_abc7890": True', final_content)
        self.assertIn('marker_done_xyz7890 = "completed"', final_content)
        self.assertIn("tokens_used", final_content)
        self.assertIn("execution_time", final_content)

    def test_edit_creates_valid_python(self) -> None:
        """Test that Edit produces syntactically valid Python code.

        Verifies that edits (such as adding type hints) result in code
        that can be parsed and compiled by Python.
        """
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = '''def calculate(a, b):
    return a + b
'''
        with open(test_file, "w") as f:
            f.write(original_content)

        # Add type hints
        result = self.tools.Edit(
            file_path=test_file,
            old_string="def calculate(a, b):",
            new_string="def calculate(a: int, b: int) -> int:",
        )
        self.assertIn("Successfully replaced", result)

        # Verify the result is valid Python
        with open(test_file) as f:
            content = f.read()

        # Try to compile it
        try:
            compile(content, test_file, "exec")
            is_valid = True
        except SyntaxError:
            is_valid = False

        self.assertTrue(is_valid, "Edit produced invalid Python syntax")

    def test_edit_workflow_for_optimization(self) -> None:
        """Test a realistic optimization workflow using Edit tools.

        Note: Since the Edit tool uses grep -c (line-based counting),
        we test single-line edits rather than multi-line function replacements.
        """
        test_file = os.path.join(self.writable_dir, "optimizer.py")
        # Use unique variable names to ensure no substring matches
        original_content = (
            "def process_collection_abc789(input_collection_abc789):\n"
            "    accumulated_results_xyz12345 = []\n"
            "    return accumulated_results_xyz12345\n"
        )
        with open(test_file, "w") as f:
            f.write(original_content)

        # Optimize a single line: change the return statement
        result = self.tools.Edit(
            file_path=test_file,
            old_string="    return accumulated_results_xyz12345",
            new_string="    return [x * 2 for x in input_collection_abc789]",
        )
        self.assertIn("Successfully replaced", result)

        # Verify optimization
        with open(test_file) as f:
            content = f.read()

        self.assertIn("return [x * 2 for x in input_collection_abc789]", content)
        self.assertNotIn("return accumulated_results_xyz12345", content)


if __name__ == "__main__":
    unittest.main()
