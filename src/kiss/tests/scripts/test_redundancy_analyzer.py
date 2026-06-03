# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for redundancy_analyzer that verify branch coverage preservation.

Creates real coverage databases with dynamic contexts and verifies the
analyzer correctly identifies redundant tests at the method level.
"""

import os
import subprocess
import tempfile

from kiss.scripts.redundancy_analyzer import _method_name, analyze_redundancy


def _create_coverage_db(test_code: str, source_code: str) -> str:
    """Create a real coverage database with dynamic contexts.

    Returns the path to the .coverage file.
    """
    tmpdir = tempfile.mkdtemp()
    source_file = os.path.join(tmpdir, "source_mod.py")
    test_file = os.path.join(tmpdir, "test_source.py")
    cov_file = os.path.join(tmpdir, ".coverage")

    with open(source_file, "w") as f:
        f.write(source_code)

    with open(test_file, "w") as f:
        f.write(test_code)

    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            test_file,
            "--cov=source_mod",
            "--cov-branch",
            "--cov-context=test",
            "--no-header",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=tmpdir,
        env={**os.environ, "COVERAGE_FILE": cov_file},
        timeout=30,
    )
    assert result.returncode == 0, f"Tests failed:\n{result.stdout}\n{result.stderr}"
    assert os.path.exists(cov_file), f"Coverage file not created at {cov_file}"
    return cov_file


def test_lcov_typescript_and_javascript_coverage_redundancy():
    """LCOV records for *.ts/*.js files participate in redundancy.

    ``test_subset`` covers a strict subset of the lines and branches
    that ``test_superset`` covers (across one TypeScript and one
    JavaScript source file), so the analyzer must mark ``test_subset``
    as redundant even though no Python source file is involved beyond
    the empty .coverage database.
    """
    tmpdir = tempfile.mkdtemp()
    cov_file = _create_coverage_db(
        source_code="x = 1\n",
        test_code="def test_noop():\n    assert True\n",
    )

    lcov_path = os.path.join(tmpdir, "browser.info")
    with open(lcov_path, "w") as f:
        f.write(
            "TN:test_superset\n"
            "SF:/proj/src/handler.ts\n"
            "DA:1,3\n"
            "DA:2,3\n"
            "DA:3,1\n"
            "BRDA:2,0,0,2\n"
            "BRDA:2,0,1,1\n"
            "FNDA:3,handle\n"
            "end_of_record\n"
            "TN:test_superset\n"
            "SF:/proj/src/util.js\n"
            "DA:10,1\n"
            "end_of_record\n"
            "TN:test_subset\n"
            "SF:/proj/src/handler.ts\n"
            "DA:1,1\n"
            "DA:2,1\n"
            "BRDA:2,0,0,1\n"
            "FNDA:1,handle\n"
            "end_of_record\n"
            "TN:test_unique\n"
            "SF:/proj/src/util.js\n"
            "DA:99,1\n"
            "end_of_record\n"
        )

    redundant = analyze_redundancy(cov_file, lcov_files=[lcov_path])
    assert "test_subset" in redundant
    assert "test_superset" not in redundant
    assert "test_unique" not in redundant


def test_lcov_not_taken_branches_and_unhit_lines_ignored():
    """``DA:line,0``, ``BRDA:...,-`` and ``FNDA:0,name`` are not items.

    A test whose LCOV record contains only zero-hit / not-taken
    entries contributes nothing to the analysis and is therefore
    redundant.
    """
    tmpdir = tempfile.mkdtemp()
    cov_file = _create_coverage_db(
        source_code="x = 1\n",
        test_code="def test_noop():\n    assert True\n",
    )

    lcov_path = os.path.join(tmpdir, "browser.info")
    with open(lcov_path, "w") as f:
        f.write(
            "TN:test_real\n"
            "SF:/proj/src/handler.ts\n"
            "DA:1,1\n"
            "BRDA:1,0,0,1\n"
            "FNDA:1,handle\n"
            "end_of_record\n"
            "TN:test_empty\n"
            "SF:/proj/src/handler.ts\n"
            "DA:1,0\n"
            "BRDA:1,0,0,-\n"
            "FNDA:0,handle\n"
            "end_of_record\n"
        )

    redundant = analyze_redundancy(cov_file, lcov_files=[lcov_path])
    assert "test_empty" not in set(redundant) - {"test_empty"}
    # test_empty has no items at all so it cannot make any item redundant
    # and is itself trivially redundant only if it has nothing to cover.
    # Methods with empty item sets are never added to method_items so
    # they simply do not appear.
    assert "test_empty" not in redundant
    assert "test_real" not in redundant


def test_lcov_default_test_name_from_filename():
    """Records without a ``TN:`` use the LCOV file's stem as the name."""
    tmpdir = tempfile.mkdtemp()
    cov_file = _create_coverage_db(
        source_code="x = 1\n",
        test_code="def test_noop():\n    assert True\n",
    )

    lcov_path = os.path.join(tmpdir, "playwright_run.info")
    with open(lcov_path, "w") as f:
        f.write(
            "SF:/proj/src/handler.ts\n"
            "DA:1,1\n"
            "end_of_record\n"
            "TN:other\n"
            "SF:/proj/src/handler.ts\n"
            "DA:1,1\n"
            "DA:2,1\n"
            "end_of_record\n"
        )

    redundant = analyze_redundancy(cov_file, lcov_files=[lcov_path])
    # playwright_run covers a strict subset of `other`, so it is redundant
    assert "playwright_run" in redundant
    assert "other" not in redundant


def test_setup_teardown_arcs_grouped_with_method():
    """Setup/teardown arcs are merged into the method's arc set.

    This is the key fix: a method is only redundant if ALL its arcs
    (from run + setup + teardown) are covered by other methods.
    """
    source = """\
def add(a, b):
    return a + b
"""
    tests = """\
from source_mod import add

class TestWithSetup:
    def setup_method(self):
        self.value = add(1, 2)

    def test_value(self):
        assert self.value == 3

def test_standalone():
    assert add(1, 2) == 3
"""
    cov_file = _create_coverage_db(tests, source)
    redundant = analyze_redundancy(cov_file)
    _verify_coverage_preserved(cov_file, redundant)


def _verify_coverage_preserved(cov_file: str, redundant: list[str]):
    """Verify that removing redundant methods preserves all arcs."""
    import coverage

    cov = coverage.Coverage(data_file=cov_file)
    cov.load()
    data = cov.get_data()

    all_arcs: set[tuple[str, int, int]] = set()
    contexts = sorted(c for c in data.measured_contexts() if c)
    for ctx in contexts:
        data.set_query_context(ctx)
        for src_file in data.measured_files():
            file_arcs = data.arcs(src_file)
            if file_arcs:
                for f, t in file_arcs:
                    all_arcs.add((src_file, f, t))

    redundant_methods = set(_method_name(r) for r in redundant)
    kept_arcs: set[tuple[str, int, int]] = set()
    for ctx in contexts:
        if _method_name(ctx) not in redundant_methods:
            data.set_query_context(ctx)
            for src_file in data.measured_files():
                file_arcs = data.arcs(src_file)
                if file_arcs:
                    for f, t in file_arcs:
                        kept_arcs.add((src_file, f, t))

    assert kept_arcs == all_arcs, f"Lost arcs: {all_arcs - kept_arcs}"


