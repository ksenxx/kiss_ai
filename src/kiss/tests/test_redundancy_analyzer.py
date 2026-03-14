"""Tests for redundancy_analyzer that verify branch coverage preservation.

Creates real coverage databases with dynamic contexts and verifies the
analyzer correctly identifies redundant tests at the method level.
"""

import os
import subprocess
import sys
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

    # Run pytest with coverage and dynamic contexts (only cover source, not tests)
    result = subprocess.run(
        [
            sys.executable,
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


def test_setup_teardown_arcs_grouped_with_method():
    """Setup/teardown arcs are merged into the method's arc set.

    This is the key fix: a method is only redundant if ALL its arcs
    (from run + setup + teardown) are covered by other methods.
    """
    source = """\
def add(a, b):
    return a + b
"""
    # Use a class with setup_method to create setup contexts
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
    # Both tests cover add(), the standalone might be redundant or
    # the class method. The key assertion: whatever is returned,
    # removing it preserves all arcs.
    _verify_coverage_preserved(cov_file, redundant)


def _verify_coverage_preserved(cov_file: str, redundant: list[str]):
    """Verify that removing redundant methods preserves all arcs."""
    import coverage

    cov = coverage.Coverage(data_file=cov_file)
    cov.load()
    data = cov.get_data()

    # Collect all arcs
    all_arcs: set[tuple[str, int, int]] = set()
    contexts = sorted(c for c in data.measured_contexts() if c)
    for ctx in contexts:
        data.set_query_context(ctx)
        for src_file in data.measured_files():
            file_arcs = data.arcs(src_file)
            if file_arcs:
                for f, t in file_arcs:
                    all_arcs.add((src_file, f, t))

    # Collect arcs from non-redundant methods only
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

