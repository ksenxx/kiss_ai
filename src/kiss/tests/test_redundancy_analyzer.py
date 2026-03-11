"""Tests for redundancy_analyzer that verify branch coverage preservation.

Creates real coverage databases with dynamic contexts and verifies the
analyzer correctly identifies redundant tests at the method level.
"""

import os
import subprocess
import tempfile

from kiss.scripts.redundancy_analyzer import _method_name, analyze_redundancy


def test_method_name_strips_run_suffix():
    assert _method_name("test_foo|run") == "test_foo"


def test_method_name_strips_setup_suffix():
    assert _method_name("test_foo|setup") == "test_foo"


def test_method_name_strips_teardown_suffix():
    assert _method_name("test_foo|teardown") == "test_foo"


def test_method_name_no_suffix():
    assert _method_name("test_foo") == "test_foo"


def test_method_name_with_class():
    assert _method_name("TestClass.test_foo|run") == "TestClass.test_foo"


def test_method_name_parametrized():
    assert _method_name("test_foo[param1]|run") == "test_foo[param1]"


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


def test_fully_redundant_method():
    """A test whose arcs are a strict subset of another test is redundant."""
    source = """\
def add(a, b):
    return a + b

def mul(a, b):
    return a * b
"""
    tests = """\
from source_mod import add, mul

def test_both():
    assert add(1, 2) == 3
    assert mul(2, 3) == 6

def test_add_only():
    assert add(1, 2) == 3
"""
    cov_file = _create_coverage_db(tests, source)
    redundant = analyze_redundancy(cov_file)
    # test_add_only is a subset of test_both, so it's redundant
    method_names = [_method_name(r) for r in redundant]
    assert any("test_add_only" in m for m in method_names)
    assert not any("test_both" in m for m in method_names)


def test_no_redundant_methods():
    """Tests with unique arcs are not redundant."""
    source = """\
def add(a, b):
    return a + b

def mul(a, b):
    return a * b
"""
    tests = """\
from source_mod import add, mul

def test_add():
    assert add(1, 2) == 3

def test_mul():
    assert mul(2, 3) == 6
"""
    cov_file = _create_coverage_db(tests, source)
    redundant = analyze_redundancy(cov_file)
    assert len(redundant) == 0


def test_multiple_redundant_methods():
    """Multiple tests can be redundant if a single test covers all their arcs."""
    source = """\
def foo(x):
    if x > 0:
        return "positive"
    return "non-positive"
"""
    tests = """\
from source_mod import foo

def test_all():
    assert foo(1) == "positive"
    assert foo(-1) == "non-positive"

def test_positive():
    assert foo(1) == "positive"

def test_negative():
    assert foo(-1) == "non-positive"
"""
    cov_file = _create_coverage_db(tests, source)
    redundant = analyze_redundancy(cov_file)
    method_names = [_method_name(r) for r in redundant]
    # test_positive and test_negative are subsets of test_all
    assert any("test_positive" in m for m in method_names)
    assert any("test_negative" in m for m in method_names)
    assert not any("test_all" in m for m in method_names)


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


def test_empty_coverage():
    """Handle coverage file with no test contexts gracefully."""
    import warnings

    tmpdir = tempfile.mkdtemp()
    cov_file = os.path.join(tmpdir, ".coverage")

    import coverage
    from coverage.exceptions import CoverageWarning

    cov = coverage.Coverage(data_file=cov_file, branch=True)
    cov.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoverageWarning)
        cov.stop()
        cov.save()

    redundant = analyze_redundancy(cov_file)
    assert redundant == []


def test_all_tests_identical():
    """When tests cover exactly the same arcs, all but one should be redundant."""
    source = """\
def add(a, b):
    return a + b
"""
    tests = """\
from source_mod import add

def test_add1():
    assert add(1, 2) == 3

def test_add2():
    assert add(1, 2) == 3

def test_add3():
    assert add(1, 2) == 3
"""
    cov_file = _create_coverage_db(tests, source)
    redundant = analyze_redundancy(cov_file)
    # 2 of 3 should be redundant
    assert len(redundant) == 2
    _verify_coverage_preserved(cov_file, redundant)
