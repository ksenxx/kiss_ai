#!/usr/bin/env python3
"""
Script to run and test the KISS tutorial notebook.

This script can:
1. Run the notebook and execute all cells
2. Convert the notebook to a Python script
3. Test the code snippets for correctness (import tests, not LLM calls)

Usage:
    # Test all imports and basic functionality (recommended first)
    uv run notebook --test

    # Run the notebook (opens in browser) - requires jupyter
    uv run notebook --run

    # Convert notebook to script - requires jupyter/nbconvert
    uv run notebook --convert

    # Run all tests and then open notebook
    uv run notebook --test --run

    # Execute notebook and update outputs
    uv run notebook --execute
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_notebook_path() -> Path:
    """Get the path to the kiss.ipynb notebook."""
    # Try to find the notebook relative to the package
    package_dir = Path(__file__).parent.parent.parent.parent  # src/kiss/scripts -> project root
    notebook_path = package_dir / "kiss.ipynb"
    if notebook_path.exists():
        return notebook_path

    # Try current working directory
    cwd_notebook = Path.cwd() / "kiss.ipynb"
    if cwd_notebook.exists():
        return cwd_notebook

    raise FileNotFoundError(
        "Could not find kiss.ipynb. Please run this command from the KISS project root."
    )


def test_imports() -> bool:
    """Test that all imports in the notebook work correctly."""
    print("=" * 60)
    print("Testing KISS Framework Imports")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: KISS version
    try:
        from kiss import __version__

        print(f"✓ kiss.__version__ = {__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import kiss.__version__: {e}")
        tests_failed += 1

    # Test 2: KISSAgent
    try:
        from kiss.core.kiss_agent import KISSAgent

        _ = KISSAgent(name="TestAgent")
        print("✓ KISSAgent created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import/create KISSAgent: {e}")
        tests_failed += 1

    # Test 3: AgentEvolver
    try:
        from kiss.agents.create_and_optimize_agent import AgentEvolver

        _ = AgentEvolver()
        print("✓ AgentEvolver created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import/create AgentEvolver: {e}")
        tests_failed += 1

    # Test 4: KISSCodingAgent
    try:
        from kiss.agents.coding_agents import KISSCodingAgent

        _ = KISSCodingAgent(name="TestCodingAgent")
        print("✓ KISSCodingAgent created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import/create KISSCodingAgent: {e}")
        tests_failed += 1

    # Test 5: ClaudeCodingAgent
    try:
        from kiss.agents.coding_agents import ClaudeCodingAgent

        _ = ClaudeCodingAgent(name="TestClaudeAgent")
        print("✓ ClaudeCodingAgent created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import/create ClaudeCodingAgent: {e}")
        tests_failed += 1

    # Test 6: GeminiCliAgent
    try:
        from kiss.agents.coding_agents import GeminiCliAgent

        _ = GeminiCliAgent(name="test_gemini_agent")
        print("✓ GeminiCliAgent created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import/create GeminiCliAgent: {e}")
        tests_failed += 1

    # Test 7: OpenAICodexAgent
    try:
        from kiss.agents.coding_agents import OpenAICodexAgent

        _ = OpenAICodexAgent(name="TestOpenAIAgent")
        print("✓ OpenAICodexAgent created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import/create OpenAICodexAgent: {e}")
        tests_failed += 1

    # Test 8: ARVO agent imports
    try:
        from kiss.evals.arvo_agent.arvo_agent import (  # noqa: F401
            find_vulnerability,
            get_all_arvo_tags,
        )

        print("✓ ARVO agent imports successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import ARVO agent: {e}")
        tests_failed += 1

    # Test 9: SimpleRAG
    try:
        from kiss.rag import SimpleRAG  # noqa: F401

        print("✓ SimpleRAG imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import SimpleRAG: {e}")
        tests_failed += 1

    # Test 10: Useful agents
    try:
        from kiss.agents.kiss import (  # noqa: F401
            get_run_simple_coding_agent,
            prompt_refiner_agent,
            run_bash_task_in_sandboxed_ubuntu_latest,
        )

        print("✓ Useful agents imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Failed to import useful agents: {e}")
        tests_failed += 1

    # Test 11: Test fibonacci test function
    try:

        def test_fibonacci(code: str) -> bool:
            """Test if the generated fibonacci code is correct."""
            try:
                namespace: dict[str, object] = {}
                exec(code, namespace)
                fib = namespace.get("fibonacci")
                if not fib or not callable(fib):
                    return False
                return bool(fib(0) == 0 and fib(1) == 1 and fib(10) == 55 and fib(20) == 6765)
            except Exception:
                return False

        correct_code = """
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]
"""
        assert test_fibonacci(correct_code) is True, "Correct code should pass"
        assert test_fibonacci("") is False, "Empty code should fail"
        assert (
            test_fibonacci("def fibonacci(n): return n") is False
        ), "Wrong impl should fail"
        print("✓ test_fibonacci function works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"✗ test_fibonacci function failed: {e}")
        tests_failed += 1

    # Test 12: Test simple coding test function
    try:

        def test_fn(code: str) -> bool:
            """Test if the generated code is correct."""
            try:
                namespace: dict[str, object] = {}
                exec(code, namespace)
                func = namespace.get("my_function")
                if not func or not callable(func):
                    return False
                return bool(func(42) == 84)
            except Exception:
                return False

        assert test_fn("def my_function(x): return x * 2") is True
        assert test_fn("def my_function(x): return x + 2") is False
        print("✓ simple coding test_fn function works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"✗ simple coding test_fn function failed: {e}")
        tests_failed += 1

    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_passed + tests_failed}")
    print("=" * 60)

    return tests_failed == 0


def check_jupyter_installed() -> bool:
    """Check if Jupyter is installed."""
    try:
        import notebook  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def run_notebook(use_lab: bool = False) -> None:
    """Open the notebook in Jupyter."""
    if not check_jupyter_installed():
        print("Jupyter notebook is not installed.")
        print("\nTo install Jupyter, run:")
        print("  uv sync --group dev")
        print("\nAlternatively, you can open the notebook in VS Code or another IDE.")
        sys.exit(1)

    notebook_path = get_notebook_path()
    if use_lab:
        print(f"Opening notebook in JupyterLab: {notebook_path}")
        subprocess.run(["jupyter", "lab", str(notebook_path)])
    else:
        print(f"Opening notebook in Jupyter Notebook: {notebook_path}")
        subprocess.run(["jupyter", "notebook", str(notebook_path)])


def convert_notebook() -> None:
    """Convert the notebook to a Python script."""
    try:
        import nbconvert  # noqa: F401
    except ImportError:
        print("nbconvert is not installed.")
        print("\nTo install nbconvert, run:")
        print("  uv sync --group dev")
        sys.exit(1)

    notebook_path = get_notebook_path()
    print("Converting notebook to script...")
    subprocess.run(["jupyter", "nbconvert", "--to", "script", str(notebook_path)])
    print("Script created: kiss.py")


def execute_notebook() -> None:
    """Execute the notebook and update outputs in place."""
    try:
        import nbconvert  # noqa: F401
    except ImportError:
        print("nbconvert is not installed.")
        print("\nTo install nbconvert, run:")
        print("  uv sync --group dev")
        sys.exit(1)

    notebook_path = get_notebook_path()
    print(f"Executing notebook: {notebook_path}")
    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(notebook_path)]
    )
    if result.returncode == 0:
        print("Notebook executed successfully!")
    else:
        print("Notebook execution completed with errors.")
        sys.exit(1)


def main() -> None:
    """Main entry point for the notebook command."""
    parser = argparse.ArgumentParser(
        description="Run and test the KISS tutorial notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run notebook --test      # Test all imports
  uv run notebook --run       # Open notebook in Jupyter Notebook
  uv run notebook --lab       # Open notebook in JupyterLab (recommended)
  uv run notebook --execute   # Execute all cells and update outputs
  uv run notebook --convert   # Convert to Python script
        """,
    )
    parser.add_argument(
        "--run", action="store_true", help="Open the notebook in Jupyter Notebook"
    )
    parser.add_argument(
        "--lab", action="store_true", help="Open the notebook in JupyterLab (recommended)"
    )
    parser.add_argument(
        "--convert", action="store_true", help="Convert notebook to Python script"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test imports and basic functionality"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute notebook and update outputs"
    )

    args = parser.parse_args()

    if not any([args.run, args.lab, args.convert, args.test, args.execute]):
        parser.print_help()
        return

    if args.test:
        success = test_imports()
        if not success:
            print("\nSome tests failed. Please check the errors above.")
            sys.exit(1)
        print("\nAll tests passed!")

    if args.execute:
        execute_notebook()

    if args.convert:
        convert_notebook()

    if args.run:
        run_notebook(use_lab=False)

    if args.lab:
        run_notebook(use_lab=True)


if __name__ == "__main__":
    main()
