# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Script to run all code quality checks: install dependencies, build, lint, and type check."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    print(f"\n✅ {description} passed")
    return True


def find_markdown_files() -> list[str]:
    """Find all markdown files in the project."""
    project_root = Path(__file__).parent.parent.parent.parent
    skip_dirs = {"node_modules", "venv", "__pycache__", "artifacts"}
    md_files = []
    for md_file in project_root.rglob("*.md"):
        # Skip hidden directories and common non-project directories
        parts = md_file.parts
        if any(part.startswith(".") or part in skip_dirs for part in parts):
            continue
        md_files.append(str(md_file))
    return sorted(md_files)


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped (e.g., inside .venv or hidden directories)."""
    skip_dirs = {".venv", "venv", ".git", "node_modules"}
    for part in path.parts:
        if part in skip_dirs:
            return True
    return False


def clean_build_artifacts() -> None:
    """Remove build artifacts and caches.

    Equivalent to:
        rm -rf build/ dist/ .pytest_cache .mypy_cache .ruff_cache && \
        find . -type d -name __pycache__ -exec rm -r {} + && \
        find . -type f -name "*.pyc" -delete
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Directories to remove from project root
    dirs_to_remove = [
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "*.egg-info",
    ]

    print(f"\n{'=' * 60}")
    print("Running: Clean build artifacts")
    print(f"{'=' * 60}\n")

    removed_count = 0

    # Remove specific directories from project root
    for pattern in dirs_to_remove:
        if "*" in pattern:
            # Handle glob patterns
            for path in project_root.glob(pattern):
                if path.is_dir():
                    print(f"  Removing: {path}")
                    shutil.rmtree(path)
                    removed_count += 1
        else:
            path = project_root / pattern
            if path.exists() and path.is_dir():
                print(f"  Removing: {path}")
                shutil.rmtree(path)
                removed_count += 1

    # Remove __pycache__ directories recursively (skip .venv and similar)
    for pycache in project_root.rglob("__pycache__"):
        if pycache.is_dir() and not _should_skip_path(pycache):
            print(f"  Removing: {pycache}")
            shutil.rmtree(pycache)
            removed_count += 1

    # Remove .pyc files recursively (skip .venv and similar)
    for pyc_file in project_root.rglob("*.pyc"):
        if not _should_skip_path(pyc_file):
            print(f"  Removing: {pyc_file}")
            pyc_file.unlink()
            removed_count += 1

    if removed_count == 0:
        print("  No artifacts to clean.")
    else:
        print(f"\n✅ Cleaned {removed_count} items")


def main() -> int:
    """Run all code quality checks."""
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build artifacts before running checks",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build artifacts, do not run checks",
    )
    args = parser.parse_args()

    # Clean if requested
    if args.clean or args.clean_only:
        clean_build_artifacts()
        if args.clean_only:
            return 0

    # Find all markdown files for linting
    md_files = find_markdown_files()

    checks = [
        (["uv", "sync"], "Install dependencies (uv sync)"),
        (["uv", "build"], "Build package"),
        (["uv", "run", "ruff", "check", "src/"], "Lint code (ruff)"),
        (["uv", "run", "mypy", "src/"], "Type check (mypy)"),
        (["uv", "run", "pyright", "src/"], "Type check (pyright)"),
    ]

    # Add markdown lint check if there are markdown files
    if md_files:
        checks.append((["uv", "run", "mdformat", "--check", *md_files], "Lint markdown (mdformat)"))

    print("\n🔍 Running all code quality checks...\n")

    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False
            break  # Stop on first failure

    if all_passed:
        print("\n" + "=" * 60)
        print("✅ All checks passed!")
        print("=" * 60 + "\n")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ Some checks failed. Please fix the errors above.")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
