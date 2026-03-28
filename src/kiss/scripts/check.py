# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Script to run all code quality checks: syntax check, lint, and type check."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful.

    Args:
        cmd: List of command arguments to execute.
        description: Human-readable description of the command for logging.

    Returns:
        True if the command exits with code 0, False otherwise.
    """
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
    """Find all non-gitignored markdown files in the project.

    Uses ``git ls-files`` to list tracked and untracked-but-not-ignored
    markdown files, automatically respecting ``.gitignore`` rules.

    Returns:
        Sorted list of absolute paths to markdown files found in the project.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        capture_output=True,
        text=True,
        cwd=project_root,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return sorted(str(project_root / f) for f in result.stdout.strip().split("\n") if f)


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped by consulting ``.gitignore`` rules.

    Uses ``git check-ignore`` to determine whether *path* (or any ancestor)
    is covered by ``.gitignore``.  The ``.git`` directory itself is always
    skipped because it is not listed in ``.gitignore`` but must never be
    cleaned.

    Args:
        path: Path object to check.

    Returns:
        True if the path is inside ``.git`` or is git-ignored, False otherwise.
    """
    if ".git" in path.parts:
        return True
    project_root = Path(__file__).parent.parent.parent.parent
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        capture_output=True,
        cwd=project_root,
        check=False,
    )
    return result.returncode == 0


def clean_build_artifacts() -> None:
    """Remove build artifacts and caches.

    Removes directories like build/, dist/, .mypy_cache/, .pytest_cache/,
    .ruff_cache/, *.egg-info/, all __pycache__ directories, and all .pyc files.
    Skips paths inside .venv, venv, .git, and node_modules directories.

    Equivalent to:
        rm -rf build/ dist/ .pytest_cache .mypy_cache .ruff_cache && \
        find . -type d -name __pycache__ -exec rm -r {} + && \
        find . -type f -name "*.pyc" -delete

    Returns:
        None.
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Directories to remove from project root
    dirs_to_remove = [
        "dist",
        "build",
        "*.egg-info",
        ".claude",
        "artifacts",
        "uv_cache",
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


def _sync_extension_version() -> None:
    """Sync VS Code extension version from _version.py into package.json.

    Reads the canonical version from ``src/kiss/_version.py`` and updates
    the ``version`` field in the extension's ``package.json`` if they differ.
    """
    import json

    project_root = Path(__file__).parent.parent.parent.parent
    version_file = project_root / "src" / "kiss" / "_version.py"
    package_json = project_root / "src" / "kiss" / "agents" / "vscode" / "package.json"

    if not version_file.exists() or not package_json.exists():
        return

    # Extract version from _version.py
    namespace: dict[str, str] = {}
    exec(version_file.read_text(), namespace)  # noqa: S102
    version = namespace.get("__version__", "")
    if not version:
        return

    # Update package.json if needed
    data = json.loads(package_json.read_text())
    if data.get("version") != version:
        data["version"] = version
        package_json.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Synced extension version to {version}")


def main() -> int:
    """Run all code quality checks.

    Parses command-line arguments and runs code quality checks including
    dependency installation, syntax checking with py_compile, linting with ruff,
    type checking with mypy, and markdown formatting checks.

    Returns:
        Exit code 0 if all checks pass, 1 if any check fails.
    """
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning build artifacts before running checks",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build artifacts, do not run checks",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run additional checks including pyright type checking",
    )
    args = parser.parse_args()

    if not args.no_clean:
        clean_build_artifacts()
        if args.clean_only:
            return 0

    # Find all markdown files for linting
    md_files = find_markdown_files()

    # Sync extension version from _version.py before running checks
    _sync_extension_version()

    checks = [
        (["uv", "sync"], "Install dependencies (uv sync)"),
        (["uv", "run", "generate-api-docs"], "Generate API docs"),
        (["uv", "run", "python", "-m", "compileall", "-q", "src/"], "Syntax check (compileall)"),
        (["uv", "run", "ruff", "check", "src/"], "Lint code (ruff)"),
        (["uv", "run", "mypy", "src/"], "Type check (mypy)"),
    ]

    if args.full:
        checks.append((["uv", "run", "pyright", "src/"], "Type check (pyright)"))

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
