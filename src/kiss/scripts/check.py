# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Script to run all code quality checks: install dependencies, build, lint, and type check."""

import subprocess
import sys


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


def main() -> int:
    """Run all code quality checks."""
    checks = [
        (["uv", "sync"], "Install dependencies (uv sync)"),
        (["uv", "build"], "Build package"),
        (["uv", "run", "ruff", "check", "src/"], "Lint code (ruff)"),
        (["uv", "run", "mypy", "src/"], "Type check (mypy)"),
    ]

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
