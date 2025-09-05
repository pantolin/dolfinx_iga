#!/usr/bin/env python3
"""Type checking script for dolfinx_iga project."""

import subprocess
import sys
from pathlib import Path


def run_mypy():
    """Run mypy type checking on the dolfinx_iga package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "dolfinx_iga/", "--show-error-codes"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode == 0:
            print("✅ Type checking passed!")
            return True
        else:
            print("❌ Type checking failed:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Error running mypy: {e}")
        return False


if __name__ == "__main__":
    success = run_mypy()
    sys.exit(0 if success else 1)
