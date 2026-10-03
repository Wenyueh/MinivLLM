"""Test configuration shared by every file in tests/.

The package uses a src layout and is not installed in CI, so the test modules
have to make `src/` importable. Each file used to do that itself, which means a
new test file that forgets the line passes locally (where the package may be
installed) and fails only in CI. Doing it once here removes that trap.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
