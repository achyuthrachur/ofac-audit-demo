"""Compatibility shim to launch the multi-page Streamlit app from Home.py."""

from __future__ import annotations

import runpy
from pathlib import Path


def _launch_home() -> None:
    home_path = Path(__file__).resolve().parent.parent / "Home.py"
    if not home_path.exists():
        raise FileNotFoundError(f"Expected Home.py at {home_path}")
    runpy.run_path(str(home_path), run_name="__main__")


if __name__ == "__main__":
    _launch_home()
else:
    # Streamlit executes the script as __main__, but we guard to be safe if imported.
    _launch_home()
