"""Single entry point for Streamlit deployments expecting `src/demo_app.py`.

This module renders a manual navigation sidebar and dispatches to the
Home/Data Generator/Audit Analysis views defined elsewhere in the repo.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Callable, Dict, Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent

PAGES = [
    ("Home", ROOT_DIR / "Home.py", "render"),
    ("📊 Data Generator", ROOT_DIR / "pages/1_📊_Data_Generator.py", "render_page"),
    ("🔍 Audit Analysis", ROOT_DIR / "pages/2_🔍_Audit_Analysis.py", "render"),
]

st.set_page_config(
    page_title="OFAC Audit Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", [title for title, _, _ in PAGES])


def _load_page(path: Path, func_name: str) -> Optional[Callable[[], None]]:
    """Execute the target module and return the desired render function."""
    page_globals: Dict[str, object] = runpy.run_path(str(path), run_name="__run__")
    func = page_globals.get(func_name)
    if callable(func):
        return func  # type: ignore[return-value]
    st.error(f"Page '{path.name}' is missing a `{func_name}()` callable.")
    return None


for title, path, func_name in PAGES:
    if selection == title:
        render_func = _load_page(path, func_name)
        if render_func:
            render_func()
        break
