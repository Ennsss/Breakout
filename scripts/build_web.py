"""Publish browser assets; account data and ML artifacts stay server-side."""

from pathlib import Path
from shutil import copytree

root = Path(__file__).resolve().parents[1]
copytree(root / "web/static", root / "public/static", dirs_exist_ok=True)
