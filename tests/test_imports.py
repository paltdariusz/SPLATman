import importlib
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH.parents[0]))
modules = [
    p for p in SRC_PATH.rglob("*.py") if p.is_file() and not p.name.startswith("__")
]


def test_imports():
    for module_file in modules:
        rel = module_file.relative_to(SRC_PATH)
        module_name = "src." + str(rel.with_suffix("")).replace("/", ".")
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            pytest.skip(f"Missing dependency for {module_name}: {e}")
