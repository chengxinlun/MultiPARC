import importlib
import pkgutil
import pathlib
import sys
import pytest


PACKAGE_NAME = "multiparc"


@pytest.fixture(scope="session", autouse=True)
def ensure_package_on_path():
    """
    Ensure 'src' is on sys.path so we can import the package
    without installation (if running directly from repo).
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def iter_modules(package_name):
    """Yield all module names in the package (recursively)."""
    package = importlib.import_module(package_name)
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        yield module_info.name


@pytest.mark.parametrize("module_name", list(iter_modules(PACKAGE_NAME)))
def test_import_module(module_name):
    """Try importing every submodule to ensure all imports succeed."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Failed to import {module_name}: {e}")