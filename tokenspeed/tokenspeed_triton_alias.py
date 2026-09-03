"""Point ``tokenspeed_triton`` at a locally built ``triton`` (development only).

tokenspeed-kernel imports every Triton symbol as ``tokenspeed_triton``, the
vendor release package (see ``tokenspeed_kernel._triton``).  A developer build
of Triton installs under the plain ``triton`` name instead, so this module
aliases the vendor name onto whatever ``triton`` the interpreter resolves,
re-using the *same* module objects -- a copy would create duplicate classes and
break ``isinstance`` across the boundary.

Build Triton however you normally do -- ``make dev-install`` once in the Triton
tree, then ``make all`` (vim ``<leader>bb``) for incremental C++ rebuilds --
then::

    python ~/scripts/tokenspeed/tokenspeed_triton_alias.py             # activate
    python ~/scripts/tokenspeed/tokenspeed_triton_alias.py --uninstall # back to the wheel

Activation is one-time per environment: the alias resolves whatever ``triton``
is installed, so rebuilding or switching Triton branches needs no re-run.

Installing copies this file into site-packages next to a ``.pth`` file that
imports it, so every interpreter in the environment picks the alias up without
a ``PYTHONPATH``.  The released ``tokenspeed-triton`` wheel stays installed and
untouched: the alias takes precedence over it, and ``TOKENSPEED_TRITON_ALIAS=0``
disables the alias for a single process.
"""

import argparse
import importlib
import importlib.abc
import importlib.util
import os
import shutil
import sys
import sysconfig
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

_SRC = "tokenspeed_triton"
_DST = "triton"

# ``zz_`` keeps the hook after the editable-install path hooks that pip writes.
_PTH_NAME = "zz_tokenspeed_triton_alias.pth"
_MODULE_NAME = "tokenspeed_triton_alias"


class _ReuseModuleLoader(importlib.abc.Loader):
    """Loader that exposes an already-imported module under an alias name."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self._module

    def exec_module(self, module: ModuleType) -> None:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Lazy ``tokenspeed_triton[.x.y]`` -> ``triton[.x.y]`` redirect finder."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != _SRC and not fullname.startswith(_SRC + "."):
            return None
        target_name = _DST + fullname[len(_SRC) :]
        try:
            module = importlib.import_module(target_name)
        except ImportError:
            return None
        is_pkg = hasattr(module, "__path__")
        spec = importlib.util.spec_from_loader(
            fullname, _ReuseModuleLoader(module), is_package=is_pkg
        )
        if is_pkg:
            spec.submodule_search_locations = module.__path__
        return spec


def activate() -> None:
    """Install the redirect finder into ``sys.meta_path``, at most once."""
    if os.environ.get("TOKENSPEED_TRITON_ALIAS", "1") == "0":
        return
    if any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _AliasFinder())


activate()


def _site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _install() -> None:
    site_packages = _site_packages()
    module_path = site_packages / f"{_MODULE_NAME}.py"
    pth_path = site_packages / _PTH_NAME
    shutil.copyfile(Path(__file__).resolve(), module_path)
    pth_path.write_text(f"import {_MODULE_NAME}\n")
    print(f"installed {module_path}")
    print(f"installed {pth_path}")

    # Resolve through the alias installed by ``activate()`` above, so a failure
    # here is a real failure of what the ``.pth`` will do in every process.
    aliased = importlib.import_module(_SRC)
    importlib.import_module(f"{_SRC}.language")
    importlib.import_module(f"{_SRC}.experimental.gluon")
    print(f"{_SRC} -> {aliased.__file__} (version {aliased.__version__})")


def _uninstall() -> None:
    site_packages = _site_packages()
    for path in (site_packages / f"{_MODULE_NAME}.py", site_packages / _PTH_NAME):
        if path.exists():
            path.unlink()
            print(f"removed {path}")
    print(f"{_SRC} now resolves to the installed tokenspeed-triton wheel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="remove the alias and fall back to the tokenspeed-triton wheel",
    )
    args = parser.parse_args()
    if args.uninstall:
        _uninstall()
    else:
        _install()


if __name__ == "__main__":
    main()
