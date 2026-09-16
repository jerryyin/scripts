"""Point ``tokenspeed_triton`` at a locally built ``triton`` (development only).

tokenspeed-kernel imports every Triton symbol as ``tokenspeed_triton``, the
vendor release package (see ``tokenspeed_kernel._triton``).  A developer build
of Triton installs under the plain ``triton`` name instead, so this module
aliases the vendor name onto ``triton`` from an explicitly selected checkout,
re-using the *same* module objects -- a copy would create duplicate classes and
break ``isinstance`` across the boundary.

Build Triton however you normally do -- ``make dev-install`` once in the Triton
tree, then ``make all`` (vim ``<leader>bb``) for incremental C++ rebuilds --
then explicitly select it::

    python ~/scripts/tokenspeed/tokenspeed_triton_alias.py \
        --install --triton-dir ~/triton                                # activate
    python ~/scripts/tokenspeed/tokenspeed_triton_alias.py --uninstall # back to the wheel

Activation is one-time per environment: the alias resolves whatever ``triton``
is in the selected checkout, so rebuilding or switching branches in that
checkout needs no re-run.

Installing copies this file into site-packages next to a ``.pth`` file that
imports it, so every interpreter in the environment picks the alias up without
a ``PYTHONPATH``.  The released ``tokenspeed-triton`` wheel stays installed and
untouched: the alias takes precedence over it, and ``TOKENSPEED_TRITON_ALIAS=0``
disables both the alias and the local checkout path for a single process.
"""

import argparse
import importlib
import importlib.abc
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

_SRC = "tokenspeed_triton"
_DST = "triton"

# ``zz_`` keeps the hook after the editable-install path hooks that pip writes.
_PTH_NAME = "zz_tokenspeed_triton_alias.pth"
_MODULE_NAME = "tokenspeed_triton_alias"
_MODULE_SIGNATURE = b'_MODULE_NAME = "tokenspeed_triton_alias"'


def _module_locations(module: ModuleType) -> list[Path]:
    file = getattr(module, "__file__", None)
    if file is not None:
        return [Path(file).resolve()]
    return [Path(path).resolve() for path in getattr(module, "__path__", ())]


def _require_module_in_tree(
    module: ModuleType, triton_python: Path, alias_name: str
) -> list[Path]:
    locations = _module_locations(module)
    if not locations or any(
        not location.is_relative_to(triton_python) for location in locations
    ):
        rendered = ", ".join(map(str, locations)) or "<no filesystem location>"
        raise ImportError(
            f"{alias_name} resolved outside selected Triton checkout "
            f"{triton_python}: {rendered}"
        )
    return locations


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

    def __init__(self, triton_python: Path) -> None:
        self.triton_python = triton_python

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
        except ImportError as error:
            raise ModuleNotFoundError(
                f"{fullname} is redirected to {target_name} in "
                f"{self.triton_python}, but that import failed"
            ) from error
        _require_module_in_tree(module, self.triton_python, fullname)
        is_pkg = hasattr(module, "__path__")
        spec = importlib.util.spec_from_loader(
            fullname, _ReuseModuleLoader(module), is_package=is_pkg
        )
        if is_pkg:
            spec.submodule_search_locations = module.__path__
        return spec


def activate(triton_python: str) -> None:
    """Select a checkout and install the redirect finder, at most once."""
    if os.environ.get("TOKENSPEED_TRITON_ALIAS", "1") == "0":
        return
    selected = Path(triton_python).expanduser().resolve()
    for name, module in tuple(sys.modules.items()):
        if module is not None and (name == "triton" or name.startswith("triton.")):
            _require_module_in_tree(module, selected, name)
    selected_string = str(selected)
    if selected_string not in sys.path:
        sys.path.insert(0, selected_string)
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder):
            if finder.triton_python != selected:
                raise RuntimeError(
                    "tokenspeed_triton alias is already active for "
                    f"{finder.triton_python}, not {selected}"
                )
            return
    sys.meta_path.insert(0, _AliasFinder(selected))


def _site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _alias_paths() -> tuple[Path, Path]:
    site_packages = _site_packages()
    return (
        site_packages / f"{_MODULE_NAME}.py",
        site_packages / _PTH_NAME,
    )


def _require_managed_paths(module_path: Path, pth_path: Path) -> None:
    if module_path.exists() and _MODULE_SIGNATURE not in module_path.read_bytes():
        raise SystemExit(f"error: refusing to overwrite unmanaged file {module_path}")
    if pth_path.exists() and _MODULE_NAME.encode() not in pth_path.read_bytes():
        raise SystemExit(f"error: refusing to overwrite unmanaged file {pth_path}")


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes, int] | None]:
    return {
        path: (path.read_bytes(), path.stat().st_mode & 0o777) if path.exists() else None
        for path in paths
    }


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore(snapshot: dict[Path, tuple[bytes, int] | None]) -> None:
    for path, previous in snapshot.items():
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            data, mode = previous
            _atomic_write(path, data, mode)


def _fresh_python(code: str, *arguments: str, disable_alias: bool = False) -> str:
    environment = os.environ.copy()
    if disable_alias:
        environment["TOKENSPEED_TRITON_ALIAS"] = "0"
    else:
        environment.pop("TOKENSPEED_TRITON_ALIAS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, *arguments],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    if result.returncode != 0:
        details = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        raise RuntimeError(details or f"fresh Python exited with {result.returncode}")
    return result.stdout.strip()


def _verify_local_alias(triton_python: Path) -> str:
    return _fresh_python(
        """
import importlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
triton = importlib.import_module("triton")
aliased = importlib.import_module("tokenspeed_triton")
native = importlib.import_module("tokenspeed_triton._C.libtriton")
if aliased is not triton:
    raise RuntimeError("tokenspeed_triton is not the selected triton module")
for label, module in (("triton", triton), ("libtriton", native)):
    location = pathlib.Path(module.__file__).resolve()
    if not location.is_relative_to(root):
        raise RuntimeError(f"{label} resolved outside {root}: {location}")
print(f"tokenspeed_triton -> {aliased.__file__} (version {aliased.__version__})")
""",
        str(triton_python),
    )


def _verify_vendor_wheel(*, disable_alias: bool) -> str:
    return _fresh_python(
        """
import importlib.metadata
import pathlib
import tokenspeed_triton

distribution = importlib.metadata.distribution("tokenspeed-triton")
package = pathlib.Path(distribution.locate_file("tokenspeed_triton")).resolve()
location = pathlib.Path(tokenspeed_triton.__file__).resolve()
if not location.is_relative_to(package):
    raise RuntimeError(f"tokenspeed_triton did not resolve to its wheel: {location}")
print(f"tokenspeed_triton wheel -> {location} (version {distribution.version})")
""",
        disable_alias=disable_alias,
    )


def _install(triton_dir: Path) -> None:
    if os.environ.get("TOKENSPEED_TRITON_ALIAS") == "0":
        raise SystemExit(
            "error: unset TOKENSPEED_TRITON_ALIAS=0 before installing the alias"
        )
    triton_python = triton_dir.expanduser().resolve() / "python"
    triton_package = triton_python / "triton" / "__init__.py"
    if not triton_package.is_file():
        raise SystemExit(
            f"error: Triton checkout not found at {triton_dir} "
            f"(missing {triton_package})"
        )

    # Validate the checkout before persisting the startup hook. In particular,
    # importing libtriton rejects a source-only checkout that has not been
    # built yet.
    activate(str(triton_python))
    aliased = importlib.import_module(_SRC)
    importlib.import_module(f"{_SRC}.language")
    importlib.import_module(f"{_SRC}.experimental.gluon")
    native = importlib.import_module(f"{_SRC}._C.libtriton")
    _require_module_in_tree(aliased, triton_python, _SRC)
    _require_module_in_tree(native, triton_python, f"{_SRC}._C.libtriton")

    # Keep a working vendor fallback, and do not overwrite unrelated files
    # that happen to use the helper's installation names.
    vendor = _verify_vendor_wheel(disable_alias=True)
    module_path, pth_path = _alias_paths()
    _require_managed_paths(module_path, pth_path)
    previous = _snapshot((module_path, pth_path))

    try:
        _atomic_write(module_path, Path(__file__).resolve().read_bytes())
        _atomic_write(
            pth_path,
            (
                f"import {_MODULE_NAME}; "
                f"{_MODULE_NAME}.activate({str(triton_python)!r})\n"
            ).encode(),
        )
        selected = _verify_local_alias(triton_python)
    except (OSError, RuntimeError) as error:
        _restore(previous)
        raise SystemExit(f"error: alias installation failed and was rolled back: {error}")

    print(f"installed {module_path}")
    print(f"installed {pth_path}")
    print(selected)
    print(f"fallback verified: {vendor}")


def _uninstall() -> None:
    module_path, pth_path = _alias_paths()
    _require_managed_paths(module_path, pth_path)
    previous = _snapshot((module_path, pth_path))

    # Verify the fallback before removing a working override, then verify again
    # without the per-process escape hatch after removal.
    _verify_vendor_wheel(disable_alias=True)
    try:
        for path in (pth_path, module_path):
            path.unlink(missing_ok=True)
        vendor = _verify_vendor_wheel(disable_alias=False)
    except (OSError, RuntimeError) as error:
        _restore(previous)
        raise SystemExit(f"error: alias removal failed and was rolled back: {error}")

    for path, contents in previous.items():
        if contents is not None:
            print(f"removed {path}")
    print(vendor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--install",
        action="store_true",
        help="persistently redirect tokenspeed_triton to a local checkout",
    )
    action.add_argument(
        "--uninstall",
        action="store_true",
        help="remove the alias and fall back to the tokenspeed-triton wheel",
    )
    parser.add_argument(
        "--triton-dir",
        type=Path,
        default=None,
        help="Triton checkout to select with --install (default: ~/triton)",
    )
    args = parser.parse_args()
    if args.uninstall:
        if args.triton_dir is not None:
            parser.error("--triton-dir requires --install")
        _uninstall()
    else:
        _install(args.triton_dir or Path.home() / "triton")


if __name__ == "__main__":
    main()
