#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

_REPO_MARKERS = (
    Path("python/triton/__init__.py"),
    Path("include/triton"),
    Path("bin/triton-lsp.cpp"),
    Path("CMakeLists.txt"),
)


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_env_path(var_name: str) -> Path | None:
    value = os.environ.get(var_name)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not _is_executable(path):
        raise FileNotFoundError(
            f"{var_name} is set but is not an executable file: {path}"
        )
    return path


def _resolve_build_dir() -> Path | None:
    build_dir = os.environ.get("TRITON_BUILD_DIR")
    if build_dir is None:
        return None
    path = Path(build_dir).expanduser() / "bin" / "triton-lsp"
    if not _is_executable(path):
        raise FileNotFoundError(
            "TRITON_BUILD_DIR is set but does not contain an executable "
            f"triton-lsp at: {path}"
        )
    return path


def _resolve_repo_root() -> Path:
    env_root = os.environ.get("TRITON_REPO_ROOT")
    if env_root is not None:
        path = Path(env_root).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"TRITON_REPO_ROOT is set but is not a directory: {path}"
            )
        return path

    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _REPO_MARKERS):
            return candidate

    raise FileNotFoundError(
        "Unable to locate Triton repo root from current working directory. "
        "Run from the repo root or set TRITON_REPO_ROOT."
    )


def _candidate_paths(root: Path, patterns: Iterable[str]) -> list[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    return [path for path in candidates if _is_executable(path)]


def _most_recent(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _find_triton_lsp() -> Path:
    env_path = _resolve_env_path("TRITON_LSP_PATH")
    if env_path is not None:
        return env_path

    which_path = shutil.which("triton-lsp")
    if which_path is not None:
        resolved = Path(which_path)
        if not _is_executable(resolved):
            raise FileNotFoundError(
                f"Found triton-lsp on PATH but it is not executable: {resolved}"
            )
        return resolved

    build_path = _resolve_build_dir()
    if build_path is not None:
        return build_path

    root = _resolve_repo_root()
    candidates = _candidate_paths(
        root,
        [
            "build/**/bin/triton-lsp",
            "build-*/bin/triton-lsp",
            "cmake-build-*/bin/triton-lsp",
        ],
    )
    selected = _most_recent(candidates)
    if selected is None:
        raise FileNotFoundError(
            "Unable to locate triton-lsp. Build Triton or set TRITON_LSP_PATH "
            "or TRITON_BUILD_DIR."
        )
    return selected


def main(argv: Sequence[str]) -> None:
    lsp_path = _find_triton_lsp()
    os.execv(str(lsp_path), [str(lsp_path), *argv[1:]])
    raise RuntimeError("execv returned unexpectedly")


if __name__ == "__main__":
    main(sys.argv)
