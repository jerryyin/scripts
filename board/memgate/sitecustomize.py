"""memgate: make a server's ranks wait for gpu-hold before their large GPU allocations.

Loaded by every Python process that has this directory on PYTHONPATH (as `sitecustomize`, which the
interpreter imports at start-up, also in multiprocessing `spawn` children). It does nothing unless
GPU_HOLD_GATE names gpu-hold's gate directory (gpu-queue serve --hold-gate-ranks exports it to jobs).
Then, when a module in GATES is imported, the named function in it is wrapped: a call writes
GATE/<pid>.wait (its phase), waits for GATE/<pid>.go from gpu-hold (at most GPU_HOLD_GATE_TIMEOUT
seconds, default 900; on timeout it goes on and says so on stderr), renames the wait file to
GATE/<pid>.in, runs the call and removes it. The target code is not changed.

GATES, TokenSpeed (the call sites name these module attributes, so patching them is enough):
- tokenspeed.runtime.model_loader.loader._initialize_model (phase weights): the model's construction,
  which allocates the weights (119-140 GiB per GPU within one 0.5 s sample).
- tokenspeed.runtime.layers.attention.registry.profile_available_cache_memory_bytes (phase kv): the
  free-memory profile that sizes the KV cache; registry.py imports the function by name, so the
  registry's binding is wrapped.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
import time
from pathlib import Path

GATES = {
    'tokenspeed.runtime.model_loader.loader': (('_initialize_model', 'weights'),),
    'tokenspeed.runtime.layers.attention.registry': (('profile_available_cache_memory_bytes', 'kv'),),
}
POLL_SECONDS = 0.05


def gated(fn, phase: str, root: Path, timeout: float, clock=time.monotonic, sleep=time.sleep):
    """`fn` wrapped to wait at the gate in `root` before each call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        wait, go, inside = root / f'{pid}.wait', root / f'{pid}.go', root / f'{pid}.in'
        tmp = root / f'{pid}.tmp'
        tmp.write_text(phase + '\n')
        os.replace(tmp, wait)
        start = clock()
        while not go.exists():
            if clock() - start >= timeout:
                print(f'memgate: no go for {phase} after {timeout:g} s; continuing', file=sys.stderr, flush=True)
                break
            sleep(POLL_SECONDS)
        os.replace(wait, inside)
        go.unlink(missing_ok=True)
        try:
            return fn(*args, **kwargs)
        finally:
            inside.unlink(missing_ok=True)

    wrapper.__memgate__ = phase
    return wrapper


def patch(module, root: Path, timeout: float) -> None:
    for name, phase in GATES.get(module.__name__, ()):
        fn = getattr(module, name, None)
        if fn is not None and not hasattr(fn, '__memgate__'):
            setattr(module, name, gated(fn, phase, root, timeout))


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path, timeout: float):
        self.root, self.timeout = root, timeout

    def find_spec(self, fullname, path, target=None):
        if fullname not in GATES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader, root, timeout = spec.loader, self.root, self.timeout
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module, root, timeout)

        loader.exec_module = exec_and_patch
        return spec


def install(environ=os.environ) -> bool:
    root = environ.get('GPU_HOLD_GATE')
    if not root:
        return False
    timeout = float(environ.get('GPU_HOLD_GATE_TIMEOUT', '900'))
    sys.meta_path.insert(0, _Finder(Path(root), timeout))
    return True


install()
