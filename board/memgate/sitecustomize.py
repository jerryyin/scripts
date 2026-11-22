"""memgate: make a server's ranks wait for gpu-hold before their large GPU allocations.

Loaded by every Python process that has this directory on PYTHONPATH (as `sitecustomize`, which the
interpreter imports at start-up, also in multiprocessing `spawn` children). It does nothing unless
GPU_HOLD_GATE names gpu-hold's gate directory (gpu-queue serve --hold-gate-ranks exports it to jobs).
Then, when a module in GATES is imported, the named function in it is wrapped: a call writes
GATE/<pid>.wait (its phase), waits for GATE/<pid>.go from gpu-hold (at most GPU_HOLD_GATE_TIMEOUT
seconds, default 900; on timeout it goes on and says so on stderr), renames the wait file to
GATE/<pid>.in, runs the call and removes it. The target code is not changed.

GATES, TokenSpeed (the call sites look these attributes up at call time, so patching them is enough):
- tokenspeed.runtime.model_loader.loader._initialize_model (phase weights): the model's construction,
  run under the target device inside startup phase `weights.allocate`, which allocates the weights
  (119-140 GiB per GPU within one 0.5 s sample).
- tokenspeed.runtime.execution.model_runner.ModelRunner.prepare_communication_runtime (phase comm):
  "allocate model communication buffers before cache planning" (startup phase
  `communication.prepare`), for the target and the draft model.
- tokenspeed.runtime.layers.attention.registry.profile_available_cache_memory_bytes (phase kv): the
  free-memory profile that sizes the KV cache; registry.py imports the function by name, so the
  registry's binding is wrapped.
- tokenspeed.runtime.execution.distributed_initializer.get_available_gpu_memory (phase dist), only its
  call with `cpu_group=` (DistributedInitializer.initialize's cross-rank minimum, which init_distributed
  returns as the device memory `gpu_memory` the KV profile takes its utilization headroom from:
  KV = free - post_profile - gpu_memory x (1 - --gpu-memory-utilization)). Read under the busy cap it is
  the cap (32 GiB), the headroom shrinks from ~29 GiB to ~3 GiB and the KV pool takes the activations'
  memory; gated, all ranks read the device's free memory. Its other calls there (two log lines, the
  per-rank balance check) are not gated. The phase's call returns its value on stderr.
A GATES entry is (attribute, phase) or (attribute, phase, keyword): with a keyword only calls that pass
it are gated.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import time
from pathlib import Path

GATES = {
    'tokenspeed.runtime.model_loader.loader': (('_initialize_model', 'weights'),),
    'tokenspeed.runtime.execution.model_runner': (('ModelRunner.prepare_communication_runtime', 'comm'),),
    'tokenspeed.runtime.layers.attention.registry': (('profile_available_cache_memory_bytes', 'kv'),),
    'tokenspeed.runtime.execution.distributed_initializer': (('get_available_gpu_memory', 'dist', 'cpu_group'),),
}
REPORT_PHASES = {'dist'}  # phases whose call's result goes to stderr (the value the server sizes with)
POLL_SECONDS = 0.05


def gated(fn, phase: str, root: Path, timeout: float, clock=time.monotonic, sleep=time.sleep,
          keyword: str | None = None):
    """`fn` wrapped to wait at the gate in `root` before each call (with `keyword`: each call passing it)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if keyword is not None and keyword not in kwargs:
            return fn(*args, **kwargs)
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
            result = fn(*args, **kwargs)
            if phase in REPORT_PHASES:
                print(f'memgate: {phase} read {result!r}', file=sys.stderr, flush=True)
            return result
        finally:
            inside.unlink(missing_ok=True)

    wrapper.__memgate__ = phase
    return wrapper


def patch(module, root: Path, timeout: float) -> None:
    for name, phase, *keyword in GATES.get(module.__name__, ()):
        owner, _, attr = name.rpartition('.')
        target = module
        for part in owner.split('.') if owner else ():
            target = getattr(target, part, None)
        fn = getattr(target, attr, None) if target is not None else None
        if fn is not None and not hasattr(fn, '__memgate__'):
            setattr(target, attr, gated(fn, phase, root, timeout, keyword=keyword[0] if keyword else None))


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


def chain() -> None:
    """Run the sitecustomize this one shadows (on Ubuntu, the system's apport hook), as Python would."""
    here = Path(__file__).resolve().parent
    rest = [p for p in sys.path if Path(p or '.').resolve() != here]
    spec = importlib.machinery.PathFinder.find_spec('sitecustomize', rest)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # as site.py does: report, never stop start-up
        print(f'memgate: shadowed sitecustomize {spec.origin} failed: {exc!r}', file=sys.stderr)


install()
chain()
