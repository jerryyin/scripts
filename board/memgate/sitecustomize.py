"""memgate: make a server's ranks wait for gpu-hold before their large GPU allocations.

Loaded by every Python process that has this directory on PYTHONPATH (as `sitecustomize`, which the
interpreter imports at start-up, also in multiprocessing `spawn` children). It does nothing unless
GPU_HOLD_GATE names gpu-hold's gate directory (gpu-queue serve --hold-gate-ranks exports it to jobs).
Then GPU_HOLD_GATE_TABLE names the gate table of the server (a JSON file, e.g. TokenSpeed's
tokenspeed/memgate-gates.json), and when a module the table names is imported, the functions it lists
there are wrapped: a call writes GATE/<pid>.wait (its phase and a new token), waits for GATE/<pid>.go
holding that token from gpu-hold (at most GPU_HOLD_GATE_TIMEOUT seconds, default 900; on timeout it
goes on and says so on stderr), renames the wait file to GATE/<pid>.in, runs the call and removes it.
The target code is not changed, so the call sites must look the wrapped attributes up at call time.

The table maps a module name to a list of gates: {"attr": "func" or "Class.method", "phase": PHASE,
"keyword": NAME (optional: gate only calls passing it), "report": true (optional: print the call's
result on stderr), "why": text (ignored)}. A missing table, or a listed attribute missing from its
module once imported, is an error: printed on stderr and written to GATE/<pid>.error, which gpu-hold
logs (the call would otherwise run ungated and its allocation fail under the busy cap).
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple


class Gate(NamedTuple):
    attr: str
    phase: str
    keyword: str | None = None
    report: bool = False


POLL_SECONDS = 0.05


def announce(root: Path, pid: int, phase: str) -> str:
    """Write root/<pid>.wait (`phase token`): `pid` waits at the gate before its `phase` call. Returns
    the token, new per call, that the go file must hold (a go left over from an earlier call does not
    count)."""
    token = os.urandom(8).hex()
    tmp = root / f'{pid}.tmp'
    tmp.write_text(f'{phase} {token}\n')
    os.replace(tmp, root / f'{pid}.wait')
    return token


def has_go(root: Path, pid: int, token: str) -> bool:
    try:
        return (root / f'{pid}.go').read_text().strip() == token
    except OSError:
        return False


def enter(root: Path, pid: int) -> None:
    """Take the go: the wait file becomes root/<pid>.in while the call runs."""
    os.replace(root / f'{pid}.wait', root / f'{pid}.in')
    (root / f'{pid}.go').unlink(missing_ok=True)


def leave(root: Path, pid: int) -> None:
    (root / f'{pid}.in').unlink(missing_ok=True)


def gated(fn, phase: str, root: Path, timeout: float, clock=time.monotonic, sleep=time.sleep,
          keyword: str | None = None, report: bool = False, pid=os.getpid):
    """`fn` wrapped to wait at the gate in `root` before each call (with `keyword`: each call passing it;
    with `report`: its result goes to stderr). `pid()` names this caller's files (the process ID)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if keyword is not None and keyword not in kwargs:
            return fn(*args, **kwargs)
        me = pid()
        token = announce(root, me, phase)
        start = clock()
        while not has_go(root, me, token):
            if clock() - start >= timeout:
                print(f'memgate: no go for {phase} after {timeout:g} s; continuing', file=sys.stderr, flush=True)
                break
            sleep(POLL_SECONDS)
        enter(root, me)
        try:
            result = fn(*args, **kwargs)
            if report:
                print(f'memgate: {phase} read {result!r}', file=sys.stderr, flush=True)
            return result
        finally:
            leave(root, me)

    wrapper.__memgate__ = phase
    return wrapper


def error(root: Path, message: str) -> None:
    """Say on stderr, and to gpu-hold in root/<pid>.error, that the gate is not where it should be."""
    line = f'memgate: ERROR: {message}'
    print(line, file=sys.stderr, flush=True)
    try:
        with (root / f'{os.getpid()}.error').open('a') as fh:
            fh.write(line + '\n')
    except OSError:
        pass


def load_table(path: Path) -> dict[str, list[Gate]]:
    """The gate table in `path`: module name -> its gates."""
    table = json.loads(path.read_text())
    return {module: [Gate(g['attr'], g['phase'], g.get('keyword'), bool(g.get('report', False)))
                     for g in gates]
            for module, gates in table.items()}


def patch(module, gates: list[Gate], root: Path, timeout: float) -> None:
    for gate in gates:
        owner, _, attr = gate.attr.rpartition('.')
        target = module
        for part in owner.split('.') if owner else ():
            target = getattr(target, part, None)
        fn = getattr(target, attr, None) if target is not None else None
        if fn is None:
            error(root, f'{module.__name__} has no {gate.attr}: its {gate.phase} call is not gated')
        elif not hasattr(fn, '__memgate__'):
            setattr(target, attr, gated(fn, gate.phase, root, timeout, keyword=gate.keyword, report=gate.report))


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, table: dict[str, list[Gate]], root: Path, timeout: float):
        self.table, self.root, self.timeout = table, root, timeout

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.table:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader, gates, root, timeout = spec.loader, self.table[fullname], self.root, self.timeout
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module, gates, root, timeout)

        loader.exec_module = exec_and_patch
        return spec


def install(environ=os.environ) -> bool:
    root = environ.get('GPU_HOLD_GATE')
    if not root:
        return False
    table_path = environ.get('GPU_HOLD_GATE_TABLE')
    if not table_path:
        error(Path(root), 'GPU_HOLD_GATE is set but GPU_HOLD_GATE_TABLE is not: nothing is gated')
        return False
    try:
        table = load_table(Path(table_path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error(Path(root), f'cannot read the gate table {table_path}: {exc!r}; nothing is gated')
        return False
    timeout = float(environ.get('GPU_HOLD_GATE_TIMEOUT', '900'))
    sys.meta_path.insert(0, _Finder(table, Path(root), timeout))
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
