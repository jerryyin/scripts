#!/usr/bin/env python3
"""Tests for memgate/sitecustomize.py: the wait/go protocol and which functions it wraps. No GPU."""
import importlib.util
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location('memgate', Path(__file__).with_name('memgate') / 'sitecustomize.py')
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)


def test_inactive_without_the_gate_variable():
    before = list(sys.meta_path)
    assert mg.install({}) is False and sys.meta_path == before


def test_a_gated_call_waits_for_its_go_and_cleans_up(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    calls = []
    fn = mg.gated(lambda x: calls.append(x) or 2 * x, 'kv', root, timeout=30)
    result = []
    thread = threading.Thread(target=lambda: result.append(fn(21)))
    thread.start()
    pid = __import__('os').getpid()
    for _ in range(200):
        if (root / f'{pid}.wait').exists():
            break
        time.sleep(0.01)
    assert (root / f'{pid}.wait').read_text().strip() == 'kv'
    time.sleep(0.1)
    assert not calls, 'no call before the go'
    (root / f'{pid}.go').touch()
    thread.join(5)
    assert result == [42] and calls == [21]
    assert not list(root.iterdir()), 'wait, go and in files are gone'


def test_a_gated_call_goes_on_after_the_timeout(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    clock = iter([0.0, 1.0, 5.0])
    fn = mg.gated(lambda: 'ran', 'weights', root, timeout=5, clock=lambda: next(clock), sleep=lambda s: None)
    assert fn() == 'ran' and not list(root.iterdir())


def test_patch_wraps_the_names_the_call_sites_use_once(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    for name, gates in mg.GATES.items():
        module = types.ModuleType(name)
        for attr, _phase in gates:
            setattr(module, attr, lambda: None)
        mg.patch(module, root, 1)
        mg.patch(module, root, 1)
        for attr, phase in gates:
            fn = getattr(module, attr)
            assert fn.__memgate__ == phase and not hasattr(fn.__wrapped__, '__memgate__')
    assert set(mg.GATES) == {'tokenspeed.runtime.model_loader.loader',
                             'tokenspeed.runtime.layers.attention.registry'}


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
