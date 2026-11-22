#!/usr/bin/env python3
"""Tests for memgate-gates.json, TokenSpeed's gate table for board/memgate (no GPU, no TokenSpeed)."""
import contextlib
import importlib.util
import io
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / 'memgate-gates.json'
spec = importlib.util.spec_from_file_location('memgate', HERE.parent / 'board' / 'memgate' / 'sitecustomize.py')
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)


def test_the_table_gates_tokenspeeds_four_start_up_allocations():
    """Only the distributed-init read passing cpu_group= (the cross-rank minimum the KV headroom is taken
    from) is gated, and its value is reported."""
    assert mg.load_table(TABLE) == {
        'tokenspeed.runtime.execution.distributed_initializer': [
            mg.Gate('get_available_gpu_memory', 'dist', 'cpu_group', True)],
        'tokenspeed.runtime.model_loader.loader': [mg.Gate('_initialize_model', 'weights')],
        'tokenspeed.runtime.execution.model_runner': [mg.Gate('ModelRunner.prepare_communication_runtime', 'comm')],
        'tokenspeed.runtime.layers.attention.registry': [mg.Gate('profile_available_cache_memory_bytes', 'kv')],
    }


def test_every_entry_wraps_its_attribute(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    for name, gates in mg.load_table(TABLE).items():
        module = types.ModuleType(name)
        for gate in gates:
            owner, _, attr = gate.attr.rpartition('.')
            target = module
            if owner:
                target = type(owner, (), {})
                setattr(module, owner, target)
            setattr(target, attr, lambda *a, **k: 'called')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            mg.patch(module, gates, root, 1)
        assert not err.getvalue(), err.getvalue()
        for gate in gates:
            owner, _, attr = gate.attr.rpartition('.')
            assert getattr(getattr(module, owner) if owner else module, attr).__memgate__ == gate.phase


if __name__ == '__main__':
    for test_name, fn in list(globals().items()):
        if test_name.startswith('test_'):
            fn()
            print('ok', test_name)
