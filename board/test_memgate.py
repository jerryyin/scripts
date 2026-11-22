#!/usr/bin/env python3
"""Tests for memgate/sitecustomize.py: the wait/go protocol and which functions it wraps. No GPU."""
import importlib.util
import os
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
    def resolve(module, name):
        owner, _, attr = name.rpartition('.')
        return (getattr(module, owner) if owner else module), attr

    for name, gates in mg.GATES.items():
        module = types.ModuleType(name)
        for dotted, _phase, *_ in gates:
            if '.' in dotted:
                setattr(module, dotted.split('.')[0], type('Runner', (), {}))
            owner, attr = resolve(module, dotted)
            setattr(owner, attr, lambda *a: 'called')
        mg.patch(module, root, 1)
        mg.patch(module, root, 1)
        for dotted, phase, *_ in gates:
            owner, attr = resolve(module, dotted)
            fn = getattr(owner, attr)
            assert fn.__memgate__ == phase and not hasattr(fn.__wrapped__, '__memgate__')
    assert set(mg.GATES) == {'tokenspeed.runtime.model_loader.loader',
                             'tokenspeed.runtime.execution.model_runner',
                             'tokenspeed.runtime.layers.attention.registry',
                             'tokenspeed.runtime.execution.distributed_initializer'}


def test_the_distributed_init_minimum_read_is_gated_and_reported_its_other_reads_are_not(tmp=None):
    """DistributedInitializer.initialize calls get_available_gpu_memory four times; only the cross-rank
    minimum (cpu_group=...) sizes anything (init_distributed returns it as the KV headroom's gpu_memory)."""
    import contextlib
    import io
    root = Path(tmp or tempfile.mkdtemp())
    module = types.ModuleType('tokenspeed.runtime.execution.distributed_initializer')
    reads = []
    module.get_available_gpu_memory = lambda device, gpu_id, distributed=False, empty_cache=True, cpu_group=None: (
        reads.append((distributed, cpu_group)) or 286.25)
    mg.patch(module, root, 30)
    fn = module.get_available_gpu_memory
    assert fn.__memgate__ == 'dist'
    assert fn('cuda', 0) == 286.25 and not list(root.iterdir()), 'a log read passes straight through'
    pid = os.getpid()
    result, err = [], io.StringIO()

    def call():
        with contextlib.redirect_stderr(err):
            result.append(fn('cuda', 0, distributed=True, cpu_group='gloo'))

    thread = threading.Thread(target=call)
    thread.start()
    for _ in range(200):
        if (root / f'{pid}.wait').exists():
            break
        time.sleep(0.01)
    assert (root / f'{pid}.wait').read_text().strip() == 'dist' and len(reads) == 1
    (root / f'{pid}.go').touch()
    thread.join(5)
    assert result == [286.25] and reads[-1] == (True, 'gloo') and not list(root.iterdir())
    assert 'memgate: dist read 286.25' in err.getvalue(), err.getvalue()


def test_a_gated_method_still_binds_to_its_instance(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    module = types.ModuleType('tokenspeed.runtime.execution.model_runner')

    class ModelRunner:
        def prepare_communication_runtime(self, max_num_tokens):
            return (self, max_num_tokens)

    module.ModelRunner = ModelRunner
    mg.patch(module, root, 0)  # timeout 0: goes on at once
    runner = ModelRunner()
    assert runner.prepare_communication_runtime(8192) == (runner, 8192)


def test_spawned_children_load_the_gate_and_the_shadowed_sitecustomize_still_runs(tmp=None):
    """A fake tokenspeed package, imported in a multiprocessing spawn child, as TokenSpeed's ranks are."""
    import subprocess
    root = Path(tmp or tempfile.mkdtemp())
    package = root / 'pkg' / 'tokenspeed' / 'runtime' / 'model_loader'
    package.mkdir(parents=True)
    for d in (package, package.parent, package.parent.parent):
        (d / '__init__.py').write_text('')
    (package / 'loader.py').write_text('def _initialize_model():\n    return "model"\n')
    shadowed = root / 'shadowed'
    shadowed.mkdir()
    (shadowed / 'sitecustomize.py').write_text('import os\nos.environ["SHADOWED_RAN"] = "1"\n')
    (root / 'gate').mkdir()
    script = (
        'import multiprocessing as mp, os\n'
        'def child(q):\n'
        '    from tokenspeed.runtime.model_loader import loader\n'
        '    q.put((getattr(loader._initialize_model, "__memgate__", None), loader._initialize_model(),\n'
        '           os.environ.get("SHADOWED_RAN")))\n'
        'if __name__ == "__main__":\n'
        '    ctx = mp.get_context("spawn"); q = ctx.Queue(); p = ctx.Process(target=child, args=(q,))\n'
        '    p.start(); print(q.get(timeout=60)); p.join()\n')
    (root / 'main.py').write_text(script)
    memgate = Path(__file__).resolve().with_name('memgate')
    env = {**os.environ, 'PYTHONPATH': f'{memgate}:{root / "pkg"}:{shadowed}', 'GPU_HOLD_GATE': str(root / 'gate'),
           'GPU_HOLD_GATE_TIMEOUT': '0', 'PYTHONDONTWRITEBYTECODE': '1'}
    out = subprocess.run([sys.executable, str(root / 'main.py')], env=env, capture_output=True, text=True,
                         timeout=120, cwd=root)
    assert out.stdout.strip() == "('weights', 'model', '1')", (out.stdout, out.stderr)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
