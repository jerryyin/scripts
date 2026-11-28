#!/usr/bin/env python3
"""Tests for memgate/sitecustomize.py: the wait/go protocol and which functions it wraps. No GPU."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
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
    phase, token = (root / f'{pid}.wait').read_text().split()
    assert phase == 'kv'
    (root / f'{pid}.go').write_text('an-earlier-call\n')
    time.sleep(0.1)
    assert not calls, 'no call before the go, and a go of an earlier call does not count'
    (root / f'{pid}.go').write_text(token + '\n')
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
    module = types.ModuleType('server.loader')
    module.Runner = type('Runner', (), {'prepare': lambda self: 'prepared'})
    module.build = lambda: 'built'
    gates = [mg.Gate('build', 'weights'), mg.Gate('Runner.prepare', 'comm')]
    mg.patch(module, gates, root, 1)
    mg.patch(module, gates, root, 1)
    assert module.build.__memgate__ == 'weights' and not hasattr(module.build.__wrapped__, '__memgate__')
    assert module.Runner.prepare.__memgate__ == 'comm' and not hasattr(module.Runner.prepare.__wrapped__, '__memgate__')


def test_the_table_file_names_modules_attributes_phases_and_options(tmp=None):
    path = Path(tmp or tempfile.mkdtemp()) / 'gates.json'
    path.write_text(json.dumps({'server.init': [{'attr': 'free_memory', 'phase': 'dist', 'keyword': 'group',
                                                 'report': True, 'why': 'ignored'}],
                                'server.loader': [{'attr': 'build', 'phase': 'weights'}]}))
    assert mg.load_table(path) == {'server.init': [mg.Gate('free_memory', 'dist', 'group', True)],
                                   'server.loader': [mg.Gate('build', 'weights', None, False)]}


def test_a_keyword_gate_gates_only_calls_passing_it_and_a_report_gate_prints_the_result(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    module = types.ModuleType('server.init')
    reads = []
    module.free_memory = lambda device, group=None: reads.append(group) or 286.25
    mg.patch(module, [mg.Gate('free_memory', 'dist', 'group', True)], root, 30)
    fn = module.free_memory
    assert fn.__memgate__ == 'dist'
    assert fn('cuda') == 286.25 and not list(root.iterdir()), 'a call without the keyword passes straight through'
    pid = os.getpid()
    result, err = [], io.StringIO()

    def call():
        with contextlib.redirect_stderr(err):
            result.append(fn('cuda', group='gloo'))

    thread = threading.Thread(target=call)
    thread.start()
    for _ in range(200):
        if (root / f'{pid}.wait').exists():
            break
        time.sleep(0.01)
    phase, token = (root / f'{pid}.wait').read_text().split()
    assert phase == 'dist' and len(reads) == 1
    (root / f'{pid}.go').write_text(token)
    thread.join(5)
    assert result == [286.25] and reads[-1] == 'gloo' and not list(root.iterdir())
    assert 'memgate: dist read 286.25' in err.getvalue(), err.getvalue()


def test_a_gated_method_still_binds_to_its_instance(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    module = types.ModuleType('server.runner')

    class Runner:
        def prepare(self, max_num_tokens):
            return (self, max_num_tokens)

    module.Runner = Runner
    mg.patch(module, [mg.Gate('Runner.prepare', 'comm')], root, 0)  # timeout 0: goes on at once
    runner = Runner()
    assert runner.prepare(8192) == (runner, 8192)


def test_a_missing_attribute_or_table_is_a_loud_error_the_holder_sees(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    module = types.ModuleType('server.loader')
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        mg.patch(module, [mg.Gate('build', 'weights')], root, 1)
    assert 'memgate: ERROR: server.loader has no build: its weights call is not gated' in err.getvalue()
    assert 'server.loader has no build' in (root / f'{os.getpid()}.error').read_text()
    before = list(sys.meta_path)
    with contextlib.redirect_stderr(err):
        assert mg.install({'GPU_HOLD_GATE': str(root)}) is False
        assert mg.install({'GPU_HOLD_GATE': str(root), 'GPU_HOLD_GATE_TABLE': str(root / 'none.json')}) is False
    assert sys.meta_path == before
    assert 'GPU_HOLD_GATE_TABLE is not' in err.getvalue() and 'cannot read the gate table' in err.getvalue()


def test_spawned_children_load_the_gate_and_the_shadowed_sitecustomize_still_runs(tmp=None):
    """A fake server package with a gate table, imported in a multiprocessing spawn child, as a
    server's ranks are."""
    root = Path(tmp or tempfile.mkdtemp())
    package = root / 'pkg' / 'server'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'loader.py').write_text('def build():\n    return "model"\n')
    (root / 'gates.json').write_text(json.dumps({'server.loader': [{'attr': 'build', 'phase': 'weights'}]}))
    shadowed = root / 'shadowed'
    shadowed.mkdir()
    (shadowed / 'sitecustomize.py').write_text('import os\nos.environ["SHADOWED_RAN"] = "1"\n')
    (root / 'gate').mkdir()
    script = (
        'import multiprocessing as mp, os\n'
        'def child(q):\n'
        '    from server import loader\n'
        '    q.put((getattr(loader.build, "__memgate__", None), loader.build(), os.environ.get("SHADOWED_RAN")))\n'
        'if __name__ == "__main__":\n'
        '    ctx = mp.get_context("spawn"); q = ctx.Queue(); p = ctx.Process(target=child, args=(q,))\n'
        '    p.start(); print(q.get(timeout=60)); p.join()\n')
    (root / 'main.py').write_text(script)
    memgate = Path(__file__).resolve().with_name('memgate')
    env = {**os.environ, 'PYTHONPATH': f'{memgate}:{root / "pkg"}:{shadowed}', 'GPU_HOLD_GATE': str(root / 'gate'),
           'GPU_HOLD_GATE_TABLE': str(root / 'gates.json'), 'GPU_HOLD_GATE_TIMEOUT': '0',
           'PYTHONDONTWRITEBYTECODE': '1'}
    out = subprocess.run([sys.executable, str(root / 'main.py')], env=env, capture_output=True, text=True,
                         timeout=120, cwd=root)
    assert out.stdout.strip() == "('weights', 'model', '1')", (out.stdout, out.stderr)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
