#!/usr/bin/env python3
"""Tests for gpu-hold.py: which processes make the holder yield, and the hold/yield cycle.

No GPU: the holder's children are `sleep` processes and the process table is a directory.
"""
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location('gpu_hold', Path(__file__).with_name('gpu-hold.py'))
gh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gh)


def fake_process(proc: Path, pid: int, kfd: bool, environ: dict[str, str]):
    entry = proc / str(pid)
    (entry / 'fd').mkdir(parents=True)
    os.symlink('/dev/kfd' if kfd else '/dev/null', entry / 'fd' / '3')
    (entry / 'environ').write_bytes(b'\0'.join(f'{k}={v}'.encode() for k, v in environ.items()))


def test_only_our_kfd_users_that_can_see_the_held_gpus_count(tmp=None):
    proc = Path(tmp or tempfile.mkdtemp())
    fake_process(proc, 10, True, {'HIP_VISIBLE_DEVICES': '2,3'})      # server on a held GPU
    fake_process(proc, 11, True, {'HIP_VISIBLE_DEVICES': '4,5,6,7'})  # another job, other GPUs
    fake_process(proc, 12, True, {})                                  # sees every GPU
    fake_process(proc, 13, False, {'HIP_VISIBLE_DEVICES': '0'})       # no GPU open
    fake_process(proc, 14, True, {'HIP_VISIBLE_DEVICES': '0'})        # the holder's own child
    assert gh.own_gpu_users({0, 1, 2, 3}, exclude={14}, proc_root=proc) == [10, 12]


def test_holds_yields_to_our_process_and_holds_again(tmp=None):
    def sleeper(_gpu):
        return subprocess.Popen(['sleep', '60'])

    holder = gh.Holder([0, 1], None, spawn=sleeper)
    # what the process table shows on each pass: nothing, our server for two passes, nothing
    script = [[], [], [99], [99], [], []]
    seen, children = [], []

    def users(exclude):
        seen.append(set(exclude))
        children.append(list(holder.children))
        return script[len(seen) - 1]

    gh.watch(holder, users, lambda: len(seen) == len(script), 0, None)
    held = [bool(c) for c in children]
    assert held == [False, True, True, False, False, True], held
    first, second = children[1], children[5]
    assert len(first) == 2 and all(c.poll() is not None for c in first), 'yield ends every child'
    assert seen[1] == {c.pid for c in first}, "the holder's own children never count as users"
    assert all(c.poll() is not None for c in second) and not holder.children, 'stop releases'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
