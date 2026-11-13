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


class FakeChild:
    """A holder child: records signals, dies on kill."""
    count = 0

    def __init__(self):
        FakeChild.count += 1
        self.pid, self.signals, self.dead = 1000 + FakeChild.count, [], False

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):
        self.dead = True

    def wait(self):
        return 0

    def poll(self):
        return 0 if self.dead else None


def run_script(script, tmp, memory=lambda step: 0):
    """Drive watch() one pass per script entry (the own GPU users seen); returns the states written."""
    tmp = Path(tmp)
    ready, state = tmp / 'ready', tmp / 'state'
    holder = gh.Holder([0, 1], None, spawn=lambda gpu: FakeChild())
    step, states = [-1], []  # stop() runs before every pass and advances to the next entry

    def users(_exclude):
        current = script[step[0]]
        if current == 'ready':
            ready.touch()
            current = [99]
        return current

    def stop():
        if state.exists() and (not states or states[-1] != state.read_text().strip()):
            states.append(state.read_text().strip())
        step[0] += 1
        return step[0] >= len(script)

    gh.watch(holder, users, lambda pids: memory(step[0]), stop, 0, None, ready, state,
             clock=lambda: step[0] * 10.0)
    return holder, states, ready


def test_busy_while_our_server_starts_then_step_aside_when_ready(tmp=None):
    tmp = tmp or tempfile.mkdtemp()
    script = [[], [], [99], [99], 'ready', [99], [], []]
    holder, states, ready = run_script(script, tmp)
    assert states == ['hold', 'busy', 'yielded', 'hold'], states
    assert not ready.exists(), 'holding again clears the ready file for the next start'
    assert not holder.children, 'stop releases'


def test_busy_ends_when_our_memory_settles(tmp=None):
    big = 2 * gh.vram_total() // 2 + (8 << 30)  # over half of two GPUs
    script = [[], [99], [99], [99], [99], [99], [99]]
    # memory grows at step 2, then stays: busy ends READY_STABLE_SECONDS (2 steps of 10 s) later
    _, states, _ = run_script(script, tmp or tempfile.mkdtemp(),
                              memory=lambda step: big if step >= 2 else 1 << 30)
    assert states == ['hold', 'busy', 'yielded'], states


def test_busy_hold_keeps_free_memory_under_the_cap_and_gives_the_rest_to_our_server():
    GiB = 1 << 30
    cap = gh.BUSY_FREE_CAP
    assert gh.busy_hold(free=286 * GiB, held=0, cap=cap) == 46 * GiB  # a 250 GiB request cannot fit
    # our server allocated 20 GiB from the 240 free: release 20 of the 46 held, free is back at 240
    assert gh.busy_hold(free=220 * GiB, held=46 * GiB, cap=cap) == 26 * GiB
    # its weights are in (153 GiB free, nothing held): hold nothing, the KV pool sees a clean GPU
    assert gh.busy_hold(free=153 * GiB, held=0, cap=cap) == 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
