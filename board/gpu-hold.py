#!/usr/bin/env python3
"""Memory holder of a local `gpu-queue serve`: holds the claimed GPUs' free memory and yields it
while one of our processes uses them."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 1 << 30
HEADROOM = 1 << 30  # left free on each GPU for our own process to start in
POLL_SECONDS = 0.2
DEVICE_ENV = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')


def note(log: Path | None, message: str) -> None:
    line = f'{dt.datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z} gpu-hold: {message}'
    print(line, flush=True)
    if log is not None:
        with log.open('a') as fh:
            fh.write(line + '\n')


def visible_devices(environ: dict[str, str]) -> set[int] | None:
    """HIP ids a process can see from its environment; None when it does not restrict them."""
    for key in DEVICE_ENV:
        value = environ.get(key)
        if value is not None:
            return {int(x) for x in value.split(',') if x.strip().isdigit()}
    return None


def _environ(pid_dir: Path) -> dict[str, str]:
    try:
        raw = (pid_dir / 'environ').read_bytes()
    except OSError:
        return {}
    return dict(item.split('=', 1) for item in raw.decode(errors='replace').split('\0') if '=' in item)


def own_gpu_users(hip: set[int], exclude: set[int], proc_root: Path = Path('/proc')) -> list[int]:
    """PIDs here, outside `exclude`, with /dev/kfd open that can see one of the `hip` GPUs."""
    users = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in exclude:
            continue
        try:
            opened = any(os.readlink(fd) == '/dev/kfd' for fd in (entry / 'fd').iterdir())
        except OSError:
            continue
        if not opened:
            continue
        seen = visible_devices(_environ(entry))
        if seen is None or seen & hip:
            users.append(int(entry.name))
    return sorted(users)


def hold_one(headroom: int) -> None:
    """Allocate the visible GPU's free memory but `headroom` and keep topping it up."""
    import torch

    blocks = []
    while True:
        free, _ = torch.cuda.mem_get_info(0)
        while free - CHUNK >= headroom:
            try:
                blocks.append(torch.empty(CHUNK, dtype=torch.uint8, device='cuda:0'))
            except torch.OutOfMemoryError:
                break
            free -= CHUNK
        time.sleep(0.5)


class Holder:
    """One child per GPU, started and stopped together."""

    def __init__(self, hip: list[int], log: Path | None, spawn=None):
        self.hip, self.log = hip, log
        self.children: list[subprocess.Popen] = []
        self._spawn = spawn or self._spawn_child

    def _spawn_child(self, gpu: int) -> subprocess.Popen:
        env = {**os.environ, 'HIP_VISIBLE_DEVICES': str(gpu)}
        for key in ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
            env.pop(key, None)
        argv = [sys.executable, os.path.abspath(__file__), '--child']
        return subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL)

    @property
    def holding(self) -> bool:
        return bool(self.children)

    def pids(self) -> set[int]:
        return {child.pid for child in self.children}

    def hold(self) -> None:
        self.children = [self._spawn(gpu) for gpu in self.hip]

    def release(self) -> None:
        for child in self.children:
            child.kill()
        for child in self.children:
            child.wait()
        self.children = []


def watch(holder: Holder, users, stop, poll: float, log: Path | None) -> None:
    """Hold while no other process of ours uses the GPUs; release while one does; until stop()."""
    while not stop():
        current = users(holder.pids())
        if current and holder.holding:
            holder.release()
            note(log, f'yield HIP {holder.hip}: own GPU process(es) {current}')
        elif not current and not holder.holding:
            holder.hold()
            note(log, f'hold HIP {holder.hip}: children {sorted(holder.pids())}')
        elif holder.holding and any(child.poll() is not None for child in holder.children):
            holder.release()
            note(log, f'a holder child exited on HIP {holder.hip}; restarting')
        time.sleep(poll)
    holder.release()
    note(log, f'stopped; released HIP {holder.hip}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hip')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        hold_one(HEADROOM)
        return 0
    if not args.hip:
        parser.error('--hip is required')
    hip = [int(x) for x in args.hip.split(',')]
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(True))
    holder = Holder(hip, args.log)
    watch(holder, lambda exclude: own_gpu_users(set(hip), exclude | {os.getpid()}),
          lambda: bool(stopping), POLL_SECONDS, args.log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
