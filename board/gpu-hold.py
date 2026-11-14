#!/usr/bin/env python3
"""Memory holder of a local `gpu-queue serve`: holds the claimed GPUs' free memory, and while one of
our processes starts on them, gives it that memory but keeps free memory capped until it is ready.

States, written to --state: `hold` (children hold the free memory), `busy` (an own process has the
GPUs open; children keep each GPU's free memory at most BUSY_FREE_CAP, giving the rest to our
process as it allocates), `yielded` (the own process is ready; no children). Busy ends when --ready exists, when our processes' GPU memory is over half of
the GPUs' and unchanged for READY_STABLE_SECONDS, or after BUSY_LIMIT_SECONDS. When no own process
has the GPUs open, the holder holds again and removes --ready."""
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
READY_STABLE_SECONDS = 20
# While our server loads, free memory stays at or below this, so a tenant that takes 250 GiB per GPU
# at once (seen repeatedly on this host) never fits; our server, which allocates gradually and sizes
# its KV pool once its weights are in (about 153 GiB free then), never waits on the holder.
BUSY_FREE_CAP = 240 << 30
BUSY_LIMIT_SECONDS = 1800
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


def drm_vram(pid: int, proc_root: Path = Path('/proc')) -> int:
    """Largest DRM fdinfo `drm-memory-vram` of a process, in bytes (its VRAM on the GPU it uses)."""
    largest = 0
    try:
        fdinfos = list((proc_root / str(pid) / 'fdinfo').iterdir())
    except OSError:
        return 0
    for fdinfo in fdinfos:
        try:
            for line in fdinfo.read_text().splitlines():
                if line.startswith('drm-memory-vram:'):
                    amount, _, unit = line.split(':', 1)[1].strip().partition(' ')
                    scale = {'': 1, 'KiB': 1 << 10, 'MiB': 1 << 20, 'GiB': 1 << 30}[unit.strip()]
                    largest = max(largest, int(amount) * scale)
        except (OSError, ValueError, KeyError):
            continue
    return largest


def vram_total(drm_root: Path = Path('/sys/class/drm')) -> int:
    """VRAM of one GPU in bytes (the largest card's; the claimed GPUs are alike)."""
    sizes = [int(p.read_text()) for p in drm_root.glob('card*/device/mem_info_vram_total')]
    return max(sizes, default=0)


def busy_hold(free: int, held: int, cap: int) -> int:
    """Bytes to hold so that free memory is at most `cap`: what is available beyond `cap`."""
    return max(0, free + held - cap)


def hold_one(headroom: int) -> None:
    """Allocate the visible GPU's free memory but `headroom` and keep topping it up; on SIGUSR1 hold
    only what keeps free memory at most BUSY_FREE_CAP."""
    import torch

    busy = []
    signal.signal(signal.SIGUSR1, lambda *_: busy.append(True))
    blocks = []
    while not busy:
        free, _ = torch.cuda.mem_get_info(0)
        while free - CHUNK >= headroom and not busy:
            try:
                blocks.append(torch.empty(CHUNK, dtype=torch.uint8, device='cuda:0'))
            except torch.OutOfMemoryError:
                break
            free -= CHUNK
        time.sleep(0.2)
    while True:
        free, _ = torch.cuda.mem_get_info(0)
        target = busy_hold(free, len(blocks) * CHUNK, BUSY_FREE_CAP)
        if len(blocks) * CHUNK > target:
            del blocks[target // CHUNK:]
            torch.cuda.empty_cache()
        else:
            while len(blocks) * CHUNK + CHUNK <= target:
                try:
                    blocks.append(torch.empty(CHUNK, dtype=torch.uint8, device='cuda:0'))
                except torch.OutOfMemoryError:
                    break
        time.sleep(0.05)


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

    def busy(self) -> None:
        for child in self.children:
            child.send_signal(signal.SIGUSR1)

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


def watch(holder: Holder, users, own_memory, stop, poll: float, log: Path | None,
          ready: Path | None = None, state_file: Path | None = None, clock=time.monotonic) -> None:
    """Hold, go busy for an own process's start, step aside once it is ready; until stop()."""
    state, busy_since, memory, stable_since = None, 0.0, 0, 0.0

    def enter(new: str, why: str) -> None:
        nonlocal state
        state = new
        if state_file is not None:
            state_file.write_text(new + '\n')
        note(log, f'{new} HIP {holder.hip}: {why}')

    while not stop():
        current = users(holder.pids())
        now = clock()
        if not current:
            if state != 'hold' or any(child.poll() is not None for child in holder.children):
                holder.release()
                if ready is not None:
                    ready.unlink(missing_ok=True)
                holder.hold()
                enter('hold', f'children {sorted(holder.pids())}')
        elif state == 'hold':
            holder.busy()
            busy_since, memory, stable_since = now, own_memory(current), now
            enter('busy', f'own GPU process(es) {current}; free memory capped at {BUSY_FREE_CAP >> 30} GiB')
        elif state == 'busy':
            used = own_memory(current)
            if abs(used - memory) > CHUNK:
                memory, stable_since = used, now
            why = None
            if ready is not None and ready.exists():
                why = 'ready'
            elif memory > len(holder.hip) * vram_total() // 2 and now - stable_since >= READY_STABLE_SECONDS:
                why = f'own memory {memory >> 30} GiB unchanged for {READY_STABLE_SECONDS} s'
            elif now - busy_since >= BUSY_LIMIT_SECONDS:
                why = f'busy for {BUSY_LIMIT_SECONDS} s'
            if why:
                holder.release()
                enter('yielded', why)
        time.sleep(poll)
    holder.release()
    if state_file is not None:
        state_file.unlink(missing_ok=True)
    note(log, f'stopped; released HIP {holder.hip}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hip')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--ready', type=Path)
    parser.add_argument('--state', type=Path)
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
          lambda pids: sum(drm_vram(pid) for pid in pids),
          lambda: bool(stopping), POLL_SECONDS, args.log, args.ready, args.state)
    return 0


if __name__ == '__main__':
    sys.exit(main())
