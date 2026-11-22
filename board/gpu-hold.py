#!/usr/bin/env python3
"""Memory holder of a local `gpu-queue serve`: holds the claimed GPUs' free memory, and while one of
our processes starts on them, gives it that memory but keeps free memory capped until it is ready.

States, written to --state: `hold` (children hold the free memory), `busy` (an own process has the
GPUs open; children keep each GPU's free memory at most --busy-free-cap, default BUSY_FREE_CAP,
giving the rest to our process as it allocates), `yielded` (the own process is ready; no children).
Busy ends when --ready exists, when our processes' GPU memory is over half of the GPUs' and unchanged
for READY_STABLE_SECONDS (not with --gate), or after BUSY_LIMIT_SECONDS. When no own process has the
GPUs open, the holder holds again and removes --ready.

With --gate DIR and --gate-ranks N (used with memgate/, which the server's ranks load with their
server's gate table), a small --busy-free-cap is safe for our server's large allocations: a rank
about to make one writes DIR/<pid>.wait (its phase, e.g. `dist`, `weights` or `comm`; `kv` for the
free-memory profile that sizes the KV cache) and waits. When N ranks wait, the holder releases
everything it holds, then writes DIR/<pid>.go for each; ranks arriving later at the same phase while
it is open get theirs at once, a rank at another phase waits for the next round. After any phase but
`kv`, when no rank waits at it or runs a gated call, the holder caps free memory again; after a `kv`
phase it yields (the cache is sized from all free memory, which our server then allocates).
memgate's errors (DIR/<pid>.error) go to the log."""
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
# While our server loads, free memory stays at or below this, so a tenant asking for more per GPU at
# once does not fit; our server, which allocates gradually and sizes its KV pool once its weights are
# in, never waits on the holder.
BUSY_FREE_CAP = 240 << 30
BUSY_LIMIT_SECONDS = 1800
UNCAPPED = 1 << 62  # a cap no GPU reaches: the children hold nothing
RELEASED_MAX = 2 << 30  # a child holding nothing keeps its context: at most this much VRAM
RELEASE_WAIT_SECONDS = 10  # gate: go after this long even if a child still reads as holding
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


def read_cap(cap_file: Path | None, default: int = BUSY_FREE_CAP) -> int:
    """The busy cap the parent last wrote (bytes); `default` when there is no readable one."""
    if cap_file is None:
        return default
    try:
        return int(cap_file.read_text().strip())
    except (OSError, ValueError):
        return default


def write_cap(cap_file: Path, cap: int) -> None:
    tmp = cap_file.with_name(cap_file.name + '.tmp')
    tmp.write_text(f'{cap}\n')
    os.replace(tmp, cap_file)


def hold_one(headroom: int, cap_file: Path | None = None) -> None:
    """Allocate the visible GPU's free memory but `headroom` and keep topping it up; on SIGUSR1 hold
    only what keeps free memory at most the cap in `cap_file` (re-read every pass; BUSY_FREE_CAP
    without one)."""
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
        target = busy_hold(free, len(blocks) * CHUNK, read_cap(cap_file))
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

    def __init__(self, hip: list[int], log: Path | None, spawn=None, cap_file: Path | None = None,
                 vram=None):
        self.hip, self.log, self.cap_file = hip, log, cap_file
        self.children: list[subprocess.Popen] = []
        self._spawn = spawn or self._spawn_child
        self._vram = vram or drm_vram

    def _spawn_child(self, gpu: int) -> subprocess.Popen:
        env = {**os.environ, 'HIP_VISIBLE_DEVICES': str(gpu)}
        if self.cap_file is not None:
            env['GPU_HOLD_CAP_FILE'] = str(self.cap_file)
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

    def set_cap(self, cap: int) -> None:
        """Free memory the busy children leave per GPU from now on (UNCAPPED: hold nothing)."""
        if self.cap_file is not None:
            write_cap(self.cap_file, cap)

    def released(self) -> bool:
        """Every child holds at most its context (after set_cap(UNCAPPED))."""
        return all(self._vram(child.pid) <= RELEASED_MAX for child in self.children)

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


class Gate:
    """The holder's side of memgate's files in one directory (see the module docstring)."""

    def __init__(self, root: Path, ranks: int):
        self.root, self.ranks = root, ranks
        self.state: str | None = None  # None (capped), 'releasing', 'open'
        self.phase, self.since = '', 0.0

    def clear(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.root.iterdir():
            if path.suffix in ('.wait', '.go', '.in', '.tmp', '.error'):
                path.unlink(missing_ok=True)
        self.state = None

    def errors(self) -> list[str]:
        """What memgate reported in <pid>.error files since the last call (the files are removed)."""
        lines = []
        for path in sorted(self.root.glob('*.error')):
            try:
                lines += [f'pid {path.stem}: {line}' for line in path.read_text().splitlines()]
            except OSError:
                continue
            path.unlink(missing_ok=True)
        return lines

    def _wait(self, pid: int) -> tuple[str, str] | None:
        """(phase, token) of `pid`'s wait file; None without one."""
        try:
            phase, _, token = (self.root / f'{pid}.wait').read_text().strip().partition(' ')
        except OSError:
            return None
        return phase, token

    def _go_token(self, pid: int) -> str | None:
        try:
            return (self.root / f'{pid}.go').read_text().strip()
        except OSError:
            return None

    def waiting(self) -> dict[int, str]:
        """PID -> phase of the ranks waiting for a go (a go file holding an older call's token is stale)."""
        found = {}
        for path in self.root.glob('*.wait'):
            if not path.stem.isdigit():
                continue
            wait = self._wait(int(path.stem))
            if wait is not None and self._go_token(int(path.stem)) != wait[1]:
                found[int(path.stem)] = wait[0]
        return found

    def busy(self) -> bool:
        """A go not taken yet or a gated call running."""
        if any(self.root.glob('*.in')):
            return True
        for path in self.root.glob('*.go'):
            wait = self._wait(int(path.stem)) if path.stem.isdigit() else None
            if wait is not None and self._go_token(int(path.stem)) == wait[1]:
                return True
        return False

    def go(self, pids) -> None:
        """Write each waiting rank's go file with the token of its current wait."""
        for pid in pids:
            wait = self._wait(pid)
            if wait is None:  # it went on without us (timeout)
                continue
            tmp = self.root / f'{pid}.go.tmp'
            tmp.write_text(wait[1] + '\n')
            os.replace(tmp, self.root / f'{pid}.go')


def watch(holder: Holder, users, own_memory, stop, poll: float, log: Path | None,
          ready: Path | None = None, state_file: Path | None = None, clock=time.monotonic,
          cap: int = BUSY_FREE_CAP, gate: Gate | None = None) -> None:
    """Hold, go busy for an own process's start, step aside once it is ready; until stop()."""
    state, busy_since, memory, stable_since = None, 0.0, 0, 0.0

    def enter(new: str, why: str) -> None:
        nonlocal state
        state = new
        if state_file is not None:
            state_file.write_text(new + '\n')
        note(log, f'{new} HIP {holder.hip}: {why}')

    while not stop():
        for line in gate.errors() if gate is not None else ():
            note(log, line)
        current = users(holder.pids())
        now = clock()
        if not current:
            if state != 'hold' or any(child.poll() is not None for child in holder.children):
                holder.release()
                if ready is not None:
                    ready.unlink(missing_ok=True)
                if gate is not None:
                    gate.clear()
                holder.set_cap(cap)
                holder.hold()
                enter('hold', f'children {sorted(holder.pids())}')
        elif state == 'hold':
            holder.busy()
            busy_since, memory, stable_since = now, own_memory(current), now
            enter('busy', f'own GPU process(es) {current}; free memory capped at {cap / (1 << 30):g} GiB'
                  + (f'; gate {gate.root} for {gate.ranks} ranks' if gate is not None else ''))
        elif state == 'busy':
            used = own_memory(current)
            if abs(used - memory) > CHUNK:
                memory, stable_since = used, now
            why = None
            if ready is not None and ready.exists():
                why = 'ready'
            elif (gate is None and memory > len(holder.hip) * vram_total() // 2
                  and now - stable_since >= READY_STABLE_SECONDS):
                why = f'own memory {memory >> 30} GiB unchanged for {READY_STABLE_SECONDS} s'
            elif now - busy_since >= BUSY_LIMIT_SECONDS:
                why = f'busy for {BUSY_LIMIT_SECONDS} s'
            if why:
                holder.release()
                enter('yielded', why)
            elif gate is not None:
                why = gate_step(holder, gate, cap, now, log)
                if why:
                    enter('yielded', why)
        elif gate is not None:
            gate.go(gate.waiting())  # yielded: nothing held, nothing to wait for
        time.sleep(poll)
    holder.release()
    if state_file is not None:
        state_file.unlink(missing_ok=True)
    note(log, f'stopped; released HIP {holder.hip}')


def gate_step(holder: Holder, gate: Gate, cap: int, now: float, log: Path | None) -> str | None:
    """One busy pass of the gate; returns why to yield (after a `kv` release), else None."""
    waiting = gate.waiting()
    if gate.state is None and len(waiting) >= gate.ranks:
        gate.state, gate.since = 'releasing', now
        gate.phase = 'kv' if 'kv' in waiting.values() else sorted(set(waiting.values()))[0]
        holder.set_cap(UNCAPPED)
        note(log, f'gate: {len(waiting)} ranks wait ({gate.phase}); releasing')
    if gate.state == 'releasing' and (holder.released() or now - gate.since >= RELEASE_WAIT_SECONDS):
        if gate.phase == 'kv':
            holder.release()
            gate.go(waiting)
            gate.state = 'open'
            return f'gate: kv profile of {sorted(waiting)}; everything released'
        gate.go(waiting)
        gate.state = 'open'
        note(log, f'gate: go {sorted(waiting)} ({gate.phase})')
    elif gate.state == 'open':
        # Only the open phase goes at once; a rank at another phase waits for a round of its own
        # (a `kv` profile must not run until everything is released).
        same = [pid for pid, phase in waiting.items() if phase == gate.phase]
        gate.go(same)
        if not same and not gate.busy():
            gate.state = None
            holder.set_cap(cap)
            note(log, f'gate: {gate.phase} done; free memory capped at {cap / (1 << 30):g} GiB again')
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hip')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--ready', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--busy-free-cap', type=float, default=BUSY_FREE_CAP / (1 << 30), metavar='GIB',
                        help=f'busy: free memory left per GPU (default {BUSY_FREE_CAP >> 30})')
    parser.add_argument('--gate', type=Path, help="directory of memgate's files (needs --gate-ranks)")
    parser.add_argument('--gate-ranks', type=int, help='ranks that wait at the gate before a release')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        cap_file = os.environ.get('GPU_HOLD_CAP_FILE')
        hold_one(HEADROOM, Path(cap_file) if cap_file else None)
        return 0
    if not args.hip:
        parser.error('--hip is required')
    if (args.gate is None) != (args.gate_ranks is None):
        parser.error('--gate and --gate-ranks go together')
    cap = int(args.busy_free_cap * (1 << 30))
    hip = [int(x) for x in args.hip.split(',')]
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(True))
    gate = None
    if args.gate is not None:
        gate = Gate(args.gate, args.gate_ranks)
        gate.clear()
    cap_dir = args.gate or (args.state.parent if args.state else Path(os.environ.get('TMPDIR', '/tmp')))
    cap_file = cap_dir / f'gpu-hold-{os.getpid()}.cap'
    holder = Holder(hip, args.log, cap_file=cap_file)
    try:
        watch(holder, lambda exclude: own_gpu_users(set(hip), exclude | {os.getpid()}),
              lambda pids: sum(drm_vram(pid) for pid in pids),
              lambda: bool(stopping), POLL_SECONDS, args.log, args.ready, args.state, cap=cap, gate=gate)
    finally:
        cap_file.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
