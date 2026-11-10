# board/ — exclusive access to, and health of, a shared GPU board

Everything to do with **getting a GPU board to yourself and keeping it in a state worth
measuring on**. If a script answers "may I touch the device right now?", "who else is
touching it?", or "is this board tuned?", it belongs here.

```
board/
├── README.md              # this file
├── gpu-lock               # run a command under an exclusive flock on the shared lock file
├── gpu-contention.sh      # who else holds /dev/kfd right now, from the HOST namespaces
├── gpu-wait.py            # wait until N GPUs have no process on them; or snapshot who is on them
├── test_gpu_wait.py       # its tests (no GPU needed)
├── lock-guard/            # make an UNLOCKED device launch fail by construction
│   └── README.md          #   the two layers, and the /proc/locks trap
└── bringup/
    └── heliosr-b0/        # post-boot tuning for one specific board
        └── README.md      #   what it does and why it is not portable
```

## Files

- `gpu-wait.py N` — polls `amd-smi process --json` until N GPUs have had no process on
  `--confirm` consecutive polls, prints their indices and exits 0 (1 on `--timeout`). Logs each
  poll with the busy GPUs' PIDs (host namespace; identify owners with `gpu-contention.sh` on the
  host). Never signals anything. For shared machines without a lock, where measurements must
  wait for idle GPUs rather than share them. `gpu-wait.py --snapshot [--gpus ...]` prints the
  current process list (PID and memory per GPU) as JSON; it is the one amd-smi parser here, used
  by `agent/unattended/measurement_record.py`.
  While another process holds the host-wide lock (`gpu-lock`'s file, `--host-lock`) every GPU
  counts as busy. On a host with an amd-tracking checkout, `--amd-tracking DIR [--lease-dir DIR]`
  takes ownership from amd-tracking's `apps/shared/gpu_admission.py` (KFD queue and VRAM owners,
  per-GPU leases) instead of amd-smi; inside amd-tracking runs, its `manage_server.py start` makes
  that check itself (with `--host-lock` for this lock). Tests: `test_gpu_wait.py`.
- `gpu-lock` — wraps a command in an exclusive `flock` on a shared lock file so only one
  wrapped application runs at a time. Cooperative, not a security boundary: anything
  launched without the wrapper bypasses it. That gap is what `lock-guard/` closes.
- `gpu-contention.sh` — reports how many processes hold `/dev/kfd`, split into *ours* and
  *foreign*. `--selftest` drives every branch against synthetic `/proc` fixtures and needs
  no GPU, no docker and no privileges.

## The contention probe must run in the HOST pid namespace

`gpu-contention.sh` is host-only in live mode, and this is a correctness requirement rather
than a convenience. Running it through `docker exec` makes **both halves of its answer
unsound**:

- a container's PID namespace cannot see neighbouring `/dev/kfd` holders at all, so the
  foreign count silently reads zero — the most dangerous possible wrong answer;
- container-relative cgroup paths do not contain the container ID, so the ownership test
  cannot classify what little it does see.

The script proves it is in a different PID namespace from the target container before it
reports any count, and refuses otherwise. Do not "fix" that check by relaxing it.

Ownership is decided by a positive property a neighbour cannot exhibit: either the holder's
cgroup names our container, **or** it is in *all* of the container's namespaces. Requiring
all of them matters — a container on host networking shares its network namespace with
every process on the box, so matching on that alone would call every tenant "ours" and the
probe would never refuse again. Anything unreadable counts as foreign.

## Configuration

| Variable | Used by | Default |
| --- | --- | --- |
| `GPU_CONTENTION_CONTAINER` | `gpu-contention.sh` | none — pass the container as argument 1 or set this |
| `GPU_LOCK_FILE` | `lock-guard/` build, `demonstrate-guard.sh` | `/data/lock/amd-gpu.lock` |
