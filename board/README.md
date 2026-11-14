# board/ — exclusive access to, and health of, a shared GPU board

Everything to do with **getting a GPU board to yourself and keeping it in a state worth
measuring on**. If a script answers "may I touch the device right now?", "who else is
touching it?", or "is this board tuned?", it belongs here.

```
board/
├── README.md              # this file
├── gpu-lock               # run a command under an exclusive flock on the shared lock file
├── gpu-contention.sh      # who else holds /dev/kfd right now, from the HOST namespaces
├── gpu-wait.py            # wait until N GPUs are idle; or snapshot who is on them
├── test_gpu_wait.py       # its tests (no GPU needed)
├── gpu-queue              # claim GPUs once on a shared host, run queued jobs back to back
├── test_gpu_queue.py      # its tests (no GPU or docker needed)
├── gpu-hold.py            # gpu-queue's memory holder
├── test_gpu_hold.py       # its tests (no GPU needed)
├── lock-guard/            # make an UNLOCKED device launch fail by construction
│   └── README.md          #   the two layers, and the /proc/locks trap
└── bringup/
    └── heliosr-b0/        # post-boot tuning for one specific board
        └── README.md      #   what it does and why it is not portable
```

## Files

- `gpu-wait.py N` — polls `amd-smi metric` until N GPUs show 0% graphics activity on
  `--confirm` consecutive polls, prints their indices and exits 0 (1 on `--timeout`). Memory
  other processes hold does not count; on a pick it logs who holds memory there (host PIDs,
  from KFD, also inside a container). Never signals anything. `--idle-by process` waits for
  no process in `amd-smi process` instead. `--snapshot` prints amd-smi's process list as JSON
  (used by `agent/unattended/measurement_record.py`). A held host-wide lock (`--host-lock`)
  makes every GPU busy; `--hip` prints HIP ids (see below). Tests: `test_gpu_wait.py`.
- `gpu-queue` — a job queue for a shared host with no scheduler, where GPUs a job releases are
  taken by other users' queues before anyone can start the next job, and idle GPUs reserve
  nothing because other launches do not check. `gpu-queue serve DIR --count N --container NAME`
  runs on the host and holds no GPU while the queue is empty. When a job is pending it waits for
  N idle GPUs (`gpu-wait.py N --hip`), then runs every pending job on those GPUs
  oldest first, each starting the moment the previous one exits, including jobs queued while
  earlier ones run; it releases when the queue is empty. A local serve holds the claimed
  GPUs' free memory from the claim until the queue is empty; while one of our processes starts on
  them it gives that process the memory but keeps each GPU's free memory capped until it signals it
  is ready (`GPU_QUEUE_READY`), so another tenant cannot take them between jobs or during a load.
  `--reclaim-after SECONDS` (host serve with `--container` only, and only where every GPU process
  is yours): when a pending job has waited that long, `serve` stops every `/dev/kfd` holder on the
  host outside `--container` (SIGTERM, SIGKILL after `--reclaim-grace`, through `sudo -n` unless
  root), logs each, and waits again. `gpu-queue add DIR [--name NAME] [--count N] -- COMMAND` (or
  `--file SCRIPT`) and `gpu-queue status DIR` run where DIR is local, normally inside the container.
  One serve runs per DIR (it holds `DIR/serve.lock`), and `add --count N` starts one for N GPUs
  when none runs; it exits when the queue empties. Serves of different DIRs do not see each other's
  claims, so share one DIR per host. Jobs get `HIP_VISIBLE_DEVICES` and the serve's environment,
  not the caller's, so a job script sets up its own directory and environment; output and exit
  status land in `DIR/done/` or `DIR/failed/`. Tests: `test_gpu_queue.py`.

  The simplest setup runs everything inside the container: amd-smi there sees every GPU, so
  the idle check and HIP numbering work, and the log lands in `DIR/serve.log` where `status`
  reads it. A container cannot see other containers' processes, so `--reclaim-after` needs a
  serve on the host.

  ```bash
  # inside the container (no reclaim); starts a serve for 4 GPUs unless one runs:
  board/gpu-queue add /path/queue --count 4 --name numerics -- 'cd /work && ./run-numerics.sh'
  board/gpu-queue status /path/queue   # waits per job, the serve, its last line, STALE if none
  # or on the host, once (survives logout), with reclaim:
  nohup setsid board/gpu-queue serve /path/in/container/queue --count 4 --container <name> \
      [--reclaim-after 1800] > /dev/null 2>&1 < /dev/null &
  tail -f ~/.local/state/gpu-queue/<name>.log
  # inside the container, any time:
  board/gpu-queue add /path/in/container/queue --name numerics -- 'cd /work && ./run-numerics.sh'
  board/gpu-queue status /path/in/container/queue
  ```

  To keep the GPUs across a sequence of experiments, queue the next job before the current one
  finishes; a job queued after the queue empties waits for a fresh claim.
- `gpu-lock` — wraps a command in an exclusive `flock` on a shared lock file so only one
  wrapped application runs at a time. Cooperative, not a security boundary: anything
  launched without the wrapper bypasses it. That gap is what `lock-guard/` closes.
- `gpu-contention.sh` — reports how many processes hold `/dev/kfd`, split into *ours* and
  *foreign*. `--selftest` drives every branch against synthetic `/proc` fixtures and needs
  no GPU, no docker and no privileges.

## The MI350X host: amd-smi's process list and its numbering are both wrong for this

- `amd-smi process` there lists every process on every GPU (all eight show the same thirteen
  PIDs, including one 246 GB holder), so `--idle-by process` never sees a free GPU while anyone
  runs anything. The default reads activity from `amd-smi metric` and memory holders from KFD's
  `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`.
- HIP enumerates GPUs in KFD topology order (`/sys/class/kfd/kfd/topology/nodes`, CPU nodes
  skipped). There HIP 0..7 are amd-smi 3, 0, 2, 1, 7, 4, 6, 5: different for every device, but the
  same *sets* {0-3} and {4-7}, which hides the mismatch until a job asks for a subset. Passing
  amd-smi indices to `HIP_VISIBLE_DEVICES` puts the job on someone else's GPUs. `--hip` maps by PCI
  bus address; checked against torch's `pci_bus_id` for all eight devices.

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
