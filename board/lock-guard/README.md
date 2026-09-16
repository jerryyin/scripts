# board/lock-guard/ — make an unlocked device launch fail by construction

`../gpu-lock` is cooperative: anything launched without the wrapper bypasses it. This closes
that gap. A HIP binary that reaches a device call without holding the shared board lock
**dies there, with exit 97**, before the call reaches the runtime.

The motivating incident: several hundred kernel executions ran on a shared board without the
lock, because a reproduction script was invoked with a subcommand that reaches its device
phase. Nobody intended to skip the lock and nobody noticed until afterwards. A line in a
document saying "do not use that subcommand" is the repair that depends on remembering, so
it is not the repair.

```
lock-guard/
├── README.md               # this file
├── lock_check.h            # THE ownership question, asked in exactly one place
├── board_lock_guard.cpp    # link-time layer: -Wl,--wrap on HIP entry points
├── board_lock_preload.cpp  # load-time layer: interposers for /etc/ld.so.preload
├── hipcc-shim.sh           # replaces hipcc so every build links the guard in
├── build-guard.sh          # build both halves; refuses to leave a half-built one
├── install-guard.sh        # install the shim and the preload, staged and reversible
└── demonstrate-guard.sh    # six cases showing it refuse AND permit (needs a GPU)
```

## Two layers, one question

1. **`board_lock_guard.o`** is linked into every HIP binary the toolchain builds, via the
   hipcc shim and `-Wl,--wrap`. Binds at **link** time; covers everything built from now on.
2. **`board_lock_preload.so`** is loaded into every process via `/etc/ld.so.preload`. Binds
   at **load** time; covers binaries built *before* the guard existed, which may not be safe
   to delete or rebuild.

Layer 1 alone leaves old binaries able to launch unlocked. Layer 2 alone is lost the moment a
binary is copied to a host without the preload configured. Both import the ownership test
from `lock_check.h` rather than reimplementing it, so the weaker one cannot quietly become
the effective policy.

## The `/proc/locks` trap — read this before "simplifying" the ownership check

The obvious implementation reads `/proc/locks`, matches the lock file's device and inode, and
checks the owning pid against the process ancestry. **That implementation is wrong**, and it
only became visible by running the demonstration.

`/proc/locks` records the pid that **CREATED** the lock. The common shell idiom is:

```bash
exec 9>>"$LOCK_PATH"   # a descriptor that outlives any child
flock 9                # blocking wait; the flock BINARY then EXITS
```

The lock survives because it belongs to the **open file description** behind fd 9, not to the
`flock` process. But that process is gone, and the kernel hides entries whose owner is not
visible in the current pid namespace. Net effect: the lock is genuinely held — a competing
`flock -n` is refused — and **`/proc/locks` reads empty**. A guard reading it would have
refused every legitimate run. It fails closed, so it was never a safety hole, but it would
have made the guard useless, and the pressure would then have been to weaken it.

So ownership is established through **descriptors**, in three steps:

1. **Is the lock held at all?** Open a fresh descriptor and try a non-blocking exclusive
   `flock`. If that *succeeds*, nobody held it — release it immediately and refuse. (Keeping
   it would be taking the board without queueing.)
2. **Is it held by us?** A descriptor opened by an ancestor is inherited across fork and
   exec, so scan `/proc/self/fd` and keep those whose device and inode match the lock file.
3. **Which inherited descriptor holds it?** A non-blocking `flock` on a descriptor whose own
   open file description already holds the lock returns success immediately — re-locking the
   same description is a no-op. On one that does not hold it, while somebody else does, it
   fails with `EWOULDBLOCK`. Given step 1, success here means *we* hold it.

**Ownership, not existence.** "Is the lock held?" is the wrong question on a shared board: if
a neighbour holds it the answer is yes, and sailing through is the exact harm.

**It fails closed and has no escape hatch.** Missing lock file, unreadable `/proc/self/fd`, a
probe that fails for any other reason — all refuse. No environment variable disables it and
no flag downgrades it to a warning; the person most likely to reach for one is the person who
needed the guard.

## Usage

```bash
GPU_LOCK_FILE=/data/lock/amd-gpu.lock ./build-guard.sh   # compile; installs nothing
sudo ./install-guard.sh                                   # shim + /etc/ld.so.preload
./demonstrate-guard.sh                                    # prove it (needs a GPU)
```

`build-guard.sh` and `install-guard.sh` are separate on purpose: writing `/etc/ld.so.preload`
can take the whole container down and must never happen as a side effect of a rebuild.
Installation is staged — the library is tested under an explicit `LD_PRELOAD` first, then the
file is written, then an external binary is run as a canary, and on failure the file is
truncated **using shell redirection alone**, because if the library were bad no external
command could be executed to remove it.

To undo by hand, with builtins only: `: > /etc/ld.so.preload`

## The lock path is compile-time, deliberately

`GPU_LOCK_FILE` is read by `build-guard.sh` and baked in with `-DGPU_LOCK_FILE`. It is **not**
read from the environment at run time: a guard that takes its lock path from `getenv()` can be
bypassed by anyone who points it at a file they can lock themselves, which is exactly the
escape hatch this is supposed not to have.

## Demonstrating it

`demonstrate-guard.sh` runs six cases, and the ones that must come back **permitted** matter
as much as the refusals — a demonstration whose every expected answer is "refused" cannot
tell a working guard from a binary that is simply broken.

| # | Case | Expected |
| --- | --- | --- |
| 1 | newly built binary, no lock | REFUSE (link-time layer) |
| 2 | pre-guard binary, no lock | REFUSE (load-time layer) |
| 3 | pre-guard binary, no device call | PROCEED (it gates device access, not process start) |
| 4 | lock held by a non-ancestor | REFUSE (ownership, not existence) |
| 5 | lock held on an inherited fd | PERMIT (the idiom above; the case v1 got wrong) |
| 6 | lock held by a live parent | PERMIT (`flock <file> <cmd>`, covered not assumed) |

Cases 5 and 6 each make one 1 KiB device allocation and free it, so **a GPU is required**.
Cases 2 and 3 need a HIP binary built *before* the guard existed — that is the only thing that
can exercise the load-time layer, since anything from the guarded toolchain would refuse via
the link-time layer and prove nothing about the other one. Without one they are **skipped**
rather than silently passing:

```bash
PREGUARD=/path/to/old-binary \
PREGUARD_DEVICE_ARGS='--code x.hsaco --args ./args --runs 1' \
PREGUARD_NODEVICE_ARGS='--check-manifest --args ./args' \
./demonstrate-guard.sh
```

## Configuration

| Variable | Used by | Default |
| --- | --- | --- |
| `GPU_LOCK_FILE` | `build-guard.sh`, `demonstrate-guard.sh` | `/data/lock/amd-gpu.lock` |
| `HIPCC` | `install-guard.sh`, `demonstrate-guard.sh` | `/opt/venv/bin/hipcc` |
| `REAL_HIPCC` | shim, build, install | `<HIPCC>.real` |
| `GUARD_DIR` | `hipcc-shim.sh` | the shim's own directory |
| `PRELOAD_FILE` | `install-guard.sh` | `/etc/ld.so.preload` |
| `DEMO_DIR`, `PROBE`, `PREGUARD` | `demonstrate-guard.sh` | see above |
