# am/ — the AMD "AM" / FFM simulators

Running kernels on the **AM** and **FFM** instruction-level simulators instead of hardware,
and reducing what comes back. AM produces instruction traces (itrace); FFM is the faster
functional model used for correctness work.

```
am/
├── README.md                      # this file
├── run_on_model.sh                # the canonical launcher: `--backend am|ffm`
├── test_run_on_model_profiles.py  # pytest: profile/topology resolution, no simulator needed
├── ffm_teardown.py                # pytest plugin working around the FFM exit hang
├── parse_am_itrace.py             # AM itrace -> per-opcode-class cycle attribution (JSON)
└── compare_att_itrace.py          # put that attribution beside hardware ATT's, per category
```

## Files

- `run_on_model.sh` — sets up the simulator package, topology and environment, then execs
  your command. Everything simulator-related should go through it rather than reproducing
  its environment by hand.

  ```bash
  run_on_model.sh --backend ffm -- python3 kernel.py
  LD_PRELOAD= GPU_ARCHS=gfx1250 run_on_model.sh --backend am -- python3 kernel.py
  ```

- `ffm_teardown.py` — FFM hangs on normal interpreter exit (192 threads stuck in
  `futex_wait_queue`). This plugin calls `hipDeviceReset` and `os._exit` after the session.
  `run_on_model.sh` loads it for you by putting this directory on `PYTHONPATH` and passing
  `-p ffm_teardown`. **In your own drivers, call `os._exit()` rather than returning from
  `main()`** — the hang is not specific to pytest.

- `parse_am_itrace.py` — turns an AM trace into per-opcode-class cycle attribution. The
  trace is a merged timeline of every wave on one WGP, so lines from different waves
  interleave freely and nothing may be parsed as a block.

- `compare_att_itrace.py` — puts AM's attribution next to hardware ATT's for the same
  kernel. This is the check a total-time comparison cannot make: a simulator can rank a set
  of configurations correctly and still be right by accident. The question worth asking is
  whether it loses cycles for the same *reasons*.

## `stall` and `span` are not the same number

`parse_am_itrace.py` emits two cycle accounts and they answer different questions:

- **`stall` is exact.** Each `TYPE_DEP` line is one cycle that one instruction spent blocked,
  tagged with the dependency that blocked it. Summing them attributes waiting to a named
  cause.
- **`span` is the gap to the next issue in the same wave** — what ItraceViz draws, and what
  its own documentation warns about. It is *not* the instruction's execution time. A
  long-latency load whose wave then stalls shows a **small** span, because the wait lands on
  whichever instruction is blocked, usually the `s_wait_loadcnt`. Read spans as occupancy of
  the issue slot, not as cost.

On the hardware side, ATT's `latency` is the analogue of `span`, and its `stall` the
analogue of `stall`. The stall columns are the ones that say *why* time went missing.

## One taxonomy, checked

Both reducers import `categorize` from `../profiling/att_analyze.py` rather than restating
it, and record its SHA-256 in their output. A comparison run with two different decoders
would be measuring the taxonomy rather than the machine, so a hash mismatch is refused
unless explicitly acknowledged. Override the location with `ATT_ANALYZE` or `SCRIPTS_ROOT`
if you are running these from outside the repo.

## Scope when comparing

ATT sees the waves on **one SIMD**; an AM summary covers **every wave of the workgroup**.
Where the waves are near enough symmetric the shares stay comparable, but they are not the
same population, and a difference of a few percent should not be read as a finding.

## Tests

```bash
cd am && python3 -m pytest test_run_on_model_profiles.py -q   # no simulator required
```
