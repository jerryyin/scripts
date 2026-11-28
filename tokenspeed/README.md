# TokenSpeed

Work that depends on TokenSpeed itself: installing it, driving and measuring its server, and
instrumenting its source. Generic pieces live in the cross-project subjects (`profiling/`,
`board/`, `agent/unattended/`); see the placement rule in the repository README.

| Path | Contents |
| --- | --- |
| `install.sh` | Install TokenSpeed's packages from a checkout into the system Python |
| `tokenspeed_triton_alias.py` | Import alias for TokenSpeed's Triton package |
| `tests-job.sh` | Run TokenSpeed test files as a GPU job (inside `board/leased-job.sh`): pytest per file from `<worktree>/tokenspeed-kernel` on the first leased GPU, or under `torchrun` on the first N on a loopback rendezvous (`--torchrun N`), `--select EXPR` (pytest `-k`), or only the collected test IDs (`--collect-only`, to compare a refactor's tests at base and head); one `results.txt` line per file with its pytest summary and the `tokenspeed`/`tokenspeed_kernel` files the same environment imports; refreshes the lease's idle grace while a file runs. Modes, exits and logs in its header. Test: `test_tests_job.sh` (no GPU) |
| [serving](serving/README.md) | Measure a running TokenSpeed server: fixed-batch decode, prefill, sustained multi-turn load, end-to-end decomposition, greedy-output comparison, server lifecycle |
| [petit-layer](petit-layer/README.md) | Time the Gluon Petit MoE layer (DeepSeek V4.1 Flash, EP4 with Petit's token exchange) alone on 4 (or 8) GPUs under `torchrun`, eager and CUDA graph, checked against TokenSpeed's distributed-test reference; Petit's own launch choice or forced producer-geometry and stage-1-tile candidates |
| [step-timing](step-timing/README.md) | Test-only instrumentation that times each component of a decode step inside the captured graph |
