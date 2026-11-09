# TokenSpeed

Work that depends on TokenSpeed itself: installing it, driving and measuring its server, and
instrumenting its source. Generic pieces live in the cross-project subjects (`profiling/`,
`board/`, `agent/unattended/`); see the placement rule in the repository README.

| Path | Contents |
| --- | --- |
| `install.sh` | Install TokenSpeed's packages from a checkout into the system Python |
| `tokenspeed_triton_alias.py` | Import alias for TokenSpeed's Triton package |
| [serving](serving/README.md) | Measure a running TokenSpeed server: fixed-batch decode, prefill, sustained multi-turn load, end-to-end decomposition, greedy-output comparison, server lifecycle |
| [step-timing](step-timing/README.md) | Test-only instrumentation that times each component of a decode step inside the captured graph |
