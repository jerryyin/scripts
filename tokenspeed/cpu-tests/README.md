# tokenspeed/cpu-tests/ — TokenSpeed's tests without a GPU

`run_cpu_tests.sh TREE SWITCH_SET [SWITCH_SET...]` runs TokenSpeed's Gluon Petit owner tests that need no
device on one source tree (a checkout or a `git archive` export with patches applied), once per
switch set, while other users hold the GPUs:

- each pytest runs under `board/no-gpu/no-gpu-run` (every `/dev/kfd` and `/dev/dri/*` open refused
  and logged), with `PYTHONPATH` pointing at the tree's `tokenspeed-kernel/python`,
  `tokenspeed-kernel-amd/python` and `python`;
- `mi350_platform_plugin.py` (`-p mi350_platform_plugin`) overrides `tokenspeed_kernel`'s platform
  with the owner fixture `mi350_platform` of `tokenspeed-kernel/test/conftest.py`, evaluated from
  that file before any conftest imports `tokenspeed_kernel`, so the tests import without a device;
- a switch set is one argument of space-separated `VAR=VALUE` pairs (`OFF` = none), at least one
  set; `run_cpu_tests.sh TREE OFF` is the minimal run;
- per set: `tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit.py` and
  `test_gluon_petit_variants.py` (extra arguments in `CPU_TESTS_KERNEL_ARGS`), then
  `test/runtime/test_gluon_petit_backend.py`; once, switches off,
  `test/runtime/test_deepseek_v4_config.py`. One summary line each; exit 1 if any failed.

```bash
tokenspeed/cpu-tests/run_cpu_tests.sh "$TREE" OFF "TOKENSPEED_PETIT_TRACE=1"
```

`test_run_cpu_tests.py` runs it on a fake tree (platform taken from the fixture, device open
refused, a failing set fails the run). `tokenspeed/petit-layer/petit_compile_check.py` compiles
Petit kernels offline; run it under `board/no-gpu/no-gpu-run` too.
