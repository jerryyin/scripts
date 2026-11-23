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
refused, a failing set fails the run).

`compile_check.py compile --ts-src TREE --kernel KERNEL` compiles one TokenSpeed MoE kernel for
gfx950 without a GPU (a stub Triton driver answers the target queries, nothing is launched) and
prints compile time, VGPRs, LDS bytes and instruction count (`--asm-out` writes the AMDGCN). The
kernels are Gluon Petit's `stage1`, `stage2`, `combine` and its opt-in variants `stage1_n128`
(`TOKENSPEED_PETIT_STAGE1_N128`), `stage2_n128` (`TOKENSPEED_PETIT_STAGE2_N128`),
`combine_residual` (`TOKENSPEED_PETIT_FUSE_POST`) and `trace_ping` (`TOKENSPEED_PETIT_TRACE=1`),
and, not Petit, `a4w4_ep_finalize`, the A4W4 EP registration's one-launch finalize
(`TOKENSPEED_A4W4_EP_FUSED_FINALIZE`). TokenSpeed's switches in the environment apply as at a
server's import. The Petit kernels need a TokenSpeed whose `tokenspeed_kernel.ops.moe.gluon` has
`_megamoe_config` (46671ec2c and later on `users/jerryyin/ep4-beats-tp4`; upstream main has none,
and the tool says so and exits). `compile_check.py compare A.s B.s` is generic: it compares the
instruction lines of any two AMDGCN listings (no directives, labels, comments or debug lines), e.g.
to show a switch-off kernel is unchanged between two trees, and exits 1 when they differ. Run
`compile` under `board/no-gpu/no-gpu-run` too; `test_compile_check.py` tests the comparison, the
usage against the kernel list and the TokenSpeed requirement message.

```bash
board/no-gpu/no-gpu-run -- python3 tokenspeed/cpu-tests/compile_check.py compile --ts-src "$TREE" \
    --kernel stage2 --asm-out stage2.s
```
