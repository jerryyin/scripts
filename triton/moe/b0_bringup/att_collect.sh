#!/bin/bash
# Collect the four B0 ATT traces (a8w4 gluon + triton moe_gfx1250, decode + prefill),
# decoded to ui_output/ + stats_ui_output_*.csv, under $OUT (default /zyin). This is a
# thin orchestrator over the generic tools/prof.sh ATT wrapper.
#
# a8w4 is driven by the shared launcher ../run_a8w4_gemm1.py with --iters
# (loops the GEMM1 so the single-CU ATT target captures it, and it exits normally on
# hardware so rocprofv3 decodes -- see README.md "never os._exit").
#
# Requires version-pinned env (see README.md): triton @ e2a04beae6 + aiter with the
# old async_gather. Payloads (moe_decode.pt / moe_prefill.pt) come from
# ../precompute_routing.py and are looked up in $OUT.
set -u
export AITER_HOME="${AITER_HOME:-/root/aiter}"
export GPU_ARCHS="${GPU_ARCHS:-gfx1250}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/zyin}"
# Derive prof.sh from THIS script's location (b0_bringup -> moe -> triton -> scripts ->
# tools), not $HOME -- in the container $HOME=/root has a stale scripts clone.
PROF="${PROF:-$(cd "$HERE/../../../tools" && pwd)/prof.sh}"
A8W4="$HERE/../run_a8w4_gemm1.py"         # shared a8w4 GEMM1 launcher
MOE="${MOE:-/root/triton/third_party/amd/python/examples/gluon/moe_gfx1250.py}"
ITERS="${ITERS:-50}"
cd /tmp

run() {  # $1=name  $2=kernel-regex  $3..=command
  local name="$1" kregex="$2"; shift 2
  echo "######## ATT: $name (kernel=$kregex) ########  $(date)"
  ATT_KERNEL_REGEX="$kregex" timeout 1500 bash "$PROF" att "$@"
  echo "PROF_RC=$? ($name)"
  rm -rf "$OUT/att_$name"
  mv /zyin/rocprof_att "$OUT/att_$name" 2>/dev/null
  local ui; ui=$(find "$OUT/att_$name" -type d -name 'ui_output*' 2>/dev/null | wc -l)
  echo "== $name -> $(du -sh "$OUT/att_$name" 2>/dev/null | cut -f1)  ui_output=$ui"
}

# Capture ONLY the GEMM kernel of each side (drops surrounding pytorch/helper kernels).
# Three a8w4 variants share one config: gluon (default gfx1250), triton (force), and
# the upstream moe_gfx1250 reference. Kernels: gluon=_moe_gemm_a8w4_{decode,prefill},
# triton=_moe_gemm_a8w4, moe_gfx1250=_matmul_swiglu_fn.
run a8w4_gluon_decode   "_moe_gemm_a8w4.*"  python "$A8W4" --backend gluon --data "$OUT/moe_decode.pt"  --iters "$ITERS"
run a8w4_gluon_prefill  "_moe_gemm_a8w4.*"  python "$A8W4" --backend gluon --data "$OUT/moe_prefill.pt" --iters "$ITERS"
# a8w4 TRITON (aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py). Requires aiter
# with the force-triton env gate (branch users/jerryyin/moe-a8w4-force-triton-env), i.e.
#   use_gluon = get_arch()=="gfx1250" and os.environ.get("AITER_FORCE_TRITON","0")!="1"
# else gfx1250 always dispatches to gluon. run_a8w4_gemm1.py --backend triton sets the env.
run a8w4_triton_decode  "_moe_gemm_a8w4"    python "$A8W4" --backend triton --data "$OUT/moe_decode.pt"  --iters "$ITERS"
run a8w4_triton_prefill "_moe_gemm_a8w4"    python "$A8W4" --backend triton --data "$OUT/moe_prefill.pt" --iters "$ITERS"
run moe_gfx1250_decode  "_matmul_swiglu_fn" python "$MOE" --action dispatch --x_dtype fp8 --w_dtype mx4 --dim1 2048 --dim2 7168 -et 256 -ea 8 -b 4
run moe_gfx1250_prefill "_matmul_swiglu_fn" python "$MOE" --action dispatch --x_dtype fp8 --w_dtype mx4 --dim1 2048 --dim2 7168 -et 256 -ea 8 -b 512

echo "######## ALL DONE $(date) ########"
ls -d "$OUT"/att_* 2>/dev/null
