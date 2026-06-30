// Standalone HIP driver for the AMDGPU ds_read_b64 / s_waitcnt miscompile.
//
// NO Triton dependency. Loads a precompiled HSACO of the _attn_fwd kernel
// (built by reproduce.sh from ir/attn_fwd.ll) and launches it with the exact
// grid/block/args captured from the originating Triton run — all embedded
// below, so the harness is fully self-contained.
//
// Verification: run the kernel N times with identical inputs; run 0 is the
// reference and each later run is compared element-wise. The -O3 build's output
// varies run to run (the race). reproduce.sh runs this on both the -O0 and -O3
// builds; the rigorous opt-level contrast is the codegen diff (see README).
//
//   build:  hipcc -O2 driver.cpp -o driver
//   usage:  ./driver <kernel.hsaco> [nruns]   (exit 2 if output is
//   nondeterministic)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

// ---- Embedded launch descriptor (captured from the real Triton launch) ------
// Kernel symbol, launch geometry, and the 30 scalar arguments. The 5 I/O
// pointers (args 0-4) and 2 null scratch pointers (args 35-36) are supplied at
// launch time; everything else is fixed. Values are strides/dims plus the
// softmax scale (1/sqrt(32)) and dropout p (0); the bug is input-independent,
// so they only need to be self-consistent with the buffer layout.
static const char *KERNEL =
    "_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_32_NUM_K_HEADS_4_BLOCK_M_128_BLOCK_N_32_"
    "BLOCK_DMODEL_16_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_0_NUM_"
    "XCD_"
    "8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0";

static const unsigned GRID[3] = {2048, 1, 1};
static const unsigned BLOCK[3] = {256, 1, 1}; // 4 warps x 64 lanes
static const unsigned SHARED_BYTES = 17472;

// Scalar kernel args 5..34, passed as 32-bit words (the two f32s — softmax
// scale 0x3e3504f3 = 1/sqrt(32), and dropout p 0.0 — are stored as their bit
// patterns).
static const uint32_t SCALAR_ARGS[30] = {
    0x00200000u, 0x00000020u, 0x00000400u, 0x00040000u, 0x00000020u,
    0x00000080u, 0x00020000u, 0x00000010u, 0x00000040u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00100000u, 0x00000010u, 0x00000200u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00010000u, 0x00000800u, 0x3e3504f3u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000800u, 0x00000800u, 0x00000004u};

static const int N_PTR_IO = 5; // args 0..4: q, k, v, out, lse  (we allocate)
static const int N_ARGS = 37;  // total kernel arguments
// Per-buffer size; generous so the kernel's strided accesses stay in bounds.
static const size_t BUF_BYTES = 128ull * 1024 * 1024;

#define CHK(x)                                                                 \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", e, hipGetErrorString(e), \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Reinterpret a bf16 bit pattern as float (bf16 = high 16 bits of an f32).
static float bf16_to_f32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <kernel.hsaco> [nruns]\n", argv[0]);
    return 1;
  }
  const char *hsaco_path = argv[1];
  int nruns = (argc >= 3) ? atoi(argv[2]) : 10;

  // --- allocate the 5 I/O buffers and fill q,k,v with fixed pseudo-random bf16
  void *dbuf[N_PTR_IO];
  for (int i = 0; i < N_PTR_IO; i++)
    CHK(hipMalloc(&dbuf[i], BUF_BYTES));
  std::vector<uint16_t> host(BUF_BYTES / 2);
  uint32_t seed = 12345; // fixed => inputs identical every run
  for (size_t j = 0; j < host.size(); j++) {
    seed = seed * 1664525u + 1013904223u;
    float f =
        ((float)((seed >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 2.0f; // ~[-1,1]
    uint32_t b;
    memcpy(&b, &f, 4);
    host[j] = (uint16_t)(b >> 16); // bf16
  }
  for (int i = 0; i < 3; i++) // args 0,1,2 = q,k,v are inputs
    CHK(hipMemcpy(dbuf[i], host.data(), BUF_BYTES, hipMemcpyHostToDevice));

  // --- load the HSACO and look up the kernel by its embedded symbol name
  std::vector<char> code;
  {
    FILE *f = fopen(hsaco_path, "rb");
    if (!f) {
      fprintf(stderr, "cannot open %s\n", hsaco_path);
      return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    code.resize(sz);
    if (fread(code.data(), 1, sz, f) != (size_t)sz) {
      fprintf(stderr, "short read on %s\n", hsaco_path);
      return 1;
    }
    fclose(f);
  }
  hipModule_t mod;
  CHK(hipModuleLoadData(&mod, code.data()));
  hipFunction_t fn;
  CHK(hipModuleGetFunction(&fn, mod, KERNEL));

  // --- build the kernel argument array: pointers to each arg's value
  void *nullp = nullptr;
  void *params[N_ARGS];
  for (int i = 0; i < N_ARGS; i++) {
    if (i < N_PTR_IO)
      params[i] = &dbuf[i]; // q, k, v, out, lse
    else if (i >= N_ARGS - 2)
      params[i] = &nullp; // args 35,36: null scratch pointers
    else
      params[i] = (void *)&SCALAR_ARGS[i - N_PTR_IO]; // args 5..34: scalars
  }

  // --- run nruns times; run 0 is the reference, compare the primary output
  size_t N = BUF_BYTES / 2;
  std::vector<uint16_t> out(N), ref(N);
  double worst = 0.0;
  for (int r = 0; r < nruns; r++) {
    CHK(hipMemset(dbuf[3], 0, BUF_BYTES)); // clear outputs before each run
    CHK(hipMemset(dbuf[4], 0, BUF_BYTES));
    CHK(hipModuleLaunchKernel(fn, GRID[0], GRID[1], GRID[2], BLOCK[0], BLOCK[1],
                              BLOCK[2], SHARED_BYTES, 0, params, nullptr));
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(out.data(), dbuf[3], BUF_BYTES, hipMemcpyDeviceToHost));
    if (r == 0) {
      ref = out;
      printf("  run 0: reference captured\n");
      continue;
    }
    double md = 0.0;
    long nbad = 0;
    for (size_t j = 0; j < N; j++) {
      float a = bf16_to_f32(out[j]), b = bf16_to_f32(ref[j]);
      if (std::isfinite(a) && std::isfinite(b)) {
        double d = fabs((double)a - b);
        if (d > md)
          md = d;
        if (d > 0.01)
          nbad++;
      }
    }
    if (md > worst)
      worst = md;
    printf("  run %d vs run0: max_abs_diff=%.5f  elems>0.01=%ld\n", r, md,
           nbad);
  }
  printf("RESULT: worst max_abs_diff across runs = %.5f (tolerance 0.01)\n",
         worst);
  for (int i = 0; i < N_PTR_IO; i++)
    CHK(hipFree(dbuf[i]));
  return worst > 0.01 ? 2 : 0;
}
