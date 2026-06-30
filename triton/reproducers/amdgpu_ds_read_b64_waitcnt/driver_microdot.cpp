#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <hip/hip_runtime.h>

#define CHK(x)                                                                 \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", e, hipGetErrorString(e), \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static float bf16_to_f32(uint16_t h) {
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <kernel.hsaco> [runs] [grid]\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  int runs = argc > 2 ? std::atoi(argv[2]) : 20;
  unsigned grid = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 2048;

  std::vector<char> code;
  FILE *f = std::fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    return 1;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  code.resize(size);
  if (std::fread(code.data(), 1, size, f) != static_cast<size_t>(size)) {
    fprintf(stderr, "short read on %s\n", path);
    return 1;
  }
  std::fclose(f);

  hipModule_t mod;
  hipFunction_t fn;
  CHK(hipModuleLoadData(&mod, code.data()));
  CHK(hipModuleGetFunction(&fn, mod, "repro"));

  constexpr size_t elems = 4096;
  uint16_t *dOut = nullptr;
  CHK(hipMalloc(&dOut, elems * sizeof(uint16_t)));
  float seed = 1.0f;
  void *args[] = {&dOut, &seed};

  std::vector<uint16_t> out(elems), ref(elems);
  double worst = 0.0;
  uint64_t totalBad = 0;
  printf("grid=%u block=256 runs=%d\n", grid, runs);
  for (int r = 0; r < runs; ++r) {
    CHK(hipMemset(dOut, 0, elems * sizeof(uint16_t)));
    CHK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 8192, 0, args,
                              nullptr));
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(out.data(), dOut, elems * sizeof(uint16_t),
                  hipMemcpyDeviceToHost));

    if (r == 0) {
      ref = out;
      printf("run 0: reference captured\n");
      continue;
    }

    double maxDiff = 0.0;
    uint64_t bad = 0;
    size_t first = static_cast<size_t>(-1);
    for (size_t i = 0; i < elems; ++i) {
      float a = bf16_to_f32(out[i]);
      float b = bf16_to_f32(ref[i]);
      double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
      if (d > maxDiff)
        maxDiff = d;
      if (out[i] != ref[i]) {
        if (first == static_cast<size_t>(-1))
          first = i;
        ++bad;
      }
    }
    if (maxDiff > worst)
      worst = maxDiff;
    totalBad += bad;
    printf("run %d: max_abs_diff=%.8f changed=%llu", r, maxDiff,
           static_cast<unsigned long long>(bad));
    if (bad)
      printf(" first={idx=%zu got=0x%04x ref=0x%04x}", first, out[first],
             ref[first]);
    printf("\n");
  }

  CHK(hipFree(dOut));
  CHK(hipModuleUnload(mod));
  printf("RESULT worst=%.8f total_changed=%llu\n", worst,
         static_cast<unsigned long long>(totalBad));
  return totalBad == 0 ? 0 : 2;
}
