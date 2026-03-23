//
// Standalone HIP harness for the asyncmark bug reproducer.
//
// Loads an HSACO, launches the matmul kernel, and compares
// against a host-computed reference.
//
// Build:  hipcc driver.cpp -o driver
// Usage:  ./driver <path.hsaco>
//
#include "hip/hip_runtime.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                    \
              hipGetErrorString(err), __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int M = 4096;
static constexpr int N = 4096;
static constexpr int K = 4096;
static constexpr int WORKGROUP_SIZE = 256;
static constexpr int NUM_WORKGROUPS = 1024;

static const char *KERNEL_NAME =
    "matmul_dispatch_0_matmul_4096x4096x4096_f32";

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <path.hsaco>\n", argv[0]);
    return 1;
  }

  const size_t elems = (size_t)M * N;
  const size_t bytes = elems * sizeof(float);

  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);
  float *h_ref = (float *)malloc(bytes);

  // Fill with varied float values derived from index
  printf("Filling inputs (%dx%d f32)...\n", M, K);
  for (size_t i = 0; i < elems; i++) {
    h_A[i] = sinf((float)(i % 1000) * 0.01f);
    h_B[i] = cosf((float)(i % 997) * 0.01f);
  }

  // Host reference matmul
  printf("Computing host reference...\n");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int p = 0; p < K; p++)
        sum += (double)h_A[i * K + p] * (double)h_B[p * N + j];
      h_ref[i * N + j] = (float)sum;
    }
  }

  // Device setup
  void *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));
  HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_C, 0, bytes));

  // Load HSACO and launch
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&function, module, KERNEL_NAME));

  struct { void *A; void *B; void *C; } args = { d_A, d_B, d_C };
  size_t arg_size = sizeof(args);
  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
    HIP_LAUNCH_PARAM_END
  };

  printf("Launching kernel (%d workgroups x %d threads)...\n",
         NUM_WORKGROUPS, WORKGROUP_SIZE);
  HIP_CHECK(hipModuleLaunchKernel(function,
                                  NUM_WORKGROUPS, 1, 1,
                                  WORKGROUP_SIZE, 1, 1,
                                  0, 0, NULL, (void **)&config));
  HIP_CHECK(hipDeviceSynchronize());

  // Compare
  HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));

  int wrong = 0;
  float max_diff = 0.0f;
  for (size_t i = 0; i < elems; i++) {
    float diff = fabsf(h_C[i] - h_ref[i]);
    if (diff > max_diff) max_diff = diff;
    if (diff > 1.0f) wrong++;
  }

  printf("\nResults:\n");
  printf("  max_abs_diff   = %f\n", max_diff);
  printf("  wrong elements = %d / %zu (%.1f%%)\n",
         wrong, elems, 100.0 * wrong / elems);
  printf("  verdict: %s\n", (wrong == 0) ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipModuleUnload(module));
  free(h_A); free(h_B); free(h_C); free(h_ref);

  return (wrong > 0) ? 1 : 0;
}
