// LD_PRELOAD shim: captures the exact hipModuleLaunchKernel launch (grid/block/
// shared + 37 kernel args) for the _attn_fwd kernel (uniquely identified by
// sharedMemBytes == 17472). Triton resolves HIP via hipGetProcAddress into a
// function table, so we wrap hipGetProcAddress and swap in our capturing launch.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define SHMEM_TAG 17472u
#define NARGS 37
// arg types: 0=ptr(8B) 1=i32(4B) 2=float(4B)  (per the kernel signature)
static const int kArgType[NARGS] = {
  0,0,0,0,0,                 // 5 I/O pointers
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 23 i32
  2,2,                       // 2 float
  1,1,1,1,1,                 // 5 i32
  0,0                        // 2 scratch pointers
};

typedef int (*launch_fn)(void* f, unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz,
                         unsigned shmem, void* stream,
                         void** kernelParams, void** extra);
static launch_fn real_launch = NULL;
static int captured = 0;

static int capturing_launch(void* f, unsigned gx, unsigned gy, unsigned gz,
                            unsigned bx, unsigned by, unsigned bz,
                            unsigned shmem, void* stream,
                            void** kernelParams, void** extra) {
  if (!captured && shmem == SHMEM_TAG && kernelParams) {
    FILE* fp = fopen("/tmp/launch_capture.bin", "wb");
    uint32_t hdr[8] = {0xCA97u, gx, gy, gz, bx, by, bz, shmem};
    fwrite(hdr, sizeof(uint32_t), 8, fp);
    uint32_t n = NARGS; fwrite(&n, 4, 1, fp);
    for (int i = 0; i < NARGS; i++) {
      uint8_t slot[8] = {0};
      int sz = (kArgType[i] == 0) ? 8 : 4;
      memcpy(slot, kernelParams[i], sz);
      uint32_t t = kArgType[i];
      fwrite(&t, 4, 1, fp);
      fwrite(slot, 8, 1, fp);
    }
    fclose(fp);
    captured = 1;
    fprintf(stderr, "[capture] dumped launch grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%u\n",
            gx, gy, gz, bx, by, bz, shmem);
  }
  return real_launch(f, gx, gy, gz, bx, by, bz, shmem, stream, kernelParams, extra);
}

typedef int (*gpa_fn)(const char*, void**, int, uint64_t, void*);
int hipGetProcAddress(const char* symbol, void** pfn, int v, uint64_t flags, void* st) {
  static gpa_fn real_gpa = NULL;
  if (!real_gpa) real_gpa = (gpa_fn)dlsym(RTLD_NEXT, "hipGetProcAddress");
  int r = real_gpa(symbol, pfn, v, flags, st);
  if (symbol && strcmp(symbol, "hipModuleLaunchKernel") == 0 && pfn && *pfn) {
    real_launch = (launch_fn)*pfn;
    *pfn = (void*)capturing_launch;
  }
  return r;
}
