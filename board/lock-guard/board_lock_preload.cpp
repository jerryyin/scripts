// board_lock_preload.cpp -- the LOAD-TIME half of the unlocked-launch guard.
//
// WHY A SECOND LAYER EXISTS.  The link-time guard beside this file covers every HIP binary the
// toolchain builds from now on.  It does not cover the binaries that already exist: in the case
// this was written for, roughly thirty-five `replay` executables built before the guard, sitting
// in directories of archived results that were not safe to delete or rebuild.  Any of them can
// still be run by hand, and if one were, the link-time guard would not be in it.  That is exactly
// the hole this closes: otherwise the rule depends on somebody remembering.
//
// This object is listed in /etc/ld.so.preload, so the dynamic loader maps it into EVERY process
// on the container before any other library.  A pre-guard binary that calls hipMalloc reaches
// this interposer first and dies there, with no cooperation from the binary required.
//
// WHY IT IS SAFE TO PUT IN ld.so.preload, WHICH IS OTHERWISE A LOADED GUN.  A broken entry there
// breaks every process on the host at once, so: this library links against nothing but libdl and
// libc, opens no device, allocates nothing at load time, and defines NO constructor.  It has no
// effect whatsoever on a process that never calls a HIP entry point -- which is every process
// here except the HIP drivers themselves.  The real symbol is resolved lazily with dlsym(RTLD_NEXT) at
// the moment of the first call, not at load.  Installation is staged and reversible with shell
// builtins alone, because if this file were bad, no external command could be run to remove it.
//
// The ownership question is NOT re-implemented here.  It comes from lock_check.h, the same
// header the link-time guard uses, so the two layers cannot drift apart and leave the weaker one
// as the effective policy.

#include "lock_check.h"

#include <dlfcn.h>

namespace {

// Resolve the real implementation lazily.  Doing this at load time would mean dlopen-ing the HIP
// runtime inside every process on the box, including ones that will never touch the device.
void *real(const char *name) {
  void *sym = dlsym(RTLD_NEXT, name);
  if (!sym) {
    fprintf(stderr,
            "BOARD LOCK GUARD: could not resolve the real %s; refusing rather than guessing.\n",
            name);
    fflush(stderr);
    _exit(96);
  }
  return sym;
}

}  // namespace

// hipError_t is an enum and returns in the integer ABI slot, so `int` is the correct return type
// here and it lets this file compile with the host compiler alone, without the HIP headers.
extern "C" {

int hipMalloc(void **ptr, size_t size) {
  check_or_die("hipMalloc (via ld.so.preload)");
  using fn_t = int (*)(void **, size_t);
  return reinterpret_cast<fn_t>(real("hipMalloc"))(ptr, size);
}

int hipModuleLoad(void *module, const char *fname) {
  check_or_die("hipModuleLoad (via ld.so.preload)");
  using fn_t = int (*)(void *, const char *);
  return reinterpret_cast<fn_t>(real("hipModuleLoad"))(module, fname);
}

int hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind) {
  check_or_die("hipMemcpy (via ld.so.preload)");
  using fn_t = int (*)(void *, const void *, size_t, int);
  return reinterpret_cast<fn_t>(real("hipMemcpy"))(dst, src, sizeBytes, kind);
}

int hipModuleLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                          unsigned by, unsigned bz, unsigned sharedMemBytes, void *stream,
                          void **kernelParams, void **extra) {
  check_or_die("hipModuleLaunchKernel (via ld.so.preload)");
  using fn_t = int (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                       unsigned, void *, void **, void **);
  return reinterpret_cast<fn_t>(real("hipModuleLaunchKernel"))(
      f, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, kernelParams, extra);
}

int hipDeviceSynchronize(void) {
  check_or_die("hipDeviceSynchronize (via ld.so.preload)");
  using fn_t = int (*)(void);
  return reinterpret_cast<fn_t>(real("hipDeviceSynchronize"))();
}

}  // extern "C"
