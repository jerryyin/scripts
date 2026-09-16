// board_lock_guard.cpp -- make an unlocked device launch FAIL BY CONSTRUCTION.
//
// WHY THIS EXISTS.  On a board shared with many tenants, device work is supposed to run only
// inside a blocking acquisition of the shared board lock.  The incident that produced this
// guard: up to 480 kernel executions ran on such a board WITHOUT the lock, because a
// reproduction script was invoked with a subcommand that reaches its device phase -- nobody
// intended to skip the lock and nobody noticed until afterwards.  The bar for the repair is
// that a device launch which is not under the lock must FAIL, by construction, without
// depending on anybody remembering anything.  A line in a document saying "do not use that
// subcommand" is the repair that depends on remembering, so it is not the repair.
//
// WHAT IT DOES.  This object is linked into every HIP binary the toolchain can produce (see
// the hipcc shim beside it), and the linker is told to divert a handful of HIP entry points
// through the wrappers below.  The first time the process tries to actually touch the device,
// the guard asks one question: DOES THIS PROCESS, OR ANY OF ITS ANCESTORS, HOLD THE BOARD
// LOCK?  If not, the process dies there, before the call reaches the runtime.
//
// WHY IT ASKS ABOUT OWNERSHIP AND NOT EXISTENCE.  "Is the lock held?" is the wrong question on
// a shared board: if a NEIGHBOUR holds it, the answer is yes and we would sail through while
// running on top of them -- which is the exact harm.  HOW ownership is established is in
// lock_check.h and is not restated here, deliberately: an earlier version of this comment
// described a /proc/locks-and-ancestry scheme that the header had already replaced, because
// /proc/locks reads EMPTY for the ordinary `exec 9>>lock; flock 9` idiom.  One description,
// in one place, beside the code that implements it.
//
// WHY IT GUARDS THE DEVICE CALLS AND NOT PROCESS START.  A constructor firing at startup would
// also kill pre-lock work that touches no device -- for instance a driver's `--check-manifest`
// verification, which deliberately runs BEFORE the lock is taken.  Guarding the HIP entry
// points instead means the honest pre-lock work still runs and only real device access is
// gated.  That is a narrower guard, not a weaker one.
//
// IT FAILS CLOSED.  If the lock file is missing, if the probe cannot be run, or if ownership
// cannot be established for any reason, the guard REFUSES.  An unreadable probe does not get
// to look like a pass.
//
// IT HAS NO ESCAPE HATCH, DELIBERATELY.  There is no environment variable that disables it and
// no flag that downgrades it to a warning.  Adding one would recreate exactly the hole this
// closes, and the person most likely to use it is the person who needed the guard, at 2am.

#include <hip/hip_runtime_api.h>


#include "lock_check.h"

// The ownership check used to live here in full.  It now lives in lock_check.h so that this
// link-time layer and the load-time preload layer cannot drift into asking different questions;
// see the header for why there are two layers at all.  Nothing about the check changed when it
// moved.

// The wrapped entry points.  These are the calls the replay driver must make before anything
// reaches the hardware: allocate, load the code object, copy input up, launch.  `--wrap` is a
// link-time diversion, so a binary built through the shim cannot call the real symbol without
// passing through here first.
extern "C" {

hipError_t __real_hipMalloc(void **ptr, size_t size);
hipError_t __wrap_hipMalloc(void **ptr, size_t size) {
  check_or_die("hipMalloc");
  return __real_hipMalloc(ptr, size);
}

hipError_t __real_hipModuleLoad(hipModule_t *module, const char *fname);
hipError_t __wrap_hipModuleLoad(hipModule_t *module, const char *fname) {
  check_or_die("hipModuleLoad");
  return __real_hipModuleLoad(module, fname);
}

hipError_t __real_hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t __wrap_hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
  check_or_die("hipMemcpy");
  return __real_hipMemcpy(dst, src, sizeBytes, kind);
}

hipError_t __real_hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz,
                                        unsigned bx, unsigned by, unsigned bz,
                                        unsigned sharedMemBytes, hipStream_t stream,
                                        void **kernelParams, void **extra);
hipError_t __wrap_hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz,
                                        unsigned bx, unsigned by, unsigned bz,
                                        unsigned sharedMemBytes, hipStream_t stream,
                                        void **kernelParams, void **extra) {
  check_or_die("hipModuleLaunchKernel");
  return __real_hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedMemBytes, stream,
                                      kernelParams, extra);
}

hipError_t __real_hipDeviceSynchronize(void);
hipError_t __wrap_hipDeviceSynchronize(void) {
  check_or_die("hipDeviceSynchronize");
  return __real_hipDeviceSynchronize();
}

}  // extern "C"
