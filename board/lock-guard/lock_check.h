// lock_check.h -- THE ONE PLACE THE OWNERSHIP QUESTION IS ASKED.
//
// The problem: on a GPU board shared by many tenants, device work must run inside a blocking
// acquisition of a shared lock so that runs queue instead of landing on top of each other. A
// written rule saying "always take the lock" is a rule somebody eventually forgets at 2am.
// This makes an unlocked device launch FAIL BY CONSTRUCTION instead.
//
// Two layers enforce it, and they must ask the IDENTICAL question, or the weaker one quietly
// becomes the real policy:
//
//   1. board_lock_guard.o    -- linked into every HIP binary this toolchain builds, via the
//      hipcc shim and `-Wl,--wrap`.  Binds at LINK time.  Covers everything built from now on.
//   2. board_lock_preload.so -- loaded into every process via /etc/ld.so.preload.  Binds at LOAD
//      time.  Covers binaries built BEFORE this guard existed, which may not be safe to delete
//      or rebuild (in the case this was written for, ~35 of them held archived results).
//
// Layer 1 alone would leave those older binaries able to launch unlocked -- the "depends on
// somebody remembering" hole this exists to close.  Layer 2 alone would be lost the moment a
// binary is copied to a host without the preload configured.
//
// ---------------------------------------------------------------------------------------------
// HOW OWNERSHIP IS DETERMINED, AND WHY NOT THE OBVIOUS WAY.
//
// The first version of this read /proc/locks, matched the lock file's device and inode, and
// checked the owning pid against our process ancestry.  THAT VERSION WAS WRONG, and only
// running the demonstration (demonstrate-guard.sh) exposed it.
//
// The common shell idiom for taking a lock is this:
//
//     exec 9>>"$LOCK_PATH"        # a descriptor that outlives any child
//     flock 9                     # blocking wait; the flock BINARY then EXITS
//
// The lock survives because it belongs to the open file description behind fd 9, not to the
// flock process.  But /proc/locks records the pid that CREATED the lock, that pid is gone, and
// the kernel hides entries whose owner is not visible in the current pid namespace.  Net effect:
// the lock is genuinely held -- a competing `flock -n` is refused -- and /proc/locks is EMPTY.
// A guard reading /proc/locks would therefore have refused every legitimate run.  It fails
// closed, so it was never a safety hole, but it would have made the guard useless, and the
// pressure would then have been to weaken it.
//
// So the question is put to the kernel directly, through descriptors, in three steps:
//
//   (1) IS THE LOCK HELD AT ALL?  Open a fresh descriptor on the lock file and try a
//       non-blocking exclusive flock.  If that SUCCEEDS, nobody held it -- release it at once
//       and refuse.  (Keeping it would be taking the board without queueing.)
//
//   (2) IF IT IS HELD, IS IT HELD BY US?  A descriptor opened by an ancestor is INHERITED across
//       fork and exec, so if a launcher up the chain holds the lock on fd 9, this process has a
//       descriptor on the same file.  Scan /proc/self/fd, fstat each one, and keep those whose
//       device and inode match the lock file's.
//
//   (3) WHICH INHERITED DESCRIPTOR, IF ANY, ACTUALLY HOLDS IT?  A non-blocking exclusive flock
//       on a descriptor whose OWN open file description already holds the lock returns success
//       immediately -- re-locking the same description is a no-op.  On a descriptor that does
//       not hold it, while somebody else does, it fails with EWOULDBLOCK.  Success on an
//       inherited descriptor, given step (1) proved the lock is held, means WE hold it.
//
// WHY IT ASKS ABOUT OWNERSHIP AND NOT MERE EXISTENCE.  "Is the lock held?" is the wrong question
// on a board with many tenants: if a NEIGHBOUR holds it the answer is yes, and we would sail
// through and land on top of their run -- the exact harm.  Steps (2) and (3) are what separate
// "held by us" from "held by someone else", and the demonstration covers that case explicitly
// rather than assuming it.
//
// IT FAILS CLOSED.  Missing lock file, unreadable /proc/self/fd, a probe that fails for any
// reason other than the lock being held: all refuse.  An unreadable probe does not get to look
// like a pass.
//
// IT HAS NO ESCAPE HATCH, DELIBERATELY.  No environment variable disables it and no flag
// downgrades it to a warning.  Adding one would recreate exactly the hole this closes, and the
// person most likely to reach for it is the person who needed the guard.

#pragma once

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// THE LOCK PATH IS A COMPILE-TIME CONSTANT ON PURPOSE.  Pass -DGPU_LOCK_FILE='"/path"' to
// build this guard for a site whose lock lives elsewhere (build-guard.sh forwards the
// GPU_LOCK_FILE environment variable for exactly that).  It is deliberately NOT read from
// the environment at run time: a guard that takes its lock path from getenv() can be
// bypassed by anyone who sets that variable to a file they can lock themselves, which is
// the escape hatch this file says three paragraphs above that it does not have.
#ifndef GPU_LOCK_FILE
#define GPU_LOCK_FILE "/data/lock/amd-gpu.lock"
#endif

namespace {

const char *kLockPath = GPU_LOCK_FILE;

void refuse(const char *what, const char *why) {
  fprintf(stderr,
          "\n"
          "================================================================================\n"
          "BOARD LOCK GUARD: REFUSING A DEVICE LAUNCH THAT IS NOT UNDER THE LOCK\n"
          "================================================================================\n"
          "  blocked call : %s\n"
          "  reason       : %s\n"
          "  lock file    : %s\n"
          "  pid          : %d\n"
          "\n"
          "This board is shared with other tenants.  Device work must run inside a\n"
          "blocking acquisition of the board lock, taken by a launcher that queues for it.\n"
          "Running a device binary by hand, or through a reproduction script's device phase,\n"
          "goes around that queue and can land on top of a neighbour who is holding it.\n"
          "\n"
          "There is no override.  Launch through a wrapper that takes the lock and holds\n"
          "it for the whole run, e.g.  flock <lockfile> <command>.\n"
          "================================================================================\n",
          what, why, kLockPath, (int)getpid());
  fflush(stderr);
  _exit(97);
}

// Returns true only if we can POSITIVELY establish that this process holds the board lock
// through a descriptor it owns or inherited.  Every other path returns false.
bool lock_held_by_us(const char **why) {
  struct stat lock_st;
  if (stat(kLockPath, &lock_st) != 0) {
    *why = "the board lock file could not be stat'd, so lock ownership is unknown; "
           "an unreadable probe is not a pass";
    return false;
  }

  // Step (1): is it held at all?  A fresh descriptor cannot be holding an inherited lock, so a
  // successful non-blocking lock here means the board was free.
  int probe = open(kLockPath, O_RDONLY | O_CLOEXEC);
  if (probe < 0) {
    *why = "the board lock file could not be opened for probing, so ownership is unknown; "
           "an unreadable probe is not a pass";
    return false;
  }
  if (flock(probe, LOCK_EX | LOCK_NB) == 0) {
    // Nobody held it.  Release immediately: keeping it would be taking the board without
    // queueing, which is the behaviour this guard exists to prevent.
    flock(probe, LOCK_UN);
    close(probe);
    *why = "the board lock is not held by anyone, so this launch never queued for the board "
           "and is not protected by the lock at all";
    return false;
  }
  if (errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR) {
    close(probe);
    *why = "probing the board lock failed for a reason other than the lock being held, so "
           "ownership is unknown; an unreadable probe is not a pass";
    return false;
  }
  close(probe);

  // Step (2): which of our descriptors, inherited or otherwise, refer to the lock file?
  DIR *d = opendir("/proc/self/fd");
  if (!d) {
    *why = "/proc/self/fd could not be read, so it cannot be established whether the held lock "
           "is ours; an unreadable probe is not a pass";
    return false;
  }

  bool ours = false;
  const int dir_fd = dirfd(d);
  struct dirent *e;
  while (!ours && (e = readdir(d)) != nullptr) {
    if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
    const int fd = atoi(e->d_name);
    if (fd == dir_fd) continue;

    struct stat fd_st;
    if (fstat(fd, &fd_st) != 0) continue;
    if (fd_st.st_dev != lock_st.st_dev || fd_st.st_ino != lock_st.st_ino) continue;

    // Step (3): re-locking a description that already holds the lock succeeds immediately.
    // Step (1) established that SOMEBODY holds it, so success here means that somebody is us.
    if (flock(fd, LOCK_EX | LOCK_NB) == 0) ours = true;
  }
  closedir(d);

  if (ours) return true;
  *why = "the board lock is held, but by a process that is not us and whose descriptor we did "
         "not inherit -- that is a NEIGHBOUR holding it, and running now would land on top of "
         "them";
  return false;
}

void check_or_die(const char *what) {
  // Checked once per process.  The first device call pays for the probe; the rest are free.
  static int decided = 0;  // 0 = not yet, 1 = permitted
  if (decided == 1) return;
  const char *why = "unknown";
  if (!lock_held_by_us(&why)) refuse(what, why);
  decided = 1;
}

}  // namespace
