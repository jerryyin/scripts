// LD_PRELOAD shim: refuse (EACCES) and log every open of /dev/kfd or /dev/dri/*, so CPU tests never open a GPU.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int gpu_path(const char *p) { return p && (!strncmp(p, "/dev/kfd", 8) || !strncmp(p, "/dev/dri", 8)); }
static void note(const char *p) {
  const char *log = getenv("KFD_GUARD_LOG");
  if (!log) return;
  FILE *f = fopen(log, "a");
  if (f) { fprintf(f, "BLOCKED open %s pid=%d\n", p, getpid()); fclose(f); }
}
#define WRAP(name, ...) \
  if (gpu_path(path)) { note(path); errno = EACCES; return -1; }
int open(const char *path, int flags, ...) {
  static int (*real)(const char *, int, ...) = 0; if (!real) real = dlsym(RTLD_NEXT, "open");
  WRAP(open); va_list a; va_start(a, flags); int m = va_arg(a, int); va_end(a); return real(path, flags, m); }
int open64(const char *path, int flags, ...) {
  static int (*real)(const char *, int, ...) = 0; if (!real) real = dlsym(RTLD_NEXT, "open64");
  WRAP(open64); va_list a; va_start(a, flags); int m = va_arg(a, int); va_end(a); return real(path, flags, m); }
int openat(int d, const char *path, int flags, ...) {
  static int (*real)(int, const char *, int, ...) = 0; if (!real) real = dlsym(RTLD_NEXT, "openat");
  WRAP(openat); va_list a; va_start(a, flags); int m = va_arg(a, int); va_end(a); return real(d, path, flags, m); }
int openat64(int d, const char *path, int flags, ...) {
  static int (*real)(int, const char *, int, ...) = 0; if (!real) real = dlsym(RTLD_NEXT, "openat64");
  WRAP(openat64); va_list a; va_start(a, flags); int m = va_arg(a, int); va_end(a); return real(d, path, flags, m); }
