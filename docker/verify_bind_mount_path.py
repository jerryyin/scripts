"""Prove that a path on the host and a path inside a container name the same bytes.

THE PROBLEM. Any workflow where a process on the host writes a file and a process inside
a container reads it (or vice versa) depends on the two paths resolving through a bind
mount to the same storage. Getting that wrong is quiet and expensive: both sides succeed
at everything they do individually, and the mismatch surfaces only when the reader looks
for a file that was really written somewhere else.

The specific trap this was built after: a job configuration handed out ONE literal path,
/tmp/<name>, as both the host directory and the container directory. /tmp is not shared
-- inside a container it lives on the container's own overlay root. The writer wrote on
the host, the reader looked in the container's private /tmp, found nothing, and refused.
Nothing about either path looked wrong; they were identical, absolute, and both existed.

WHY STRING COMPARISON CANNOT SETTLE IT. Two path strings that are EQUAL are not evidence
they name the same bytes (the case above). Two that DIFFER are not evidence they do
either. The only thing that settles it is the mount table, so this module reads
/proc/self/mountinfo and refuses on the mapping rather than on the strings. It also
rejects roots that sit on a filesystem which cannot be shared with the host at all --
overlay, tmpfs and friends -- since a path there is the /tmp defect wearing another name.

Needs no privileges and no running container: it reads /proc/self/mountinfo and does
string algebra. No docker command, no network, no device.

WHAT THIS CANNOT SEE -- A REAL LIMITATION, NOT A CAVEAT. mountinfo field 4 is the path of
the mount's root WITHIN ITS SOURCE FILESYSTEM, which equals the host's path only when
that filesystem is mounted at / on the host. That is true in the common case (the bind
source is on the host root disk) and it is NOT guaranteed. If a host mounted the source
disk somewhere else, field 4 would be correct about the filesystem and wrong about the
host path, and this module would have no way to know.

That is exactly why `resolve_shared_roots` takes the caller's DECLARED host/container
pair and requires the measurement to agree with it, rather than deriving the host path
and trusting it. Either one drifting alone is caught; neither is trusted on its own. If
you need certainty beyond this, cross-check with `docker inspect` on the host.

USAGE
    python3 verify_bind_mount_path.py --selftest       # fixtures, runs anywhere
    python3 verify_bind_mount_path.py <host> <container>   # check a declared pair here
    from verify_bind_mount_path import resolve_shared_roots, read_mounts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

MOUNTINFO_PATH = Path("/proc/self/mountinfo")

# Filesystems that exist inside the container but are not a window onto host storage.
# A declared root on one of these is exactly the /tmp defect wearing a different name.
NON_SHAREABLE_FSTYPES = frozenset({
    "overlay", "tmpfs", "devtmpfs", "proc", "sysfs", "devpts", "mqueue",
    "cgroup", "cgroup2", "securityfs", "pstore", "bpf", "tracefs", "debugfs",
    "configfs", "fusectl", "hugetlbfs", "ramfs", "autofs", "binfmt_misc",
})


class SharedPathError(RuntimeError):
    """The two declared roots do not provably name the same bytes. Never downgraded."""


@dataclass(frozen=True)
class BindMount:
    """One row of /proc/self/mountinfo, reduced to what the mapping question needs."""

    source_path: PurePosixPath   # field 4: the mount's root within its source filesystem
    mount_point: PurePosixPath   # field 5: where the container sees it
    device: str                  # field 3: major:minor
    fstype: str                  # the field after the " - " separator

    def shareable(self, root_device: str) -> bool:
        """Is this a window onto storage the board host also sees?

        The container's own root filesystem is an overlay assembled from image layers;
        anything on it -- /tmp included -- exists only inside the container even though
        the path looks perfectly ordinary from in here. Sharing a device id with / is
        therefore the discriminating test, and the fstype list is a second, independent
        reason to reject the same cases.
        """
        return (
            str(self.mount_point) != "/"
            and self.device != root_device
            and self.fstype not in NON_SHAREABLE_FSTYPES
        )

    def to_container(self, source: PurePosixPath) -> PurePosixPath:
        return self.mount_point / source.relative_to(self.source_path)

    def to_source(self, container: PurePosixPath) -> PurePosixPath:
        return self.source_path / container.relative_to(self.mount_point)


def parse_mountinfo(text: str) -> list[BindMount]:
    """Parse mountinfo. Optional fields sit between field 6 and the ' - ' separator."""
    mounts: list[BindMount] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            head, tail = line.split(" - ", 1)
        except ValueError as error:
            raise SharedPathError(f"mountinfo line has no ' - ' separator: {line!r}") from error
        head_fields = head.split()
        tail_fields = tail.split()
        if len(head_fields) < 5 or not tail_fields:
            raise SharedPathError(f"mountinfo line is too short to use: {line!r}")
        mounts.append(BindMount(
            source_path=PurePosixPath(head_fields[3]),
            mount_point=PurePosixPath(head_fields[4]),
            device=head_fields[2],
            fstype=tail_fields[0],
        ))
    if not mounts:
        raise SharedPathError("mountinfo is empty; the mapping cannot be established")
    return mounts


def root_device(mounts: list[BindMount]) -> str:
    for mount in mounts:
        if str(mount.mount_point) == "/":
            return mount.device
    raise SharedPathError("mountinfo declares no root mount; the mapping cannot be established")


def read_mounts(path: Path = MOUNTINFO_PATH) -> list[BindMount]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SharedPathError(f"cannot read the mount table at {path}: {error}") from error
    return parse_mountinfo(text)


def shareable_mounts(mounts: list[BindMount]) -> list[BindMount]:
    """Longest mount point first, so a nested bind wins over its parent."""
    device = root_device(mounts)
    found = [mount for mount in mounts if mount.shareable(device)]
    return sorted(found, key=lambda mount: len(str(mount.mount_point)), reverse=True)


def resolve_shared_roots(
    host_root: str, container_root: str, mounts: list[BindMount]
) -> BindMount:
    """Refuse unless the two declared roots provably name the same bytes.

    Returns the bind mount that carries them, so the caller can record WHICH mount it
    relied on rather than merely that some mount existed.

    Equality of the two strings is checked first and separately. It is a necessary
    condition and nowhere near a sufficient one -- two different wrong paths would sail
    past a non-equality assertion, which is why the mount lookup below is the real test
    and this one is only here to name the identical-path defect explicitly when it recurs.
    """
    if not host_root or not container_root:
        raise SharedPathError("both declared roots must be declared; one of them is empty")
    host = PurePosixPath(host_root)
    container = PurePosixPath(container_root)
    if not host.is_absolute() or not container.is_absolute():
        raise SharedPathError(f"declared roots must be absolute: {host_root!r}, {container_root!r}")
    if host == container:
        raise SharedPathError(
            f"the host and container declared roots are the same literal path ({host_root}). "
            "That is one name for two different filesystems unless the container binds it "
            "at the identical location. This is the identical-path defect verbatim."
        )

    candidates = shareable_mounts(mounts)
    if not candidates:
        raise SharedPathError(
            "this container has no mount that is a window onto host storage, so no shared "
            "directory here can be shared with the board host"
        )
    for mount in candidates:
        try:
            expected_host = mount.to_source(container)
        except ValueError:
            continue
        if expected_host == host:
            return mount
        raise SharedPathError(
            f"container root {container_root} is carried by the bind "
            f"{mount.source_path} -> {mount.mount_point}, which puts it at {expected_host} on "
            f"the board host, but the contract declares {host_root}. The two roots name "
            "different bytes."
        )

    visible = ", ".join(f"{m.source_path} -> {m.mount_point}" for m in candidates)
    raise SharedPathError(
        f"container root {container_root} is not under any host-shared mount; this container "
        f"shares only [{visible}]. A file written here is invisible to the board host, "
        "which is the identical-path defect in another form."
    )


def describe(mount: BindMount) -> str:
    return f"{mount.source_path} -> {mount.mount_point} ({mount.fstype}, dev {mount.device})"


# ---------------------------------------------------------------------------------------
# Self-test. Drives the real predicate against synthetic mount tables, so the mapping is
# exercised with no container, no docker and no privileges. It runs anywhere.
#
# The design constraint worth keeping in view: asserting that the two roots are merely
# UNEQUAL is not enough, because two different WRONG paths would sail past that. The
# TMPFS_TABLE case below is exactly that, and it must still be refused.
# ---------------------------------------------------------------------------------------

# A container with two real bind mounts from the host root disk. /tmp is deliberately
# absent, because on a real container it is absent too: it lives on the overlay root,
# which is the whole reason a /tmp path silently fails to be shared.
_BOUND_TABLE = """\
425 388 0:51 / / rw,relatime - overlay overlay rw,lowerdir=/x,upperdir=/y,workdir=/z
434 425 259:4 /data /data rw,relatime - ext4 /dev/nvme0n1p2 rw
435 425 259:4 /home/alice /work rw,relatime - ext4 /dev/nvme0n1p2 rw
"""

# A container that DOES have host-shared storage and also has /tmp as its own tmpfs. The
# honest near-miss: shared storage exists, so a refusal cannot come from "nothing here is
# shared" -- it has to come from the roots not being on the part that IS shared.
_TMPFS_TABLE = """\
425 388 0:51 / / rw,relatime - overlay overlay rw
434 425 259:4 /data /data rw,relatime - ext4 /dev/nvme0n1p2 rw
440 425 0:77 / /tmp rw,relatime - tmpfs tmpfs rw
"""


def _refuses(host: str, container: str, table: str, because: str) -> str:
    try:
        resolve_shared_roots(host, container, parse_mountinfo(table))
    except SharedPathError as error:
        return str(error)
    raise AssertionError(f"accepted a pair it must refuse ({because}): {host} / {container}")


def _self_test() -> int:
    failures = 0

    def ok(message: str) -> None:
        print(f"  ok  {message}")

    def case(name: str, body: Callable[[], None]) -> None:
        nonlocal failures
        try:
            body()
        except AssertionError as error:
            failures += 1
            print(f"  FAIL {name}: {error}")

    def accepts_a_real_bind_pair() -> None:
        mount = resolve_shared_roots(
            "/home/alice/project/out", "/work/project/out", parse_mountinfo(_BOUND_TABLE)
        )
        assert str(mount.mount_point) == "/work", f"resolved through the wrong mount: {describe(mount)}"
        ok(f"a genuinely bind-mounted pair resolves, through {describe(mount)}")

    def refuses_the_identical_path_defect() -> None:
        identical = "/tmp/shared-scratch"
        message = _refuses(identical, identical, _BOUND_TABLE, "identical /tmp paths")
        assert "same literal path" in message, f"refused for the wrong reason: {message}"
        ok("the identical-path defect is refused, and named as itself")

    def refuses_two_different_private_paths() -> None:
        # Unequal, absolute, and still wrong: neither is on shared storage. This is the
        # case that proves the check is a mapping test rather than a string test.
        message = _refuses(
            "/tmp/host-side", "/tmp/container-side", _TMPFS_TABLE, "two unequal private paths"
        )
        assert "not under any host-shared mount" in message, f"wrong reason: {message}"
        ok("two DIFFERENT paths on container-private storage are refused, not just equal ones")

    def refuses_a_mismapped_pair() -> None:
        message = _refuses(
            "/home/bob/project/out",
            "/work/project/out",
            _BOUND_TABLE,
            "host root does not map onto the container root",
        )
        assert "name different bytes" in message, f"wrong reason: {message}"
        ok("a real shared mount with a mismatched host name is refused")

    def refuses_a_relative_root() -> None:
        message = _refuses("relative/path", "/work/x", _BOUND_TABLE, "a relative root")
        assert "must be absolute" in message, f"wrong reason: {message}"
        ok("a non-absolute root is refused")

    def reads_the_live_table() -> None:
        # Informational, not an assertion about this machine: the fixtures above are what
        # pin the behaviour. All this proves is that the real parser survives the real
        # /proc/self/mountinfo, which is worth knowing and is true on hosts and containers
        # alike. A machine with no host-shared mount is a normal answer here, not a failure.
        mounts = read_mounts()
        shared = shareable_mounts(mounts)
        summary = ", ".join(describe(m) for m in shared) or "<none>"
        ok(f"live mount table parsed: {len(mounts)} mounts, host-shareable = [{summary}]")

    print("verify_bind_mount_path --selftest")
    case("accepts a real bind pair", accepts_a_real_bind_pair)
    case("refuses the identical-path defect", refuses_the_identical_path_defect)
    case("refuses two different private paths", refuses_two_different_private_paths)
    case("refuses a mismapped pair", refuses_a_mismapped_pair)
    case("refuses a relative root", refuses_a_relative_root)
    case("reads the live table", reads_the_live_table)

    if failures:
        print(f"selftest FAILED: {failures}")
        return 1
    print("selftest PASS")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 1 and argv[0] == "--selftest":
        return _self_test()
    if len(argv) == 2:
        try:
            mount = resolve_shared_roots(argv[0], argv[1], read_mounts())
        except SharedPathError as error:
            print(f"REFUSED: {error}")
            return 1
        print(f"OK: the two roots name the same bytes, through {describe(mount)}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
