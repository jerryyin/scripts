#!/usr/bin/env python3
"""Temporarily force MI350 MFMA source-A/B VGPR reads.

The ``read`` command is read-only.  The ``run`` command changes SQ_CONFIG1
bit 21 on every XCC, runs one command, and restores only that bit to its
original per-XCC state.  It also leaves a state file in this reproducer's
``build`` directory until restoration succeeds, so an interrupted run has an
explicit recovery path.
"""
import argparse
import fcntl
import json
import os
import pwd
import signal
import struct
import subprocess
import sys
import tempfile

REGS2_IOCTL_SET_STATE_V2 = 0xC02C2001
REGS2_STATE = struct.Struct("=11I")
U32 = struct.Struct("=I")
FEATURES = {
    "mfma-srcab-read-skip": (0x037A, 1 << 21),
    "mai-coexec": (0x0300, 1 << 5),
    "valu-coexec": (0x037A, 1 << 8),
}
XCC_COUNT = 8
GC_HW_ID = 11
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")


def regs_path(bdf):
    return f"/sys/kernel/debug/dri/{bdf}/amdgpu_regs2"


def select_xcc(fd, xcc):
    broadcast = 0xFFFFFFFF
    state = REGS2_STATE.pack(
        0, 1, 0, broadcast, broadcast, broadcast, 0, 0, 0, 0, xcc
    )
    fcntl.ioctl(fd, REGS2_IOCTL_SET_STATE_V2, state)


def register_offsets(bdf, register_dwords):
    discovery = f"/sys/bus/pci/devices/{bdf}/ip_discovery/die/0/{GC_HW_ID}"
    offsets = [None] * XCC_COUNT
    for entry in os.scandir(discovery):
        if not entry.is_dir():
            continue
        with open(os.path.join(entry.path, "num_instance"), encoding="utf-8") as stream:
            xcc = int(stream.read(), 0)
        if xcc >= XCC_COUNT:
            continue
        with open(os.path.join(entry.path, "base_addr"), encoding="utf-8") as stream:
            base_dwords = int(stream.read().split()[0], 16)
        offsets[xcc] = (base_dwords + register_dwords) * 4
    if any(offset is None for offset in offsets):
        raise RuntimeError("GC IP discovery did not provide all eight XCC bases")
    return offsets


def read_values(fd, offsets):
    values = []
    for xcc in range(XCC_COUNT):
        select_xcc(fd, xcc)
        data = os.pread(fd, U32.size, offsets[xcc])
        if len(data) != U32.size:
            raise OSError(f"short SQ_CONFIG1 read on XCC {xcc}")
        values.append(U32.unpack(data)[0])
    return values


def write_values(fd, offsets, values):
    for xcc, value in enumerate(values):
        select_xcc(fd, xcc)
        written = os.pwrite(fd, U32.pack(value), offsets[xcc])
        if written != U32.size:
            raise OSError(f"short SQ_CONFIG1 write on XCC {xcc}")


def print_values(values, feature):
    _, mask = FEATURES[feature]
    bit = mask.bit_length() - 1
    for xcc, value in enumerate(values):
        enabled = bool(value & mask)
        print(f"XCC {xcc}: value=0x{value:08x} bit{bit}={int(enabled)}",
              flush=True)


def restore_from_state(state_path, remove_state=True):
    with open(state_path, encoding="utf-8") as stream:
        state = json.load(stream)
    feature = state["feature"]
    register_dwords, mask = FEATURES[feature]
    offsets = register_offsets(state["bdf"], register_dwords)
    if state.get("offsets") != offsets:
        raise RuntimeError("state file has unexpected SQ_CONFIG1 offsets")
    if state.get("mask") != mask:
        raise RuntimeError("state file has an unexpected register mask")
    original = state["values"]
    if len(original) != XCC_COUNT:
        raise RuntimeError("state file does not contain all eight XCCs")
    path = regs_path(state["bdf"])
    fd = os.open(path, os.O_RDWR)
    try:
        current = read_values(fd, offsets)
        restored = [
            (now & ~mask) | (before & mask)
            for now, before in zip(current, original)
        ]
        write_values(fd, offsets, restored)
        verified = read_values(fd, offsets)
        if any(
            (now ^ before) & mask
            for now, before in zip(verified, original)
        ):
            raise RuntimeError(f"{feature} restoration did not verify")
    finally:
        os.close(fd)

    print(f"Restored original {feature} state:", file=sys.stderr)
    print_values(verified, feature)
    if remove_state:
        os.unlink(state_path)


class ForwardedSignal(Exception):
    def __init__(self, signum):
        self.signum = signum


def run_as_invoking_user(command):
    uid = int(os.environ.get("SUDO_UID", "0"))
    gid = int(os.environ.get("SUDO_GID", "0"))
    child = None

    def demote():
        if uid:
            groups = os.environ.get("SQ_CONFIG1_CALLER_GROUPS")
            if not groups:
                raise RuntimeError("SQ_CONFIG1_CALLER_GROUPS is required")
            os.setgroups([int(group) for group in groups.split(",")])
            os.setgid(gid)
            os.setuid(uid)

    env = os.environ.copy()
    if uid:
        account = pwd.getpwuid(uid)
        env.update(HOME=account.pw_dir, USER=account.pw_name,
                   LOGNAME=account.pw_name)
    try:
        child = subprocess.Popen(command, env=env, preexec_fn=demote)
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def command_read(args):
    register_dwords, _ = FEATURES[args.feature]
    offsets = register_offsets(args.bdf, register_dwords)
    fd = os.open(regs_path(args.bdf), os.O_RDONLY)
    try:
        print_values(read_values(fd, offsets), args.feature)
    finally:
        os.close(fd)
    return 0


def command_restore(args):
    if os.geteuid() != 0:
        raise PermissionError("restore requires sudo/root")
    restore_from_state(args.state_file)
    return 0


def command_run(args):
    if os.geteuid() != 0:
        raise PermissionError("run requires sudo/root")
    if not args.exclusive_gpu:
        raise RuntimeError("run requires --exclusive-gpu acknowledgement")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("no command supplied after --")
    path = regs_path(args.bdf)
    register_dwords, mask = FEATURES[args.feature]
    offsets = register_offsets(args.bdf, register_dwords)
    fd = os.open(path, os.O_RDWR)
    state_path = None
    old_handlers = {}
    try:
        original = read_values(fd, offsets)
        os.makedirs(STATE_DIR, exist_ok=True)
        state_fd, state_path = tempfile.mkstemp(
            prefix=f"sq-config-{args.feature}-{args.bdf.replace(':', '_')}-",
            suffix=".json",
            dir=STATE_DIR,
        )
        state = {
            "bdf": args.bdf,
            "feature": args.feature,
            "offsets": offsets,
            "mask": mask,
            "values": original,
        }
        with os.fdopen(state_fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())

        print(f"Recovery state: {state_path}", file=sys.stderr)
        print(f"Emergency restore: sudo {os.path.abspath(__file__)} restore "
              f"{state_path}", file=sys.stderr)

        forced = [value | mask for value in original]
        write_values(fd, offsets, forced)
        verified = read_values(fd, offsets)
        if any(not value & mask for value in verified):
            raise RuntimeError(f"{args.feature} set did not verify on every XCC")
        print(f"Enabled {args.feature} control:", file=sys.stderr)
        print_values(verified, args.feature)
        os.close(fd)
        fd = None

        def handle_signal(signum, _frame):
            raise ForwardedSignal(signum)

        for signum in (signal.SIGHUP, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handle_signal)
        try:
            return run_as_invoking_user(command)
        except KeyboardInterrupt:
            return 128 + signal.SIGINT
        except ForwardedSignal as error:
            return 128 + error.signum
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if fd is not None:
            os.close(fd)
        if state_path is not None:
            restore_from_state(state_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    read_parser = subparsers.add_parser("read", help="read all XCCs; no writes")
    read_parser.add_argument("--bdf", required=True, help="PCI BDF, e.g. 0000:66:00.0")
    read_parser.add_argument("--feature", choices=FEATURES,
                             default="mfma-srcab-read-skip")
    read_parser.set_defaults(function=command_read)

    run_parser = subparsers.add_parser("run", help="set bit, run command, restore bit")
    run_parser.add_argument("--bdf", required=True, help="PCI BDF")
    run_parser.add_argument("--feature", choices=FEATURES,
                            default="mfma-srcab-read-skip")
    run_parser.add_argument("--exclusive-gpu", action="store_true",
                            help="acknowledge that the whole physical GPU is reserved")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(function=command_run)

    restore_parser = subparsers.add_parser("restore", help="restore an interrupted run")
    restore_parser.add_argument("state_file")
    restore_parser.set_defaults(function=command_restore)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        return args.function(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
