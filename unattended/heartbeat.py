"""Status file for a long unattended run, written so a stale field cannot pass as a fresh one.

A status file that someone checks while a job runs overnight is only useful if a reader can
trust it without verifying it by hand. The failure this is built against is specific: a
status file that went on asserting work in flight that had already finished -- a download
that had completed, a self-test with no process behind it. A wrong answer that LOOKS like a
current answer has to be independently disproved before anything can be done with it, which
makes it worse than no file at all. This one is structured so that cannot happen.

Import it and call `write()` from a long-running script, or run it directly for a manual
write. Set HEARTBEAT_FILE to choose the path (default ~/HEARTBEAT.json).

THREE PROPERTIES, and they are requirements rather than style:

  1. EVERY FIELD CARRIES THE TIME IT WAS ESTABLISHED, and the file carries the time it was
     written. Fields of different vintage read AS different vintages. `age_seconds` is
     computed at write time so a reader does not have to subtract.

  2. NO FIELD SAYS SOMETHING IS RUNNING UNLESS THAT WAS CHECKED IN THIS WRITE. A claim of
     running is not a value here; it is a `Process` whose liveness is re-read from /proc at
     every write. A pid that has exited becomes `finished`, with the time it was last seen
     alive. If liveness cannot be read, the field says `unknown` and says why -- an
     unreadable answer is not a zero, and it is not a yes either.

  3. WRITTEN WHEN THE PICTURE CHANGES. `note` carries what changed. A write that changes
     nothing is still a write, and says so, because "nothing changed and I know it" and
     "I stopped updating" must not look identical from outside.

The file is rewritten whole from the fields given plus the fields carried forward, so a key
nobody updates cannot survive undated. Carried-forward fields keep their ORIGINAL
established time -- that is the entire point.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

HEARTBEAT = Path(os.environ.get("HEARTBEAT_FILE", Path.home() / "HEARTBEAT.json"))


def now() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip()


def _epoch() -> int:
    return int(subprocess.check_output(["date", "-u", "+%s"]).decode().strip())


def _epoch_of(stamp: str) -> int | None:
    try:
        return int(subprocess.check_output(["date", "-u", "-d", stamp, "+%s"]).decode().strip())
    except subprocess.CalledProcessError:
        return None


class Process:
    """A claim that something is running, which is re-verified at every write."""

    def __init__(self, pid: int, what: str, log: str | None = None) -> None:
        self.pid = int(pid)
        self.what = what
        self.log = log

    def observe(self, stamp: str, previous: dict[str, Any] | None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "what": self.what, "pid": self.pid, "log": self.log,
            "liveness_checked_at": stamp,
        }
        proc = Path(f"/proc/{self.pid}")
        try:
            alive = proc.is_dir()
            if alive:
                state = (proc / "stat").read_text().split(") ", 1)[1].split(" ", 1)[0]
                alive = state != "Z"
                record["proc_state"] = state
        except (OSError, IndexError) as exc:
            record["status"] = "unknown"
            record["unknown_because"] = f"{type(exc).__name__}: {exc}"
            record["last_seen_alive"] = (previous or {}).get("last_seen_alive")
            return record
        if alive:
            record["status"] = "running"
            record["last_seen_alive"] = stamp
        else:
            record["status"] = "finished"
            record["last_seen_alive"] = (previous or {}).get("last_seen_alive")
            record["noticed_finished_at"] = stamp
        return record


def write(note: str, doing: str, **fields: Any) -> dict[str, Any]:
    """Rewrite the file whole. Given fields are stamped now; the rest keep their vintage."""
    stamp = now()
    epoch = _epoch()
    old = {}
    if HEARTBEAT.exists():
        try:
            old = json.loads(HEARTBEAT.read_text())
        except json.JSONDecodeError:
            old = {}
    old_fields = old.get("fields", {}) if isinstance(old.get("fields"), dict) else {}

    out: dict[str, Any] = {}
    for name, entry in old_fields.items():
        if name not in fields:
            out[name] = entry  # carried forward, ORIGINAL established time kept
    for name, value in fields.items():
        previous = old_fields.get(name, {})
        if isinstance(value, Process):
            out[name] = {"established": stamp,
                         "value": value.observe(stamp, previous.get("value"))}
        else:
            unchanged = previous.get("value") == value
            out[name] = {
                "established": previous.get("established", stamp) if unchanged else stamp,
                "value": value,
            }
            if unchanged and "established" in previous:
                out[name]["reconfirmed_at"] = stamp

    # Re-verify every carried-forward process claim too, so a finished job cannot keep
    # asserting itself just because nobody named it in this write.
    for name, entry in out.items():
        value = entry.get("value")
        if isinstance(value, dict) and "pid" in value and "status" in value:
            if value.get("status") == "running":
                refreshed = Process(value["pid"], value["what"], value.get("log")).observe(
                    stamp, value
                )
                entry["value"] = refreshed
                entry["reverified_at"] = stamp

    for name, entry in out.items():
        established = _epoch_of(entry.get("established", stamp))
        entry["age_seconds"] = None if established is None else epoch - established

    document = {
        "written": stamp,
        "written_by": f"{Path(__file__).name} pid {os.getpid()}",
        "note": note,
        "doing": doing,
        "how_to_read_this": (
            "Every field carries `established` (when that value became true) and "
            "`age_seconds` at write time. A field with a `pid` also carries `status`, "
            "re-read from /proc during THIS write: running, finished, or unknown. "
            "`unknown` means liveness could not be read -- that is not a zero and not a "
            "yes. Nothing in this file asserts that a process is running unless /proc "
            "said so at `liveness_checked_at`."
        ),
        "fields": out,
    }
    HEARTBEAT.write_text(json.dumps(document, indent=2) + "\n")
    return document


if __name__ == "__main__":
    import sys

    print(json.dumps(write(sys.argv[1] if len(sys.argv) > 1 else "manual write",
                           sys.argv[2] if len(sys.argv) > 2 else "unspecified"), indent=2)[:600])
