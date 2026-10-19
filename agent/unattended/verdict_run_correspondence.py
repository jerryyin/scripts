"""A go/no-go verdict file describes exactly one run of the checker, or it describes nothing.

THE PATTERN THIS SERVES. A periodic "gate" or preflight script checks whether some shared
resource is fit to use, writes a one-line verdict to a well-known file, and logs what it
did. Anything about to use the resource reads the verdict file. The hazard is that the
verdict file is a LATCH: if a later run of the gate fails to publish, the previous run's
permission stays on disk and reads exactly like a current one.

    A verdict older than the newest gate run is not a weak permission. It is an
    ABSENT one, and absent means refused.

THE FAILURE IT WAS BUILT AGAINST, because it is subtler than "the file was old". A gate
run refused at 08:49:19Z while the verdict file went on reading ALLOWED, stamped 08:39:15Z
from the previous run. The publishing step was itself gated on a status value that could
not be read -- and that same unreadable status was one of the inputs that produced the
refusal. So a single unreadable input both manufactured a blocker AND suppressed
publication of the verdict that blocker produced, leaving the earlier run's permission
standing as live authorization. Fixing the writer is necessary; this is the reader-side
half, which does not have to trust that the writer got it right.

THIS IS NOT AN AGE THRESHOLD, AND MUST NOT BECOME ONE. Nothing here compares an elapsed
duration to a limit: no window, no maximum age, no timeout. The only comparisons are
between two observed events -- the verdict and the newest gate run -- asking whether they
are the SAME run. A verdict five seconds old fails this check if a newer gate run exists;
a verdict five hours old passes it if none does. That is a correspondence test, not an age
test, and the difference is the whole point: an age threshold rejects healthy verdicts
during a slow period and accepts stale ones during a fast one.

IT REFUSES BY SENDING YOU BACK TO THE GATE. Re-run the gate; this never overrides a
verdict and never manufactures one.

FAIL CLOSED. Every way of not knowing -- no verdict file, an unreadable one, no gate log
to compare against, a log whose verdict cannot be parsed, a verdict or a log that cannot
be tied to a run -- returns ABSENT. Not knowing whether a permission is current is the
same as not having one.

HOW CORRESPONDENCE IS ESTABLISHED, AND THE RULE THIS REPLACED. The first version compared
the two mtimes and refused when the gate log was newer. That was wrong, and wrong in the
direction that makes a check useless rather than dangerous. The gate publishes the verdict
FIRST and then logs the line confirming it landed, so on a healthy run the log is always a
few milliseconds newer than the file it confirms -- one real pair came out 11:25:05.078 for
the log against 11:25:05.070 for the verdict, and the old rule called that correctly
published verdict absent. It survived review for two reasons worth knowing: publication had
never once succeeded while a log sat in the directory, and the self-test fixture wrote the
log BEFORE the verdict, an ordering the gate cannot actually produce. A rule that can only
ever say no is as broken as one that can only ever say yes; it just fails in the expensive
direction instead of the dangerous one.

Correspondence is therefore RUN IDENTITY, two comparisons, neither of them a duration:
  1. The gate stamps the same instant into its concluding log line ("[11:23:30] ===
     DEVICE_WORK: BLOCKED ===") and into the verdict file ("checked_at_utc:
     2026-09-11T11:23:30Z"). Those times of day must match, and so must the two verdicts.
  2. The two files must have been WRITTEN on the same UTC calendar date. A shared time of
     day is not an identity on its own -- yesterday's verdict would alias onto a run
     concluding at the same second today -- and comparing the writes to each other rather
     than to the claim also survives a run that concludes at 23:59:59 and publishes after
     midnight, because both writes land on the new date together.
This is immune to which of the two writes happened first, which is the property the old
rule lacked.

WHICH LOGS IT READS. It globs GATE_LOG_GLOB under GATE_LOG_DIR and takes the newest by
mtime. Those must be the LIVE logs the gate writes as it runs, not archived copies of
them: a copy is made after the fact and carries a fresh mtime, so pointing this at an
archive directory would compare the verdict against the copy's write time and silently
stop testing anything.

CONFIGURATION (all optional, defaults shown):
    VERDICT_FILE    ~/PREFLIGHT.txt    the published verdict
    GATE_LOG_DIR    $HOME             where the gate writes its live logs
    GATE_LOG_GLOB   preflight*.log    which of them to consider

USAGE
    python3 verdict_run_correspondence.py            # print the check as JSON
    python3 verdict_run_correspondence.py selftest   # run the fixtures, no I/O outside tmp
    from verdict_run_correspondence import require   # raises VerdictAbsent unless current
"""

# DELIBERATELY NO `from __future__ import annotations` HERE. Under PEP 563 every annotation
# becomes a string that @dataclass resolves through sys.modules[__name__]. A caller that
# loads this file through an importlib spec without registering it in sys.modules gets None
# there, and the class body raises AttributeError at import. Real annotation objects
# sidestep that entirely, and `str | None` needs no future import on a modern interpreter.

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VERDICT_PATH = Path(os.environ.get("VERDICT_FILE", Path.home() / "PREFLIGHT.txt"))
GATE_LOG_DIR = Path(os.environ.get("GATE_LOG_DIR", Path.home()))
GATE_LOG_GLOB = os.environ.get("GATE_LOG_GLOB", "preflight*.log")

# The gate stamps its own conclusion into its log behind the same [HH:MM:SS] prefix every
# other line carries. Anchoring this at the start of the line without allowing that prefix
# read a finished run as an unfinished one -- a wrong reason for a right refusal, which is
# its own kind of failure. BOTH the stamp and the word are captured, because the stamp is
# what ties the line to a run. The prefix is optional in the pattern and REQUIRED in
# practice: a concluding line without one cannot be tied to a run and is refused below
# rather than guessed at.
_LOG_VERDICT = re.compile(
    r"^(?:\[([0-9]{2}:[0-9]{2}:[0-9]{2})\]\s*)?===\s*DEVICE_WORK:\s*(\w+)\s*===\s*$",
    re.MULTILINE,
)
_FILE_VERDICT = re.compile(r"^DEVICE_WORK:\s*(\w+)\s*$", re.MULTILINE)
_CHECKED_AT = re.compile(r"^checked_at_utc:\s*(\S+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Correspondence:
    """Whether the verdict on disk describes the newest gate read, and why not if not."""

    corresponds: bool
    reason: str
    verdict: str | None
    verdict_checked_at_utc: str | None
    newest_log: str | None
    newest_log_verdict: str | None

    def as_record(self) -> dict[str, object]:
        """The shape that goes on a row. Every field, including the ones that are None."""
        return {
            "check": "preflight_verdict_correspondence",
            "corresponds": self.corresponds,
            "reason": self.reason,
            "verdict": self.verdict,
            "verdict_checked_at_utc": self.verdict_checked_at_utc,
            "newest_gate_log": self.newest_log,
            "newest_gate_log_verdict": self.newest_log_verdict,
            "this_is_not_an_age_check": (
                "No elapsed duration is compared to any limit. The only comparison is "
                "whether the verdict and the newest gate read are the same run."
            ),
        }


def _date_utc(mtime: float) -> str:
    """The UTC calendar date of an mtime, as YYYY-MM-DD."""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")


def _newest_gate_log(log_dir: Path, glob: str) -> Path | None:
    """The most recently written gate-read log, or None if there is not one."""
    logs = [path for path in log_dir.glob(glob) if path.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def check(
    verdict_path: Path = VERDICT_PATH,
    log_dir: Path = GATE_LOG_DIR,
    glob: str = GATE_LOG_GLOB,
) -> Correspondence:
    """Decide whether the verdict on disk describes the newest gate read.

    Returns rather than raises, so a caller can record the answer on a row in the
    cases where it is not gating anything. `require()` is the gating form.
    """
    absent = lambda reason: Correspondence(  # noqa: E731 - one shape, six call sites
        corresponds=False,
        reason=reason,
        verdict=None,
        verdict_checked_at_utc=None,
        newest_log=None,
        newest_log_verdict=None,
    )

    if not verdict_path.is_file():
        return absent(f"no verdict file at {verdict_path}")
    try:
        verdict_text = verdict_path.read_text(encoding="utf-8")
    except OSError as error:
        return absent(f"verdict file {verdict_path} is unreadable: {error}")

    verdict_match = _FILE_VERDICT.search(verdict_text)
    if verdict_match is None:
        return absent(f"verdict file {verdict_path} carries no DEVICE_WORK line")
    verdict = verdict_match.group(1)
    checked_match = _CHECKED_AT.search(verdict_text)
    checked_at = checked_match.group(1) if checked_match else None

    newest = _newest_gate_log(log_dir, glob)
    if newest is None:
        return Correspondence(
            corresponds=False,
            reason=(
                f"no gate-read log matching {glob} under {log_dir}, so there is nothing to "
                "establish which run this verdict describes. Not knowing is not a yes."
            ),
            verdict=verdict,
            verdict_checked_at_utc=checked_at,
            newest_log=None,
            newest_log_verdict=None,
        )

    try:
        log_text = newest.read_text(encoding="utf-8")
    except OSError as error:
        return Correspondence(
            corresponds=False,
            reason=f"newest gate log {newest} is unreadable: {error}",
            verdict=verdict,
            verdict_checked_at_utc=checked_at,
            newest_log=str(newest),
            newest_log_verdict=None,
        )

    log_matches = _LOG_VERDICT.findall(log_text)
    log_stamp, log_verdict = log_matches[-1] if log_matches else (None, None)

    partial = lambda reason: Correspondence(  # noqa: E731
        corresponds=False,
        reason=reason,
        verdict=verdict,
        verdict_checked_at_utc=checked_at,
        newest_log=str(newest),
        newest_log_verdict=log_verdict,
    )

    if log_verdict is None:
        return partial(
            f"the newest gate log {newest} carries no concluding DEVICE_WORK line, so it "
            "either did not finish or did not finish in a shape this can read. An "
            "unfinished gate read is not a permission."
        )

    # IDENTITY, NOT ORDER, AND STILL NOT AGE. This used to compare the two mtimes and
    # refuse when the log was newer. That was WRONG, and wrong in the direction that makes
    # a check useless rather than dangerous: the gate publishes the verdict FIRST and then
    # logs the line confirming it landed, so on a healthy run the log is always a few
    # milliseconds newer than the file it is confirming. One real healthy pair
    # published correctly and the mtimes came out 11:25:05.078 for the log against
    # 11:25:05.070 for the verdict -- eight milliseconds, and the old rule called a
    # correctly published verdict absent. It had never been caught because publication had
    # never once succeeded while a log sat in this directory, and because the self-test
    # fixture wrote the log BEFORE the verdict, which is an ordering the gate cannot
    # produce. A rule that can only ever say no is as broken as one that can only say yes.
    #
    # So correspondence is now established by RUN IDENTITY: the gate stamps the same
    # instant into its concluding log line ("[11:23:30] === DEVICE_WORK: BLOCKED ===") and
    # into the verdict file ("checked_at_utc: 2026-09-11T11:23:30Z"). Same instant and same
    # conclusion means same run. This is immune to the order of the two writes, and no
    # elapsed duration is compared to any limit anywhere in it -- there is still no age
    # threshold here and section 7 still forbids one.
    if checked_at is None:
        return partial(
            f"verdict file {verdict_path.name} carries no checked_at_utc line, so it "
            "cannot be tied to a run at all. The gate always stamps one; a verdict "
            "without it is not a verdict this can vouch for."
        )
    # findall with two groups hands back an empty string, not None, when the optional
    # prefix did not match -- so this tests falsiness. Getting that wrong let an unstamped
    # line fall through to the time comparison and refuse for the wrong reason.
    if not log_stamp:
        return partial(
            f"the concluding line of {newest.name} carries no [HH:MM:SS] stamp, so the "
            "run it describes cannot be identified. Not knowing which run a log is is "
            "not a yes."
        )
    checked_time = checked_at[11:19] if len(checked_at) >= 19 else None
    if checked_time != log_stamp:
        return partial(
            f"gate read {newest.name} concluded at {log_stamp} and the verdict on disk is "
            f"stamped {checked_at} -- different runs. That read's own conclusion "
            f"({log_verdict}) was never published, so the file in front of the launcher "
            "describes an earlier run. A verdict describes exactly one run."
        )
    # A shared time of day is not by itself an identity: yesterday's verdict carries
    # yesterday's 08:39:15 and would alias onto a run concluding at 08:39:15 today. What
    # kills that is comparing the two WRITES to each other rather than to the content
    # stamp. The gate publishes the verdict and logs its confirmation milliseconds apart,
    # so a real pair always shares a calendar date, while a day-old file on disk does not
    # share one with today's log. Comparing the writes rather than the claim also survives
    # a run that concludes at 23:59:59 and publishes after midnight: both writes land on
    # the new date together, and the content stamps still match each other. Two observed
    # events compared for identity; no duration, no limit, no threshold.
    log_date = _date_utc(newest.stat().st_mtime)
    verdict_date = _date_utc(verdict_path.stat().st_mtime)
    if log_date != verdict_date:
        return partial(
            f"gate read {newest.name} was written on {log_date} and {verdict_path.name} "
            f"was written on {verdict_date}. The times of day agree but the two files were "
            "not written by the same run, so the match is a coincidence of the clock."
        )

    if log_verdict != verdict:
        return partial(
            f"the newest gate read {newest.name} concluded {log_verdict} and the verdict "
            f"file says {verdict}. Two readings that disagree stay two readings; the "
            "permission is not the one to pick."
        )

    return Correspondence(
        corresponds=True,
        reason=(
            f"verdict {verdict} corresponds to gate read {newest.name}, which concluded "
            f"{log_verdict}"
        ),
        verdict=verdict,
        verdict_checked_at_utc=checked_at,
        newest_log=str(newest),
        newest_log_verdict=log_verdict,
    )


class VerdictAbsent(RuntimeError):
    """The verdict on disk does not describe the newest gate read."""


def require(
    verdict_path: Path = VERDICT_PATH,
    log_dir: Path = GATE_LOG_DIR,
    glob: str = GATE_LOG_GLOB,
) -> Correspondence:
    """Refuse unless the verdict corresponds AND allows. Raises VerdictAbsent otherwise.

    Two separate refusals, deliberately kept apart in the message: a verdict that does
    not correspond is ABSENT and the answer is to re-run the gate; a verdict that
    corresponds and says anything other than ALLOWED is a live refusal and the answer is
    to obey it.
    """
    result = check(verdict_path, log_dir, glob)
    if not result.corresponds:
        raise VerdictAbsent(
            "the device-work verdict does not describe the newest gate read, so there is "
            f"no current permission: {result.reason}. Re-run the gate; do not proceed on "
            "the file in front of you."
        )
    if result.verdict != "ALLOWED":
        raise VerdictAbsent(
            f"the current gate verdict is {result.verdict}, not ALLOWED "
            f"({result.reason})."
        )
    return result


def _self_test() -> None:
    """Drive every branch through the real code path on synthetic fixtures.

    No device, no network, nothing outside a temporary directory.
    """
    import os
    import tempfile

    # THE GATE'S REAL WRITE ORDER, which the earlier version of this fixture got backwards.
    # The gate publishes the verdict and THEN logs the line saying
    # it landed, so the log is always a shade newer than the verdict it confirms. Every
    # fixture below writes them in that order, and `log_first` now only exists to build the
    # one shape the gate cannot produce -- a log older than the verdict -- which must not
    # change the answer either way, because correspondence no longer looks at the order.
    day = "2026-09-11"

    def fixture(
        directory: Path, verdict: str | None, log: str | None, log_first: bool = False
    ) -> tuple[Path, Path]:
        verdict_path = directory / "PREFLIGHT.txt"
        log_path = directory / "preflight-synthetic.log"
        # Real epoch seconds on the fixture day, so the date comparison sees a real date
        # rather than 1970. 11:25:05Z on 2026-09-11.
        base = datetime(2026, 9, 11, 11, 25, 5, tzinfo=timezone.utc).timestamp()
        if log is not None and log_first:
            log_path.write_text(log, encoding="utf-8")
            os.utime(log_path, (base - 1, base - 1))
        if verdict is not None:
            verdict_path.write_text(verdict, encoding="utf-8")
            os.utime(verdict_path, (base, base))
        if log is not None and not log_first:
            log_path.write_text(log, encoding="utf-8")
            # Eight milliseconds later, exactly as read 11 came out on the real box.
            os.utime(log_path, (base + 0.008, base + 0.008))
        return verdict_path, directory

    allowed = f"DEVICE_WORK: ALLOWED\nchecked_at_utc: {day}T08:39:15Z\n"
    blocked = f"DEVICE_WORK: BLOCKED\nchecked_at_utc: {day}T09:18:22Z\n"
    log_allowed = "[08:39:15] ok\n[08:39:15] === DEVICE_WORK: ALLOWED ===\n"
    log_blocked = "[09:18:22] BLOCK: ...\n[09:18:22] === DEVICE_WORK: BLOCKED ===\n"
    # A concluding line with no stamp cannot be tied to a run, so it must refuse.
    log_unstamped = "[09:18:22] BLOCK: ...\n=== DEVICE_WORK: BLOCKED ===\n"
    # Same time of day, different run: the shape the date check exists to catch.
    log_allowed_other_day = "[08:39:15] === DEVICE_WORK: ALLOWED ===\n"

    cases: list[tuple[str, str | None, str | None, bool, bool, str]] = [
        # name, verdict, log, log_first, expect_corresponds, expect_reason_contains
        # THE REGRESSION THIS FIXTURE EXISTS FOR: healthy publication, log written after
        # the verdict by milliseconds, and it must CORRESPOND. The old rule failed here.
        ("matching allowed, log written after", allowed, log_allowed, False, True, "corresponds"),
        ("matching blocked, log written after", blocked, log_blocked, False, True, "corresponds"),
        # The order the gate cannot produce must give the same answer: order is not the test.
        ("matching allowed, log written first", allowed, log_allowed, True, True, "corresponds"),
        # The 2026-09-11 failure itself: the file is an earlier run's, times disagree.
        ("stale verdict, newer read", allowed, log_blocked, False, False, "different runs"),
        ("no verdict file", None, log_blocked, False, False, "no verdict file"),
        ("no gate log", allowed, None, False, False, "nothing to establish"),
        ("unfinished read", allowed, "[09:03] started\n", False, False, "did not finish"),
        ("malformed verdict", "nothing useful\n", log_allowed, False, False, "no DEVICE_WORK"),
        ("verdict with no checked_at", "DEVICE_WORK: ALLOWED\n", log_allowed, False, False,
         "no checked_at_utc"),
        ("log verdict line unstamped", blocked, log_unstamped, False, False, "no [HH:MM:SS]"),

    ]

    for name, verdict, log, log_first, expect_ok, expect_text in cases:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            verdict_path, log_dir = fixture(directory, verdict, log, log_first)
            result = check(verdict_path, log_dir, "preflight*.log")
            assert result.corresponds is expect_ok, f"{name}: {result}"
            assert expect_text in result.reason, f"{name}: {result.reason}"
            print(f"  ok  {name}: corresponds={result.corresponds} -- {result.reason[:88]}")

    # THE CLOCK-ALIAS CASE, which needs its own fixture because it turns on the two files
    # being WRITTEN a day apart while their content stamps agree to the second. Yesterday's
    # verdict, today's log, same 08:39:15 on both. It must refuse.
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        verdict_path = directory / "PREFLIGHT.txt"
        log_path = directory / "preflight-synthetic.log"
        verdict_path.write_text(
            f"DEVICE_WORK: ALLOWED\nchecked_at_utc: {day}T08:39:15Z\n", encoding="utf-8"
        )
        log_path.write_text(log_allowed_other_day, encoding="utf-8")
        yesterday = datetime(2026, 9, 10, 8, 39, 15, tzinfo=timezone.utc).timestamp()
        today = datetime(2026, 9, 11, 8, 39, 15, tzinfo=timezone.utc).timestamp()
        os.utime(verdict_path, (yesterday, yesterday))
        os.utime(log_path, (today, today))
        result = check(verdict_path, directory, "preflight*.log")
        assert result.corresponds is False, result
        assert "coincidence of the clock" in result.reason, result.reason
        print(f"  ok  same time of day, written a day apart: {result.reason[:78]}")

    # require() refuses on both kinds, and passes only on a corresponding ALLOWED.
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        # A correctly published, correctly
        # corresponding BLOCKED. It must be recognised as current AND still refuse.
        verdict_path, log_dir = fixture(directory, blocked, log_blocked)
        assert check(verdict_path, log_dir, "preflight*.log").corresponds is True
        try:
            require(verdict_path, log_dir, "preflight*.log")
        except VerdictAbsent as error:
            assert "not ALLOWED" in str(error), error
            print(f"  ok  require refuses a published, corresponding BLOCKED: {str(error)[:70]}")
        else:
            raise AssertionError("require() let a BLOCKED verdict through")

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        verdict_path, log_dir = fixture(directory, allowed, log_blocked)
        try:
            require(verdict_path, log_dir, "preflight*.log")
        except VerdictAbsent as error:
            assert "newest gate read" in str(error), error
            print(f"  ok  require refuses a stale ALLOWED: {str(error)[:70]}")
        else:
            raise AssertionError("require() let a stale ALLOWED through")

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        verdict_path, log_dir = fixture(directory, allowed, log_allowed)
        result = require(verdict_path, log_dir, "preflight*.log")
        print(f"  ok  require passes a corresponding ALLOWED: {result.reason[:70]}")

    print("self-test: all branches exercised, no device, no network")


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _self_test()
    else:
        print(json.dumps(check().as_record(), indent=2))
