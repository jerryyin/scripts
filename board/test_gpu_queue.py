#!/usr/bin/env python3
"""Tests for gpu-queue: add, status, serve's claim/drain/release loop, crash recovery, and the
one-serve-per-queue lock that lets add start a serve.

No GPU or docker: the queue is a local directory and the GPU wait is replaced.
"""
import argparse
import os
import importlib.machinery
import importlib.util
import io
import subprocess
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
loader = importlib.machinery.SourceFileLoader('gpu_queue', str(HERE / 'gpu-queue'))
spec = importlib.util.spec_from_loader('gpu_queue', loader)
gq = importlib.util.module_from_spec(spec)
sys.modules['gpu_queue'] = gq
loader.exec_module(gq)


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


@contextmanager
def served(queue: Path, count: int = 4):
    """Hold the queue's serve lock, as a running serve would."""
    queue.mkdir(parents=True, exist_ok=True)
    lock = gq.ServeLock(gq.Store(str(queue)), count)
    assert lock.acquire()
    try:
        yield
    finally:
        lock.release()


class NoHold:
    """Stands in for the memory holder, which needs a GPU."""

    def __init__(self, hip, log):
        pass

    def start(self):
        return {}

    def stop(self):
        pass


def serve_once(queue: Path, log: Path, claims: list[str], hold=None):
    args = argparse.Namespace(count=4, container=None, poll=0, once=True, reclaim_after=None)

    def fake_wait(_args, _log):
        claims.append('claim')
        return '5,7,6,4'
    return quiet(gq.serve, args, gq.Store(str(queue)), log, wait=fake_wait, hold=hold or NoHold)


def test_queue_runs_back_to_back_in_one_claim(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue, log = root / 'q', root / 'serve.log'
    tool = HERE / 'gpu-queue'
    # First job reports its environment and queues a follow-up while it runs: the follow-up
    # must run in the same claim, with no wait for GPUs in between.
    first = (f'echo "hip=$HIP_VISIBLE_DEVICES job=$GPU_QUEUE_JOB"; '
             f'{sys.executable} {tool} add {queue} --name follow-up -- "echo second; exit 0"')
    script = root / 'fails.sh'
    script.write_text('#!/usr/bin/env bash\necho failing; exit 3\n')
    with served(queue):
        quiet(gq.main, ['add', str(queue), '--name', 'first', '--', first])
        quiet(gq.main, ['add', str(queue), '--file', str(script)])
    claims: list[str] = []
    assert serve_once(queue, log, claims)[0] == 0
    assert claims == ['claim'], 'all jobs, including the follow-up, must share one claim'
    done = sorted(p.name for p in (queue / 'done').glob('*.sh'))
    failed = sorted(p.name for p in (queue / 'failed').glob('*.sh'))
    assert [n.split('-', 1)[1] for n in done] == ['first.sh', 'follow-up.sh'], done
    assert [n.split('-', 1)[1] for n in failed] == ['fails.sh'], failed
    first_log = (queue / 'done' / f'{done[0]}.log').read_text()
    assert first_log.startswith(f'hip=5,7,6,4 job={done[0]}'), first_log
    assert (queue / 'done' / f'{done[1]}.log').read_text() == 'second\n'
    assert (queue / 'failed' / f'{failed[0]}.exit').read_text() == '3\n'
    assert not list((queue / 'running').iterdir()) and not list((queue / 'pending').iterdir())
    _, listing = quiet(gq.main, ['status', str(queue)])
    assert 'done: 2' in listing and 'failed: 1' in listing and 'exit=3' in listing


def test_hold_spans_the_claim_and_names_its_group_to_jobs(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue, log = root / 'q', root / 'serve.log'
    events: list[str] = []

    class FakeHold:
        def __init__(self, hip, _log):
            events.append(f'init {hip}')

        def start(self):
            events.append('start')
            return {'GPU_ADMISSION_OWN_PGIDS': '4242'}

        def stop(self):
            events.append('stop')

    with served(queue):
        for name in ('a', 'b'):
            quiet(gq.main, ['add', str(queue), '--name', name, '--',
                            f'echo "pgids=$GPU_ADMISSION_OWN_PGIDS"; echo ran-{name} >> {root}/order'])
    assert serve_once(queue, log, [], hold=FakeHold)[0] == 0
    assert events == ['init 5,7,6,4', 'start', 'stop'], 'one hold spans every job of the claim'
    assert (root / 'order').read_text() == 'ran-a\nran-b\n'
    for job in (queue / 'done').glob('*.sh'):
        assert (queue / 'done' / f'{job.name}.log').read_text().startswith('pgids=4242\n')


def test_empty_queue_claims_nothing_and_recovers_interrupted_jobs(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue, log = root / 'q', root / 'serve.log'
    claims: list[str] = []
    assert serve_once(queue, log, claims)[0] == 0
    assert claims == [], 'an empty queue must not claim GPUs'
    stale = queue / 'running' / '20261110T000000000000Z-crashed.sh'
    stale.write_text('#!/usr/bin/env bash\n')
    (queue / 'running' / f'{stale.name}.log').write_text('partial\n')
    serve_once(queue, log, claims)
    assert (queue / 'failed' / stale.name).exists()
    assert (queue / 'failed' / f'{stale.name}.log').read_text() == 'partial\n'
    assert (queue / 'failed' / f'{stale.name}.exit').read_text() == 'interrupted\n'
    assert claims == []


def test_add_rejects_bad_names(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    try:
        quiet(gq.main, ['add', str(root / 'q'), '--name', '../escape', '--', 'true'])
    except SystemExit as error:
        assert 'job name' in str(error)
    else:
        raise AssertionError('a name with "/" must be refused')


def test_reclaim_runs_only_after_a_timed_out_wait(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue, log = root / 'q', root / 'serve.log'
    with served(queue):
        quiet(gq.main, ['add', str(queue), '--name', 'job', '--', 'true'])
    args = argparse.Namespace(count=4, container=None, poll=0, once=True,
                              reclaim_after=600, reclaim_grace=7)
    answers = iter([None, None, '4,5,6,7'])  # two timed-out waits, then a claim
    reclaimed: list[int] = []
    quiet(gq.serve, args, gq.Store(str(queue)), log, wait=lambda _a, _l: next(answers),
          reclaim=lambda _spare, grace, _log: reclaimed.append(grace), hold=NoHold)
    assert reclaimed == [7, 7], reclaimed
    assert [p.name.split('-', 1)[1] for p in (queue / 'done').glob('*.sh')] == ['job.sh']


def test_reclaim_needs_a_host_serve(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    try:
        quiet(gq.main, ['serve', str(root / 'q'), '--count', '4', '--reclaim-after', '60'])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('--reclaim-after without --container must be refused')


def test_holder_listing_finds_kfd_openers_and_never_pid_1():
    import subprocess
    import time
    if not Path('/dev/kfd').exists():
        print('skip: no /dev/kfd')
        return
    child = subprocess.Popen([sys.executable, '-c',
                              'import time; f = open("/dev/kfd", "rb"); time.sleep(30)'])
    try:
        time.sleep(0.5)
        out = subprocess.run(['sh', '-c', gq.KFD_HOLDERS], capture_output=True, text=True).stdout
        holders = gq.parse_holders(out)
        assert child.pid in [pid for pid, _ in holders], out
        assert 1 not in [pid for pid, _ in holders]
        if os.geteuid() == 0:  # kfd_holders lists through sudo -n otherwise
            assert child.pid in [pid for pid, _ in gq.kfd_holders()]
            own_cgroup = Path(f'/proc/{child.pid}/cgroup').read_text().strip()
            spared = [pid for pid, _ in gq.kfd_holders(spare=own_cgroup)]
            assert child.pid not in spared, 'a holder in the spared container must not be listed'
    finally:
        child.kill()
    assert gq.parse_holders('1\tinit\n12\tpython x\nnoise\n') == [(12, 'python x')]


def test_status_shows_waits_and_flags_a_queue_with_no_serve(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue = root / 'q'
    log = queue / gq.SERVE_LOG
    with served(queue):
        quiet(gq.main, ['add', str(queue), '--name', 'waiting', '--', 'true'])
        job = next((queue / 'pending').glob('*.sh'))
        queued = gq.dt.datetime.strptime(job.name.split('-', 1)[0], '%Y%m%dT%H%M%S%fZ').replace(
            tzinfo=gq.dt.timezone.utc).timestamp()
        log.write_text('2026-11-10T16:00:00-0500 free=[] busy={4: ["gfx:100%"]}\n')
        _, out = quiet(gq.status, argparse.Namespace(dir=str(queue)), now=queued + 2 * 3600 + 5 * 60)
        assert 'queued 2h05m ago' in out and 'busy={4:' in out, out
        assert 'serve: running, 4 GPUs per claim' in out and 'STALE' not in out, out
    _, out = quiet(gq.status, argparse.Namespace(dir=str(queue)), now=queued + 60)
    assert 'serve: none' in out and 'STALE' in out, out


def test_one_serve_per_queue_and_add_starts_one(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue = root / 'q'
    started: list[int] = []

    def add(*argv):
        return quiet(gq.add, gq.parse(['add', str(queue), *argv, '--', 'true']),
                     spawn=lambda _queue, count: started.append(count) or 4242)
    try:
        add()
    except SystemExit as error:
        assert 'no serve' in str(error), error
    else:
        raise AssertionError('a job for an unserved queue without --count must be refused')
    assert not (queue / 'pending').exists(), 'a refused job must not be queued'
    add('--count', '4')
    assert started == [4], 'add --count must start a serve when none runs'
    with served(queue, 4):
        claims: list[str] = []
        assert serve_once(queue, root / 'serve.log', claims)[0] == 1 and claims == [], \
            'a second serve of one queue must exit without claiming'
        add('--count', '4')
        assert started == [4], 'a running serve takes the job; no second serve'
        try:
            add('--count', '8')
        except SystemExit as error:
            assert 'claims 4 GPUs' in str(error), error
        else:
            raise AssertionError('a job needing more GPUs than the serve claims must be refused')
    assert len(list((queue / 'pending').glob('*.sh'))) == 2


def test_once_serve_takes_a_job_queued_as_it_stops(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue = root / 'q'
    late = queue / 'pending' / '20261111T000000000000Z-late.sh'

    class LateJob(gq.ServeLock):
        """The first release lands a job, as an add that saw the lock still held would leave it."""
        def release(self):
            super().release()
            if not late.exists() and not (queue / 'done' / late.name).exists():
                late.write_text('#!/usr/bin/env bash\ntrue\n')
                late.chmod(0o755)
    store = gq.Store(str(queue))
    args = argparse.Namespace(count=4, container=None, poll=0, once=True, reclaim_after=None)
    claims: list[str] = []
    result, _ = quiet(gq.serve, args, store, root / 'serve.log',
                      wait=lambda _a, _l: claims.append('claim') or '0,1,2,3', lock=LateJob(store, 4),
                      hold=NoHold)
    assert result == 0 and claims == ['claim'], claims
    assert (queue / 'done' / late.name).exists(), 'the late job must run before the serve exits'


def test_container_lock_holder_keeps_the_lock_until_stdin_closes(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue = root / 'q'
    queue.mkdir()
    hold = ['sh', '-c', gq.HOLD_LOCK, 'gpu-queue', str(queue / gq.LOCK)]
    holder = subprocess.Popen([*hold, '4'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == 'locked'
    assert gq.serving(queue) == 4
    second = subprocess.run([*hold, '8'], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert second.stdout == '' and gq.serving(queue) == 4, 'a second holder must neither lock nor overwrite'
    holder.stdin.close()
    holder.wait()
    assert gq.serving(queue) is None


if __name__ == '__main__':
    test_queue_runs_back_to_back_in_one_claim()
    test_empty_queue_claims_nothing_and_recovers_interrupted_jobs()
    test_add_rejects_bad_names()
    test_reclaim_runs_only_after_a_timed_out_wait()
    test_reclaim_needs_a_host_serve()
    test_holder_listing_finds_kfd_openers_and_never_pid_1()
    test_status_shows_waits_and_flags_a_queue_with_no_serve()
    test_one_serve_per_queue_and_add_starts_one()
    test_once_serve_takes_a_job_queued_as_it_stops()
    test_container_lock_holder_keeps_the_lock_until_stdin_closes()
    print('ok')
