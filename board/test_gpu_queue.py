#!/usr/bin/env python3
"""Tests for gpu-queue: add, status, serve's claim/drain/release loop and crash recovery.

No GPU or docker: the queue is a local directory and the GPU wait is replaced.
"""
import argparse
import os
import importlib.machinery
import importlib.util
import io
import sys
import tempfile
from contextlib import redirect_stdout
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


def serve_once(queue: Path, log: Path, claims: list[str]):
    args = argparse.Namespace(count=4, container=None, poll=0, once=True, reclaim_after=None)

    def fake_wait(_args, _log):
        claims.append('claim')
        return '5,7,6,4'
    return quiet(gq.serve, args, gq.Store(str(queue)), log, wait=fake_wait)


def test_queue_runs_back_to_back_in_one_claim(tmp=None):
    root = Path(tmp or tempfile.mkdtemp())
    queue, log = root / 'q', root / 'serve.log'
    tool = HERE / 'gpu-queue'
    # First job reports its environment and queues a follow-up while it runs: the follow-up
    # must run in the same claim, with no wait for GPUs in between.
    first = (f'echo "hip=$HIP_VISIBLE_DEVICES job=$GPU_QUEUE_JOB"; '
             f'{sys.executable} {tool} add {queue} --name follow-up -- "echo second; exit 0"')
    quiet(gq.main, ['add', str(queue), '--name', 'first', '--', first])
    script = root / 'fails.sh'
    script.write_text('#!/usr/bin/env bash\necho failing; exit 3\n')
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
    quiet(gq.main, ['add', str(queue), '--name', 'job', '--', 'true'])
    args = argparse.Namespace(count=4, container=None, poll=0, once=True,
                              reclaim_after=600, reclaim_grace=7)
    answers = iter([None, None, '4,5,6,7'])  # two timed-out waits, then a claim
    reclaimed: list[int] = []
    quiet(gq.serve, args, gq.Store(str(queue)), log, wait=lambda _a, _l: next(answers),
          reclaim=lambda _spare, grace, _log: reclaimed.append(grace))
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


def test_status_shows_waits_and_flags_a_silent_serve(tmp=None):
    import os as _os
    root = Path(tmp or tempfile.mkdtemp())
    queue = root / 'q'
    quiet(gq.main, ['add', str(queue), '--name', 'waiting', '--', 'true'])
    job = next((queue / 'pending').glob('*.sh'))
    queued = gq.dt.datetime.strptime(job.name.split('-', 1)[0], '%Y%m%dT%H%M%S%fZ').replace(
        tzinfo=gq.dt.timezone.utc).timestamp()
    log = queue / gq.SERVE_LOG
    log.write_text('2026-11-10T16:00:00-0500 free=[] busy={4: ["vram:235960MB"]}\n')
    _os.utime(log, (queued, queued))
    _, out = quiet(gq.status, argparse.Namespace(dir=str(queue)), now=queued + 2 * 3600 + 5 * 60)
    assert 'queued 2h05m ago' in out, out
    assert 'busy={4:' in out and 'STALE' in out, out
    _os.utime(log, (queued + 2 * 3600, queued + 2 * 3600))
    _, out = quiet(gq.status, argparse.Namespace(dir=str(queue)), now=queued + 2 * 3600 + 60)
    assert 'STALE' not in out, 'a serve that polled a minute ago is alive'


if __name__ == '__main__':
    test_queue_runs_back_to_back_in_one_claim()
    test_empty_queue_claims_nothing_and_recovers_interrupted_jobs()
    test_add_rejects_bad_names()
    test_reclaim_runs_only_after_a_timed_out_wait()
    test_reclaim_needs_a_host_serve()
    test_holder_listing_finds_kfd_openers_and_never_pid_1()
    test_status_shows_waits_and_flags_a_silent_serve()
    print('ok')
