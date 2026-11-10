#!/usr/bin/env python3
"""Tests for gpu-queue: add, status, serve's claim/drain/release loop and crash recovery.

No GPU or docker: the queue is a local directory and the GPU wait is replaced.
"""
import argparse
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
    args = argparse.Namespace(count=4, container=None, poll=0, once=True)

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


if __name__ == '__main__':
    test_queue_runs_back_to_back_in_one_claim()
    test_empty_queue_claims_nothing_and_recovers_interrupted_jobs()
    test_add_rejects_bad_names()
    print('ok')
