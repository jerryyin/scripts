#!/usr/bin/env python3
"""CPU tests for claim_run.py (run with pytest or directly): no GPU, no gpu-queue, no amd-tracking run.

The host (clock, processes, git, the GPU observation) is a fake that records every command."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, HERE / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


cr = load('claim_run.py')
DEADLINE = '2026-11-23T05:41:40Z'
COMMIT = '100621d171dee9d0168f5d661d55cf68ca87f823'


class FakeHost(cr.Host):
    """Time advances only through sleep and the jobs it 'runs'; every command is recorded."""

    def __init__(self, start: float, foreign=None, head: str = COMMIT, porcelain: str = '',
                 job_seconds: float = 600.0, on_job=None, server_status: str = 'stopped'):
        self.t = start
        self.foreign_at = foreign or (lambda t: [])
        self.head, self.porcelain = head, porcelain
        self.job_seconds, self.on_job, self.server_status = job_seconds, on_job, server_status
        self.calls: list[list[str]] = []
        self.jobs: list[tuple[str, dict, float]] = []
        self.core_limited = False

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    def limit_core(self) -> None:
        self.core_limited = True

    def capture(self, argv: list[str]) -> str:
        self.calls.append(argv)
        if argv[-2:] == ['rev-parse', 'HEAD']:
            return self.head + '\n'
        if argv[-2:] == ['status', '--porcelain']:
            return self.porcelain
        if 'status' in argv:  # manage_server status
            return json.dumps({'status': self.server_status, 'matched': self.server_status == 'running'})
        if 'add' in argv:  # gpu-queue add prints the pending job's path
            return '/q/pending/20261123T000000000000Z-claim-run-out.sh\n'
        raise AssertionError(argv)

    def call(self, step, cwd=None) -> int:
        self.calls.append(step.argv)
        if '--emit-job' in step.argv:
            job = Path(step.argv[step.argv.index('--emit-job') + 1])
            job.write_text('#!/usr/bin/env bash\ntrue\n')
        return 0

    def foreign(self, plan, gpu_ids, own_pgids) -> list[str]:
        return self.foreign_at(self.t)

    def run_job(self, script, env, timeout, grace) -> int:
        self.jobs.append((script, env, timeout))
        if self.on_job:
            self.on_job(script)
        self.t += self.job_seconds
        return 0


def plan_argv(root: Path, start_method: str, extra: list[str]) -> list[str]:
    """`run` arguments for two configurations under root (creating the files they name)."""
    at, ts = root / 'at', root / 'ts'
    (at / Path(cr.MANAGE_SERVER).parent).mkdir(parents=True, exist_ok=True)
    (at / cr.MANAGE_SERVER).write_text('')
    ts.mkdir(exist_ok=True)
    for name in ('a.json', 'b.json'):
        (root / name).write_text('{}')
    return ['run', '--out', str(root / 'out'), '--queue', str(root / 'q'), '--gpu-count', '4',
            '--host-gpu-count', '8', '--amd-tracking', str(at), '--tokenspeed', str(ts),
            '--tokenspeed-commit', COMMIT, '--deadline-utc', DEADLINE, '--quiet-seconds', '120',
            '--start-method', start_method, '--model-path', '/m',
            '--client-venv', '/venv', '--idle-by', 'process', '--confirm', '20', '--stop-grace-seconds', '60',
            '--config', f'fused={root / "a.json"}', '--min-seconds', 'fused=2700',
            '--clients', 'fused=smoke,canonical-L50000-O500',
            '--config', f'unfused={root / "b.json"}', '--min-seconds', 'unfused=1500',
            '--forbid-env', 'TOKENSPEED_PETIT_', *extra]


def make_plan(root: Path, start_method: str = 'unguarded', extra: list[str] = ()):
    return cr.build_plan(parse_run(plan_argv(root, start_method, list(extra))))


def parse_run(argv: list[str]):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    cr.add_run_arguments(sub.add_parser('run'))
    return ap.parse_args(argv)


def start_of(seconds_before_deadline: float) -> float:
    return cr.parse_deadline(DEADLINE) - seconds_before_deadline


def claim(plan, host, env=None):
    (Path(plan.out)).mkdir(parents=True, exist_ok=True)
    env = {'GPU_QUEUE_HIP': '1,3,2,0', 'GPU_ADMISSION_OWN_PGIDS': '77', **(env or {})}
    with contextlib.redirect_stdout(io.StringIO()):
        rc = cr.Claim(plan, host, env, 99).run()
    return rc, json.loads((Path(plan.out) / 'claim-run.json').read_text())


def test_every_campaign_value_is_required_and_checked() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        plan = make_plan(root)
        assert [c.name for c in plan.configs] == ['fused', 'unfused'], 'configurations keep the order given'
        assert plan.configs[0].clients == ['smoke', 'canonical-L50000-O500'] and plan.configs[1].clients == []
    with tempfile.TemporaryDirectory() as d:
        argv = plan_argv(Path(d), 'unguarded', [])
        for flag in ('--quiet-seconds', '--tokenspeed-commit', '--deadline-utc', '--gpu-count', '--host-gpu-count',
                     '--amd-tracking', '--tokenspeed', '--queue', '--start-method', '--stop-grace-seconds'):
            i = argv.index(flag)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    parse_run(argv[:i] + argv[i + 2:])
            except SystemExit:
                continue
            raise AssertionError(f'{flag} must be required')
    for extra, message in (([], None),
                           (['--min-seconds', 'nope=1'], 'names no --config'),
                           (['--decompose', 'canonical-L50000-O500'], '--prompt-tokens'),
                           (['--summarize', 'x'], '--summary-key')):
        with tempfile.TemporaryDirectory() as d:
            try:
                make_plan(Path(d), extra=extra)
            except SystemExit as e:
                assert message and message in str(e), (extra, e)
                continue
            assert message is None, f'{extra} must be refused'
    with tempfile.TemporaryDirectory() as d:
        try:
            make_plan(Path(d), start_method='guarded')
        except SystemExit as e:
            assert '--hold-gate-ranks' in str(e)
        else:
            raise AssertionError('guarded without the hold options must be refused')


def test_dry_run_prints_the_owner_commands_and_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        plan = make_plan(root, extra=['--stop-on-foreign', 'fused', '--facts', '--decompose', 'canonical-L50000-O500',
                                      '--prompt-tokens', '50000', '--decode-log-interval', '40'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cr.print_dry_run(plan, Path(plan.out) / 'plan.json')
        text = out.getvalue()
        assert not Path(plan.out).exists()
        assert (f"manage_server.py run --config {root / 'a.json'} --client smoke --client canonical-L50000-O500 "
                f"--emit-job {plan.out}/jobs/fused.sh --run-dir {plan.out}/fused") in text
        assert '--start-method unguarded --deadline-utc 2026-11-23T05:41:40Z' in text
        assert (f"gpu-queue add {root / 'q'} --name claim-run-out --count 4 --idle-by process --confirm 20 -- "
                in text), 'one queue job for every configuration'
        assert text.count('gpu-queue add') == 1
        assert f"claim_run.py job {plan.out}/plan.json" in text
        assert 'fused: needs 2700 s' in text and 'STOP, guard needed' in text.split('unfused:')[0]
        assert f'git -C {plan.tokenspeed} status --porcelain' in text
        assert f"manage_server.py stop --state-file {plan.out}/unfused/server.json --grace-seconds 60" in text
        assert f"manage_server.py facts {plan.out}/fused {plan.out}/unfused --markdown --recompute" in text
        assert '--prompt-tokens 50000 --decode-log-interval 40' in text


def test_quiet_wait_restarts_on_a_foreign_process_and_gives_up_before_the_deadline() -> None:
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d))
        t0 = start_of(10_000)
        host = FakeHost(t0, foreign=lambda t: ['4242:hip1'] if t < t0 + 30 else [])
        with contextlib.redirect_stdout(io.StringIO()):
            wait = cr.quiet_wait(plan, host, [1, 3, 2, 0], [99], 2700)
        assert wait.status == 'quiet' and wait.foreign == ['4242:hip1']
        assert wait.waited_s == 147, wait  # polls every 3 s: last foreign sighting at 27 s, then 120 s quiet
        host = FakeHost(start_of(2800), foreign=lambda t: ['4242:hip1'])
        with contextlib.redirect_stdout(io.StringIO()):
            wait = cr.quiet_wait(plan, host, [1, 3, 2, 0], [99], 2700)
        assert wait.status == 'deadline' and wait.waited_s > 100


def test_claim_runs_each_configuration_and_stops_what_it_started() -> None:
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d))
        for cfg in plan.configs:
            Path(cfg.run_dir).mkdir(parents=True)
            Path(cfg.run_dir, 'server.json').write_text('{}')
        host = FakeHost(start_of(10_000), server_status='running')
        rc, record = claim(plan, host)
        assert rc == 0 and host.core_limited
        assert [r['status'] for r in record['configs']] == ['ran', 'ran']
        assert record['own_pgids'] == [77, 99] and record['gpus'] == [1, 3, 2, 0]
        assert [Path(j[0]).name for j in host.jobs] == ['fused.sh', 'unfused.sh']
        stops = [c for c in host.calls if 'stop' in c]
        assert len(stops) == 2 and stops[0][stops[0].index('--state-file') + 1].endswith('fused/server.json')
        git = [c[-2:] for c in host.calls if c[0] == 'git']
        assert git == [['rev-parse', 'HEAD'], ['status', '--porcelain']] * 2, 'clean-tree check before each start'
        assert 'PYTHONPATH' not in host.jobs[0][1] or str(cr.MEMGATE) not in host.jobs[0][1]['PYTHONPATH']


def test_a_server_already_stopped_is_not_stopped_again() -> None:
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d))
        Path(plan.configs[0].run_dir).mkdir(parents=True)
        Path(plan.configs[0].run_dir, 'server.json').write_text('{}')
        host = FakeHost(start_of(10_000), server_status='stopped')
        claim(plan, host)
        assert not [c for c in host.calls if 'stop' in c]


def test_stop_marker_cases() -> None:
    # a foreign process during a --stop-on-foreign wait: nothing runs
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d), extra=['--stop-on-foreign', 'fused'])
        t0 = start_of(10_000)
        host = FakeHost(t0, foreign=lambda t: ['5:hip0'] if t < t0 + 10 else [])
        rc, record = claim(plan, host)
        assert rc == 1 and not host.jobs and 'guard needed' in plan.stop_file.read_text()
        assert record['configs'][1]['reason'].startswith('stop marker: guard needed')
    # a worktree with a change, or at another commit
    for kwargs in ({'porcelain': ' M python/x.py\n'}, {'head': 'f' * 40}):
        with tempfile.TemporaryDirectory() as d:
            plan = make_plan(Path(d))
            host = FakeHost(start_of(10_000), **kwargs)
            rc, record = claim(plan, host)
            assert rc == 1 and not host.jobs and 'TokenSpeed' in plan.stop_file.read_text(), kwargs
    # a forbidden switch in the job environment
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d))
        host = FakeHost(start_of(10_000))
        rc, _ = claim(plan, host, env={'TOKENSPEED_PETIT_FUSION': '1'})
        assert rc == 1 and not host.jobs and 'TOKENSPEED_PETIT_FUSION=1' in plan.stop_file.read_text()
    # an out-of-memory line in the first server's log: the second does not run
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d))

        def oom(script: str) -> None:
            run_dir = Path(plan.out) / Path(script).stem
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'server.stderr.log').write_text('ok\ntorch.OutOfMemoryError: HIP out of memory\n')

        host = FakeHost(start_of(10_000), on_job=oom)
        rc, record = claim(plan, host)
        assert len(host.jobs) == 1 and record['configs'][0]['oom'] and 'out of memory in fused' in \
            plan.stop_file.read_text()


def test_too_little_time_skips_only_that_configuration() -> None:
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d))
        host = FakeHost(start_of(2000))  # fused needs 2700 s, unfused 1500 s (+ 120 s quiet)
        rc, record = claim(plan, host)
        assert [r['status'] for r in record['configs']] == ['not-run', 'ran'] and rc == 1
        assert not plan.stop_file.exists()
        assert host.jobs[0][2] < 2000 - 120 + 1, 'the job is given the time left before the deadline'


def test_guarded_needs_the_gate_and_puts_memgate_first() -> None:
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d), start_method='guarded',
                         extra=['--hold-busy-free-cap', '32', '--hold-gate-ranks', '4'])
        host = FakeHost(start_of(10_000))
        rc, _ = claim(plan, host)
        assert rc == 1 and 'GPU_HOLD_GATE' in plan.stop_file.read_text()
    with tempfile.TemporaryDirectory() as d:
        plan = make_plan(Path(d), start_method='guarded',
                         extra=['--hold-busy-free-cap', '32', '--hold-gate-ranks', '4'])
        assert '--hold-busy-free-cap 32 --hold-gate-ranks 4' in ' '.join(cr.queue_argv(plan, Path('p')))
        host = FakeHost(start_of(10_000))
        rc, _ = claim(plan, host, env={'GPU_HOLD_GATE': '/q/hold.gate', 'PYTHONPATH': '/x'})
        assert rc == 0 and host.jobs[0][1]['PYTHONPATH'] == f'{cr.MEMGATE}:/x'


def test_run_emits_queues_once_waits_and_post_processes() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        plan = make_plan(root, extra=['--facts', '--summarize', 'canonical-L50000-O500', '--summary-key',
                                      'concurrency', '--summary-metric', 'summary.Avg TPOT (ms)',
                                      '--post-command', 'echo $CLAIM_RUN_DIR_FUSED'])
        plan.queue = str(root / 'q')
        host = FakeHost(start_of(10_000))

        def finish(seconds: float) -> None:  # gpu-queue finishes the job while we wait
            done = root / 'q' / 'done'
            done.mkdir(parents=True, exist_ok=True)
            (done / '20261123T000000000000Z-claim-run-out.sh.exit').write_text('0\n')
            fused = Path(plan.configs[0].run_dir)
            (fused / 'canonical-L50000-O500' / 'tokenspeed').mkdir(parents=True)
            (fused / 'canonical-L50000-O500' / 'tokenspeed' / 'benchmark.json').write_text('{}')
            (fused / 'run.json').write_text('{}')

        host.sleep = finish
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cr.submit(plan, host, wait=True)
        assert rc == 0
        assert [c for c in host.calls if '--emit-job' in c][1][-4:] == [
            '--start-method', 'unguarded', '--deadline-utc', DEADLINE]
        assert len([c for c in host.calls if 'add' in c]) == 1
        saved = cr.Plan.load(Path(plan.out) / 'plan.json')
        assert saved == plan, 'the job reads back the plan run wrote'
        facts = [c for c in host.calls if 'facts' in c]
        assert facts and facts[0][3:] == [plan.configs[0].run_dir, '--markdown', '--recompute'], \
            'facts only over run dirs that have a run.json'
        summaries = [c for c in host.calls if 'summarize' in c]
        assert len(summaries) == 1
        assert summaries[0][-1] == str(Path(plan.configs[0].run_dir, 'canonical-L50000-O500/tokenspeed/benchmark.json'))
        assert host.calls[-1] == ['bash', '-c', 'echo $CLAIM_RUN_DIR_FUSED']
        try:
            cr.submit(plan, host, wait=False)
        except SystemExit as e:
            assert 'exists' in str(e)
        else:
            raise AssertionError('a second run into the same --out must be refused')


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'


def test_real_host_stops_the_job_session_and_reads_gpu_admission() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        pidfile = root / 'child.pid'
        script = root / 'job.sh'
        script.write_text(f'sleep 300 & echo $! > {pidfile}\nsleep 300\n')
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cr.Host().run_job(str(script), dict(os.environ), timeout=1, grace=1)
        assert rc == cr.TIMED_OUT
        assert not alive(int(pidfile.read_text())), 'the background child of the job session is stopped too'
        script.write_text(f'sleep 300 & echo $! > {pidfile}\nexit 3\n')
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cr.Host().run_job(str(script), dict(os.environ), timeout=60, grace=1)
        assert rc == 3
        assert not alive(int(pidfile.read_text())), 'what the job left in its session is stopped after it exits'
        plan = make_plan(root)
        admission = Path(plan.amd_tracking) / cr.GPU_ADMISSION
        admission.parent.mkdir(parents=True)
        admission.write_text(
            'def observe_gpu_set(topology, processes, proc, gpu_count, gpu_ids, own_pgids=()):\n'
            '    assert gpu_count == 8 and list(own_pgids) == [77, 99], (gpu_count, own_pgids)\n'
            '    return {"owners": {gpu_ids[1]: [{"pid": 4242}]}}\n')
        assert cr.Host().foreign(plan, [1, 3, 2, 0], [77, 99]) == ['4242:hip3']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('ok')
