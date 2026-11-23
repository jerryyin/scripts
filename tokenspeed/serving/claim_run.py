#!/usr/bin/env python3
"""Run several serving configurations on one GPU claim: quiet wait, manage_server (clean-tree check), stop.

  claim_run.py run --out OUT --queue DIR --gpu-count N --host-gpu-count N --amd-tracking AT
                   --tokenspeed TS --tokenspeed-commit SHA --deadline-utc ISO --quiet-seconds S
                   --start-method unguarded|guarded --model-path M --client-venv V
                   --idle-by usage|process --confirm N --stop-grace-seconds S
                   --config NAME=FILE --min-seconds NAME=S [--clients NAME=A,B,...] ...
                   [post-processing options] [--no-wait] [--dry-run]
      For each --config, in the order given, amd-tracking's manage_server.py writes the gpu-queue job
      (`run --config FILE --emit-job OUT/jobs/NAME.sh --run-dir OUT/NAME ...`); OUT/plan.json records
      every input; ONE board/gpu-queue job (`claim_run.py job OUT/plan.json`) is queued, so every
      configuration runs on one claim of N GPUs. Waits for that job (unless --no-wait), then runs the
      post-processing. --dry-run prints every command, including the in-claim steps, and writes nothing.
  claim_run.py job OUT/plan.json
      What gpu-queue runs on the claimed GPUs (GPU_QUEUE_HIP or HIP_VISIBLE_DEVICES). Core files off
      (RLIMIT_CORE 0, inherited by the server). Per configuration: skip it if OUT/STOP exists or fewer
      than its --min-seconds remain before the deadline; require no --forbid-env match in the job
      environment; wait until the claimed GPUs have had no other user's process for --quiet-seconds
      (amd-tracking `gpu_admission.py quiet`, giving up --min-seconds before the deadline; own = this
      job's process group and GPU_ADMISSION_OWN_PGIDS; verdict in OUT/quiet-NAME.json); run the
      emitted job in its own session (killed at the deadline, SIGKILL --stop-grace-seconds later),
      whose `manage_server.py run` refuses to start unless the TokenSpeed worktree is at
      --tokenspeed-commit with no change (emitted with --require-tokenspeed-commit and
      --require-clean-tokenspeed; run.json status tokenspeed-source-refused); stop that session's
      leftovers and, if `manage_server.py status` says it is still up, the server it started
      (`manage_server.py stop`). OUT/STOP stops the remaining configurations; it is written when
      another user's process shows up during the quiet wait of a --stop-on-foreign configuration
      (an unguarded start is not safe; the wait ends at that sighting), when manage_server refused
      the tree, when a --forbid-env variable is set, and when the server log has an out-of-memory
      line. With --start-method guarded the job needs GPU_HOLD_GATE (gpu-queue's
      --hold-gate-ranks) and puts board/memgate first on PYTHONPATH. Per configuration result:
      OUT/claim-run.json. Exit 0 when every configuration ran with exit 0.
  claim_run.py post OUT/plan.json
      The post-processing of the plan again, on whatever run dirs exist; outputs in OUT/post/:
      --facts: manage_server.py facts RUN_DIRS --markdown --recompute (facts.md) and the JSON
        (facts-recompute.json);
      --decompose CLIENT: e2e_decompose.py --benchmark-json OUT/NAME/CLIENT/tokenspeed/benchmark.json
        --server-log OUT/NAME/server.stderr.log (NAME-CLIENT-decompose.json), with --prompt-tokens and
        --decode-log-interval;
      --summarize CLIENT: numerics/range_compare.py summarize --key --metric ... --where ...
        --rows-field results over the same benchmark.json (NAME-CLIENT-summary.csv);
      --post-command CMD: bash -c CMD after those, with CLAIM_RUN_OUT, CLAIM_RUN_POST and
        CLAIM_RUN_DIR_<NAME> (NAME upper-cased, - as _) set (comparisons across runs: range_compare
        compare/decide, e2e_decompose --accept-split, ...).

Every campaign value (paths, configurations, commit, deadline, GPU counts, quiet time) is an input,
none has a default. The tool runs owners and does not redo their jobs: board/gpu-queue claims the
GPUs, manage_server.py emits, runs and stops the server (and checks the client venv against the
lock, and refuses a TokenSpeed tree not clean or not at the commit), gpu_admission.py waits for the
GPUs to be quiet, e2e_decompose.py and range_compare.py analyse.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import resource
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parents[1]
GPU_QUEUE = TOOLKIT / 'board' / 'gpu-queue'
MEMGATE = TOOLKIT / 'board' / 'memgate'
E2E_DECOMPOSE = HERE / 'e2e_decompose.py'
RANGE_COMPARE = TOOLKIT / 'numerics' / 'range_compare.py'
MANAGE_SERVER = 'skills/inference_engine/benchmark/scripts/manage_server.py'  # in --amd-tracking
GPU_ADMISSION = 'apps/shared/gpu_admission.py'  # in --amd-tracking
QUIET_EXIT = {0: 'quiet', 3: 'foreign', 4: 'deadline'}  # gpu_admission.py quiet's exit status per verdict
SOURCE_REFUSED = 'tokenspeed-source-refused'  # manage_server run.json status: tree not clean / not at the commit
OOM = re.compile(r'OutOfMemory|out of memory|HIPCachingAllocator.*OOM')
TIMED_OUT = 124  # the exit status coreutils `timeout` gives a command it stopped
QUEUE_POLL_SECONDS = 30.0
LIVE_STATES = ('starting', 'running')  # manage_server state-file statuses of a server that may be up
MATRIX_ROWS = 'results'  # the per-phase rows of an amd-tracking matrix benchmark.json


@dataclasses.dataclass
class Config:
    name: str
    config: str
    clients: list[str]
    min_seconds: int
    stop_on_foreign: bool
    run_dir: str
    job: str


@dataclasses.dataclass
class Post:
    facts: bool
    decompose: list[str]
    prompt_tokens: int | None
    decode_log_interval: int | None
    summarize: list[str]
    summary_key: str | None
    summary_metrics: list[str]
    summary_where: list[str]
    commands: list[str]


@dataclasses.dataclass
class Plan:
    out: str
    queue: str
    gpu_count: int
    host_gpu_count: int
    amd_tracking: str
    tokenspeed: str
    tokenspeed_commit: str
    deadline_utc: str
    quiet_seconds: float
    quiet_poll_seconds: float
    start_method: str
    stop_grace_seconds: float
    forbid_env: list[str]
    model_path: str
    client_venv: str
    evalscope_lock: str | None
    env_files: list[str]
    idle_by: str
    confirm: int
    hold_busy_free_cap: float | None
    hold_gate_ranks: int | None
    configs: list[Config]
    post: Post

    @property
    def manage_server(self) -> str:
        return str(Path(self.amd_tracking) / MANAGE_SERVER)

    @property
    def gpu_admission(self) -> str:
        return str(Path(self.amd_tracking) / GPU_ADMISSION)

    @property
    def deadline(self) -> float:
        return parse_deadline(self.deadline_utc)

    @property
    def stop_file(self) -> Path:
        return Path(self.out) / 'STOP'

    @property
    def post_dir(self) -> Path:
        return Path(self.out) / 'post'

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + '\n')

    @staticmethod
    def load(path: Path) -> 'Plan':
        data = json.loads(path.read_text())
        data['configs'] = [Config(**c) for c in data['configs']]
        data['post'] = Post(**data['post'])
        return Plan(**data)


def parse_deadline(text: str) -> float:
    when = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if when.tzinfo is None:
        raise ValueError(f'deadline {text!r} needs a time zone, e.g. a trailing Z')
    return when.timestamp()


def utc(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(message: str) -> None:
    print(f'{utc(time.time())} claim_run: {message}', flush=True)


# ---------------------------------------------------------------- commands (also what --dry-run prints)

def emit_argv(plan: Plan, cfg: Config) -> list[str]:
    argv = [sys.executable, plan.manage_server, 'run', '--config', cfg.config]
    for client in cfg.clients:
        argv += ['--client', client]
    argv += ['--emit-job', cfg.job, '--run-dir', cfg.run_dir, '--model-path', plan.model_path,
             '--tokenspeed-source', plan.tokenspeed, '--client-venv', plan.client_venv]
    if plan.evalscope_lock:
        argv += ['--evalscope-lock', plan.evalscope_lock]
    for env_file in plan.env_files:
        argv += ['--env-file', env_file]
    argv += ['--require-tokenspeed-commit', plan.tokenspeed_commit, '--require-clean-tokenspeed']
    return argv + ['--start-method', plan.start_method, '--deadline-utc', plan.deadline_utc]


def quiet_file(plan: Plan, cfg: Config) -> Path:
    return Path(plan.out) / f'quiet-{cfg.name}.json'


def quiet_argv(plan: Plan, cfg: Config, gpus: str, own_pgids: list[str]) -> list[str]:
    """gpu_admission.py quiet on the claimed GPUs, giving up when fewer than cfg.min_seconds would remain."""
    argv = [sys.executable, plan.gpu_admission, 'quiet', '--gpus', gpus, '--seconds', f'{plan.quiet_seconds:g}',
            '--give-up-utc', utc(plan.deadline - cfg.min_seconds), '--poll-seconds', f'{plan.quiet_poll_seconds:g}',
            '--gpu-count', str(plan.host_gpu_count)]
    for pgid in own_pgids:
        argv += ['--own-pgid', pgid]
    return argv + (['--stop-on-foreign'] if cfg.stop_on_foreign else [])


def job_argv(plan_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), 'job', str(plan_path)]


def queue_argv(plan: Plan, plan_path: Path) -> list[str]:
    argv = [str(GPU_QUEUE), 'add', plan.queue, '--name', 'claim-run-' + Path(plan.out).name,
            '--count', str(plan.gpu_count), '--idle-by', plan.idle_by, '--confirm', str(plan.confirm)]
    if plan.hold_busy_free_cap is not None:
        argv += ['--hold-busy-free-cap', f'{plan.hold_busy_free_cap:g}']
    if plan.hold_gate_ranks is not None:
        argv += ['--hold-gate-ranks', str(plan.hold_gate_ranks)]
    return argv + ['--', *job_argv(plan_path)]


def status_argv(plan: Plan, cfg: Config) -> list[str]:
    return [sys.executable, plan.manage_server, 'status', '--state-file', str(Path(cfg.run_dir) / 'server.json')]


def stop_argv(plan: Plan, cfg: Config) -> list[str]:
    return [sys.executable, plan.manage_server, 'stop', '--state-file', str(Path(cfg.run_dir) / 'server.json'),
            '--grace-seconds', f'{plan.stop_grace_seconds:g}', '--reason', f'claim_run: after {cfg.name}',
            '--execute']


@dataclasses.dataclass
class Step:
    argv: list[str]
    stdout: Path | None
    env: dict[str, str] | None = None


def benchmark_json(cfg: Config, client: str) -> Path:
    return Path(cfg.run_dir) / client / 'tokenspeed' / 'benchmark.json'


def post_steps(plan: Plan, only_existing: bool) -> list[Step]:
    """The post-processing commands; with only_existing, only those whose inputs exist."""
    post, out = plan.post, plan.post_dir
    run_dirs = [c.run_dir for c in plan.configs
                if not only_existing or (Path(c.run_dir) / 'run.json').is_file()]
    steps = []
    if post.facts and run_dirs:
        facts = [sys.executable, plan.manage_server, 'facts', *run_dirs]
        steps += [Step(facts + ['--markdown', '--recompute'], out / 'facts.md'),
                  Step(facts + ['--recompute'], out / 'facts-recompute.json')]
    for cfg in plan.configs:
        for client in post.decompose:
            if not only_existing or benchmark_json(cfg, client).is_file():
                steps.append(decompose_step(plan, cfg, client))
        for client in post.summarize:
            if not only_existing or benchmark_json(cfg, client).is_file():
                steps.append(summarize_step(plan, cfg, client))
    env = {'CLAIM_RUN_OUT': plan.out, 'CLAIM_RUN_POST': str(out)}
    env.update({'CLAIM_RUN_DIR_' + c.name.upper().replace('-', '_').replace('.', '_'): c.run_dir
                for c in plan.configs})
    steps += [Step(['bash', '-c', command], None, env) for command in post.commands]
    return steps


def decompose_step(plan: Plan, cfg: Config, client: str) -> Step:
    stem = plan.post_dir / f'{cfg.name}-{client}'
    return Step([sys.executable, str(E2E_DECOMPOSE), '--benchmark-json', str(benchmark_json(cfg, client)),
                 '--server-log', str(Path(cfg.run_dir) / 'server.stderr.log'),
                 '--slices-dir', f'{stem}-slices', '--prompt-tokens', str(plan.post.prompt_tokens),
                 '--decode-log-interval', str(plan.post.decode_log_interval),
                 '--out-json', f'{stem}-decompose.json'], Path(f'{stem}-decompose.out'))


def summarize_step(plan: Plan, cfg: Config, client: str) -> Step:
    post = plan.post
    argv = [sys.executable, str(RANGE_COMPARE), 'summarize', '--key', post.summary_key]
    for metric in post.summary_metrics:
        argv += ['--metric', metric]
    argv += ['--rows-field', MATRIX_ROWS]
    for where in post.summary_where:
        argv += ['--where', where]
    argv.append(str(benchmark_json(cfg, client)))
    return Step(argv, plan.post_dir / f'{cfg.name}-{client}-summary.csv')


# ---------------------------------------------------------------- the host (replaced in tests)

class Host:
    """Everything the tool does to the machine: clock and processes."""

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def limit_core(self) -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def call(self, step: Step, cwd: str | None = None) -> int:
        env = {**os.environ, **(step.env or {})}
        if step.stdout is None:
            return subprocess.run(step.argv, env=env, cwd=cwd).returncode
        step.stdout.parent.mkdir(parents=True, exist_ok=True)
        with step.stdout.open('w') as fh:
            return subprocess.run(step.argv, stdout=fh, env=env, cwd=cwd).returncode

    def capture(self, argv: list[str]) -> str:
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout

    def run_job(self, script: str, env: dict[str, str], timeout: float, grace: float) -> int:
        """bash SCRIPT in its own session; at the timeout SIGTERM the session, SIGKILL grace s later.
        Whatever is left in the session after the script exits is stopped the same way."""
        child = subprocess.Popen(['bash', script], env=env, start_new_session=True)
        try:
            rc = child.wait(timeout=max(timeout, 0))
        except subprocess.TimeoutExpired:
            log(f'deadline: stopping {script} (session {child.pid})')
            stop_group(child.pid, grace)
            child.wait()
            return TIMED_OUT
        stop_group(child.pid, grace)
        return rc


def stop_group(pgid: int, grace: float) -> None:
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        log(f'sent {sig.name} to process group {pgid}')
        end = time.monotonic() + wait
        while time.monotonic() < end:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)


# ---------------------------------------------------------------- in the claim

def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def env_problem(plan: Plan, env: dict[str, str]) -> str | None:
    hits = [f'{k}={v}' for k, v in sorted(env.items()) for rx in plan.forbid_env if re.search(rx, f'{k}={v}')]
    if hits:
        return f'forbidden variable(s) in the job environment: {hits}'
    if plan.start_method == 'guarded' and not env.get('GPU_HOLD_GATE'):
        return 'start method guarded but GPU_HOLD_GATE is unset (queue with --hold-gate-ranks)'
    return None


def claimed_gpus(plan: Plan, env: dict[str, str]) -> list[int]:
    text = env.get('GPU_QUEUE_HIP') or env.get('HIP_VISIBLE_DEVICES')
    if not text:
        raise SystemExit('claim_run job: no GPU_QUEUE_HIP or HIP_VISIBLE_DEVICES; run it under gpu-queue')
    gpus = [int(x) for x in text.split(',')]
    if len(gpus) != plan.gpu_count:
        raise SystemExit(f'claim_run job: needs {plan.gpu_count} GPUs, got {text}')
    return gpus


def job_environment(plan: Plan, env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    if plan.start_method == 'guarded':
        env['PYTHONPATH'] = str(MEMGATE) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def mark_stop(plan: Plan, reason: str) -> None:
    log(f'STOP: {reason}')
    with plan.stop_file.open('a') as fh:
        fh.write(reason + '\n')


class Claim:
    """One gpu-queue job: every configuration of the plan in order on the claimed GPUs."""

    def __init__(self, plan: Plan, host: Host, env: dict[str, str], own_pgid: int):
        self.plan, self.host, self.env = plan, host, env
        self.gpus = claimed_gpus(plan, env)
        self.own = [int(x) for x in env.get('GPU_ADMISSION_OWN_PGIDS', '').split(',') if x] + [own_pgid]
        self.results: list[dict] = []
        self.record = Path(plan.out) / 'claim-run.json'

    def run(self) -> int:
        self.host.limit_core()
        log(f'GPUs {self.gpus}, own process groups {self.own}, core files off, deadline {self.plan.deadline_utc}')
        for cfg in self.plan.configs:
            result = {'name': cfg.name, 'start_utc': utc(self.host.now())}
            result.update(self.one(cfg))
            result['end_utc'] = utc(self.host.now())
            log(f'{cfg.name}: {result["status"]} {result.get("reason", "")} rc={result.get("rc")}')
            self.results.append(result)
            self.save()
        return 0 if all(r['status'] == 'ran' and r['rc'] == 0 for r in self.results) else 1

    def save(self) -> None:
        self.record.write_text(json.dumps({'gpus': self.gpus, 'own_pgids': self.own,
                                           'configs': self.results}, indent=2) + '\n')

    def one(self, cfg: Config) -> dict:
        plan = self.plan
        if plan.stop_file.exists():
            return {'status': 'not-run', 'reason': 'stop marker: ' + plan.stop_file.read_text().strip()}
        if plan.deadline - self.host.now() < cfg.min_seconds:
            return {'status': 'not-run', 'reason': f'fewer than {cfg.min_seconds} s left before the deadline'}
        problem = env_problem(plan, self.env)
        if problem:
            mark_stop(plan, problem)
            return {'status': 'not-run', 'reason': problem}
        verdict_path = quiet_file(plan, cfg)
        rc = self.host.call(Step(quiet_argv(plan, cfg, ','.join(map(str, self.gpus)), list(map(str, self.own))),
                                 verdict_path))
        verdict = read_json(verdict_path)
        if rc not in QUIET_EXIT or verdict is None or verdict.get('status') != QUIET_EXIT[rc]:
            problem = f'gpu_admission.py quiet failed (exit {rc}; {verdict_path})'
            mark_stop(plan, problem)
            return {'status': 'not-run', 'reason': problem}
        result = {'quiet_wait': {'verdict': str(verdict_path),
                                 **{k: verdict[k] for k in ('status', 'waited_seconds', 'foreign_pids')}}}
        if verdict['status'] == 'deadline':
            return {**result, 'status': 'not-run', 'reason': 'deadline during the quiet wait'}
        if verdict['status'] == 'foreign':
            mark_stop(plan, f'guard needed: foreign process during the quiet wait of {cfg.name} '
                            f'(PIDs {verdict["foreign_pids"]})')
            return {**result, 'status': 'not-run', 'reason': 'foreign process during the quiet wait'}
        log(f'{cfg.name}: quiet; running {cfg.job}')
        rc = self.host.run_job(cfg.job, job_environment(plan, self.env), plan.deadline - self.host.now(),
                               plan.stop_grace_seconds)
        run = read_json(Path(cfg.run_dir) / 'run.json') or {}
        if run.get('status') == SOURCE_REFUSED:
            problem = f'manage_server refused the TokenSpeed tree: {run.get("error")}'
            mark_stop(plan, problem)
            return {**result, 'status': 'not-run', 'reason': problem, 'rc': rc}
        result.update(status='ran', rc=rc, stop_rc=self.stop_server(cfg))
        oom = oom_lines(Path(cfg.run_dir) / 'server.stderr.log')
        if oom:
            result['oom'] = oom
            mark_stop(plan, f'out of memory in {cfg.name}: {oom[0]}')
        return result

    def stop_server(self, cfg: Config) -> int | None:
        """manage_server stop, only for a server its state file says is still up (a stop rewrites the
        recorded stop reason)."""
        if not (Path(cfg.run_dir) / 'server.json').is_file():
            return None
        state = json.loads(self.host.capture(status_argv(self.plan, cfg)))
        if not state.get('matched') and state.get('status') not in LIVE_STATES:
            return None
        log(f'{cfg.name}: server state {state.get("status")}, matched {state.get("matched")}: stopping it')
        return self.host.call(Step(stop_argv(self.plan, cfg), Path(self.plan.out) / f'stop-{cfg.name}.json'))


def oom_lines(server_log: Path, limit: int = 20) -> list[str]:
    if not server_log.is_file():
        return []
    with server_log.open(errors='replace') as fh:
        return [line.rstrip() for line in fh if OOM.search(line)][:limit]


# ---------------------------------------------------------------- submit, wait, post

def run_post(plan: Plan, host: Host) -> int:
    failed = []
    plan.post_dir.mkdir(parents=True, exist_ok=True)
    for step in post_steps(plan, only_existing=True):
        rc = host.call(step)
        if rc != 0:
            failed.append(f'rc={rc}: {shlex.join(step.argv)}')
    for line in failed:
        log('post-processing failed ' + line)
    return 1 if failed else 0


def wait_for_job(queue: Path, job_name: str, host: Host) -> int:
    """Exit status gpu-queue recorded for the job (done/ or failed/), once it has one."""
    while True:
        for state in ('done', 'failed'):
            exit_file = queue / state / f'{job_name}.exit'
            if exit_file.is_file():
                text = exit_file.read_text().strip()
                log(f'queue job {job_name}: {state}, exit {text}')
                return int(text) if text.lstrip('-').isdigit() else 1
        host.sleep(QUEUE_POLL_SECONDS)


def print_dry_run(plan: Plan, plan_path: Path) -> None:
    q = shlex.join
    print(f'# claim_run --dry-run: nothing is written. Plan: {plan_path}')
    print('# 1. emit one gpu-queue job per configuration (manage_server.py, HIP_VISIBLE_DEVICES empty)')
    for cfg in plan.configs:
        print(f'HIP_VISIBLE_DEVICES= {q(emit_argv(plan, cfg))} > {cfg.job}.emit.json && bash -n {cfg.job}')
    print('# 2. one claim: queue the in-claim job')
    print(q(queue_argv(plan, plan_path)))
    print(f'# 3. in the claim (ulimit -c 0), per configuration, unless {plan.stop_file} exists '
          '(CLAIMED_GPUS: GPU_QUEUE_HIP or HIP_VISIBLE_DEVICES; --own-pgid OWN_PGID once per GPU_ADMISSION_OWN_PGIDS '
          'entry and for the claim job\'s process group):')
    for cfg in plan.configs:
        print(f'#   {cfg.name}: needs {cfg.min_seconds} s before {plan.deadline_utc}'
              + (f'; no job environment variable matches {plan.forbid_env}' if plan.forbid_env else '')
              + ('; exit 3 (foreign): STOP, guard needed' if cfg.stop_on_foreign else '') + '; exit 4 (deadline): skip')
        print(f'{q(quiet_argv(plan, cfg, "CLAIMED_GPUS", ["OWN_PGID"]))} > {quiet_file(plan, cfg)}')
        pythonpath = f'PYTHONPATH={MEMGATE}:$PYTHONPATH ' if plan.start_method == 'guarded' else ''
        print(f'{pythonpath}setsid bash {cfg.job}  # killed at {plan.deadline_utc}, '
              f'SIGKILL {plan.stop_grace_seconds:g} s later; run.json status {SOURCE_REFUSED}: STOP')
        print(f'{q(status_argv(plan, cfg))}  # if server.json exists')
        print(f'{q(stop_argv(plan, cfg))} > {Path(plan.out) / f"stop-{cfg.name}.json"}  # if still starting/running')
        print(f'grep -E {shlex.quote(OOM.pattern)} {Path(cfg.run_dir) / "server.stderr.log"}  # a line: STOP')
    print(f'# 4. after the queue job, post-processing into {plan.post_dir} (each step only if its input exists)')
    for step in post_steps(plan, only_existing=False):
        env = ' '.join(f'{k}={shlex.quote(v)}' for k, v in (step.env or {}).items())
        print((env + ' ' if env else '') + q(step.argv) + (f' > {step.stdout}' if step.stdout else ''))


def submit(plan: Plan, host: Host, wait: bool) -> int:
    out = Path(plan.out)
    plan_path = out / 'plan.json'
    if plan_path.exists():
        raise SystemExit(f'{plan_path} exists; give a new --out')
    (out / 'jobs').mkdir(parents=True, exist_ok=True)
    for cfg in plan.configs:
        step = Step(emit_argv(plan, cfg), Path(cfg.job + '.emit.json'), {'HIP_VISIBLE_DEVICES': ''})
        if host.call(step, cwd=str(out)) != 0:
            raise SystemExit(f'manage_server.py --emit-job failed for {cfg.name}; see {step.stdout}')
        if host.call(Step(['bash', '-n', cfg.job], None)) != 0:
            raise SystemExit(f'emitted job {cfg.job} is not valid bash')
    plan.save(plan_path)
    queued = Path(host.capture(queue_argv(plan, plan_path)).strip().splitlines()[0])
    log(f'queued {queued}')
    if not wait:
        post = [sys.executable, str(Path(__file__).resolve()), 'post', str(plan_path)]
        log(f'not waiting; after the queue job: {shlex.join(post)}')
        return 0
    rc = wait_for_job(Path(plan.queue), queued.name, host)
    return run_post(plan, host) or rc


# ---------------------------------------------------------------- command line

def named(items: list[str], flag: str) -> dict[str, str]:
    result = {}
    for item in items:
        name, sep, value = item.partition('=')
        if not sep or not name or not value:
            raise SystemExit(f'{flag} {item!r}: expected NAME=VALUE')
        if name in result:
            raise SystemExit(f'{flag}: {name} given twice')
        result[name] = value
    return result


def build_plan(a: argparse.Namespace) -> Plan:
    out = Path(a.out).resolve()
    configs = named(a.config, '--config')
    minimum = named(a.min_seconds, '--min-seconds')
    clients = named(a.clients, '--clients')
    for flag, names in (('--min-seconds', minimum), ('--clients', clients),
                        ('--stop-on-foreign', dict.fromkeys(a.stop_on_foreign))):
        unknown = set(names) - set(configs)
        if unknown:
            raise SystemExit(f'{flag} names no --config: {sorted(unknown)}')
    if set(configs) - set(minimum):
        raise SystemExit(f'--min-seconds missing for {sorted(set(configs) - set(minimum))}')
    for name, path in configs.items():
        if not re.fullmatch(r'[A-Za-z0-9._-]+', name):
            raise SystemExit(f'--config name {name!r} may use only letters, digits, ".", "_" and "-"')
        if not Path(path).is_file():
            raise SystemExit(f'--config {name}: no file {path}')
    if a.start_method == 'guarded' and (a.hold_busy_free_cap is None or a.hold_gate_ranks is None):
        raise SystemExit('--start-method guarded needs --hold-busy-free-cap and --hold-gate-ranks')
    if a.decompose and (a.prompt_tokens is None or a.decode_log_interval is None):
        raise SystemExit('--decompose needs --prompt-tokens and --decode-log-interval')
    if a.summarize and (a.summary_key is None or not a.summary_metric):
        raise SystemExit('--summarize needs --summary-key and --summary-metric')
    parse_deadline(a.deadline_utc)
    for path, flag in ((a.amd_tracking, '--amd-tracking'), (a.tokenspeed, '--tokenspeed')):
        if not Path(path).is_dir():
            raise SystemExit(f'{flag}: no directory {path}')
    if not (Path(a.amd_tracking) / MANAGE_SERVER).is_file():
        raise SystemExit(f'--amd-tracking {a.amd_tracking} has no {MANAGE_SERVER}')
    return Plan(
        out=str(out), queue=str(Path(a.queue).resolve()), gpu_count=a.gpu_count, host_gpu_count=a.host_gpu_count,
        amd_tracking=str(Path(a.amd_tracking).resolve()), tokenspeed=str(Path(a.tokenspeed).resolve()),
        tokenspeed_commit=a.tokenspeed_commit, deadline_utc=a.deadline_utc, quiet_seconds=a.quiet_seconds,
        quiet_poll_seconds=a.quiet_poll_seconds, start_method=a.start_method,
        stop_grace_seconds=a.stop_grace_seconds,
        forbid_env=a.forbid_env, model_path=a.model_path, client_venv=a.client_venv,
        evalscope_lock=a.evalscope_lock, env_files=[str(Path(p).resolve()) for p in a.env_file],
        idle_by=a.idle_by, confirm=a.confirm, hold_busy_free_cap=a.hold_busy_free_cap,
        hold_gate_ranks=a.hold_gate_ranks,
        configs=[Config(name, str(Path(path).resolve()), [c for c in clients.get(name, '').split(',') if c],
                        int(minimum[name]), name in a.stop_on_foreign, str(out / name),
                        str(out / 'jobs' / f'{name}.sh'))
                 for name, path in configs.items()],
        post=Post(a.facts, a.decompose, a.prompt_tokens, a.decode_log_interval, a.summarize, a.summary_key,
                  a.summary_metric, a.summary_where, a.post_command))


def add_run_arguments(r: argparse.ArgumentParser) -> None:
    req = r.add_argument_group('required')
    for flag, kind, text in (
            ('--out', str, 'new output directory: jobs/, one run dir per configuration, plan.json, post/'),
            ('--queue', str, 'gpu-queue directory'),
            ('--gpu-count', int, 'GPUs to claim (the configurations\' GPU count)'),
            ('--host-gpu-count', int, 'GPUs on the host (gpu_admission numbering)'),
            ('--amd-tracking', str, 'amd-tracking checkout (manage_server.py, gpu_admission.py, configurations)'),
            ('--tokenspeed', str, 'TokenSpeed worktree the server runs'),
            ('--tokenspeed-commit', str, 'full commit the worktree must be at, without changes '
                                         '(manage_server --require-tokenspeed-commit, --require-clean-tokenspeed)'),
            ('--deadline-utc', str, 'ISO 8601 with zone: emitted into every job, ends waits and jobs'),
            ('--quiet-seconds', float, 'no foreign process on the claimed GPUs for this long before each start '
                                      '(gpu_admission.py quiet --seconds)'),
            ('--model-path', str, 'checkpoint (manage_server --model-path)'),
            ('--client-venv', str, 'EvalScope venv (manage_server --client-venv)'),
            ('--confirm', int, 'gpu-queue serve --confirm, if this starts the serve'),
            ('--stop-grace-seconds', float, 'SIGTERM-to-SIGKILL time for the job session and manage_server stop')):
        req.add_argument(flag, type=kind, required=True, help=text)
    req.add_argument('--start-method', required=True, choices=('unguarded', 'guarded'))
    req.add_argument('--idle-by', required=True, choices=('usage', 'process'), help='gpu-queue serve --idle-by')
    req.add_argument('--config', action='append', required=True, metavar='NAME=FILE',
                     help='serving configuration; repeatable, run in this order')
    req.add_argument('--min-seconds', action='append', required=True, metavar='NAME=S',
                     help='skip NAME when fewer seconds remain before the deadline (one per --config)')
    r.add_argument('--clients', action='append', default=[], metavar='NAME=A,B',
                   help="manage_server --client for NAME's configuration (default: all its clients)")
    r.add_argument('--evalscope-lock', help='manage_server --evalscope-lock')
    r.add_argument('--env-file', action='append', default=[], help='manage_server --env-file (repeatable)')
    r.add_argument('--forbid-env', action='append', default=[], metavar='REGEX',
                   help='refuse to start (STOP) when a NAME=VALUE of the job environment matches')
    r.add_argument('--stop-on-foreign', action='append', default=[], metavar='NAME',
                   help="STOP (guard needed) when a foreign process appears during NAME's quiet wait")
    r.add_argument('--quiet-poll-seconds', type=float, default=3.0, help='gpu_admission.py quiet --poll-seconds')
    r.add_argument('--hold-busy-free-cap', type=float, metavar='GIB', help='gpu-queue (needed when guarded)')
    r.add_argument('--hold-gate-ranks', type=int, metavar='N', help='gpu-queue (needed when guarded)')
    p = r.add_argument_group('post-processing')
    p.add_argument('--facts', action='store_true', help='manage_server.py facts over the run dirs')
    p.add_argument('--decompose', action='append', default=[], metavar='CLIENT', help='e2e_decompose.py')
    p.add_argument('--prompt-tokens', type=int)
    p.add_argument('--decode-log-interval', type=int)
    p.add_argument('--summarize', action='append', default=[], metavar='CLIENT', help='range_compare.py summarize')
    p.add_argument('--summary-key')
    p.add_argument('--summary-metric', action='append', default=[])
    p.add_argument('--summary-where', action='append', default=[])
    p.add_argument('--post-command', action='append', default=[], metavar='CMD')
    r.add_argument('--no-wait', action='store_true', help='return once the job is queued')
    r.add_argument('--dry-run', action='store_true', help='print every command; write nothing')


def main(argv: list[str] | None = None, host: Host | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    add_run_arguments(sub.add_parser('run', help='emit, queue on one claim, wait, post-process'))
    sub.add_parser('job', help='in the claim (gpu-queue runs this)').add_argument('plan', type=Path)
    sub.add_parser('post', help='post-processing of a plan').add_argument('plan', type=Path)
    a = ap.parse_args(argv)
    host = host or Host()
    if a.cmd == 'run':
        plan = build_plan(a)
        if a.dry_run:
            print_dry_run(plan, Path(plan.out) / 'plan.json')
            return 0
        return submit(plan, host, wait=not a.no_wait)
    plan = Plan.load(a.plan.resolve())
    if a.cmd == 'job':
        return Claim(plan, host, dict(os.environ), os.getpgrp()).run()
    return run_post(plan, host)


if __name__ == '__main__':
    sys.exit(main())
