#!/usr/bin/env python3
"""Run one command in a SPUR allocation after checking device access."""

import argparse
import os
from pathlib import Path
import re
import shlex
import sys


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def duration(value: str) -> str:
    match = re.fullmatch(r'(\d+):([0-5]\d):([0-5]\d)', value)
    if match is None or not any(int(part) for part in match.groups()):
        raise argparse.ArgumentTypeError('use a positive HH:MM:SS duration')
    return value


def check_allocation(gpus: int, cpus: int) -> None:
    job = os.environ.get('SPUR_JOB_ID', '')
    if not job.isdecimal():
        raise RuntimeError('the allocated command has no valid SPUR_JOB_ID')
    available_cpus = len(os.sched_getaffinity(0))
    if available_cpus < cpus:
        raise RuntimeError(f'job {job} exposes {available_cpus} CPUs; requested {cpus}')
    render_devices = sorted(Path('/dev/dri').glob('renderD*'))
    if len(render_devices) != gpus:
        raise RuntimeError(
            f'job {job} exposes {len(render_devices)} render devices; requested {gpus}')
    for device in [Path('/dev/kfd'), *render_devices]:
        try:
            descriptor = os.open(device, os.O_RDWR)
        except OSError as error:
            raise RuntimeError(
                f'job {job} cannot open {device}: {error.strerror}. '
                'The test command was not started; ask IT about GPU device access.') from error
        os.close(descriptor)


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    parser.add_argument('--qos', required=True)
    parser.add_argument('--partition', default='amd-spur-v2')
    parser.add_argument('--node', help='request a particular compute node through SPUR')
    parser.add_argument('--gpus', type=positive_integer, choices=range(1, 9), required=True)
    parser.add_argument('--cpus', type=positive_integer, required=True)
    parser.add_argument('--time', type=duration, required=True, metavar='HH:MM:SS')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allocated', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    command = options.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('provide a command after --')
    if options.allocated:
        if options.dry_run:
            parser.error('--allocated cannot be combined with --dry-run')
        check_allocation(options.gpus, options.cpus)
        # SPUR's physical indices can hide GPUs already isolated by the job.
        os.environ.pop('ROCR_VISIBLE_DEVICES', None)
        print(f'job {os.environ["SPUR_JOB_ID"]}: device access checked; starting command', flush=True)
        os.execvp(command[0], command)
    if os.environ.get('SPUR_JOB_ID') or os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('already in a job; run the command directly instead of allocating again')
    launch = [
        'srun', '--account', options.account, '--qos', options.qos,
        '--partition', options.partition, '--gres', f'gpu:mi355x:{options.gpus}',
        '--cpus-per-task', str(options.cpus), '--time', options.time,
    ]
    if options.node:
        launch.extend(['--nodelist', options.node])
    launch.extend([
        sys.executable, str(Path(__file__).resolve()), '--allocated',
        '--account', options.account, '--qos', options.qos,
        '--partition', options.partition, '--gpus', str(options.gpus),
        '--cpus', str(options.cpus), '--time', options.time, '--', *command,
    ])
    if options.dry_run:
        print(shlex.join(launch))
        return 0
    if not sys.platform.startswith('linux'):
        raise RuntimeError('run this launcher on the SPUR Linux login node')
    print(
        f'Requesting {options.gpus} GPU(s), {options.cpus} CPU(s), {options.time}; '
        f'account={options.account} qos={options.qos}', flush=True)
    os.execvp(launch[0], launch)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, RuntimeError) as error:
        print(f'spur-run: {error}', file=sys.stderr)
        sys.exit(1)
