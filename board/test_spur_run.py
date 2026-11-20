"""Exercise the launcher with a fake scheduler; no GPU or SPUR service needed."""

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).with_name('spur-run.py')
RESOURCES = ['--account', 'test-team', '--qos', 'test-qos',
             '--gpus', '1', '--cpus', '2', '--time', '00:05:00']


@unittest.skipUnless(sys.platform.startswith('linux'), 'launcher executes Linux programs')
class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        scheduler = self.root / 'srun'
        scheduler.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, pathlib, sys\n'
            'pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
            'sys.exit(7)\n')
        scheduler.chmod(0o755)
        self.environment = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                                CAPTURE=str(self.root / 'arguments.json'))
        for name in ('SPUR_JOB_ID', 'SLURM_JOB_ID'):
            self.environment.pop(name, None)

    def run_launcher(self, arguments):
        return subprocess.run([sys.executable, str(TOOL), *arguments],
                              env=self.environment, capture_output=True, text=True)

    def test_one_allocation_preserves_command_arguments_and_scheduler_failure(self):
        command = ['python3', 'test group.py', '--pattern', 'a b; $HOME']
        result = self.run_launcher([*RESOURCES, '--', *command])
        self.assertEqual(result.returncode, 7, result.stderr)
        submitted = json.loads((self.root / 'arguments.json').read_text())
        self.assertEqual(submitted[:12], [
            '--account', 'test-team', '--qos', 'test-qos', '--partition', 'amd-spur-v2',
            '--gres', 'gpu:mi355x:1', '--cpus-per-task', '2', '--time', '00:05:00'])
        self.assertEqual(submitted[12:15], [sys.executable, str(TOOL.resolve()), '--allocated'])
        self.assertEqual(submitted[-len(command):], command)

    def test_missing_or_invalid_resource_requests_do_not_submit(self):
        requests = [
            ['--account', 'test-team', '--qos', 'test-qos', '--gpus', '1', '--cpus', '2'],
            [*RESOURCES, '--time', '00:00:00'],
            [*RESOURCES, '--time', '00:60:00'],
            [*RESOURCES, '--gpus', '0'],
            [*RESOURCES, '--cpus', '0'],
        ]
        for arguments in requests:
            with self.subTest(arguments=arguments):
                result = self.run_launcher([*arguments, '--', 'true'])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / 'arguments.json').exists())

    def test_existing_allocation_does_not_submit_another_job(self):
        for name in ('SPUR_JOB_ID', 'SLURM_JOB_ID'):
            with self.subTest(name=name):
                self.environment[name] = '123'
                result = self.run_launcher([*RESOURCES, '--', 'true'])
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('already in a job', result.stderr)
                self.assertFalse((self.root / 'arguments.json').exists())
                del self.environment[name]

    def test_dry_run_does_not_submit(self):
        result = self.run_launcher([*RESOURCES, '--dry-run', '--', 'true'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--gres gpu:mi355x:1', result.stdout)
        self.assertNotIn('--dry-run', result.stdout)
        self.assertFalse((self.root / 'arguments.json').exists())

    def test_node_request_still_goes_through_scheduler(self):
        result = self.run_launcher([*RESOURCES, '--node', 'test-node', '--', 'true'])
        self.assertEqual(result.returncode, 7, result.stderr)
        submitted = json.loads((self.root / 'arguments.json').read_text())
        self.assertEqual(submitted[12:14], ['--nodelist', 'test-node'])

    def test_allocated_mode_requires_a_real_job_environment(self):
        result = self.run_launcher(['--allocated', *RESOURCES, '--', 'true'])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('no valid SPUR_JOB_ID', result.stderr)
        self.assertFalse((self.root / 'arguments.json').exists())

    def test_device_permission_failure_prevents_command_execution(self):
        spec = importlib.util.spec_from_file_location('spur_run', TOOL)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with (patch.dict(os.environ, SPUR_JOB_ID='123'),
              patch.object(os, 'sched_getaffinity', return_value={0, 1}),
              patch.object(Path, 'glob', return_value=[Path('/dev/dri/renderD128')]),
              patch.object(os, 'open', side_effect=PermissionError(13, 'Permission denied')),
              patch.object(os, 'execvp') as execute):
            with self.assertRaisesRegex(RuntimeError, 'cannot open /dev/kfd'):
                launcher.main(['--allocated', *RESOURCES, '--', 'true'])
            execute.assert_not_called()


@unittest.skipUnless(os.environ.get('SPUR_RUN_LIVE_GPU_TEST') == '1' and
                     os.environ.get('SPUR_JOB_ID'), 'explicit live GPU allocation required')
class LiveGpuTests(unittest.TestCase):
    def test_one_gpu_memory_round_trip(self):
        runtime = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
        runtime.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        runtime.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        runtime.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.c_int]
        runtime.hipFree.argtypes = [ctypes.c_void_p]
        count = ctypes.c_int()
        self.assertEqual(runtime.hipGetDeviceCount(ctypes.byref(count)), 0)
        self.assertEqual(count.value, 1)
        device = ctypes.c_void_p()
        expected = [42, 42, 42, 42]
        self.assertEqual(runtime.hipMalloc(ctypes.byref(device), len(expected)), 0)
        try:
            self.assertEqual(runtime.hipMemset(device, 42, len(expected)), 0)
            actual = (ctypes.c_ubyte * len(expected))()
            # HIP's device-to-host copy kind is 2.
            self.assertEqual(runtime.hipMemcpy(actual, device, len(expected), 2), 0)
            self.assertEqual(list(actual), expected)
            print(f'GPU memory round trip: {list(actual)}', flush=True)
        finally:
            self.assertEqual(runtime.hipFree(device), 0)


if __name__ == '__main__':
    unittest.main()
