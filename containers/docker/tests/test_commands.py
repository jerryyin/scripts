"""Run the installed Docker commands without a Docker daemon or shell profile."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux', 'Linux Docker commands')
class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='docker commands ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "Someone's scripts/containers/docker"
        self.checkout.mkdir(parents=True)
        shutil.copytree(ROOT / 'bin', self.checkout / 'bin')
        self.compose = self.checkout / 'docker-compose.yml'
        self.compose.write_text('services: {}\n')
        self.bin = self.root / 'home/.local/bin'
        self.bin.mkdir(parents=True)
        for command in (self.checkout / 'bin').iterdir():
            command.chmod(0o755)
            (self.bin / command.name).symlink_to(command)
        self.log = self.root / 'calls.jsonl'
        self.environment = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                                USER='fixture', HOME=str(self.root / 'home'), CALL_LOG=str(self.log))
        for name in ('COMPOSE_FILE', 'COMPOSE_PROJECT_NAME', 'FAIL_DOCKER_COMMAND', 'DOCKER_NAMES'):
            self.environment.pop(name, None)
        self.executable('docker', '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'project': os.environ.get('COMPOSE_PROJECT_NAME')}) + '\\n')
if os.environ.get('FAIL_DOCKER_COMMAND') == sys.argv[1]:
    sys.exit(23)
if sys.argv[1] == 'ps':
    print(os.environ.get('DOCKER_NAMES', ''))
if sys.argv[1] == 'compose' and 'config' in sys.argv:
    print('triton\\nbase')
''')
        self.executable('date', '#!/bin/sh\nprintf 1010-1234\n')
        self.executable('sudo', '#!/bin/sh\nexec "$@"\n')

    def executable(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)

    def invoke(self, name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([name, *arguments], env=self.environment, cwd=self.root,
                              text=True, capture_output=True)

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_build_resolves_its_checkout_through_installed_links_and_preserves_arguments(self) -> None:
        result = self.invoke('dbuild', '--build-arg', 'VALUE=two words', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [{'args': ['compose', '-f', str(self.compose), 'build',
                                                 '--build-arg', 'VALUE=two words', 'triton'],
                                        'project': None}])

    def test_direct_compose_and_relative_override(self) -> None:
        override = self.root / 'alternate model.yml'
        override.write_text('services: {}\n')
        self.environment['COMPOSE_FILE'] = override.name
        self.environment['COMPOSE_PROJECT_NAME'] = 'existing-project'
        result = self.invoke('dcompose', 'logs', '--tail', '10', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]['args'], ['compose', '-f', str(override), 'logs', '--tail', '10', 'triton'])

    def test_run_identity_is_reused_for_logs_and_down(self) -> None:
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        project = self.calls()[-1]['project']
        self.assertIn('Compose project: ' + project, result.stdout)
        self.environment['COMPOSE_PROJECT_NAME'] = project
        self.executable('date', '#!/bin/sh\nprintf 1010-2359\n')
        for arguments in (('logs', 'triton'), ('down', 'triton')):
            result = self.invoke('dcompose', *arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.calls()[-1]['project'], project)

    def test_explicit_project_is_preserved_and_cli_can_override_it(self) -> None:
        self.environment['COMPOSE_PROJECT_NAME'] = 'chosen-project'
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1]['project'], 'chosen-project')
        result = self.invoke('dcompose', '-p', 'other-project', 'down')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1]['project'], 'other-project')

    def test_lifecycle_without_project_is_rejected_before_docker(self) -> None:
        for arguments in (('logs', 'triton'), ('down',), ('-p',)):
            result = self.invoke('dcompose', *arguments)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_run_validates_service_and_never_removes_existing_containers(self) -> None:
        self.environment['DOCKER_NAMES'] = 'other\nzyin-tools-setup\nzyin-tools-setup-backup'
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call['args'] for call in self.calls()], [
            ['compose', '-f', str(self.compose), '--profile', 'triton', 'config', '--services'],
            ['compose', '-f', str(self.compose), 'up', '-d', '--no-build', 'triton']])

    def test_unknown_service_never_starts_or_removes_a_container(self) -> None:
        result = self.invoke('drun', 'unknown')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('Unknown Compose service', result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn('config', self.calls()[0]['args'])

    def test_invalid_model_stops_before_launch(self) -> None:
        self.environment['DOCKER_NAMES'] = 'zyin-tools-setup'
        self.environment['FAIL_DOCKER_COMMAND'] = 'compose'
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn('config', self.calls()[0]['args'])

    def test_missing_arguments_or_compose_file_never_launch_docker(self) -> None:
        for command in ('drun', 'dbuild', 'dcompose'):
            with self.subTest(command=command):
                result = self.invoke(command)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Usage:', result.stderr)
        self.environment['COMPOSE_FILE'] = 'missing.yml'
        for command in ('dbuild', 'drun'):
            result = self.invoke(command, 'triton')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Compose file not found:', result.stderr)
        self.assertEqual(self.calls(), [])

    def test_raw_gpu_runner_preserves_arguments_and_uses_distinct_names(self) -> None:
        names = []
        for _ in range(2):
            result = self.invoke('dockrun', 'fixture:image', 'bash', '-c', 'echo two words')
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = self.calls()[-1]['args']
            self.assertEqual(arguments[-4:], ['fixture:image', 'bash', '-c', 'echo two words'])
            self.assertIn('--device=/dev/kfd', arguments)
            self.assertIn(str(self.root / 'home') + ':/zyin', arguments)
            names.append(arguments[arguments.index('--name') + 1])
        self.assertNotEqual(names[0], names[1])


if __name__ == '__main__':
    unittest.main()
