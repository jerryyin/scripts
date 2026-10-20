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
        for name in ('COMPOSE_FILE', 'FAIL_DOCKER_COMMAND', 'DOCKER_NAMES'):
            self.environment.pop(name, None)
        self.executable('docker', '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'project': os.environ.get('COMPOSE_PROJECT_NAME')}) + '\\n')
if os.environ.get('FAIL_DOCKER_COMMAND') == sys.argv[1]:
    sys.exit(23)
if sys.argv[1] == 'ps':
    print(os.environ.get('DOCKER_NAMES', ''))
''')
        self.executable('date', '#!/bin/sh\nprintf 1010-1234\n')

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
                                        'project': 'fixture-triton-1010-1234'}])

    def test_direct_compose_and_relative_override(self) -> None:
        override = self.root / 'alternate model.yml'
        override.write_text('services: {}\n')
        self.environment['COMPOSE_FILE'] = override.name
        result = self.invoke('dcompose', 'logs', '--tail', '10', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]['args'], ['compose', '-f', str(override), 'logs', '--tail', '10', 'triton'])

    def test_run_removes_only_the_existing_tools_container_and_skips_build(self) -> None:
        self.environment['DOCKER_NAMES'] = 'other\nzyin-tools-setup\nzyin-tools-setup-backup'
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call['args'] for call in self.calls()], [
            ['ps', '-a', '--format', '{{.Names}}'], ['rm', '-f', 'zyin-tools-setup'],
            ['compose', '-f', str(self.compose), 'up', '-d', '--no-build', 'triton']])

    def test_run_leaves_similarly_named_containers_alone(self) -> None:
        self.environment['DOCKER_NAMES'] = 'zyin-tools-setup-backup\nold-zyin-tools-setup'
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call['args'][0] for call in self.calls()], ['ps', 'compose'])

    def test_docker_errors_propagate_and_stop_later_operations(self) -> None:
        self.environment['DOCKER_NAMES'] = 'zyin-tools-setup'
        for failure, expected in (('ps', ['ps']), ('rm', ['ps', 'rm']), ('compose', ['ps', 'rm', 'compose'])):
            with self.subTest(failure=failure):
                self.log.unlink(missing_ok=True)
                self.environment['FAIL_DOCKER_COMMAND'] = failure
                result = self.invoke('drun', 'triton')
                self.assertEqual(result.returncode, 23, result.stderr)
                self.assertEqual([call['args'][0] for call in self.calls()], expected)

    def test_missing_arguments_or_compose_file_never_launch_docker(self) -> None:
        for command in ('drun', 'dbuild', 'dcompose'):
            with self.subTest(command=command):
                result = self.invoke(command)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Usage:', result.stderr)
        self.environment['COMPOSE_FILE'] = 'missing.yml'
        result = self.invoke('dbuild', 'triton')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Compose file not found:', result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()
