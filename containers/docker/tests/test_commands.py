"""Run the installed Docker commands without a Docker daemon or shell profile."""

import json
import os
from pathlib import Path
import shutil
import stat
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
        for name in ('session.py',):
            if (ROOT / name).exists():
                shutil.copyfile(ROOT / name, self.checkout / name)
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
        for scope in ('work', 'homelab'):
            vault = self.root / 'home' / f'vault-{scope}'
            vault.mkdir()
            (vault / '.vault-scope').write_text(scope + '\n')
        self.rc = self.root / 'home/rc_files/.setup'
        self.rc.mkdir(parents=True)
        (self.rc / '__main__.py').write_text('')
        for name in ('COMPOSE_FILE', 'COMPOSE_PROJECT_NAME', 'FAIL_DOCKER_COMMAND', 'DOCKER_NAMES',
                     'VAULT_SCOPE', 'VAULT_DIR', 'DEVENV_ROLE_ROOT', 'DEVENV_MODELS_DIR', 'RC_FILES_DIR',
                     'DOCKER_CONFIG'):
            self.environment.pop(name, None)
        self.executable('docker', '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'project': os.environ.get('COMPOSE_PROJECT_NAME')}) + '\\n')
with open(os.environ['CALL_LOG'] + '.env', 'w') as stream:
    json.dump({name: os.environ.get(name) for name in ('VAULT_SCOPE', 'DEVENV_ROLE_ROOT', 'DEVENV_VAULT_DIR', 'DEVENV_MODELS_DIR', 'DOCKER_CONFIG', 'SSH_AUTH_SOCK')}, stream)
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
            ['ps', '-a', '--format', '{{.Names}} {{.Label "com.docker.compose.project"}}'],
            ['compose', '-f', str(self.compose), 'up', '-d', '--no-build', 'triton']])

    def test_run_names_are_short_unique_and_take_an_optional_label(self) -> None:
        self.environment['DOCKER_NAMES'] = 'fixture-triton-1010-1234\nfixture-triton-1010-1234-2'
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1]['project'], 'fixture-triton-1010-1234-3')
        result = self.invoke('drun', 'triton', 'dsv41')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[-1]['project'], 'fixture-triton-1010-1234-dsv41')
        self.assertEqual(self.calls()[-1]['args'][-1], 'triton')

    def test_invalid_label_never_launches_docker(self) -> None:
        for arguments in (('triton', 'Not valid'), ('triton', '-x'), ('triton', 'one', 'two')):
            with self.subTest(arguments=arguments):
                result = self.invoke('drun', *arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Usage:', result.stderr)
        self.environment['COMPOSE_PROJECT_NAME'] = 'chosen-project'
        result = self.invoke('drun', 'triton', 'dsv41')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

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
            self.assertIn(str(self.root / 'home/.local/share/devenv/work') + ':/zyin', arguments)
            self.assertIn(str(self.root / 'home/.local/share/devenv/models') + ':/models:ro', arguments)
            self.assertNotIn(str(self.root / 'home') + ':/zyin', arguments)
            names.append(arguments[arguments.index('--name') + 1])
        self.assertNotEqual(names[0], names[1])

    def test_build_and_run_use_only_the_selected_vault_and_docker_login(self) -> None:
        for scope in ('work', 'homelab'):
            self.environment['VAULT_SCOPE'] = scope
            self.environment['SSH_AUTH_SOCK'] = '/host/agent'
            for command in ('dbuild', 'drun'):
                result = self.invoke(command, 'triton')
                self.assertEqual(result.returncode, 0, result.stderr)
                env = json.loads(Path(str(self.log) + '.env').read_text())
                root = self.root / 'home/.local/share/devenv' / scope
                self.assertEqual(env['DEVENV_ROLE_ROOT'], str(root))
                self.assertEqual(env['DEVENV_VAULT_DIR'], str(self.root / 'home' / f'vault-{scope}'))
                self.assertEqual(env['DOCKER_CONFIG'], str(root / '.docker'))
                self.assertEqual(env['DEVENV_MODELS_DIR'], str(self.root / 'home/.local/share/devenv/models'))
                self.assertIsNone(env['SSH_AUTH_SOCK'])

    def test_models_directory_follows_a_link_and_never_exposes_the_home(self) -> None:
        disk = self.root / 'raid/models'
        disk.mkdir(parents=True)
        link = self.root / 'home/.local/share/devenv/models'
        link.parent.mkdir(parents=True)
        link.symlink_to(disk)
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(Path(str(self.log) + '.env').read_text())['DEVENV_MODELS_DIR'], str(disk))
        calls = len(self.calls())
        for models in (self.root / 'home', self.root / 'home/.local/share/devenv', self.root):
            with self.subTest(models=models):
                self.environment['DEVENV_MODELS_DIR'] = str(models)
                result = self.invoke('drun', 'triton')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('models directory', result.stderr)
                self.assertEqual(len(self.calls()), calls)

    def test_seeded_keys_reach_the_role_directory_without_replacing_imports(self) -> None:
        source = self.root / 'home/.ssh'
        source.mkdir()
        for name in ('id_rsa', 'id_enterprise'):
            (source / name).write_text(f'{name} private\n')
            (source / f'{name}.pub').write_text(f'{name} public\n')
        seed = [sys.executable, str(self.checkout / 'session.py'), 'seed-ssh', 'id_rsa', 'id_enterprise']
        result = self.invoke(*seed)
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / 'home/.local/share/devenv/work/.ssh'
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
        for name, mode in (('id_rsa', 0o600), ('id_rsa.pub', 0o644),
                           ('id_enterprise', 0o600), ('id_enterprise.pub', 0o644)):
            self.assertEqual((target / name).read_bytes(), (source / name).read_bytes())
            self.assertEqual(stat.S_IMODE((target / name).stat().st_mode), mode)
        (target / 'id_rsa').write_text('imported earlier\n')
        result = self.invoke(*seed)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((target / 'id_rsa').read_text(), 'imported earlier\n')
        result = self.invoke('drun', 'triton')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_vault_or_host_home_mount_is_rejected_before_docker(self) -> None:
        self.environment['VAULT_DIR'] = str(self.root / 'home/vault-homelab')
        result = self.invoke('drun', 'triton')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('scope', result.stderr)
        self.assertEqual(self.calls(), [])
        self.environment.pop('VAULT_DIR')
        self.environment['DEVENV_ROLE_ROOT'] = str(self.root / 'home')
        result = self.invoke('drun', 'triton')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('home', result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()
