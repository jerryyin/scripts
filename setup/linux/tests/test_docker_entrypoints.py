"""Exercise the actual Dockerfile RUN and Compose startup shell commands."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import unittest

from support import Fixture, ROOT, executable, recorder

try:
    import yaml
except ImportError:
    yaml = None

DOCKER_CONFIG = ROOT / 'containers/docker'


@unittest.skipUnless(yaml and (DOCKER_CONFIG / 'docker-compose.yml').exists(),
                     'PyYAML required')
class DockerEntrypointTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.compose = yaml.safe_load((DOCKER_CONFIG / 'docker-compose.yml').read_text())
        dockerfile = (DOCKER_CONFIG / 'base.dockerfile').read_text().replace('\\\n', ' ')
        self.build_command = next(command for command in re.findall(r'^RUN (.*)$', dockerfile, re.M)
                                  if 'git clone' in command)
        self.binaries = self.temp / 'bin'
        executable(self.binaries / 'apt-get', 'exit 0')
        executable(self.binaries / 'git', '''
printf '%s\\n' "$*" >> "$GIT_LOG"
if [[ "$1" == clone ]]; then
    [[ "${FAIL_CLONE:-}" != 1 ]] || exit 41
    for argument; do destination="$argument"; done
    mkdir -p "$destination"
    cp -a "$SOURCE_TREE/." "$destination/"
fi
''')
        recorder(self.binaries / 'zsh', 'shell')
        self.env.update(PATH=str(self.binaries) + os.pathsep + self.env['PATH'],
                        GIT_LOG=str(self.temp / 'git.log'), SOURCE_TREE=str(self.repo),
                        SCRIPTS_REPOSITORY='https://example.invalid/scripts.git', SCRIPTS_REF='')
        self.image_repo = self.temp / 'image' / 'scripts'

    def build(self, **overrides):
        shutil.rmtree(self.image_repo, ignore_errors=True)
        self.clear_log()
        # Only relocate the container's /root/scripts path. Execute the actual RUN
        # shell syntax and real profile runner, with external installers substituted.
        command = self.build_command.replace('/root/scripts', shlex.quote(str(self.image_repo)))
        env = {**self.env, 'SERVICE_NAME': '', 'SETUP_PROFILE': '', **overrides}
        # Docker RUN uses /bin/sh; the selected entry point then invokes Bash.
        return subprocess.run(['sh', '-c', command], env=env, cwd=self.temp,
                              text=True, capture_output=True)

    def test_every_compose_build_profile_keeps_its_project_setup(self):
        for service, config in self.compose['services'].items():
            if not config.get('build', {}).get('dockerfile', '').endswith('base.dockerfile'):
                continue
            with self.subTest(service=service):
                args = config['build']['args']
                result = self.build(SERVICE_NAME=args['SERVICE_NAME'],
                                    SETUP_PROFILE=args.get('SETUP_PROFILE', ''))
                self.assertEqual(result.returncode, 0, result.stderr)
                steps = [row[0] for row in self.calls()]
                self.assertIn('bootstrap.sh', steps)
                self.assertNotIn('credentials/activate.sh', steps)
                self.assertTrue(all(row[1] == '' for row in self.calls()))
                if service.startswith('triton'):
                    self.assertIn('profiles/triton/deps.sh', steps)
                    self.assertIn('packages/tokenspeed.sh', steps)
                    self.assertIn('profiles/triton/workspace.sh', steps)
                if service == 'iree':
                    self.assertIn('profiles/iree/deps.sh', steps)
                    self.assertIn('profiles/iree/workspace.sh', steps)

    def test_service_name_build_argument_remains_supported(self):
        result = self.build(SERVICE_NAME='iree')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('profiles/iree/workspace.sh', [row[0] for row in self.calls()])

    def test_explicit_profile_overrides_service_name_and_clone_ref_is_forwarded(self):
        result = self.build(SERVICE_NAME='base', SETUP_PROFILE='triton', SCRIPTS_REF='fixture-branch')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('profiles/triton/workspace.sh', [row[0] for row in self.calls()])
        self.assertIn('--branch fixture-branch https://example.invalid/scripts.git',
                      (self.temp / 'git.log').read_text())

    def test_missing_setup_entrypoint_fails_the_build(self):
        (self.repo / 'setup/linux/setup.sh').unlink()
        result = self.build(SERVICE_NAME='base')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_clone_and_installer_failures_fail_the_build(self):
        result = self.build(SERVICE_NAME='base', FAIL_CLONE='1')
        self.assertEqual(result.returncode, 41)
        self.assertFalse(self.log.exists())
        result = self.build(SERVICE_NAME='triton', FAIL_STEP='profiles/triton/deps.sh')
        self.assertEqual(result.returncode, 23)
        self.assertNotIn('profiles/triton/workspace.sh', [row[0] for row in self.calls()])

    def startup(self, command, **overrides):
        command = command.replace('$$', '$').replace('/bin/zsh', shlex.quote(str(self.binaries / 'zsh')))
        return self.run_shell('-c', command, cwd=self.repo.parent,
                              env=dict(self.env, **overrides), check=False)

    def test_compose_startup_activates_credentials_then_starts_shell(self):
        (self.repo.parent / 'scripts').symlink_to(self.repo, target_is_directory=True)
        command = self.compose['services']['triton']['command'][2]
        result = self.startup(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0], ['credentials/activate.sh', '/zyin', ''])
        self.assertEqual(self.calls()[-1][0], 'shell')

    def test_activation_failure_prevents_shell_startup(self):
        (self.repo.parent / 'scripts').symlink_to(self.repo, target_is_directory=True)
        result = self.startup(self.compose['services']['iree']['command'][2],
                              FAIL_STEP='credentials/activate.sh')
        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(self.calls()), 1)

    def test_missing_activation_entrypoint_prevents_shell_startup(self):
        (self.repo.parent / 'scripts').symlink_to(self.repo, target_is_directory=True)
        (self.repo / 'setup/linux/credentials/activate.sh').unlink()
        result = self.startup(self.compose['services']['base']['command'][2])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_base_image_pull_uses_build_sources_instead_of_service_image_tags(self):
        model = self.temp / 'compose.json'
        model.write_text(json.dumps(self.compose))
        executable(self.binaries / 'docker', '''
if [[ "$1" == compose ]]; then
    if [[ "$2" == version ]]; then exit 0; fi
    cat "$COMPOSE_MODEL"
elif [[ "$1" == pull ]]; then
    printf '%s|%s|%s\\n' pull '' "$2" >> "$CALL_LOG"
else
    exit 97
fi
''')
        self.run_shell(ROOT / 'containers/docker/pull-base-images.sh', 'triton', 'iree',
                       env=dict(self.env, COMPOSE_MODEL=str(model),
                                COMPOSE_FILE=str(DOCKER_CONFIG / 'docker-compose.yml')))
        self.assertEqual([row[2] for row in self.calls()], ['rocm/pytorch:latest', 'rocm/dev-ubuntu-24.04:latest'])


if __name__ == '__main__':
    unittest.main()
