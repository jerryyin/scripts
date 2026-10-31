"""Check the real Compose model without building images or using a daemon."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = [os.environ['COMPOSE_CLI']] if os.environ.get('COMPOSE_CLI') else ['docker', 'compose']


@unittest.skipUnless(sys.platform == 'linux' and shutil.which(CLI[0]), 'Linux Docker Compose CLI is required; no daemon needed')
class ComposeRoles(unittest.TestCase):
    def test_every_service_preserves_scope_mounts_and_rejects_old_startup(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            for scope in ('work', 'homelab'):
                root, vault = directory / scope, directory / f'vault-{scope}'
                env = dict(os.environ, VAULT_SCOPE=scope, DEVENV_ROLE_ROOT=str(root), DEVENV_VAULT_DIR=str(vault))
                command = [*CLI, '-f', str(ROOT / 'docker-compose.yml')]
                profiles = subprocess.check_output([*command, 'config', '--profiles'], env=env, text=True).splitlines()
                self.assertIn('triton-b0', profiles)
                for profile in profiles:
                    with self.subTest(scope=scope, profile=profile):
                        model = json.loads(subprocess.check_output([*command, '--profile', profile, 'config', '--format', 'json'], env=env, text=True))
                        service = model['services'][profile]
                        mounts = {mount['target']: mount for mount in service['volumes'] if mount['type'] == 'bind'}
                        self.assertEqual(mounts['/zyin']['source'], str(root))
                        selected = mounts[f'/zyin/vault-{scope}']
                        self.assertEqual(selected['source'], str(vault))
                        self.assertTrue(selected['read_only'])
                        self.assertNotIn('/data', mounts)
                        self.assertNotIn(str(Path.home()), [item['source'] for item in mounts.values()])
                        self.assertEqual(service['environment']['VAULT_SCOPE'], scope)
                        self.assertEqual(service['build']['context'], str(ROOT))
                        result = subprocess.run(service['command'], cwd=directory, env=env, text=True, capture_output=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('Rebuild this image', result.stderr)


if __name__ == '__main__':
    unittest.main()
