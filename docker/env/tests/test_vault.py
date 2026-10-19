import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'linux', 'Linux vault bootstrap')
class VaultTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='vault-setup-')
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.vault = self.home / 'vault'
        self.vault.mkdir()
        self.script = Path(__file__).resolve().parents[1] / 'vault.sh'
        self.environment = dict(os.environ, HOME=str(self.home))
        for name in ('DOCKER_CONFIG', 'DOCKER_AUTH_FILE', 'DOCKER_PLACEHOLDER',
                     'GH_CONFIG_DIR', 'GH_JERRYYIN_KEY_FILE', 'GH_AMDENG_KEY_FILE',
                     'GH_JERRYYIN_PLACEHOLDER', 'GH_AMDENG_PLACEHOLDER'):
            self.environment.pop(name, None)

    def run_vault(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', str(self.script), *arguments], env=self.environment,
                              text=True, capture_output=True)

    def test_docker_setup_reruns_and_rotation_still_work(self) -> None:
        directory = self.home / '.docker'
        directory.mkdir()
        (directory / 'config.json.template').write_text('{"auths":{"mkmhub.amd.com":{"auth":"__DOCKER_KEY__"}}}\n')
        secret = self.vault / 'docker_mkmhub_auth.txt'
        config = directory / 'config.json'
        for password in ('first', 'second'):
            encoded = base64.b64encode(('fixture:' + password).encode()).decode()
            secret.write_text(encoded)
            result = self.run_vault('docker')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(config.read_text())['auths']['mkmhub.amd.com']['auth'], encoded)
            timestamp = config.stat().st_mtime_ns
            self.assertEqual(self.run_vault('docker').returncode, 0)
            self.assertEqual(config.stat().st_mtime_ns, timestamp)

    def test_github_accounts_and_local_state_are_preserved(self) -> None:
        directory = self.home / '.config/gh'
        directory.mkdir(parents=True)
        (directory / 'hosts.yml.template').write_text('''github.com:
    users:
        first:
            oauth_token: __GH_TOKEN_JERRYYIN__
        second:
            oauth_token: __GH_TOKEN_AMDENG__
''')
        (self.vault / 'gh_token_jerryyin.txt').write_text('fixture-first')
        second = self.vault / 'gh_token_amdeng.txt'
        second.write_text('fixture-second')
        self.assertEqual(self.run_vault('gh').returncode, 0)
        config = directory / 'hosts.yml'
        self.assertIn('oauth_token: fixture-first', config.read_text())
        self.assertIn('oauth_token: fixture-second', config.read_text())
        config.write_text(config.read_text() + '    user: first\n')
        before = config.read_bytes(), config.stat().st_mtime_ns
        second.write_text('fixture-rotated')
        result = self.run_vault('gh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('refusing to re-seed', result.stdout)
        self.assertEqual((config.read_bytes(), config.stat().st_mtime_ns), before)

    def test_retired_claude_profile_cannot_overwrite_state(self) -> None:
        config = self.home / '.claude.json'
        config.write_text('{"projects":{"keep":{}},"machineID":"keep"}')
        (self.home / '.claude.json.template').write_text('{"env":{"key":"__CLAUDE_SUB_KEY__"}}')
        (self.vault / 'claude_key.txt').write_text('fixture-private-key')
        before = config.read_bytes(), config.stat().st_mtime_ns
        for arguments in (('claude',), ('claude', '--force'), ('claude', '--status')):
            result = self.run_vault(*arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((config.read_bytes(), config.stat().st_mtime_ns), before)
            self.assertNotIn('fixture-private-key', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
