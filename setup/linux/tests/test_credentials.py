import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'linux', 'Linux session activation')
class ActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='session-credentials-')
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.vault = self.home / 'vault'
        self.vault.mkdir()
        (self.vault / 'fake').touch()
        self.storage = self.home / 'storage'
        self.storage.mkdir()
        self.rc = self.home / 'rc_files'
        self.rc.mkdir()
        setup = self.rc / '.setup'
        setup.mkdir()
        (setup / '__main__.py').write_text('''import json, sys
from pathlib import Path
with (Path.home() / 'calls').open('a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
''')
        ssh = self.home / '.ssh'
        ssh.mkdir()
        (ssh / 'id_ed25519').write_text('fake-existing-key')
        self.root = Path(__file__).resolve().parents[1]
        self.script = self.root / 'credentials/activate.sh'
        self.environment = dict(os.environ, HOME=str(self.home), VAULT_DIR=str(self.vault),
                                RC_FILES_DIR=str(self.rc), DEVENV_PERSISTENT_ROOT=str(self.storage))

    def activate(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', str(self.script)], env=self.environment,
                              text=True, capture_output=True)

    def test_runtime_delegates_only_credentials_after_vault_availability(self) -> None:
        for _ in range(2):
            result = self.activate()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in (self.home / 'calls').read_text().splitlines()]
        self.assertEqual(calls, [['credentials', '--vault', str(self.vault),
                                 '--persistent-root', str(self.storage)]] * 2)
        self.assertEqual((self.home / '.ssh/id_ed25519').read_text(), 'fake-existing-key')

    def test_persistence_opt_out_and_custom_vault(self) -> None:
        self.environment['DEVENV_PERSISTENT_ROOT'] = ''
        result = self.activate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.home / 'calls').read_text()),
                         ['credentials', '--vault', str(self.vault)])

    def test_missing_configuration_stops_and_propagates_hook_failure(self) -> None:
        entry = self.rc / '.setup/__main__.py'
        entry.write_text('raise SystemExit(23)')
        self.assertEqual(self.activate().returncode, 23)
        entry.unlink()
        result = self.activate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing rc_files setup', result.stderr)

    def test_ssh_restore_keeps_existing_configuration(self) -> None:
        fixture = self.home / 'fixture'
        (fixture / 'credentials').mkdir(parents=True)
        (fixture / 'lib').mkdir()
        shutil.copyfile(self.script, fixture / 'credentials/activate.sh')
        shutil.copyfile(self.root / 'lib/find_persistent_root.sh', fixture / 'lib/find_persistent_root.sh')
        (fixture / 'lib/fix_hostname.sh').write_text('fix_hostname() { :; }\n')
        self.script = fixture / 'credentials/activate.sh'
        (self.vault / '.git').mkdir()
        (self.home / '.ssh/id_ed25519').unlink()
        config = self.home / '.ssh/config'
        config.write_text('existing host aliases')
        keys = self.storage / '.ssh'
        keys.mkdir()
        (keys / 'id_ed25519').write_text('fake-restored-key')
        (keys / 'config').write_text('do not import host configuration')
        result = self.activate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(config.read_text(), 'existing host aliases')
        self.assertEqual((self.home / '.ssh/id_ed25519').read_text(), 'fake-restored-key')
        self.assertEqual((self.home / '.ssh/id_ed25519').stat().st_mode & 0o777, 0o600)

    def test_build_phase_excludes_activation(self) -> None:
        result = subprocess.run(['bash', str(self.root / 'setup.sh'), 'base', '--phase', 'build', '--plan'],
                                env=self.environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('credentials/activate.sh', result.stdout)
        self.assertFalse((self.home / 'calls').exists())

    @unittest.skipUnless(os.environ.get('RC_FILES_TEST_ROOT'), 'Set RC_FILES_TEST_ROOT for real package hooks')
    def test_actual_packages_activate_fake_credentials_without_installation(self) -> None:
        self.environment['RC_FILES_DIR'] = os.environ['RC_FILES_TEST_ROOT']
        (self.vault / 'claude_key.txt').write_text('fake-claude-key')
        (self.vault / 'docker_mkmhub_auth.txt').write_text('ZmFrZTpzZWNyZXQ=')
        (self.vault / 'gh_token_jerryyin.txt').write_text('fake-personal-token')
        (self.vault / 'gh_token_amdeng.txt').write_text('fake-work-token')
        (self.vault / 'atlartifactory_token.txt').write_text('fake-artifactory-token')
        codex = self.storage / '.codex'
        codex.mkdir()
        (codex / 'auth.json').write_text('{"fake":"auth"}')
        for name in ('CLAUDE_CONFIG_DIR', 'CLAUDE_KEY_FILE', 'DOCKER_CONFIG', 'DOCKER_AUTH_FILE',
                     'GH_CONFIG_DIR', 'CODEX_HOME', 'GH_JERRYYIN_KEY_FILE', 'GH_AMDENG_KEY_FILE',
                     'GIST_KEY_FILE', 'GIST_CONFIG', 'ARTIFACTORY_KEY_FILE'):
            self.environment.pop(name, None)
        for _ in range(2):
            result = self.activate()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('fake-claude-key', result.stdout + result.stderr)
        settings = json.loads((self.home / '.claude/settings.json').read_text())
        self.assertEqual(settings['env']['ANTHROPIC_CUSTOM_HEADERS'], 'Ocp-Apim-Subscription-Key: fake-claude-key')
        self.assertEqual(json.loads((self.home / '.docker/config.json').read_text())['auths']['mkmhub.amd.com']['auth'],
                         'ZmFrZTpzZWNyZXQ=')
        self.assertEqual((self.home / '.codex/auth.json').resolve(), codex / 'auth.json')
        self.assertFalse((self.home / '.codex').is_symlink())


if __name__ == '__main__':
    unittest.main()
