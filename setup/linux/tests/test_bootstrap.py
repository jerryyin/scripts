"""Required bootstrap steps must not hide failures behind a later installer."""

import os
from pathlib import Path
import unittest

from support import Fixture, ROOT, executable, recorder


class BootstrapTests(Fixture, unittest.TestCase):
    def test_docker_command_installation_links_sources_and_is_repeatable(self):
        rc_root = Path(os.environ.get('RC_FILES_ROOT', ROOT.parent / 'rc_files'))
        if not (rc_root / '.setup/__main__.py').exists():
            self.skipTest('rc_files setup is required for the command installation check')
        (self.home / 'rc_files').symlink_to(rc_root, target_is_directory=True)
        source = (ROOT / 'setup/linux/bootstrap.sh').read_text()
        installer = self.temp / 'install-commands.sh'
        installer.write_text('set -e\nSCRIPT_DIR="$COMMAND_SETUP_DIR"\n' +
                             source[source.index('# Expose this checkout'):])
        environment = dict(self.env, COMMAND_SETUP_DIR=str(ROOT / 'setup/linux'))
        self.run_shell(installer, env=environment)
        commands = list((ROOT / 'containers/docker/bin').iterdir())
        self.assertEqual({path.name for path in commands}, {'drun', 'dbuild', 'dcompose'})
        stamps = {}
        for command in commands:
            target = self.home / '.local/bin' / command.name
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), command)
            stamps[command.name] = target.lstat().st_mtime_ns
        self.run_shell(installer, env=environment)
        self.assertEqual({name: (self.home / '.local/bin' / name).lstat().st_mtime_ns for name in stamps}, stamps)

    def test_credential_refresh_cannot_install_applications_and_propagates_failure(self):
        entry = self.repo / 'setup/linux/credentials/activate.sh'
        entry.write_text((ROOT / 'setup/linux/credentials/activate.sh').read_text())
        (self.home / '.ssh').mkdir()
        (self.home / '.ssh/id_ed25519').touch()
        (self.home / 'vault').mkdir()
        (self.home / 'vault/fixture').touch()
        setup = self.home / 'rc_files/.setup/__main__.py'
        setup.parent.mkdir(parents=True)
        setup.touch()
        binaries = self.temp / 'bin'
        recorder(binaries / 'python3', 'rc_files')
        for fail in ('rc_files', ''):
            with self.subTest(fail=fail):
                self.clear_log()
                result = self.run_shell(entry, check=False, env=dict(
                    self.env, FAIL_STEP=fail, PATH=str(binaries) + os.pathsep + self.env['PATH']))
                self.assertEqual(result.returncode, 23 if fail else 0, result.stderr)
                calls = self.calls()
                self.assertEqual(calls[0][2], str(self.home / 'rc_files/.setup') +
                                 ' credentials --vault ' + str(self.home / 'vault'))
                self.assertEqual(len(calls), 1)

    def test_project_profiles_leave_the_shared_editor_language_server_to_rc_files(self):
        for profile in ('triton', 'iree'):
            with self.subTest(profile=profile):
                source = (ROOT / 'setup/linux/profiles' / profile / 'deps.sh').read_text()
                self.assertNotRegex(source, r'\bclangd\b')

    def test_required_installers_propagate_failures(self):
        source = (ROOT / 'setup/linux/bootstrap.sh').read_text()
        # Execute the actual mandatory prerequisite/dotfile section. Earlier
        # repository probes and later host configuration are outside this test.
        start = source.index('# Machine and project prerequisites.')
        script = self.temp / 'required-bootstrap.sh'
        script.write_text('SCRIPT_DIR="$FIXTURE_LINUX"\n'
                          'wait_for_dpkg_lock() { :; }\n' + source[start:source.index('# Clone scripts', start)] +
                          '\npython3 "$HOME/rc_files/.setup"\n' +
                          '\nprintf "continued|unused|\\n" >> "$CALL_LOG"\n')
        (self.home / 'rc_files').mkdir()
        binaries = self.temp / 'bin'
        executable(binaries / 'sudo', 'exit "${APT_RESULT:-0}"')
        executable(binaries / 'python3', 'exit "${RC_RESULT:-0}"')
        for failure in ('apt', 'rc_files', None):
            with self.subTest(failure=failure):
                self.clear_log()
                env = dict(self.env, FIXTURE_LINUX=str(self.repo / 'setup/linux'),
                           PATH=str(binaries) + os.pathsep + self.env['PATH'],
                           APT_RESULT='23' if failure == 'apt' else '0',
                           RC_RESULT='23' if failure == 'rc_files' else '0')
                result = self.run_shell(script, check=False, env=env)
                self.assertEqual(result.returncode, 23 if failure else 0, result.stderr)
                self.assertEqual(any(row[0] == 'continued' for row in self.calls()), failure is None)


if __name__ == '__main__':
    unittest.main()
