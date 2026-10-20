"""Required bootstrap steps must not hide failures behind a later installer."""

import os
import unittest

from support import Fixture, ROOT, executable, recorder


class BootstrapTests(Fixture, unittest.TestCase):
    def test_credential_refresh_cannot_install_applications_and_propagates_failure(self):
        entry = self.repo / 'setup/linux/credentials/activate.sh'
        entry.write_text((ROOT / 'setup/linux/credentials/activate.sh').read_text())
        (self.home / '.ssh').mkdir()
        (self.home / '.ssh/id_ed25519').touch()
        (self.home / 'vault').mkdir()
        (self.home / 'vault/fixture').touch()
        claude = self.home / 'rc_files/claude/.setup/__main__.py'
        claude.parent.mkdir(parents=True)
        claude.touch()
        binaries = self.temp / 'bin'
        recorder(binaries / 'python3', 'rc_files')
        for fail in ('rc_files', ''):
            with self.subTest(fail=fail):
                self.clear_log()
                result = self.run_shell(entry, check=False, env=dict(
                    self.env, FAIL_STEP=fail, PATH=str(binaries) + os.pathsep + self.env['PATH']))
                self.assertEqual(result.returncode, 23 if fail else 0, result.stderr)
                calls = self.calls()
                self.assertEqual(calls[1][2], str(self.home / 'rc_files/.setup') +
                                 ' --package claude --no-install')
                self.assertEqual(len(calls), 2 if fail else 6)

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
