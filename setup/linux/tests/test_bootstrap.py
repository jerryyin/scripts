"""Required bootstrap steps must not hide failures behind a later installer."""

import os
import unittest

from support import Fixture, ROOT, executable


class BootstrapTests(Fixture, unittest.TestCase):
    def test_required_installers_propagate_failures(self):
        source = (ROOT / 'setup/linux/bootstrap.sh').read_text()
        # Execute the actual mandatory prerequisite/dotfile section. Earlier
        # repository probes and later host configuration are outside this test.
        start = source.index('# Repository probes above')
        end = source.index('\n. "$SCRIPT_DIR/lib/login_shell.sh"')
        script = self.temp / 'required-bootstrap.sh'
        script.write_text('SCRIPT_DIR="$FIXTURE_LINUX"\n'
                          'wait_for_dpkg_lock() { :; }\n' + source[start:end] +
                          '\nprintf "continued|unused|\\n" >> "$CALL_LOG"\n')
        (self.home / 'rc_files').mkdir()
        binaries = self.temp / 'bin'
        executable(binaries / 'sudo', 'exit "${APT_RESULT:-0}"')
        executable(binaries / 'python3', 'exit "${RC_RESULT:-0}"')
        for failure in ('apt', 'node', 'rc_files', None):
            with self.subTest(failure=failure):
                self.clear_log()
                env = dict(self.env, FIXTURE_LINUX=str(self.repo / 'setup/linux'),
                           PATH=str(binaries) + os.pathsep + self.env['PATH'],
                           APT_RESULT='23' if failure == 'apt' else '0',
                           RC_RESULT='23' if failure == 'rc_files' else '0',
                           FAIL_STEP='packages/node.sh' if failure == 'node' else '')
                result = self.run_shell(script, check=False, env=env)
                self.assertEqual(result.returncode, 23 if failure else 0, result.stderr)
                self.assertEqual(any(row[0] == 'continued' for row in self.calls()), failure is None)


if __name__ == '__main__':
    unittest.main()
