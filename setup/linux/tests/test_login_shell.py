import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('zsh'), 'Zsh required')
class LoginShellTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='login-shell-')
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.helper = Path(__file__).resolve().parents[1] / 'lib/login_shell.sh'

    def configure(self, current='/bin/bash', sudo_result=1):
        script = '''
source "$1"
id() { echo fixture; }
getent() { printf 'fixture:x:1000:1000:Fixture:/home/fixture:%s\n' "$LOGIN_SHELL"; }
sudo() { printf '%s\n' "$*" >> "$HOME/sudo.log"; return "$SUDO_RESULT"; }
configure_login_shell
'''
        subprocess.run(['bash', '-c', script, 'test', str(self.helper)], check=True,
                       env=dict(os.environ, HOME=str(self.home), LOGIN_SHELL=current,
                                SUDO_RESULT=str(sudo_result)), capture_output=True)

    def test_selected_shell_needs_no_account_or_bash_changes(self):
        self.configure(current=shutil.which('zsh'))
        self.assertFalse((self.home / 'sudo.log').exists())
        self.assertFalse((self.home / '.bashrc').exists())

    def test_successful_noninteractive_account_change_needs_no_fallback(self):
        self.configure(sudo_result=0)
        self.assertIn('-n chsh -s ', (self.home / 'sudo.log').read_text())
        self.assertFalse((self.home / '.bashrc').exists())

    def test_fallback_preserves_local_content_and_is_repeatable(self):
        bashrc = self.home / '.bashrc'
        bashrc.write_text('# personal settings\n')
        self.configure()
        before = bashrc.stat().st_mtime_ns
        self.configure()
        self.assertEqual(bashrc.stat().st_mtime_ns, before)
        self.assertTrue(bashrc.read_text().startswith('# personal settings\n'))
        self.assertEqual(bashrc.read_text().count('auto-exec zsh (chsh unavailable) >>>'), 1)
        self.assertIn('$HOME/.config/zsh/interactive.sh', bashrc.read_text())
