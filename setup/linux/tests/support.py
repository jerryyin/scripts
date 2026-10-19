"""Isolated setup fixtures: external installation steps only append to a log."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/bash\nset -e\n' + body + '\n')
    path.chmod(0o755)


def recorder(path, name):
    executable(path, f'''
printf '%s|%s|%s\\n' '{name}' "${{DEVENV_PERSISTENT_ROOT-unset}}" "$*" >> "$CALL_LOG"
if [[ "${{FAIL_STEP:-}}" == '{name}' ]]; then exit 23; fi
''')


class Fixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='scripts refactor ')
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.home = self.temp / 'home'
        self.home.mkdir()
        self.repo = self.temp / 'relocated checkout'
        self.log = self.temp / 'calls.log'
        self.env = dict(os.environ, HOME=str(self.home), CALL_LOG=str(self.log),
                        DEVENV_PERSISTENT_ROOT='', PYTHONDONTWRITEBYTECODE='1')
        self.env.pop('FAIL_STEP', None)
        self.env.pop('DEVENV_WORKSPACE_ROOT', None)
        # A missed installer substitute must fail, never install onto the host.
        guards = self.temp / 'blocked commands'
        for command in ('sudo', 'apt', 'apt-get', 'pip', 'pip3', 'npm', 'curl', 'wget', 'ssh', 'scp'):
            executable(guards / command,
                       'echo "Unexpected external operation in fixture: $0 $*" >&2; exit 97')
        self.env['PATH'] = str(guards) + os.pathsep + os.environ['PATH']
        shutil.copytree(ROOT / 'setup/linux', self.repo / 'setup/linux',
                        ignore=shutil.ignore_patterns('tests', '__pycache__', '.*.sw?'))
        for script in (self.repo / 'setup/linux').rglob('*.sh'):
            relative = script.relative_to(self.repo / 'setup/linux')
            is_plan = relative.parts[0] == 'profiles' and relative.name == 'profile.sh'
            if relative.parts[0] != 'lib' and relative != Path('setup.sh') and not is_plan:
                recorder(script, str(relative))

    def run_shell(self, *args, check=True, env=None, cwd=None):
        return subprocess.run(['bash', *map(str, args)], env=env or self.env,
                              cwd=cwd or self.temp, text=True, capture_output=True, check=check)

    def calls(self):
        return [line.split('|') for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def clear_log(self):
        self.log.unlink(missing_ok=True)
