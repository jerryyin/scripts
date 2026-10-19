import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import unittest

from support import Fixture, ROOT, executable, recorder


class ProfileTests(Fixture, unittest.TestCase):
    def setup_profile(self, *args, **kwargs):
        return self.run_shell(self.repo / 'setup/linux/setup.sh', *args, **kwargs)

    def test_triton_keeps_dependencies_and_orders_credentials_before_workspace(self):
        self.setup_profile('triton')
        self.assertEqual([row[0] for row in self.calls()], [
            'bootstrap.sh', 'credentials/activate.sh', 'profiles/triton/deps.sh',
            'packages/tokenspeed.sh', 'profiles/triton/workspace.sh'])

    def test_build_has_workspaces_without_credentials_or_persistent_storage(self):
        self.setup_profile('iree', '--phase', 'build',
                           env=dict(self.env, DEVENV_PERSISTENT_ROOT='/should-not-be-used'))
        self.assertEqual(self.calls(), [
            ['bootstrap.sh', '', ''], ['profiles/iree/deps.sh', '', ''],
            ['profiles/iree/workspace.sh', '', '']])

    def test_activation_and_workspace_can_run_without_installers(self):
        for phase, expected in [('activate', 'credentials/activate.sh'),
                                ('workspace', 'profiles/iree/workspace.sh')]:
            with self.subTest(phase=phase):
                self.clear_log()
                self.setup_profile('iree', '--phase', phase)
                self.assertEqual([row[0] for row in self.calls()], [expected])

    def test_all_declared_profiles_have_valid_plans_and_no_side_effects(self):
        names = self.setup_profile('--list').stdout.splitlines()
        self.assertEqual(set(names), {'base', 'triton', 'iree'})
        for name in names:
            with self.subTest(profile=name):
                self.setup_profile(name, '--plan')
        self.assertFalse(self.log.exists())

    def test_invalid_profile_and_phase_fail_before_installation(self):
        for args in [('unknown',), ('aiter',), ('../base',),
                     ('base', '--phase', 'typo'), ('base', '--phase')]:
            with self.subTest(args=args):
                self.assertNotEqual(self.setup_profile(*args, check=False).returncode, 0)
        self.assertFalse(self.log.exists())

    def test_bootstrap_failure_stops_activation_and_workspace_creation(self):
        result = self.setup_profile('triton', check=False,
                                    env=dict(self.env, FAIL_STEP='bootstrap.sh'))
        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(self.calls()), 1)

    def test_project_install_failure_stops_workspace_creation(self):
        result = self.setup_profile('triton', check=False,
                                    env=dict(self.env, FAIL_STEP='packages/tokenspeed.sh'))
        self.assertEqual(result.returncode, 23)
        self.assertEqual([row[0] for row in self.calls()], [
            'bootstrap.sh', 'credentials/activate.sh', 'profiles/triton/deps.sh',
            'packages/tokenspeed.sh'])


class ModuleTests(Fixture, unittest.TestCase):
    def test_iree_driver_uses_numerics_helpers(self):
        entry = self.repo / 'iree/test.sh'
        entry.parent.mkdir()
        shutil.copy2(ROOT / 'iree/test.sh', entry)
        numerics = self.repo / 'numerics'
        numerics.mkdir()
        binaries = self.temp / 'bin'
        executable(binaries / 'python', '''
printf '%s\\n' "$1" >> "$CALL_LOG"
case "$1" in
    */numerics/genRandInput.py) touch "$2" ;;
    */numerics/compare.py) ;;
    *) exit 97 ;;
esac
''')
        executable(binaries / 'iree-compile', '''
while (( $# )); do
    if [[ "$1" == -o ]]; then touch "$2"; exit 0; fi
    shift
done
exit 97
''')
        executable(binaries / 'iree-run-module', '''
for argument; do
    if [[ "$argument" == --output=@* ]]; then touch "${argument#--output=@}"; exit 0; fi
done
exit 97
''')
        model = self.temp / 'fixture.mlir'
        model.touch()
        self.run_shell(entry, '-f', model, '-d', 'f32', '-i', '1', '--cpu',
                       env=dict(self.env, PATH=str(binaries) + os.pathsep + self.env['PATH']))
        self.assertEqual(self.log.read_text().splitlines(), [
            str(numerics / 'genRandInput.py'), str(numerics / 'compare.py')])

    def test_rc_shell_finds_ffm_plugin_in_am_directory(self):
        rc_root = Path(os.environ.get('RC_FILES_ROOT', ROOT.parent / 'rc_files'))
        source = (rc_root / 'zsh/.linux/.zshrc').read_text()
        start = source.index('function load_ffm_env()')
        end = source.index('\nload_ffm_env', start)
        function = source[start:end]
        vendor = self.temp / 'ffmlite_env.sh'
        vendor.write_text('# Disposable simulator environment\n')
        function = function.replace('/am-ffm/ffmlite_env.sh', shlex.quote(str(vendor)))
        plugin = self.home / 'scripts/am/ffm_teardown.py'
        plugin.parent.mkdir(parents=True)
        shutil.copy2(ROOT / 'am/ffm_teardown.py', plugin)
        code = '''import importlib.util; print(importlib.util.find_spec("ffm_teardown").origin)'''
        command = function + '\nload_ffm_env\npython3 -B -c ' + shlex.quote(code)
        result = subprocess.run(['zsh', '-f', '-c', command], cwd=self.temp,
                                env=dict(self.env, _TRITON_MODEL_PKG='', PYTHONPATH=''),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(plugin))

    def test_libraries_are_sourceable_without_clobbering_callers_directory(self):
        for library in (self.repo / 'setup/linux/lib').glob('*.sh'):
            with self.subTest(library=library.name):
                self.run_shell('-c', 'SCRIPT_DIR=caller; source "$1"; test "$SCRIPT_DIR" = caller',
                               'fixture', library)

    def test_gdb_utility_preserves_injected_gdb_and_exports_helpers(self):
        class Gdb:
            error = RuntimeError

            @staticmethod
            def execute(command, to_string=False):
                return '$1 = 42'

        namespace = runpy.run_path(str(ROOT / 'gdb/print_lanes.py'),
                                  init_globals={'gdb': Gdb})
        self.assertTrue(callable(namespace['print_variable_for_lanes']))
        self.assertIs(namespace['print_variable_for_lanes'].__globals__['gdb'], Gdb)

    def test_standalone_remote_helper_survives_scp_and_preserves_repository(self):
        target = self.repo / 'setup/linux/remote/remote-provision-host.sh'
        shutil.copy2(ROOT / 'setup/linux/remote/remote-provision-host.sh', target)
        (self.home / 'scripts').symlink_to(self.repo, target_is_directory=True)
        (self.repo / '.git').mkdir()
        ssh = self.home / '.ssh'
        ssh.mkdir()
        (ssh / 'fixture').write_text('not a key')
        (ssh / 'fixture.pub').write_text('not a key')
        binaries = self.temp / 'bin'
        recorder(binaries / 'git', 'git')
        env = dict(self.env, NO_BASE_PULL='1', PATH=str(binaries) + os.pathsep + self.env['PATH'])
        copied = self.home / 'remote-provision-host.sh'
        # Copy the canonical standalone helper, as batch-provision does with scp.
        shutil.copy2(target, copied)
        self.run_shell(copied, 'fixture', env=env)
        self.assertFalse(copied.exists())
        self.assertTrue(target.exists())
        self.assertEqual([row[0] for row in self.calls()], [
            'git', 'bootstrap.sh', 'credentials/activate.sh'])
        self.assertEqual(self.calls()[-1][2], '--force')

    def test_remote_bootstrap_failure_stops_activation_and_removes_only_the_copy(self):
        target = self.repo / 'setup/linux/remote/remote-provision-host.sh'
        shutil.copy2(ROOT / 'setup/linux/remote/remote-provision-host.sh', target)
        (self.home / 'scripts').symlink_to(self.repo, target_is_directory=True)
        (self.repo / '.git').mkdir()
        binaries = self.temp / 'bin'
        recorder(binaries / 'git', 'git')
        for standalone in (False, True):
            with self.subTest(standalone=standalone):
                self.clear_log()
                entry = target
                if standalone:
                    entry = self.home / 'remote-provision-host.sh'
                    shutil.copy2(target, entry)
                result = self.run_shell(entry, check=False, env=dict(
                    self.env, FAIL_STEP='bootstrap.sh', NO_BASE_PULL='1',
                    PATH=str(binaries) + os.pathsep + self.env['PATH']))
                self.assertEqual(result.returncode, 23, result.stderr)
                self.assertEqual([row[0] for row in self.calls()], ['git', 'bootstrap.sh'])
                self.assertTrue(target.exists())
                if standalone:
                    self.assertFalse(entry.exists())


if __name__ == '__main__':
    unittest.main()
