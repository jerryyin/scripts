import os
import shutil
import unittest

from support import Fixture, ROOT, executable, recorder


class ProfileTests(Fixture, unittest.TestCase):
    def setup_profile(self, *args, **kwargs):
        return self.run_shell(self.repo / 'setup/linux/setup.sh', *args, **kwargs)

    def test_successful_setup_remembers_only_its_selected_shell_profile(self):
        selection = self.home / '.config/scripts/setup-profile'
        self.setup_profile('iree', '--phase', 'build')
        self.assertEqual(selection.read_text(), 'iree\n')
        for arguments in (('triton', '--plan'), ('triton', '--phase', 'activate')):
            self.setup_profile(*arguments)
            self.assertEqual(selection.read_text(), 'iree\n')
        result = self.setup_profile('triton', check=False,
                                    env=dict(self.env, FAIL_STEP='profiles/triton/deps.sh'))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(selection.read_text(), 'iree\n')
        self.setup_profile('triton', '--phase', 'workspace')
        self.assertEqual(selection.read_text(), 'triton\n')

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

    def test_libraries_are_sourceable_without_clobbering_callers_directory(self):
        for library in (self.repo / 'setup/linux/lib').glob('*.sh'):
            with self.subTest(library=library.name):
                self.run_shell('-c', 'SCRIPT_DIR=caller; source "$1"; test "$SCRIPT_DIR" = caller',
                               'fixture', library)

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

    def test_remote_optional_prefetch_failure_reports_warning_after_setup(self):
        target = self.repo / 'setup/linux/remote/remote-provision-host.sh'
        shutil.copy2(ROOT / 'setup/linux/remote/remote-provision-host.sh', target)
        (self.home / 'scripts').symlink_to(self.repo, target_is_directory=True)
        (self.repo / '.git').mkdir()
        binaries = self.temp / 'bin'
        recorder(binaries / 'git', 'git')
        recorder(self.repo / 'containers/docker/pull-base-images.sh', 'prefetch')
        result = self.run_shell(target, env=dict(
            self.env, NO_BASE_PULL='', FAIL_STEP='prefetch',
            PATH=str(binaries) + os.pathsep + self.env['PATH']))
        self.assertIn('Optional base-image prefetch failed', result.stderr)
        self.assertEqual([row[0] for row in self.calls()], [
            'git', 'bootstrap.sh', 'credentials/activate.sh', 'prefetch'])


if __name__ == '__main__':
    unittest.main()
