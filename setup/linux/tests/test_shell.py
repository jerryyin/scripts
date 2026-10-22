import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from support import ROOT


class ShellTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='project shell ')
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.env = dict(os.environ, HOME=str(self.home), AM_FFM_DIR=str(self.home / 'model'),
                        XDG_CONFIG_HOME=str(self.home / '.config'),
                        PYTHONPATH='existing-python', PYTEST_PLUGINS='existing_plugin',
                        GLIBC_TUNABLES='glibc.malloc.trim_threshold=1024')
        for name in ('_TRITON_MODEL_PKG', 'TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN',
                     'DEVENV_WORKSPACE_ROOT', 'TRITON_DIR', 'IREE_DIR', 'SETUP_PROFILE'):
            self.env.pop(name, None)

    def activate(self):
        code = '''import json, os, importlib.util
keys = ['PATH', 'PYTHONPATH', 'PYTEST_PLUGINS', 'GLIBC_TUNABLES', '_TRITON_MODEL_PKG', 'TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN', 'FIXTURE_MODEL_LOADS']
values = {key: os.environ.get(key) for key in keys}
plugin = importlib.util.find_spec('ffm_teardown')
values['plugin'] = plugin.origin if plugin else None
values['cwd'] = os.getcwd()
print(json.dumps(values))'''
        result = subprocess.run(['zsh', '-f', '-c', 'source "$1" || exit; source "$1" || exit; python3 -B -c "$2"',
                                 'fixture', str(ROOT / 'setup/linux/shell.zsh'), code],
                                cwd=self.home, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_missing_projects_leave_environment_unchanged(self):
        values = self.activate()
        for key in ('PATH', 'PYTHONPATH', 'PYTEST_PLUGINS', 'GLIBC_TUNABLES'):
            self.assertEqual(values[key], self.env[key])
        self.assertIsNone(values['_TRITON_MODEL_PKG'])
        self.assertIsNone(values['TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN'])
        self.assertEqual(values['cwd'], str(self.home))

    def test_project_paths_are_existing_idempotent_and_preserve_other_settings(self):
        binary = self.home / 'iree/build/dbg/tools'
        bindings = self.home / 'iree/build/model/compiler/bindings/python'
        triton = self.home / 'triton-mi450/python'
        for directory in (binary, bindings, triton):
            directory.mkdir(parents=True)
        for profile in ('', 'base', 'iree', 'triton'):
            with self.subTest(profile=profile):
                self.env['SETUP_PROFILE'] = profile
                values = self.activate()
                self.assertEqual(values['PATH'].split(':').count(str(binary)), int(profile == 'iree'))
                self.assertEqual(values['PYTHONPATH'].split(':').count(str(bindings)), int(profile == 'iree'))
                self.assertEqual(values['PYTHONPATH'].split(':').count(str(triton)), int(profile == 'triton'))
                self.assertNotIn(str(self.home / 'iree/build/model/runtime/bindings/python'), values['PYTHONPATH'])
                expected_tls = self.env['GLIBC_TUNABLES'] + (':glibc.rtld.optional_static_tls=4096' if profile == 'iree' else '')
                self.assertEqual(values['GLIBC_TUNABLES'], expected_tls)
                self.assertTrue(values['PYTHONPATH'].endswith('existing-python'))

    def test_model_finds_plugin_and_preserves_existing_plugins(self):
        model = self.home / 'model'
        model.mkdir()
        (model / 'ffmlite_env.sh').write_text('export FIXTURE_MODEL_LOADS="${FIXTURE_MODEL_LOADS:-}ffm"\n')
        (model / 'am_env.sh').write_text('export FIXTURE_MODEL_LOADS="${FIXTURE_MODEL_LOADS:-}am"\n')
        for profile in ('base', 'iree', 'triton'):
            with self.subTest(profile=profile):
                self.env['SETUP_PROFILE'] = profile
                values = self.activate()
                if profile == 'triton':
                    self.assertEqual(values['_TRITON_MODEL_PKG'], str(model))
                    self.assertEqual(values['plugin'], str(ROOT / 'am/ffm_teardown.py'))
                    self.assertEqual(values['PYTEST_PLUGINS'], 'existing_plugin,ffm_teardown')
                    self.assertEqual(values['FIXTURE_MODEL_LOADS'], 'ffm')
                else:
                    self.assertIsNone(values['_TRITON_MODEL_PKG'])
                    self.assertIsNone(values['plugin'])
                    self.assertEqual(values['PYTEST_PLUGINS'], 'existing_plugin')
                    self.assertIsNone(values['FIXTURE_MODEL_LOADS'])
        (model / 'ffmlite_env.sh').unlink()
        self.env['SETUP_PROFILE'] = 'triton'
        self.assertEqual(self.activate()['FIXTURE_MODEL_LOADS'], 'am')

    def test_explicit_project_paths_and_tls_setting_are_preserved(self):
        root = self.home / 'custom project'
        (root / 'python').mkdir(parents=True)
        (root / 'build/dbg/tools').mkdir(parents=True)
        self.env.update(IREE_DIR=str(root), TRITON_DIR=str(root),
                        GLIBC_TUNABLES='glibc.rtld.optional_static_tls=8192',
                        TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN='explicit-fixture')
        self.env['SETUP_PROFILE'] = 'iree'
        values = self.activate()
        self.assertIn(str(root / 'build/dbg/tools'), values['PATH'].split(':'))
        self.assertNotIn(str(root / 'python'), values['PYTHONPATH'].split(':'))
        self.assertEqual(values['GLIBC_TUNABLES'], self.env['GLIBC_TUNABLES'])
        self.assertEqual(values['TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN'], 'explicit-fixture')

    def test_saved_profile_is_used_and_environment_can_override_it(self):
        triton = self.home / 'triton-mi450/python'
        triton.mkdir(parents=True)
        selection = self.home / '.config/scripts/setup-profile'
        selection.parent.mkdir(parents=True)
        selection.write_text('triton\n')
        self.assertIn(str(triton), self.activate()['PYTHONPATH'].split(':'))
        self.env['SETUP_PROFILE'] = 'base'
        self.assertNotIn(str(triton), self.activate()['PYTHONPATH'].split(':'))

    def test_triton_token_is_loaded_only_for_triton_profile(self):
        vault = self.home / 'vault'
        vault.mkdir()
        (vault / 'gh_token_amdeng.txt').write_text(' fixture-token \n')
        self.assertIsNone(self.activate()['TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN'])
        self.env['SETUP_PROFILE'] = 'triton'
        self.assertEqual(self.activate()['TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN'], 'fixture-token')

    def test_rc_hook_is_optional_and_uses_an_explicit_scripts_checkout(self):
        rc_root = Path(os.environ.get('RC_FILES_ROOT', ROOT.parent / 'rc_files'))
        source = (rc_root / 'zsh/.linux/.zshrc').read_text()
        start = source.index('# Optional project environments')
        end = source.index('\nfi', start) + len('\nfi')
        hook = source[start:end] + '\nprint -r -- "loaded=${FIXTURE_LOADED:-no}"'
        checkout = self.home / 'relocated scripts'
        environment = dict(self.env, SCRIPTS_ROOT=str(checkout))
        for present in (False, True):
            if present:
                entry = checkout / 'setup/linux/shell.zsh'
                entry.parent.mkdir(parents=True)
                entry.write_text('export FIXTURE_LOADED=yes\n')
            result = subprocess.run(['zsh', '-f', '-c', hook], env=environment,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'loaded=yes' if present else 'loaded=no')


if __name__ == '__main__':
    unittest.main()
