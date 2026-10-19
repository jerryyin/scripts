import os
from pathlib import Path
import shutil
import subprocess
import unittest

from support import Fixture, ROOT, executable


class WorkspaceTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.workspace = self.temp / 'project workspaces'
        self.env['DEVENV_WORKSPACE_ROOT'] = str(self.workspace)

    def test_real_local_git_clone_uses_explicit_workspace_root_and_reuses_checkout(self):
        source = self.temp / 'source'
        subprocess.run(['git', 'init', '-q', str(source)], check=True, env=self.env)
        (source / 'file.txt').write_text('source content\n')
        subprocess.run(['git', '-C', str(source), 'add', '.'], check=True, env=self.env)
        subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'],
                       check=True, env=self.env)
        helper = self.repo / 'setup/linux/lib/git_workspace.sh'
        script = 'source "$1"; setup_workspace sample "$2"'
        self.run_shell('-c', script, 'fixture', helper, source)
        checkout = self.workspace / 'sample'
        self.assertEqual((checkout / 'file.txt').read_text(), 'source content\n')
        (checkout / 'file.txt').write_text('local work\n')
        self.run_shell('-c', script, 'fixture', helper, source)
        self.assertEqual((checkout / 'file.txt').read_text(), 'local work\n')
        self.assertFalse((self.home / 'sample').exists())

    def test_storage_override_empty_and_invalid_paths(self):
        helper = self.repo / 'setup/linux/lib/find_persistent_root.sh'
        for value in ('', str(self.home), str(self.temp / 'missing')):
            with self.subTest(value=value):
                result = self.run_shell('-c', 'source "$1"; find_persistent_root', 'fixture', helper,
                                        env=dict(self.env, DEVENV_PERSISTENT_ROOT=value), check=False)
                if value.endswith('/missing'):
                    self.assertNotEqual(result.returncode, 0)
                else:
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, value + '\n')

    def test_iree_workspace_finds_helpers_and_presets_from_relocated_repository(self):
        target = self.repo / 'setup/linux/profiles/iree/workspace.sh'
        shutil.copy2(ROOT / 'setup/linux/profiles/iree/workspace.sh', target)
        for relative in ('iree/CMakePresets.json', 'llvm/CMakeUserPresets.json'):
            preset = self.repo / relative
            preset.parent.mkdir(parents=True, exist_ok=True)
            preset.write_text('{}\n')
        binaries = self.temp / 'bin'
        executable(binaries / 'git', '''
if [[ "$1" == clone ]]; then
    mkdir -p "$3/third_party/llvm-project/llvm" "$3/.git"
fi
''')
        executable(binaries / 'pre-commit', 'exit 0')
        self.run_shell(target, env=dict(self.env, PATH=str(binaries) + os.pathsep + self.env['PATH']))
        checkout = self.workspace / 'iree'
        self.assertEqual((checkout / 'CMakePresets.json').resolve(), self.repo / 'iree/CMakePresets.json')
        self.assertEqual((checkout / 'third_party/llvm-project/llvm/CMakeUserPresets.json').resolve(),
                         self.repo / 'llvm/CMakeUserPresets.json')

    def test_tokenspeed_rerun_preserves_existing_checkout_and_local_work(self):
        target = self.repo / 'setup/linux/packages/tokenspeed.sh'
        shutil.copy2(ROOT / 'setup/linux/packages/tokenspeed.sh', target)
        checkout = self.workspace / 'tokenspeed'
        subprocess.run(['git', 'init', '-q', str(checkout)], check=True, env=self.env)
        subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', 'https://example.invalid'],
                       check=True, env=self.env)
        requirements = checkout / 'tokenspeed-kernel/python/requirements/rocm.txt'
        requirements.parent.mkdir(parents=True)
        requirements.write_text('fixture-requirement\n')
        local_file = checkout / 'local-work.txt'
        local_file.write_text('keep me')
        binaries = self.temp / 'bin'
        executable(binaries / 'sudo', 'exit 0')
        executable(binaries / 'python3', 'exit 0')
        env = dict(self.env, PATH=str(binaries) + os.pathsep + self.env['PATH'])
        for _ in range(2):
            self.run_shell(target, env=env)
        self.assertEqual(local_file.read_text(), 'keep me')
        remotes = subprocess.run(['git', '-C', str(checkout), 'remote'], text=True,
                                  capture_output=True, check=True, env=self.env).stdout.splitlines()
        self.assertEqual(remotes, ['jerryyin', 'origin'])


if __name__ == '__main__':
    unittest.main()
