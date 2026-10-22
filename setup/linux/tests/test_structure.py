"""Keep environment setup and Docker integration under their canonical owners."""
import unittest

from support import ROOT


class StructureTests(unittest.TestCase):
    def test_setup_root_contains_only_platforms_and_documentation(self):
        self.assertEqual({entry.name for entry in (ROOT / 'setup').iterdir()
                          if not entry.name.startswith('.')},
                         {'README.md', 'linux', 'windows'})

    def test_setup_and_docker_have_one_canonical_location(self):
        self.assertFalse((ROOT / 'devenv').exists())
        self.assertFalse((ROOT / 'docker').exists())
        self.assertTrue((ROOT / 'containers/docker/pull-base-images.sh').is_file())
        self.assertTrue((ROOT / 'setup/linux/tests/docker/integration.py').is_file())
        for name in ('env', 'workspace', 'lib', 'host', 'setup-service.sh'):
            with self.subTest(name=name):
                self.assertFalse((ROOT / 'containers/docker' / name).exists())

    def test_profiles_declare_steps_without_forwarding_to_other_profiles(self):
        plans = list((ROOT / 'setup/linux/profiles').glob('*/profile.sh'))
        self.assertEqual({plan.parent.name for plan in plans}, {'iree', 'triton'})
        self.assertFalse(list((ROOT / 'setup/linux/profiles').glob('*.sh')))
        self.assertFalse((ROOT / 'setup/linux/projects').exists())
        for profile in plans:
            with self.subTest(profile=profile.parent.name):
                lines = [line for line in profile.read_text().splitlines()
                         if line.strip() and not line.startswith('#')]
                self.assertEqual(len(lines), 2)
                self.assertTrue(lines[0].startswith('INSTALL_STEPS=('))
                self.assertTrue(lines[1].startswith('WORKSPACE_STEPS=('))

    def test_subject_directories_and_agent_utilities_have_canonical_owners(self):
        self.assertTrue((ROOT / 'tools/AGENTS.md').is_file())
        self.assertTrue((ROOT / 'tools/gdb_print_lanes.py').is_file())
        self.assertFalse((ROOT / 'gdb/print_lanes.py').exists())
        self.assertFalse((ROOT / 'kubernetes').exists())
        self.assertTrue((ROOT / 'containers/kubernetes').is_dir())
        for name in ('am', 'profiling', 'board', 'tools', 'bench', 'numerics',
                     'triton', 'iree', 'llvm', 'tokenspeed'):
            with self.subTest(subject=name):
                self.assertTrue((ROOT / name).is_dir())
        self.assertFalse((ROOT / 'unattended').exists())
        for name in ('claude', 'cursor', 'unattended'):
            with self.subTest(agent=name):
                self.assertTrue((ROOT / 'agent' / name).is_dir())

    def test_ssh_pod_bootstrap_does_not_install_an_editor(self):
        source = (ROOT / 'containers/kubernetes/interactive/pod-ssh-ephemeral.yml').read_text()
        install = next(line for line in source.splitlines() if 'apt-get install' in line)
        self.assertNotIn('vim', install.split())


if __name__ == '__main__':
    unittest.main()
