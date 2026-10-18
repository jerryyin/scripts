import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CodexSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='codex-setup-')
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        binaries = self.home / '.local/bin'
        binaries.mkdir(parents=True)
        for name, body in {
            'npm': 'case "$1" in prefix) echo "$HOME/.local" ;; view) echo 0.0.0 ;; esac',
            'codex': 'echo "codex 0.0.0"',
        }.items():
            executable = binaries / name
            executable.write_text('#!/bin/sh\n' + body + '\n')
            executable.chmod(0o755)
        self.environment = dict(os.environ, HOME=str(self.home), PATH=str(binaries) + os.pathsep + os.environ['PATH'])
        self.template = self.home / '.codex.config.toml.template'
        self.template.write_text('model_reasoning_effort = "max"\n')
        self.config = self.home / '.codex/config.toml'

    def setup_codex(self):
        return subprocess.run(
            ['bash', str(Path(__file__).resolve().parents[1] / 'codex.sh')],
            env=self.environment, text=True, capture_output=True, check=True,
        )

    def test_existing_local_configuration_and_agent_links_are_preserved(self):
        self.config.parent.mkdir()
        self.config.write_text('local_setting = "preserve"\n')
        os.utime(self.config, (1, 1))
        before = self.config.stat().st_mtime_ns
        instructions = self.home / 'instructions.md'
        instructions.write_text('local instructions')
        agent_link = self.config.parent / 'AGENTS.md'
        agent_link.symlink_to(instructions)
        skills = self.home / '.agents/skills'
        skills.mkdir(parents=True)
        (skills / 'local-skill').mkdir()
        self.setup_codex()
        self.assertEqual(self.config.read_text(), 'local_setting = "preserve"\n')
        self.assertEqual(self.config.stat().st_mtime_ns, before)
        self.assertEqual(agent_link.resolve(), instructions)
        self.assertTrue((skills / 'local-skill').is_dir())

    def test_missing_config_is_initialized_once(self):
        self.setup_codex()
        self.assertEqual(self.config.read_bytes(), self.template.read_bytes())
        before = self.config.stat().st_mtime_ns
        self.setup_codex()
        self.assertEqual(self.config.stat().st_mtime_ns, before)


if __name__ == '__main__':
    unittest.main()
