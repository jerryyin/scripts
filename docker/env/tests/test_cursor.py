import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CursorRuleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cursor-rules-')
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.rule = self.home / '.claude/rules/general.md'
        self.rule.parent.mkdir(parents=True)
        self.rule.write_text('---\ndescription: general guidance\n---\noriginal\n')
        workspace = self.home / 'project'
        (workspace / '.git').mkdir(parents=True)
        self.target = workspace / '.cursor/rules/general.mdc'
        self.target.parent.mkdir(parents=True)
        rc_root = os.environ.get('RC_ROOT', str(Path.home() / 'rc_files'))
        self.environment = dict(os.environ, HOME=str(self.home), RC_ROOT=rc_root)

    def setup_rules(self):
        return subprocess.run(
            ['bash', str(Path(__file__).resolve().parents[1] / 'cursor.sh')],
            env=self.environment, text=True, capture_output=True,
        )

    def test_old_hardlink_becomes_a_source_link_that_survives_source_replacement(self):
        os.link(self.rule, self.target)
        result = self.setup_rules()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(self.target.resolve(), self.rule)
        replacement = self.rule.with_suffix('.new')
        replacement.write_text('updated by an atomic save\n')
        os.replace(replacement, self.rule)
        self.assertEqual(self.target.read_text(), 'updated by an atomic save\n')
        before = self.target.lstat().st_mtime_ns
        self.assertEqual(self.setup_rules().returncode, 0)
        self.assertEqual(self.target.lstat().st_mtime_ns, before)

    def test_differing_local_rule_is_not_replaced(self):
        self.target.write_text('local changes\n')
        result = self.setup_rules()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Configuration conflict', result.stdout + result.stderr)
        self.assertEqual(self.target.read_text(), 'local changes\n')


if __name__ == '__main__':
    unittest.main()
