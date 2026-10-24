"""Existing checkouts follow their remote while keeping local edits."""

from pathlib import Path
import subprocess
import tempfile
import unittest

LIBRARY = Path(__file__).resolve().parents[1] / 'lib/sync_checkout.sh'
# Git treats edits on adjacent lines as conflicting; separate independent edits.
SPACER = ''.join(f'# unchanged {number}\n' for number in range(5))


def git(*arguments: str, cwd: Path) -> str:
    return subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                           '-c', 'commit.gpgsign=false', *arguments],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class SyncCheckoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.source = root / 'source'
        self.source.mkdir()
        git('init', '--quiet', '--initial-branch=main', cwd=self.source)
        (self.source / 'settings.conf').write_text(f'color = old\nsize = 10\n{SPACER}')
        git('add', '.', cwd=self.source)
        git('commit', '--quiet', '-m', 'initial', cwd=self.source)
        self.clone = root / 'clone'
        git('clone', '--quiet', str(self.source), str(self.clone), cwd=root)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def publish(self, text: str) -> None:
        (self.source / 'settings.conf').write_text(text)
        git('commit', '--quiet', '-am', 'update', cwd=self.source)

    def sync(self) -> str:
        return subprocess.run(['bash', '-c', f'. "{LIBRARY}"; sync_checkout "$0"', str(self.clone)],
                              check=True, capture_output=True, text=True).stdout

    def test_update_restores_nonconflicting_local_edits(self) -> None:
        self.publish(f'color = new\nsize = 10\n{SPACER}')
        (self.clone / 'settings.conf').write_text(f'color = old\nsize = 10\n{SPACER}font = mine\n')
        (self.clone / 'local.txt').write_text('untracked note\n')
        self.sync()
        self.assertEqual((self.clone / 'settings.conf').read_text(), f'color = new\nsize = 10\n{SPACER}font = mine\n')
        self.assertTrue((self.clone / 'local.txt').exists())
        self.assertEqual(git('stash', 'list', cwd=self.clone), '')

    def test_unchanged_upstream_preserves_staged_edits(self) -> None:
        (self.clone / 'settings.conf').write_text('color = staged\n')
        git('add', 'settings.conf', cwd=self.clone)
        self.sync()
        self.assertEqual(git('diff', '--cached', '--name-only', cwd=self.clone), 'settings.conf')
        self.assertEqual(git('diff', '--name-only', cwd=self.clone), '')

    def test_conflicting_local_edits_yield_to_upstream_and_stay_stashed(self) -> None:
        self.publish('color = new\nsize = 10\n')
        (self.clone / 'settings.conf').write_text('color = mine\nsize = 10\n')
        output = self.sync()
        self.assertEqual((self.clone / 'settings.conf').read_text(), 'color = new\nsize = 10\n')
        self.assertIn('kept in git stash list', output)
        self.assertIn('bootstrap sync', git('stash', 'list', cwd=self.clone))

    def test_renamed_default_branch_is_followed(self) -> None:
        git('checkout', '--quiet', '-b', 'trunk', cwd=self.source)
        self.publish('color = trunk\nsize = 10\n')
        git('symbolic-ref', 'HEAD', 'refs/heads/trunk', cwd=self.source)
        git('branch', '--quiet', '-D', 'main', cwd=self.source)
        self.sync()
        self.assertEqual(git('rev-parse', '--abbrev-ref', 'HEAD', cwd=self.clone), 'trunk')
        self.assertEqual((self.clone / 'settings.conf').read_text(), 'color = trunk\nsize = 10\n')

    def test_unreachable_origin_only_warns(self) -> None:
        git('remote', 'set-url', 'origin', str(self.source.parent / 'missing'), cwd=self.clone)
        self.assertIn('cannot reach origin', self.sync())


if __name__ == '__main__':
    unittest.main()
