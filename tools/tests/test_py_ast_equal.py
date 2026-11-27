"""Tests for py_ast_equal.py on small synthetic sources and a throwaway git repository."""

from __future__ import annotations

import io
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

TOOL = Path(__file__).resolve().parents[1] / "py_ast_equal.py"
pae = SimpleNamespace(**runpy.run_path(str(TOOL)))

OLD = '''"""Module docstring."""
import os

LIMIT = 4  # a comment


class Box:
    """A box."""

    size = 1

    def grow(self, n):
        """Grow."""
        def step(k):
            return k + 1
        return step(n) * LIMIT

    def shrink(self):
        return self.size - 1


def helper(x):
    return -(-x // LIMIT)
'''


class PyAstEqualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = pae.main(list(argv))
        return rc, out.getvalue()

    def pair(self, old: str, new: str) -> tuple[str, str]:
        a, b = self.root / "a.py", self.root / "b.py"
        a.write_text(old, encoding="utf-8")
        b.write_text(new, encoding="utf-8")
        return str(a), str(b)

    def test_comments_and_formatting_are_equal(self):
        new = OLD.replace("LIMIT = 4  # a comment", "# moved comment\nLIMIT = (\n    4\n)")
        rc, out = self.run_main(*self.pair(OLD, new))
        self.assertEqual(rc, 0)
        self.assertTrue(out.startswith("equal "))

    def test_docstrings_differ_unless_ignored(self):
        new = OLD.replace('"""Grow."""', '"""Grow by n steps."""').replace('"""Module docstring."""', "")
        files = self.pair(OLD, new)
        rc, out = self.run_main(*files)
        self.assertEqual(rc, 1)
        self.assertIn("  changed Box.grow\n", out)
        self.assertIn("  changed <module>\n", out)
        rc, out = self.run_main("--ignore-docstrings", *files)
        self.assertEqual((rc, out.splitlines()[1:]), (0, []))

    def test_lists_only_the_innermost_changed_definition(self):
        new = OLD.replace("return k + 1", "return k + 2")
        rc, out = self.run_main(*self.pair(OLD, new))
        self.assertEqual(rc, 1)
        self.assertEqual(out.splitlines()[1:], ["  changed Box.grow.step"])

    def test_moved_function_is_removed_and_module_changed(self):
        new = OLD.replace("def helper(x):\n    return -(-x // LIMIT)\n", "from elsewhere import helper\n")
        rc, out = self.run_main(*self.pair(OLD, new))
        self.assertEqual(rc, 1)
        self.assertEqual(out.splitlines()[1:], ["  changed <module>", "  removed helper"])

    def test_duplicate_names_are_numbered(self):
        src = "if X:\n    def f():\n        return 1\nelse:\n    def f():\n        return 2\n"
        self.assertEqual(sorted(pae.definitions(src, "x.py", False)), ["<module>", "f", "f#2"])
        rc, out = self.run_main(*self.pair(src, src.replace("return 2", "return 3")))
        self.assertEqual(out.splitlines()[1:], ["  changed f#2"])

    def test_syntax_error_exits_2(self):
        rc, _ = self.run_main(*self.pair(OLD, "def broken(:\n"))
        self.assertEqual(rc, 2)

    def test_git_mode_reads_two_revisions(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        repo = self.root / "repo"
        repo.mkdir()
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git + ["init", "-q"], check=True)
        (repo / "m.py").write_text(OLD, encoding="utf-8")
        subprocess.run(git + ["add", "m.py"], check=True)
        subprocess.run(git + ["commit", "-qm", "a"], check=True)
        (repo / "m.py").write_text(OLD.replace('"""A box."""', '"""A bigger box."""'), encoding="utf-8")
        (repo / "n.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(git + ["add", "m.py", "n.py"], check=True)
        subprocess.run(git + ["commit", "-qm", "b"], check=True)
        args = ["--git", str(repo), "HEAD~1", "HEAD"]
        rc, out = self.run_main("--ignore-docstrings", *args, "m.py")
        self.assertEqual((rc, out), (0, "equal m.py\n"))
        rc, out = self.run_main(*args, "m.py", "n.py")
        self.assertEqual(rc, 1)
        self.assertEqual(out.splitlines(), ["differs m.py", "  changed Box", "differs n.py", "  added   <module>"])
        rc, _ = self.run_main(*args, "missing.py")
        self.assertEqual(rc, 2)

    def test_runs_copied_out_of_checkout(self):
        copied = shutil.copy2(TOOL, self.root / "elsewhere.py")  # runs copied out of the checkout
        a, b = self.pair(OLD, OLD.replace("self.size - 1", "self.size - 2"))
        proc = subprocess.run(
            [sys.executable, "-B", str(copied), a, b],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(proc.stdout.splitlines()[1:], ["  changed Box.shrink"])


if __name__ == "__main__":
    unittest.main()
