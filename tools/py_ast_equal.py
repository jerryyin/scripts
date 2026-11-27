#!/usr/bin/env python3
"""Prove that a Python file's code is unchanged between two versions by comparing their ASTs.

Use case: a comment-, docstring- or move-only edit must not change what a module executes.
`ast` already drops comments and formatting; `--ignore-docstrings` also drops the docstring of the
module and of every class and function. When the two ASTs differ, the tool lists exactly which
definitions differ, by qualified name (`Class.method`, `outer.inner`; `<module>` is the module's
own statements outside any definition), each as changed, added or removed.

    py_ast_equal.py [--ignore-docstrings] OLD.py NEW.py
    py_ast_equal.py [--ignore-docstrings] --git REPO REV_A REV_B PATH...

With --git, each PATH (relative to the repository root) is read at both revisions with
`git -C REPO show REV:PATH`; a path missing at one revision is reported as added or removed.
Prints one `equal PATH` or `differs PATH` line per file, then the differing definitions indented.
Exit status: 0 when every file is equal, 1 when any differs, 2 on a usage, read or syntax error.
Standard library only (Python 3.8+); --git needs `git` on PATH.
"""

from __future__ import annotations

import argparse
import ast
import copy
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
MODULE = "<module>"


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the leading string-constant statement of the module and of every class and function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module,) + DEFS) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:]
    return tree


class _Splitter(ast.NodeTransformer):
    """Replace each nested definition by a named placeholder and record it under its qualname."""

    def __init__(self, prefix: str, out: Dict[str, str]) -> None:
        self.prefix = prefix
        self.out = out

    def _definition(self, node: ast.AST) -> ast.AST:
        name = node.name  # type: ignore[attr-defined]
        qualname = _unique(self.prefix + name, self.out)
        _record(node, qualname + ".", qualname, self.out)
        return ast.Expr(value=ast.Constant(value="<definition %s>" % qualname))

    visit_FunctionDef = _definition
    visit_AsyncFunctionDef = _definition
    visit_ClassDef = _definition


def _unique(qualname: str, out: Dict[str, str]) -> str:
    """A second definition of the same name (a conditional def, a property setter) gets #2, #3 ..."""
    if qualname not in out:
        return qualname
    n = 2
    while "%s#%d" % (qualname, n) in out:
        n += 1
    return "%s#%d" % (qualname, n)


def _record(node: ast.AST, prefix: str, qualname: str, out: Dict[str, str]) -> None:
    out[qualname] = ""  # reserve the name before its children so nesting order is stable
    shell = copy.copy(node)
    splitter = _Splitter(prefix, out)
    # the definition's own fields (decorators, arguments, bases, body) with nested definitions cut out
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, ast.AST):
                    item = splitter.visit(copy.deepcopy(item))
                items.append(item)
            setattr(shell, field, items)
        elif isinstance(value, ast.AST):
            setattr(shell, field, splitter.visit(copy.deepcopy(value)))
    out[qualname] = ast.dump(shell)


def definitions(source: str, filename: str, ignore_docstrings: bool) -> Dict[str, str]:
    """Map each qualname (and `<module>`) to the dump of its own statements."""
    tree = ast.parse(source, filename=filename)
    if ignore_docstrings:
        strip_docstrings(tree)
    out: Dict[str, str] = {}
    _record(tree, "", MODULE, out)
    return out


def compare(
    old: Optional[str], new: Optional[str], filename: str, ignore_docstrings: bool
) -> List[Tuple[str, str]]:
    """Return (kind, qualname) for every definition that differs; empty when the ASTs are equal."""
    a = definitions(old, filename, ignore_docstrings) if old is not None else {}
    b = definitions(new, filename, ignore_docstrings) if new is not None else {}
    result = []
    for name in a:
        if name not in b:
            result.append(("removed", name))
        elif a[name] != b[name]:
            result.append(("changed", name))
    result.extend(("added", name) for name in b if name not in a)
    return result


def git_blob(repo: str, rev: str, path: str) -> Optional[str]:
    """The file's text at REV, or None when REV does not contain PATH."""
    exists = subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", "%s:%s" % (rev, path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "--quiet", rev + "^{commit}"],
                       check=True, stdout=subprocess.DEVNULL)
        return None
    blob = subprocess.run(
        ["git", "-C", repo, "show", "%s:%s" % (rev, path)], check=True, stdout=subprocess.PIPE
    )
    return blob.stdout.decode("utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ignore-docstrings", action="store_true",
                        help="drop module, class and function docstrings before comparing")
    parser.add_argument("--git", nargs=3, metavar=("REPO", "REV_A", "REV_B"),
                        help="read each PATH at two revisions of REPO")
    parser.add_argument("paths", nargs="+", metavar="PATH",
                        help="OLD.py NEW.py, or with --git the repository-relative paths")
    args = parser.parse_args(argv)
    if args.git is None and len(args.paths) != 2:
        parser.error("give OLD.py NEW.py, or --git REPO REV_A REV_B PATH...")
    try:
        if args.git is None:
            with open(args.paths[0], encoding="utf-8") as fa, open(args.paths[1], encoding="utf-8") as fb:
                pairs = [("%s %s" % tuple(args.paths), fa.read(), fb.read())]
        else:
            repo, rev_a, rev_b = args.git
            pairs = [(p, git_blob(repo, rev_a, p), git_blob(repo, rev_b, p)) for p in args.paths]
        differs = False
        for label, old, new in pairs:
            if old is None and new is None:
                print("py_ast_equal: %s exists at neither revision" % label, file=sys.stderr)
                return 2
            diffs = compare(old, new, label, args.ignore_docstrings)
            print("%s %s" % ("differs" if diffs else "equal", label))
            for kind, name in diffs:
                print("  %-7s %s" % (kind, name))
            differs = differs or bool(diffs)
    except (OSError, SyntaxError, UnicodeDecodeError, subprocess.CalledProcessError) as exc:
        print("py_ast_equal: %s" % exc, file=sys.stderr)
        return 2
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
