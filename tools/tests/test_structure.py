import ast
from pathlib import Path
import re
import unittest


TOOLS = Path(__file__).resolve().parents[1]


def dependencies(source: str, siblings: set[str], repository_modules: set[str]) -> list[str]:
    errors = []
    forbidden = repository_modules | siblings | {"tools", "tests", "importlib", "runpy"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                errors.append("relative import")
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        errors.extend(f"forbidden import: {name}" for name in names if name in forbidden)
        if isinstance(node, ast.Name) and node.id == "__import__":
            errors.append("dynamic import")
        if isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "sys":
            errors.append("import-path access")
        if isinstance(node, ast.ImportFrom) and node.module == "sys" and any(alias.name == "path" for alias in node.names):
            errors.append("import-path access")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for sibling in siblings:
                if re.search(r"(?<![\w])" + re.escape(sibling) + r"\.py\b", node.value):
                    errors.append(f"sibling file dependency: {sibling}.py")
    return errors


class StructureTests(unittest.TestCase):
    def test_flat_documented_independent_tools(self):
        scripts = list(TOOLS.glob("*.py"))
        self.assertTrue(scripts, "tools must contain at least one documented utility")
        readme = (TOOLS / "README.md").read_text(encoding="utf-8")
        modules = {path.name for path in TOOLS.parent.iterdir() if path.is_dir() and not path.name.startswith(".")}
        modules |= {path.stem for path in TOOLS.parent.glob("*.py")}
        for entry in TOOLS.iterdir():
            if entry.name == "__pycache__":
                continue
            if entry.is_dir():
                self.assertEqual(entry.name, "tests", f"Shared runtime directory is not allowed: {entry.name}")
            else:
                self.assertTrue(entry.name in {"README.md", "AGENTS.md"} or entry.suffix == ".py", entry.name)
        for script in scripts:
            with self.subTest(tool=script.name):
                self.assertNotEqual(script.name, "__init__.py", "tools must not become a package")
                source = script.read_text(encoding="utf-8")
                self.assertTrue(ast.get_docstring(ast.parse(source)), "State this tool's dedicated use case")
                self.assertIn(f"| `{script.name}` |", readme, "Document the use case, invocation, and prerequisites")
                self.assertTrue((TOOLS / "tests" / f"test_{script.stem}.py").is_file(), "Add dedicated tests for this tool")
                siblings = {path.stem for path in scripts if path != script}
                self.assertEqual(dependencies(source, siblings, modules), [])

    def test_detects_imports_and_direct_calls_to_other_tools(self):
        cases = [
            "import sibling", "from tools import sibling", "from . import sibling",
            "from numerics import compare", "import importlib", "import runpy",
            "__import__('sibling')", "sys.path.insert(0, '..')", "from sys import path",
            "subprocess.run(['python', '../tools/sibling.py'])",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertTrue(dependencies(source, {"sibling"}, {"numerics"}))
        self.assertEqual(dependencies("import subprocess\nsubprocess.run(command_from_user)", {"sibling"}, {"numerics"}), [])


if __name__ == "__main__":
    unittest.main()
