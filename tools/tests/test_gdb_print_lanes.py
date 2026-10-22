import contextlib
import io
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "gdb_print_lanes.py"


class GdbTests(unittest.TestCase):
    def test_standalone_tool_uses_injected_gdb_and_prints_requested_lanes(self):
        commands = []

        class Gdb:
            error = RuntimeError

            @staticmethod
            def execute(command, to_string=False):
                commands.append(command)
                return "$1 = 42"

        with tempfile.TemporaryDirectory() as directory:
            copied = shutil.copy2(TOOL, directory)
            namespace = runpy.run_path(copied, init_globals={"gdb": Gdb})
            self.assertIs(namespace["print_variable_for_lanes"].__globals__["gdb"], Gdb)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                namespace["print_variable_for_lanes"]("value", 0, 1)
        self.assertEqual(commands, ["lane 0", "frame 0", "p value", "lane 1", "frame 0", "p value"])
        self.assertEqual(output.getvalue(), "value values: [42, 42]\n")


if __name__ == "__main__":
    unittest.main()
