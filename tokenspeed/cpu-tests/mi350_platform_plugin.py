"""pytest plugin (tokenspeed/cpu-tests): override tokenspeed_kernel's platform with the owner fixture `mi350_platform`
(tokenspeed-kernel/test/conftest.py), evaluated from that file at plugin import (before any
conftest imports tokenspeed_kernel), so CPU tests import without a device."""
import ast
import os
import pathlib

_src = (pathlib.Path(os.environ["TS_KERNEL_TEST"]) / "conftest.py").read_text()
_fn = next(n for n in ast.parse(_src).body
           if isinstance(n, ast.FunctionDef) and n.name == "mi350_platform")
_ret = next(n for n in _fn.body if isinstance(n, ast.Return))
from tokenspeed_kernel import platform as _P  # noqa: E402

_P.Platform.override(eval(compile(ast.Expression(_ret.value), "conftest", "eval"),
                          {k: getattr(_P, k) for k in dir(_P)}))
