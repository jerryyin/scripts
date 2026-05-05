"""Pytest plugin: call hipDeviceReset + os._exit after the session to avoid
the FFM simulator teardown hang (192 threads stuck in futex_wait_queue).

Load with:  pytest -p ffm_teardown ...
Requires:   ~/scripts/tools on PYTHONPATH (run_on_model.sh does this).
"""

import atexit
import ctypes
import os

_pytest_exit_status = None


def _force_clean_exit():
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipDeviceReset()
    except Exception:
        pass
    os._exit(_pytest_exit_status or 0)


def pytest_sessionfinish(session, exitstatus):
    global _pytest_exit_status
    _pytest_exit_status = exitstatus
    atexit.register(_force_clean_exit)
