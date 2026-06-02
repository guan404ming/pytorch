"""
Custom pytest plugin that records, per test, the set of torch Python files whose
code was *executed at runtime* during that test.

Enable with ``--pytest-trace=<path>``; the plugin writes a JSON file:

    {
      "test/test_foo.py::TestBar::test_baz": {"deps": ["torch/...", ...]},
      ...
    }

Mechanism: ``sys.monitoring`` (PEP 669, Python 3.12+) PY_START events. The callback
records the file of every Python code object that starts executing during the test,
filtered to the torch package.

Import-time code is deliberately excluded so deps reflect runtime execution only and
are independent of test order / what ran before. A module body executes once per
process (at first import), so without this exclusion the first test to trigger an
import would absorb hundreds of files that every other test "uses" too. We bracket
imports by wrapping ``importlib._bootstrap._find_and_load`` (the single funnel for
both ``import`` statements and ``importlib.import_module``): while an import is in
progress, ``_import_depth > 0`` and PY_START events are attributed to the import, not
the test. Crucially we do NOT DISABLE a code location seen during an import, so a
function first entered while importing is still captured if the test later calls it
at runtime.

``DISABLE`` is returned for runtime events to suppress repeat callbacks for an
already-seen code location (bounding overhead to roughly one callback per distinct
function per test); ``restart_events`` re-enables them at the start of each test so
every test records its own full set.

Limitations:
* Only Python-level execution is captured. C++/CUDA kernels invoked by torch do
  not map to Python source and will not appear.
* Single process only. pytest-xdist workers would each need to write/merge a
  partial file; not supported here.
* Subprocesses spawned by a test are not traced.
"""

import importlib._bootstrap as _bootstrap
import json
import os
import sys

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser


def pytest_addoptions(parser: Parser) -> None:
    group = parser.getgroup("trace")
    group.addoption(
        "--pytest-trace",
        action="store",
        dest="pytest_trace",
        default=None,
        metavar="path",
        help="Record per-test executed torch files to the given JSON file.",
    )


class PytestTracePlugin:
    def __init__(self, config: Config) -> None:
        import torch

        self._torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
        # Base for repo-relative paths so deps look like "torch/...".
        self._repo_root = os.path.dirname(self._torch_dir)
        self._out_path = config.getoption("pytest_trace")
        self._results: dict[str, dict[str, list[str]]] = {}
        self._current: set[str] | None = None
        # > 0 while a module import is executing; used to exclude import-time code.
        self._import_depth = 0
        self._orig_find_and_load = self._install_import_bracket()

        self._tid = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
        sys.monitoring.use_tool_id(self._tid, "pytorch_pytest_trace")
        sys.monitoring.register_callback(
            self._tid, sys.monitoring.events.PY_START, self._on_py_start
        )

    def _install_import_bracket(self):
        orig = _bootstrap._find_and_load

        def wrapped(name, import_):
            self._import_depth += 1
            try:
                return orig(name, import_)
            finally:
                self._import_depth -= 1

        _bootstrap._find_and_load = wrapped
        return orig

    def _on_py_start(self, code, instruction_offset):
        # Keep the location armed during imports so a later runtime call is caught.
        if self._import_depth > 0:
            return None
        if self._current is not None and code.co_filename.startswith(self._torch_dir):
            self._current.add(code.co_filename)
        return sys.monitoring.DISABLE

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self._current = set()
        # Re-enable locations that previous tests DISABLEd so this test records
        # its own complete set.
        sys.monitoring.restart_events()
        sys.monitoring.set_events(self._tid, sys.monitoring.events.PY_START)
        try:
            yield
        finally:
            sys.monitoring.set_events(self._tid, 0)
            deps = sorted(os.path.relpath(f, self._repo_root) for f in self._current)
            self._results[item.nodeid] = {"deps": deps}
            self._current = None

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        with open(self._out_path, "w") as f:
            json.dump(self._results, f, indent=2, sort_keys=True)
        sys.monitoring.set_events(self._tid, 0)
        sys.monitoring.free_tool_id(self._tid)
        _bootstrap._find_and_load = self._orig_find_and_load
