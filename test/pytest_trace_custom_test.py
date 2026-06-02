"""
Tests for the pytest_trace_custom plugin (``--pytest-trace``).

Verifies per-test dependency tracing is consistent between a whole-file run and
per-test runs. Each individual test is traced in its own subprocess, which is the
complete "cold" footprint and the source of truth. The whole-file run shares one
process, so process-global one-time init (e.g. JitTestCase._restored_warnings in
torch/testing/_internal/jit_utils.py, which makes only the first test in a process
touch torch/jit/_trace.py) is paid once rather than per test. Hence the whole-file
deps for a test are always a SUBSET of that test's isolated deps: the fast
whole-file run never invents a dependency, it only ever misses process-global init.
"""

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from torch.testing._internal.common_utils import run_tests, slowTest, TestCase


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "test/test_tensorexpr.py"


def _run_trace(targets: list[str]) -> dict[str, set[str]]:
    """Run ``pytest --pytest-trace=<tmp> <targets>`` and return {nodeid: {deps}}."""
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"--pytest-trace={out_path}",
                *targets,
                "-q",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if os.path.getsize(out_path) == 0:
            raise RuntimeError(
                f"pytest produced no trace for {targets}; "
                f"stderr tail:\n{proc.stderr.decode()[-2000:]}"
            )
        with open(out_path) as f:
            data = json.load(f)
    finally:
        os.unlink(out_path)
    return {nid: set(v["deps"]) for nid, v in data.items()}


class TestPytestTrace(TestCase):
    @slowTest
    def test_whole_file_deps_subset_of_isolated(self):
        # 1) One whole-file (single-process) run -> warm per-test deps.
        whole = _run_trace([TARGET])
        self.assertTrue(whole, "whole-file trace produced no results")
        nodeids = sorted(whole)

        # 2) Each test in its own subprocess -> cold/complete per-test deps.
        def trace_one(nid: str) -> tuple[str, set[str] | None]:
            return nid, _run_trace([nid]).get(nid)

        max_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            isolated = dict(ex.map(trace_one, nodeids))

        # 3) Whole-file deps must be a subset of the isolated deps for every test.
        violations = {}
        for nid in nodeids:
            iso = isolated[nid]
            self.assertIsNotNone(iso, f"isolated run for {nid} produced no deps")
            extra = whole[nid] - iso
            if extra:
                violations[nid] = sorted(extra)
        self.assertEqual(
            violations,
            {},
            f"whole-file deps not a subset of isolated deps for: {violations}",
        )


if __name__ == "__main__":
    run_tests()
