from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_c_research_long_query_does_not_crash() -> None:
    """Regression test for the C research agent buffer overflow.

    Runs in a subprocess so a native crash does not take down the pytest worker.
    """
    repo_root = Path(__file__).resolve().parents[1]
    code = r"""
import specialized_agents_c

q = "x" * 10000
r = specialized_agents_c.research(q)
assert isinstance(r, str)
print("ok", len(r))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

