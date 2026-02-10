from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import re

def _run_research_subprocess(query: str) -> tuple[int, str, str]:
    """Helper to run specialized_agents_c.research in a subprocess."""
    repo_root = Path(__file__).resolve().parents[1]
    code = f"""
import specialized_agents_c
import time

# Initialize agents, which also loads the Python web search module
specialized_agents_c.create_agents()

r = specialized_agents_c.research({repr(query)})
assert isinstance(r, str)
print("ok", len(r))
print("---RESULT_START---")
print(r)
print("---RESULT_END---")
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
    return proc.returncode, proc.stdout, proc.stderr


def test_c_research_long_query_does_not_crash() -> None:
    """Regression test for the C research agent buffer overflow with a long query."""
    query = "x" * 10000
    returncode, stdout, stderr = _run_research_subprocess(query)
    assert returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "ok" in stdout
    # The actual content for a garbage query doesn't matter, just that it doesn't crash
    # and returns something.

def test_c_research_returns_valid_results() -> None:
    """Test that the C research agent returns actual web search results."""
    query = "Python C-API PyObject reference counting"
    returncode, stdout, stderr = _run_research_subprocess(query)
    assert returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "ok" in stdout

    # Extract the result string from stdout
    match = re.search(r"---RESULT_START---\n(.*?)\n---RESULT_END---", stdout, re.DOTALL)
    assert match, f"Could not find result in stdout:\n{stdout}"
    result_str = match.group(1)

    print(f"--- Captured Result for '{query}' ---")
    print(result_str)
    print("------------------------------------")

    # Assert that the result string contains typical web search result elements
    assert "Research Results for 'Python C-API PyObject reference counting':" in result_str
    assert "Title:" in result_str
    assert "URL:" in result_str
    assert "Snippet:" in result_str
    # Check for at least a few results (DuckDuckGo usually returns something for this query)
    assert result_str.count("Title:") >= 1, "Expected at least one search result."
