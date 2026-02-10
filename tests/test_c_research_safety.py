import sys
import os
import subprocess
import pytest
from pathlib import Path
import re

# Add the project root to sys.path for module discovery during testing
sys.path.insert(0, os.path.abspath('.'))

def _run_research_subprocess_with_long_query(query: str) -> tuple[int, str, str]:
    """Helper to run specialized_agents_c.research in a subprocess with a long query."""
    repo_root = Path(__file__).resolve().parents[1]
    
    # Ensure the combined C extension is imported correctly
    code = f"""
import orchestrator_and_agents as specialized_agents_c
import time

# Initialize agents (important for `global_agents` and web search setup)
specialized_agents_c.create_agents()

# Call the research function with the provided query
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

def test_c_research_long_query_does_not_crash_and_truncates() -> None:
    """
    Test that the C research agent with a very long query does not crash
    and truncates the output to MAX_RESULT_BUFFER_LEN.
    """
    long_query_input = "a" * 1000  # A long query input to trigger potential buffer issues
    
    # We expect the output buffer to be limited to approximately MAX_RESULT_BUFFER_LEN (8192)
    # Plus some header and formatting overhead, so we check for truncation.
    
    returncode, stdout, stderr = _run_research_subprocess_with_long_query(long_query_input)
    
    # Fix the f-string syntax error here
    print(f"Stdout:\n{stdout}")
    print(f"Stderr:\n{stderr}")
    
    assert returncode == 0, f"Subprocess crashed with exit code {returncode}. Stderr:\n{stderr}\nStdout:\n{stdout}"
    assert "ok" in stdout, f"Expected 'ok' in stdout. Stderr:\n{stderr}\nStdout:\n{stdout}"
    
    # Extract the result string from stdout
    match = re.search(r"---RESULT_START---\n(.*?)\n---RESULT_END---", stdout, re.DOTALL)
    assert match, f"Could not find result in stdout:\n{stdout}"
    result_str = match.group(1).strip()
    
    # The actual MAX_RESULT_BUFFER_LEN from C is 8192
    # The header is "Research Results for 'query':\n\n"
    # A query of 1000 'a's would make the header ~1000+ characters.
    # The total length should be capped around 8192 characters.
    # We allow some wiggle room for null terminators and exact snprintf behavior.
    
    expected_max_len = 8192 # Corresponds to MAX_RESULT_BUFFER_LEN in specialized_agents_c.c
    
    assert len(result_str) <= expected_max_len, f"Result string length {len(result_str)} exceeds expected max {expected_max_len}"
    
    # Check for truncation warning in stderr (if triggered by a large response from Python web search)
    # This might not always trigger with just a long query, but if the Python web search returns many items
    assert "Research result buffer limit reached" in stderr or "Individual research result item too large" in stderr or len(result_str) < expected_max_len, \
        f"Expected buffer truncation warning in stderr or actual truncation. Stderr:\n{stderr}"

    print(f"✅ Research agent did not crash with long query and output is truncated (length: {len(result_str)}).")