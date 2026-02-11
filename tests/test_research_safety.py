import pytest
from unittest.mock import MagicMock, patch
import os
import sys # Added import sys

# Assuming specialized_agents_c and multi_agent_orchestrator_c are combined
# into a single extension module named 'orchestrator_and_agents'.

# Mock the Python web search function directly
@patch('sam_web_search.search_web_with_sam')
def test_research_agent_long_query_and_no_results_safety(mock_search_web_with_sam):
    # Dynamically import the C extension module
    try:
        import orchestrator_and_agents as c_agents
    except ImportError:
        pytest.skip("orchestrator_and_agents module not found, skipping C extension tests.")
    
    # Configure the mock to return no results for our test case
    mock_search_web_with_sam.return_value = {
        "query": "very obscure non-existent topic that will never return any results on duckduckgo",
        "results": [],
        "source": "mock_duckduckgo",
        "timestamp": 123456789.0,
    }

    # Define a query that is very long, exceeding MAX_QUERY_LEN in C if not handled
    long_query_part = "a" * 600 # Exceeds MAX_QUERY_LEN (512)
    obscure_query = f"this is a {long_query_part} very obscure non-existent topic that will never return any results on duckduckgo and should trigger the no results path"

    # Initialize the C agents system
    c_agents.create_agents() # This will internally call init_python_web_search() and create the researcher agent

    # Call the exposed Python binding for research
    result_str = c_agents.research(obscure_query)
    
    assert result_str is not None, "research_agent_perform_search returned None"

    # Verify that the mock was called correctly
    mock_search_web_with_sam.assert_called_once_with(
        obscure_query, save_to_drive=False, max_results=5
    )

    # Verify that the "no results found" message is present
    assert "No relevant results found." in result_str

    # Verify that the overall result string is within MAX_RESULT_BUFFER_LEN (8192)
    # The C side is responsible for truncation. We check if Python receives a reasonable size.
    # We can't directly inspect MAX_RESULT_BUFFER_LEN from Python, but we can check if it's
    # a reasonable length and doesn't seem to have overflowed.
    # The actual max length is controlled by the C code, so we assert it's not excessively large,
    # and certainly less than an arbitrary large number.
    assert len(result_str) < 9000 # Just a sanity check for truncation

    # The query in the result should be truncated due to MAX_QUERY_LEN and safe_copy
    # It should reflect the truncation logic used in C for display_query
    # It's hard to assert exact truncation without knowing internal C defines,
    # but the log_query also uses truncation. The result header uses display_query (MAX_QUERY_LEN).
    # The full query should *not* be in the header if it was longer than MAX_QUERY_LEN.
    # The header format is 'Research Results for \'%s\':\n\n'
    # MAX_QUERY_LEN is 512, so the query itself should be truncated at ~512 chars + other header text.
    expected_truncated_query_in_header = obscure_query[:512] # MAX_QUERY_LEN
    assert f"Research Results for '{expected_truncated_query_in_header}'" in result_str[:len(f"Research Results for '{expected_truncated_query_in_header}'")]

    # Free the agent and the result string allocated in C


# This test case verifies proper handling when the Python web search initialization fails
# This test case verifies proper handling when the Python web search initialization fails
# To run these tests, you would typically need to build the C extension first.
# Example build command (adjust as needed for your setup.py or Makefile):
# python setup.py build_ext --inplace

# Make sure sam_web_search.py is discoverable in Python path.
