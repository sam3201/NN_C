import orchestrator_and_agents
import pytest

# Define a very long query string
VERY_LONG_QUERY = "A" * 4096  # Assuming a typical buffer size might be 4KB

def test_research_agent_long_query_no_overflow():
    """
    Test orchestrator_and_agents.research with a very long query
    to ensure it does not cause a buffer overflow or crash.
    """
    try:
        # Step 1: Initialize individual agents
        orchestrator_and_agents.create_agents()
        # Step 2: Initialize the multi-agent system, which should now find agents initialized
        orchestrator_and_agents.create_multi_agent_system()
        
        # Directly call the 'research' function, assuming it handles agent creation internally
        result = orchestrator_and_agents.research(VERY_LONG_QUERY)
        
        # Assert that the function returned without crashing.
        # Depending on the expected behavior, further assertions can be added
        # e.g., that the result indicates an error, or is truncated.
        assert isinstance(result, str) or result is None
        print(f"Test passed: research handled long query gracefully. Result: {result[:100]}...")

    except Exception as e:
        pytest.fail(f"orchestrator_and_agents.research crashed or raised an unexpected exception with long query: {e}")
