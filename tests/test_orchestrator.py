import orchestrator_and_agents as multi_agent_orchestrator_c
import time

print("Testing multi_agent_orchestrator_c module...")

try:
    print("Calling multi_agent_orchestrator_c.create_agents()...")
    multi_agent_orchestrator_c.create_agents()
    print("Specialized agents created.")

    print("Calling multi_agent_orchestrator_c.create_multi_agent_system()...")
    multi_agent_orchestrator_c.create_multi_agent_system()
    print("System created.")

    print("Calling multi_agent_orchestrator_c.start_orchestrator()...")
    multi_agent_orchestrator_c.start_orchestrator()
    print("Orchestrator started (background thread).")

    # Give the orchestrator a moment to process the message in its thread
    time.sleep(1) 

    print("Calling multi_agent_orchestrator_c.get_orchestrator_status()...")
    status = multi_agent_orchestrator_c.get_orchestrator_status()
    print(f"Orchestrator Status: {status}")

    # Verify key status elements
    assert status['agent_count'] > 0
    assert status['name'] == 'SAM_MultiAgent'

    print("✅ Multi-agent orchestrator C module test successful!")

except Exception as e:
    print(f"❌ Multi-agent orchestrator C module test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)