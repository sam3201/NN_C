
import sys
import os
from pathlib import Path

# Add src/python to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src" / "python"))

from complete_sam_unified import UnifiedSAMSystem

def test_governance_veto():
    print("🛡️ Testing SAM-D Governance Quorum Gate...")
    
    # Initialize system in a minimal mode for testing
    os.environ["SAM_STRICT_LOCAL_ONLY"] = "1"
    os.environ["SAM_AUTONOMOUS_ENABLED"] = "0"
    system = UnifiedSAMSystem()
    
    # 1. Test safe code change (should pass)
    safe_proposal = {
        "patch": "def hello(): print('hello')",
        "risk": 0.1
    }
    approved, log = system._governance_quorum_vote("code_change", safe_proposal)
    print(f"Safe proposal approved: {approved} (Quorum: {log['quorum']})")
    assert approved == True
    
    # 2. Test dangerous code change (should be vetoed by SAV)
    dangerous_proposal = {
        "patch": "import os; os.system('rm -rf /')",
        "risk": 0.9
    }
    approved, log = system._governance_quorum_vote("code_change", dangerous_proposal)
    print(f"Dangerous proposal approved: {approved} (Quorum: {log['quorum']})")
    print(f"SAV Vote: {log['votes']['SAV']}")
    assert approved == False
    assert log['votes']['SAV']['vote'] == 0
    
    # 3. Test high drift change (should be vetoed by LOVE)
    # We simulate this by mocking identity_drift
    system.love_agent.compute_identity_drift = lambda: 0.2 # Above 0.1 threshold
    drift_proposal = {
        "patch": "def mutate(): pass",
        "risk": 0.2
    }
    approved, log = system._governance_quorum_vote("code_change", drift_proposal)
    print(f"High-drift proposal approved: {approved} (Quorum: {log['quorum']})")
    print(f"LOVE Vote: {log['votes']['LOVE']}")
    assert log['votes']['LOVE']['vote'] == 0
    
    print("✅ Governance verification successful!")

if __name__ == "__main__":
    try:
        test_governance_veto()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
