🔧 INTEGRATION STEPS FOR RAM-AWARE MODEL SWITCHING:

1. Add this import at the top of complete_sam_unified.py:
from ram_model_switcher import (
    RAMAwareModelSwitcher, 
    ConversationDiversityManager,
    initialize_ram_aware_switching,
    get_optimal_model_for_task,
    check_model_switch_needed,
    should_allow_meta_agent_message,
    record_conversation_message
)

2. Add this to UnifiedSAMSystem.__init__ after MetaAgent initialization:
        # Initialize RAM-aware model switching and conversation diversity
        try:
            initialize_ram_aware_switching(self)
            optimal = get_optimal_model_for_task(self, "general")
            print(f"🎯 Optimal model: {optimal['provider']}/{optimal['model']} (RAM: {optimal.get('ram_percent', 'unknown')}%)")
        except Exception as e:
            print(f"⚠️ RAM-aware features failed: {e}")

3. Add this method to MetaAgent class:
    def request_help(self, issue_description: str, urgency: str = "medium") -> bool:
        """Request help from groupchat participants for complex issues"""
        if not hasattr(self.system, 'diversity_manager'):
            self._send_groupchat_message(f"🤖 META-AGENT HELP NEEDED: {issue_description}")
            return True
        
        if self.system.diversity_manager.should_allow_meta_agent_message("help_request"):
            message = f"🤖 META-AGENT HELP REQUEST ({urgency}): {issue_description}"
            self._send_groupchat_message(message)
            self.system.diversity_manager.record_message("meta_agent", "help_request", message)
            return True
        else:
            print(f"🎭 Diversity control: Postponing help request to maintain conversation balance")
            return False
    
    def _send_groupchat_message(self, message: str):
        """Send a message to the groupchat"""
        print(f"📢 GROUPCHAT: {message}")
        # Integrate with your actual chat system here

4. Test with: python3 -c "from ram_model_switcher import *; print('✅ RAM features available')"

Then restart the system with: ./run_sam.sh
