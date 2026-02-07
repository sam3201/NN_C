#!/usr/bin/env python3
"""
SAM 2.0 Multi-Agent Debate System
Implements Fixer, Skeptic, and Judge agents for patch validation
"""

import logging
from typing import Dict, Any, Optional
from ollama_integration import query_ollama

logger = logging.getLogger(__name__)

class DebateAgent:
    """Multi-agent debate system for patch validation"""

    def __init__(self):
        self.agents = {
            'fixer': self._create_fixer(),
            'skeptic': self._create_skeptic(),
            'judge': self._create_judge()
        }

    def _create_fixer(self) -> Dict[str, str]:
        """Create the Fixer agent"""
        return {
            'name': 'Fixer',
            'role': 'Generate minimal unified diffs to fix errors',
            'prompt_template': """You are an expert Python software engineer. Generate a minimal unified diff to fix this error.

Error details:
{error_context}

Requirements:
- Output ONLY a unified diff
- NO explanations or comments
- Start directly with --- filename
- Include proper @@ hunks
- Fix the root cause
- Make minimal changes
- Use correct unified diff format

Generate the fix:""",
            'max_tokens': 600
        }

    def _create_skeptic(self) -> Dict[str, str]:
        """Create the Skeptic agent"""
        return {
            'name': 'Skeptic',
            'role': 'Critically analyze patches for risks and issues',
            'prompt_template': """You are a senior code reviewer. Critically analyze this proposed patch.

Error that was fixed:
{error_context}

Proposed patch:
{patch}

Identify:
1. Potential risks or side effects
2. Missing test cases or edge cases
3. Code quality issues
4. Security concerns
5. Performance implications

Be thorough but concise. List specific concerns:""",
            'max_tokens': 400
        }

    def _create_judge(self) -> Dict[str, str]:
        """Create the Judge agent"""
        return {
            'name': 'Judge',
            'role': 'Final decision maker on patch safety',
            'prompt_template': """You are the final authority on patch safety. Review the analysis and decide if this patch should be applied.

Error context:
{error_context}

Proposed patch:
{patch}

Skeptic's analysis:
{skeptic_analysis}

Safety requirements:
- Must not introduce new bugs
- Must handle edge cases properly
- Must follow security best practices
- Must maintain code quality
- Must be minimal and targeted

Answer ONLY: YES or NO
Provide a one-sentence reason.

Decision:""",
            'max_tokens': 50
        }

    def debate_patch(self, error_context: str, patch: str) -> Dict[str, Any]:
        """
        Run full debate process: Fixer → Skeptic → Judge

        Args:
            error_context: Description of the error to fix
            patch: The proposed patch to debate

        Returns:
            Dict with debate results and final decision
        """
        logger.info("Starting multi-agent debate...")

        # Step 1: Skeptic analyzes the patch
        skeptic_result = self._query_agent('skeptic', {
            'error_context': error_context,
            'patch': patch
        })

        # Step 2: Judge makes final decision
        judge_result = self._query_agent('judge', {
            'error_context': error_context,
            'patch': patch,
            'skeptic_analysis': skeptic_result.get('response', '')
        })

        # Parse judge's decision
        judge_response = judge_result.get('response', '').strip().upper()
        approved = judge_response.startswith('YES')

        # Extract reason
        reason = ""
        if 'reason' in judge_result.get('metadata', {}):
            reason = judge_result['metadata']['reason']
        elif len(judge_response.split('\n')) > 1:
            reason = judge_response.split('\n')[1].strip()

        result = {
            'approved': approved,
            'reason': reason,
            'skeptic_analysis': skeptic_result.get('response', ''),
            'judge_response': judge_response,
            'debate_complete': True,
            'confidence': self._calculate_debate_confidence(skeptic_result, judge_result)
        }

        logger.info(f"Debate complete: {'APPROVED' if approved else 'REJECTED'} ({reason})")
        return result

    def _query_agent(self, agent_name: str, context: Dict[str, str]) -> Dict[str, Any]:
        """Query a specific agent"""
        if agent_name not in self.agents:
            return {'error': f'Unknown agent: {agent_name}'}

        agent = self.agents[agent_name]
        prompt = agent['prompt_template'].format(**context)

        try:
            response = query_ollama(prompt, max_tokens=agent['max_tokens'])

            return {
                'agent': agent_name,
                'response': response,
                'success': True,
                'metadata': {
                    'prompt_length': len(prompt),
                    'response_length': len(response)
                }
            }

        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return {
                'agent': agent_name,
                'error': str(e),
                'success': False
            }

    def _calculate_debate_confidence(self, skeptic_result: Dict, judge_result: Dict) -> float:
        """Calculate confidence in the debate outcome"""
        confidence = 0.5  # Base confidence

        # Judge's decision strength
        judge_response = judge_result.get('response', '').strip().upper()
        if 'YES' in judge_response:
            confidence += 0.2
        elif 'NO' in judge_response:
            confidence += 0.1  # NO decisions are also confident

        # Skeptic analysis quality (longer, more detailed analysis = higher confidence)
        skeptic_response = skeptic_result.get('response', '')
        if len(skeptic_response) > 100:  # Substantial analysis
            confidence += 0.2
        elif len(skeptic_response) > 50:
            confidence += 0.1

        # Response consistency
        if skeptic_result.get('success') and judge_result.get('success'):
            confidence += 0.1

        return min(1.0, max(0.0, confidence))

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about all agents"""
        return {
            agent_name: {
                'name': info['name'],
                'role': info['role'],
                'max_tokens': info['max_tokens']
            }
            for agent_name, info in self.agents.items()
        }

# Global debate system instance
debate_system = DebateAgent()

# Convenience functions
def debate_patch(error_context: str, patch: str) -> Dict[str, Any]:
    """Run debate on a patch"""
    return debate_system.debate_patch(error_context, patch)

def get_agent_info() -> Dict[str, Any]:
    """Get agent information"""
    return debate_system.get_agent_info()

if __name__ == "__main__":
    # Test debate system
    print("🧪 Testing Multi-Agent Debate System")

    # Test debate
    error_context = "NameError: name 'undefined_var' is not defined in test.py line 5"
    test_patch = """--- a/test.py
+++ b/test.py
@@ -4,1 +4,1 @@
-undefined_var = 42
+defined_var = 42"""

    try:
        result = debate_patch(error_context, test_patch)

        print(f"Debate result: {'APPROVED' if result['approved'] else 'REJECTED'}")
        print(f"Reason: {result['reason']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Skeptic analysis length: {len(result['skeptic_analysis'])}")

        # Get agent info
        agents = get_agent_info()
        print(f"Available agents: {list(agents.keys())}")

    except Exception as e:
        print(f"❌ Debate test failed: {e}")

    print("✅ Debate system test complete")
