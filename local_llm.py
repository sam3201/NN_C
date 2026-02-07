#!/usr/bin/env python3
"""
SAM 2.0 Local LLM Interface - Ollama Only
Completely removed CodeLlama due to memory crashes. Uses Ollama exclusively.
"""

import logging

logger = logging.getLogger(__name__)

# Legacy compatibility - redirect everything to Ollama
def generate_llm_response(prompt: str, max_tokens: int = 100) -> str:
    """Legacy function - redirects to Ollama"""
    try:
        from ollama_integration import query_ollama
        return query_ollama(prompt, max_tokens)
    except Exception as e:
        logger.error(f"Ollama unavailable: {e}")
        return "Local LLM not available - using fallback mode"

def analyze_message_coherence(message: str, context: list = None) -> dict:
    """Legacy function - redirects to Ollama"""
    try:
        from ollama_integration import analyze_message_coherence as ollama_analyze
        return ollama_analyze(message, context)
    except Exception as e:
        logger.error(f"Ollama coherence analysis failed: {e}")
        return {"coherence_score": 0.5, "issues": ["LLM not available"]}

def get_improvement_suggestions(message: str, issues: list) -> str:
    """Legacy function - redirects to Ollama"""
    try:
        from ollama_integration import get_improvement_suggestions as ollama_suggest
        return ollama_suggest(message, issues)
    except Exception as e:
        logger.error(f"Ollama suggestions failed: {e}")
        return "Could not generate suggestions due to LLM error"

def check_llm_status() -> dict:
    """Legacy function - redirects to Ollama"""
    try:
        from ollama_integration import check_llm_status as ollama_status
        return ollama_status()
    except Exception as e:
        logger.error(f"Ollama status check failed: {e}")
        return {"status": "error", "message": str(e)}

# Stub class for legacy compatibility
class LocalLLM:
    """Stub class - all functionality moved to Ollama"""

    def __init__(self, *args, **kwargs):
        logger.warning("LocalLLM is deprecated. Using Ollama integration instead.")
        self.model = None
        self.tokenizer = None

    def generate_response(self, *args, **kwargs):
        return generate_llm_response(*args, **kwargs)

    def analyze_coherence(self, *args, **kwargs):
        return analyze_message_coherence(*args, **kwargs)

    def is_available(self):
        try:
            from ollama_integration import is_ollama_available
            return is_ollama_available()
        except:
            return False

    def get_model_info(self):
        return check_llm_status()

# Global instance for legacy compatibility
local_llm = LocalLLM()

if __name__ == "__main__":
    print("🧪 Testing Local LLM Interface (Ollama Only)")

    status = check_llm_status()
    print(f"LLM Status: {status}")

    if local_llm.is_available():
        print("✅ Ollama integration working")
    else:
        print("⚠️ Ollama not available - run 'ollama serve' and pull a model")
