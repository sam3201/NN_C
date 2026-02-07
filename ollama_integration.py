#!/usr/bin/env python3
"""
SAM 2.0 Ollama Integration
Provides local LLM access via Ollama for SWE-agent functionality
"""

import requests
import json
import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:7b"):
        self.base_url = base_url
        self.model = model
        self.timeout = 120  # 2 minutes timeout

    def query(self, prompt: str, max_tokens: int = 100, temperature: float = 0.1) -> str:
        """
        Query Ollama model with a prompt

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            str: Generated response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            logger.debug(f"Querying Ollama: {prompt[:100]}...")
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

            result = response.json()
            response_text = result.get("response", "")

            elapsed = time.time() - start_time
            logger.info(f"Ollama query complete: {len(response_text)} chars in {elapsed:.1f}s")

            return response_text

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return f"Ollama unavailable: {e}"
        except Exception as e:
            logger.error(f"Ollama query error: {e}")
            return f"Ollama error: {e}"

    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return {"status": "error", "message": f"HTTP {response.status_code}"}

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            if self.model in model_names:
                return {"status": "ready", "model": self.model}
            else:
                return {
                    "status": "model_missing",
                    "message": f"Model '{self.model}' not found. Available: {model_names}"
                }

        except requests.exceptions.RequestException as e:
            return {"status": "unavailable", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except:
            return []

# Global Ollama client instance
ollama_client = OllamaClient()

# Legacy compatibility functions (for existing code)
def generate_llm_response(prompt: str, max_tokens: int = 100) -> str:
    """Generate response using Ollama (legacy compatibility)"""
    return ollama_client.query(prompt, max_tokens)

def analyze_message_coherence(message: str, context: list = None) -> dict:
    """Analyze message coherence using Ollama"""
    context_str = "\n".join(context) if context else "No context"

    prompt = f"""Analyze the coherence of this message in context:

Context:
{context_str}

Message to analyze:
{message}

Provide a coherence score (0.0-1.0) and identify any issues:
"""

    response = ollama_client.query(prompt, max_tokens=150)

    # Parse response for score and issues
    coherence_score = 0.5  # default
    issues = []

    # Simple parsing of response
    if "score" in response.lower():
        import re
        score_match = re.search(r'score[:\s]*([0-9.]+)', response.lower())
        if score_match:
            try:
                coherence_score = min(1.0, max(0.0, float(score_match.group(1))))
            except:
                pass

    # Extract issues
    response_lower = response.lower()
    if "incoherent" in response_lower or "inconsistency" in response_lower:
        issues.append("Potential incoherence detected")
    if "repetition" in response_lower:
        issues.append("Repetitive content")
    if "unclear" in response_lower or "confusing" in response_lower:
        issues.append("Unclear or confusing")

    return {
        "coherence_score": coherence_score,
        "issues": issues,
        "analysis": response[:200]  # Truncate for brevity
    }

def get_improvement_suggestions(message: str, issues: list) -> str:
    """Get improvement suggestions"""
    issues_str = "\n".join(f"- {issue}" for issue in issues)

    prompt = f"""Given this message and its issues, suggest specific improvements:

Message: {message}

Issues identified:
{issues_str}

Provide 2-3 specific, actionable suggestions for improvement:
"""

    return ollama_client.query(prompt, max_tokens=200)

def check_llm_status() -> dict:
    """Check LLM status"""
    health = ollama_client.check_health()
    return {
        "status": health["status"],
        "model": ollama_client.model,
        "base_url": ollama_client.base_url
    }

# SWE-agent specific functions
def query_ollama(prompt: str, max_tokens: int = 100, temperature: float = 0.1) -> str:
    """Direct Ollama query for SWE-agent components"""
    return ollama_client.query(prompt, max_tokens, temperature)

def is_ollama_available() -> bool:
    """Check if Ollama is available"""
    health = ollama_client.check_health()
    return health["status"] == "ready"

if __name__ == "__main__":
    # Test Ollama integration
    print("🧪 Testing Ollama Integration")

    # Check health
    health = ollama_client.check_health()
    print(f"Ollama status: {health}")

    if health["status"] == "ready":
        # Test basic query
        response = generate_llm_response("Hello, how are you?")
        print(f"Test response: {response[:100]}...")

        # Test coherence analysis
        analysis = analyze_message_coherence("This is a test message")
        print(f"Coherence analysis: {analysis}")

        print("✅ Ollama integration working!")
    else:
        print("❌ Ollama not ready - make sure it's running and model is pulled")
        print("Run: ollama serve")
        print("Then: ollama pull qwen2.5-coder:7b")
