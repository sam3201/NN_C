#!/usr/bin/env python3
"""
SAM 2.0 Meta-Agent LLM Interface
Uses Ollama for intelligent patch generation with memory and learning
"""

import re
import logging
from typing import Optional
from ollama_integration import query_ollama, is_ollama_available, check_llm_status
from config import LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS
from memory_system import query_memory
from teacher_loop import get_few_shot_examples

logger = logging.getLogger(__name__)

def generate_patch(prompt: str) -> str:
    """
    Generate a unified diff patch using Ollama with memory and learning

    Args:
        prompt: Description of the issue and context

    Returns:
        str: Unified diff patch ONLY, or empty string if generation fails
    """
    if not is_ollama_available():
        logger.error("Ollama not available for patch generation")
        return ""

    try:
        # Get few-shot examples for better generation
        few_shot_examples = get_few_shot_examples(prompt, max_examples=2)

        # Query memory for similar past fixes
        memory_results = query_memory(prompt, k=3)

        # Build enhanced prompt with memory and examples
        enhanced_prompt = _build_patch_prompt(prompt, few_shot_examples, memory_results)

        logger.info("Generating patch with Ollama + memory integration...")
        response = query_ollama(enhanced_prompt, max_tokens=LLM_MAX_TOKENS)

        # Clean and validate the response
        if not response or not response.strip():
            logger.error("Empty response from Ollama")
            return ""

        # Extract unified diff from response
        diff = _extract_unified_diff(response.strip())

        if not diff:
            logger.error("No valid unified diff found in response")
            logger.debug(f"Raw response: {response[:500]}...")
            return ""

        logger.info(f"Generated code fix patch with {len(diff.splitlines())} lines")
        return diff

    except Exception as e:
        logger.error(f"Patch generation failed: {e}")
        return ""

def _build_patch_prompt(prompt: str, few_shot_examples: list, memory_results: list) -> str:
    """
    Build enhanced prompt with memory and few-shot examples

    Args:
        prompt: Original error prompt
        few_shot_examples: Few-shot learning examples
        memory_results: Memory retrieval results

    Returns:
        str: Enhanced prompt for Ollama
    """
    prompt_parts = []

    # System instruction
    prompt_parts.append("""You are an expert Python software engineer. Analyze the error and generate a minimal unified diff fix.

CRITICAL REQUIREMENTS:
- Output ONLY the unified diff (--- and +++ lines, @@ hunks, +/- changes)
- NO explanations, comments, or markdown formatting
- Start directly with --- a/filename
- Use proper unified diff format with correct paths
- Fix the root cause, not just symptoms
- Make minimal, targeted changes
- Ensure syntax is correct
- Follow Python best practices""")

    # Add few-shot examples if available
    if few_shot_examples:
        prompt_parts.append("\n\nEXAMPLES OF SIMILAR FIXES:")
        for i, example in enumerate(few_shot_examples, 1):
            status = "SUCCESSFUL" if example.get("success") else "FAILED"
            prompt_parts.append(f"\nExample {i} ({status}):")
            prompt_parts.append(f"Error: {example['input']}")
            prompt_parts.append(f"Fix: {example['output']}")

    # Add memory context if available
    if memory_results:
        prompt_parts.append("\n\nRELEVANT PAST FIXES:")
        for memory in memory_results[:2]:  # Limit to 2 most relevant
            metadata = memory.get("metadata", {})
            success = "SUCCESSFUL" if metadata.get("success") else "FAILED"
            prompt_parts.append(f"- {success}: {memory['text'][:100]}...")

    # Add current error
    prompt_parts.append(f"""

CURRENT ERROR TO FIX:
{prompt}

Generate the unified diff patch now:""")

    return "\n".join(prompt_parts)

def _extract_unified_diff(response: str) -> str:
    """
    Extract unified diff from LLM response.

    Args:
        response: Raw LLM response

    Returns:
        str: Clean unified diff or empty string
    """
    lines = response.splitlines()

    # Find the start of unified diff (--- or +++ line)
    diff_start = -1
    for i, line in enumerate(lines):
        if line.startswith('--- ') or line.startswith('+++ '):
            diff_start = i
            break

    if diff_start == -1:
        return ""

    # Extract diff lines
    diff_lines = lines[diff_start:]

    # Validate it's a proper unified diff
    if not _is_valid_unified_diff(diff_lines):
        return ""

    return '\n'.join(diff_lines)

def _is_valid_unified_diff(lines: list) -> bool:
    """
    Basic validation that this looks like a unified diff.

    Args:
        lines: Diff lines to validate

    Returns:
        bool: True if it looks like a valid unified diff
    """
    if len(lines) < 3:
        return False

    # Should start with --- or +++
    if not (lines[0].startswith('--- ') or lines[0].startswith('+++ ')):
        return False

    # Should have at least one @@ line
    has_hunk = any(line.startswith('@@ ') for line in lines)
    if not has_hunk:
        return False

    # Should have some actual changes (+ or - lines)
    has_changes = any(line.startswith('+') or line.startswith('-') for line in lines)
    if not has_changes:
        return False

    return True

def explain_patch(diff: str) -> str:
    """
    Generate a human-readable explanation of what the patch does.

    Args:
        diff: Unified diff to explain

    Returns:
        str: Human explanation
    """
    if not is_ollama_available():
        return "Ollama not available for explanation"

    try:
        prompt = f"""Explain what this unified diff does in simple terms:

{diff}

Provide a brief, clear explanation of the changes:"""

        return query_ollama(prompt, max_tokens=150)

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return f"Could not generate explanation: {e}"

def validate_patch_reasoning(diff: str, error_context: str) -> dict:
    """
    Use Ollama to validate if a patch makes sense for the given error.

    Args:
        diff: The proposed patch
        error_context: Original error description

    Returns:
        dict: Validation result with score and reasoning
    """
    if not is_ollama_available():
        return {"valid": False, "score": 0, "reasoning": "Ollama not available"}

    try:
        prompt = f"""Evaluate if this patch correctly fixes the given error.

Error:
{error_context}

Patch:
{diff}

Rate the patch on a scale of 0-10 (10 being perfect fix).
Provide your rating and brief reasoning.

Format: SCORE: X/10
REASONING: [your explanation]"""

        response = query_ollama(prompt, max_tokens=200)

        # Parse response
        score = 0
        reasoning = response

        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+)/10', response, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))

        return {
            "valid": score >= 7,  # Consider valid if 7+ out of 10
            "score": score,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"Patch validation failed: {e}")
        return {"valid": False, "score": 0, "reasoning": f"Validation error: {e}"}

if __name__ == "__main__":
    # Test the LLM interface
    print("🧪 Testing Meta-Agent LLM Interface")

    try:
        status = check_llm_status()

        if status["status"] == "ready":
            print("✅ Ollama loaded for meta-agent")

            # Test basic functionality
            test_prompt = "Fix NameError: undefined_var not defined"
            response = generate_patch(test_prompt)
            print(f"Generated patch: {response[:200]}...")

            if response:
                print("✅ Patch generation working")
            else:
                print("❌ Patch generation failed")
        else:
            print(f"❌ Ollama not ready: {status}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
