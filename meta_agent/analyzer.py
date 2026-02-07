#!/usr/bin/env python3
"""
Exception Analyzer for Meta-Agent
Extracts context from exceptions and stack traces
"""

import traceback
import inspect
import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def extract_context(exc: Exception, additional_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract comprehensive context from an exception.

    Args:
        exc: The exception that occurred
        additional_context: Optional additional context information

    Returns:
        dict: Comprehensive exception context
    """
    try:
        # Get basic exception info
        exc_type = type(exc).__name__
        exc_message = str(exc)
        exc_module = getattr(exc, '__module__', 'builtins')

        # Get full traceback
        tb_str = traceback.format_exc()

        # Extract file and line information
        tb_obj = traceback.extract_tb(exc.__traceback__)
        if tb_obj:
            last_frame = tb_obj[-1]
            file_path = last_frame.filename
            line_number = last_frame.lineno
            function_name = last_frame.name
            code_line = last_frame.line
        else:
            file_path = "unknown"
            line_number = 0
            function_name = "unknown"
            code_line = "unknown"

        # Get affected files from traceback
        affected_files = extract_affected_files(tb_str)

        # Get local variables from the frame where exception occurred
        local_vars = {}
        if exc.__traceback__:
            try:
                frame = exc.__traceback__.tb_frame
                local_vars = {k: repr(v) for k, v in frame.f_locals.items()
                            if not k.startswith('_')}  # Skip private vars
            except:
                pass

        # Get system information
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "pid": os.getpid()
        }

        # Build comprehensive context
        context = {
            "exception": {
                "type": exc_type,
                "message": exc_message,
                "module": exc_module,
                "full_message": f"{exc_type}: {exc_message}"
            },
            "traceback": {
                "full": tb_str,
                "summary": traceback.format_exception_only(exc_type, exc),
                "extracted": traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
            },
            "location": {
                "file": file_path,
                "line": line_number,
                "function": function_name,
                "code": code_line
            },
            "affected_files": affected_files,
            "local_variables": local_vars,
            "system_info": system_info,
            "additional_context": additional_context or {}
        }

        # Log the extraction
        logger.info(f"Extracted context for {exc_type} in {file_path}:{line_number}")

        return context

    except Exception as e:
        logger.error(f"Failed to extract exception context: {e}")
        return {
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "full_message": f"{type(exc).__name__}: {str(exc)}"
            },
            "traceback": {
                "full": traceback.format_exc(),
                "error": f"Context extraction failed: {e}"
            }
        }

def extract_affected_files(traceback_str: str) -> List[Dict[str, Any]]:
    """
    Extract all files mentioned in the traceback.

    Args:
        traceback_str: Full traceback string

    Returns:
        list: List of affected file information
    """
    affected_files = []

    try:
        # Parse traceback to extract file information
        tb_lines = traceback_str.splitlines()

        for line in tb_lines:
            if 'File "' in line and '", line ' in line:
                try:
                    # Extract file path, line number, and function
                    parts = line.split('File "')[1].split('", line ')
                    file_path = parts[0]
                    rest = parts[1].split(', in ')

                    line_num = int(rest[0])
                    function_name = rest[1] if len(rest) > 1 else "unknown"

                    # Get relative path if possible
                    try:
                        rel_path = os.path.relpath(file_path)
                    except:
                        rel_path = file_path

                    affected_files.append({
                        "absolute_path": file_path,
                        "relative_path": rel_path,
                        "line_number": line_num,
                        "function": function_name,
                        "exists": os.path.exists(file_path)
                    })

                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse traceback line: {line} - {e}")
                    continue

    except Exception as e:
        logger.error(f"Failed to extract affected files: {e}")

    # Remove duplicates based on file path
    seen_paths = set()
    unique_files = []
    for file_info in affected_files:
        if file_info["absolute_path"] not in seen_paths:
            seen_paths.add(file_info["absolute_path"])
            unique_files.append(file_info)

    return unique_files

def get_file_context(file_path: str, line_number: int, context_lines: int = 3) -> Dict[str, Any]:
    """
    Get context around a specific line in a file.

    Args:
        file_path: Path to the file
        line_number: Line number (1-indexed)
        context_lines: Number of lines of context to include

    Returns:
        dict: File context information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines):
            return {"error": f"Line number {line_number} out of range"}

        # Get context lines (0-indexed)
        start_line = max(0, line_number - context_lines - 1)
        end_line = min(len(lines), line_number + context_lines)

        context_lines_content = []
        for i in range(start_line, end_line):
            marker = ">>> " if i + 1 == line_number else "    "
            context_lines_content.append(f"{marker}{i + 1:4d}: {lines[i].rstrip()}")

        return {
            "file_path": file_path,
            "target_line": line_number,
            "context_lines": context_lines_content,
            "total_lines": len(lines),
            "context_start": start_line + 1,
            "context_end": end_line
        }

    except Exception as e:
        return {"error": f"Could not read file context: {e}"}

def generate_fix_prompt(context: Dict[str, Any]) -> str:
    """
    Generate a comprehensive prompt for the LLM to create a fix.

    Args:
        context: Exception context from extract_context()

    Returns:
        str: Formatted prompt for LLM
    """
    exc_info = context.get("exception", {})
    location = context.get("location", {})
    affected_files = context.get("affected_files", [])

    prompt = f"""You are an expert Python developer. Fix the following exception by generating a unified diff.

EXCEPTION DETAILS:
- Type: {exc_info.get('type', 'Unknown')}
- Message: {exc_info.get('message', 'Unknown')}
- Location: {location.get('file', 'Unknown')}:{location.get('line', 'Unknown')}
- Function: {location.get('function', 'Unknown')}

TRACEBACK SUMMARY:
{context.get('traceback', {}).get('full', 'No traceback available')[:1000]}

AFFECTED FILES:
"""

    for file_info in affected_files[:3]:  # Limit to first 3 files
        prompt += f"- {file_info.get('relative_path', 'Unknown')}: line {file_info.get('line_number', 'Unknown')}\n"

        # Add file context if available
        file_context = get_file_context(
            file_info.get('absolute_path', ''),
            file_info.get('line_number', 0),
            context_lines=2
        )

        if "context_lines" in file_context:
            prompt += "  Context:\n"
            for line in file_context["context_lines"]:
                prompt += f"  {line}\n"

    prompt += """
INSTRUCTIONS:
1. Generate ONLY a unified diff that fixes the root cause
2. Do not add comments or explanations
3. Start directly with the diff (--- filename)
4. Make minimal, targeted changes
5. Ensure the fix addresses the actual error, not just symptoms

Generate the unified diff now:
"""

    return prompt

# Test function
if __name__ == "__main__":
    # Test the analyzer
    print("🧪 Testing Exception Analyzer")

    try:
        # Create a test exception
        raise ValueError("Test exception for analyzer")

    except Exception as e:
        context = extract_context(e)
        print(f"✅ Extracted context: {context['exception']['full_message']}")
        print(f"📁 Affected files: {len(context['affected_files'])}")

        # Generate fix prompt
        prompt = generate_fix_prompt(context)
        print(f"📝 Generated prompt length: {len(prompt)} characters")
