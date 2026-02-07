#!/usr/bin/env python3
"""
Meta-Agent Configuration
Centralized configuration for the self-healing meta-agent system
"""

import os
from pathlib import Path

# Project Configuration
PROJECT_ROOT = str(Path(__file__).parent.parent)
TEST_COMMAND = ["python3", "-m", "pytest", "-q", "--tb=short"]
MAX_PATCH_LINES = 200
ALLOW_FILES = [".py", ".md", ".txt", ".json", ".yaml", ".yml"]

# LLM Configuration
LLM_MODEL_NAME = "microsoft/DialoGPT-small"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.3  # Lower temperature for more deterministic code generation

# Safety Configuration
MAX_PATCH_SIZE_BYTES = 10240  # 10KB max patch size
DANGEROUS_PATTERNS = [
    "rm -rf",
    "subprocess",
    "os.system",
    "eval(",
    "exec(",
    "__import__",
    "importlib",
    "sys.exit",
    "os._exit",
    "quit(",
    "exit(",
    "import os; os.",
    "import subprocess",
    "from subprocess",
    "shell=True",
    "capture_output=False"
]

# Git Configuration
GIT_COMMIT_MESSAGE_PREFIX = "🤖 Auto-fix by meta-agent:"
GIT_COMMIT_MESSAGE_TEMPLATE = "{prefix} {description}"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "meta_agent.log"

# Timeout Configuration
LLM_TIMEOUT_SECONDS = 30
TEST_TIMEOUT_SECONDS = 60
GIT_TIMEOUT_SECONDS = 10

# Validation Configuration
REQUIRE_TESTS_PASS = True  # Enforce the key rule: cannot modify code unless tests pass
VALIDATE_SYNTAX = True
VALIDATE_SAFETY = True
VALIDATE_SIZE = True

# Monitoring Configuration
ENABLE_METRICS = True
METRICS_FILE = "meta_agent_metrics.json"

# Emergency Stop
EMERGENCY_STOP_FILE = ".meta_agent_stop"
if os.path.exists(EMERGENCY_STOP_FILE):
    print("🚨 EMERGENCY STOP FILE DETECTED - META-AGENT DISABLED")
    ENABLED = False
else:
    ENABLED = True
