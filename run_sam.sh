#!/bin/bash

# SAM 2.0 Unified Complete System - Production Launch Script
# Ensures proper virtual environment, dependencies, and system startup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"
REQUIREMENTS_FILE="requirements.txt"
SAM_SYSTEM="complete_sam_unified.py"
PORT=5004

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please adapt for your system."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    print_header "CREATING VIRTUAL ENVIRONMENT"
    print_status "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"

    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_header "ACTIVATING VIRTUAL ENVIRONMENT"
print_status "Activating virtual environment..."
source "$VENV_ACTIVATE"

if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" != *"$VENV_DIR" ]]; then
    print_error "Virtual environment not properly activated"
    exit 1
fi

print_status "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
print_header "UPGRADING PIP"
print_status "Upgrading pip to latest version..."
pip install --upgrade pip

# Install Python dependencies
print_header "INSTALLING PYTHON DEPENDENCIES"
if [ -f "$REQUIREMENTS_FILE" ]; then
    print_status "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"

    if [ $? -ne 0 ]; then
        print_error "Failed to install Python dependencies"
        exit 1
    fi
else
    print_warning "Requirements file not found: $REQUIREMENTS_FILE"
    print_status "Installing basic dependencies..."
    pip install flask flask-socketio eventlet requests python-dotenv psutil
fi

# Install additional system dependencies
print_header "INSTALLING SYSTEM DEPENDENCIES"
print_status "Ensuring system dependencies are available..."

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    print_warning "Ollama not found. Install from: https://ollama.ai/download"
    print_status "Continuing without Ollama - system will use available models"
else
    print_status "Ollama found - checking available models..."
    ollama list || print_warning "Ollama models not accessible"
fi

# Build C extensions
print_header "BUILDING C EXTENSIONS"
if [ -f "setup.py" ]; then
    print_status "Building C extensions..."
    python setup.py build_ext --inplace

    if [ $? -ne 0 ]; then
        print_error "Failed to build C extensions"
        print_warning "Continuing without C extensions - some features may be limited"
    else
        print_status "C extensions built successfully"
    fi
else
    print_warning "setup.py not found - skipping C extension build"
fi

# Check for API keys (optional)
print_header "CHECKING API CONFIGURATION"
API_KEYS_CONFIGURED=0

if [ -n "$ANTHROPIC_API_KEY" ]; then
    ((API_KEYS_CONFIGURED++))
    print_status "Anthropic API key configured"
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    ((API_KEYS_CONFIGURED++))
    print_status "Google API key configured"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    ((API_KEYS_CONFIGURED++))
    print_status "OpenAI API key configured"
fi

if [ $API_KEYS_CONFIGURED -eq 0 ]; then
    print_warning "No API keys configured - system will use available local models"
else
    print_status "$API_KEYS_CONFIGURED API key(s) configured"
fi

# Run system health check
print_header "SYSTEM HEALTH CHECK"
print_status "Running pre-launch health checks..."

# Check if main system file exists
if [ ! -f "$SAM_SYSTEM" ]; then
    print_error "SAM system file not found: $SAM_SYSTEM"
    exit 1
fi

# Quick syntax check
python3 -m py_compile "$SAM_SYSTEM"
if [ $? -ne 0 ]; then
    print_error "Syntax error in $SAM_SYSTEM"
    exit 1
fi

# Check imports
python3 -c "
try:
    import sys
    sys.path.insert(0, '.')
    from complete_sam_unified import UnifiedSAMSystem
    print('✅ Main system imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️ Import warning: {e}')
"

# Check C extensions
python3 -c "
try:
    import consciousness_algorithmic
    print('✅ Consciousness module available')
except ImportError:
    print('⚠️ Consciousness module not available - using fallback')

try:
    import specialized_agents_c
    print('✅ Specialized agents available')
except ImportError:
    print('⚠️ Specialized agents not available - using fallback')

try:
    import multi_agent_orchestrator_c
    print('✅ Multi-agent orchestrator available')
except ImportError:
    print('⚠️ Multi-agent orchestrator not available - using fallback')
"

print_status "Health checks completed"

# Launch the system
print_header "LAUNCHING SAM 2.0 UNIFIED COMPLETE SYSTEM"
print_status "Starting SAM system on port $PORT..."
print_status "Dashboard will be available at: http://localhost:$PORT"
print_status "Press Ctrl+C to stop the system gracefully"
echo ""

# Set environment variables
export FLASK_ENV=production
export PYTHONPATH="$PWD:$PYTHONPATH"

# Launch with proper error handling
python3 "$SAM_SYSTEM" || {
    print_error "SAM system exited with error"
    print_status "Check logs above for details"
    exit 1
}
