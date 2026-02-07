#!/bin/bash
# SAM 2.0 Complete Setup Script
# Sets up everything needed for SAM 2.0 AGI System
# Run this after cloning the repository

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get OS type
get_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check system prerequisites
check_prerequisites() {
    log_info "Checking system prerequisites..."

    OS=$(get_os)
    log_info "Detected OS: $OS"

    # Check Python
    if ! command_exists python3; then
        log_error "Python 3 is required but not found. Please install Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"

    # Check pip
    if ! command_exists pip3; then
        log_warning "pip3 not found, will install pip"
    fi

    # Check git
    if ! command_exists git; then
        log_error "Git is required but not found. Please install Git"
        exit 1
    fi

    # Check for C compiler
    if [[ "$OS" == "macos" ]]; then
        if ! command_exists clang || ! command_exists gcc; then
            log_error "C compiler (clang/gcc) is required for macOS. Install Xcode Command Line Tools:"
            log_error "xcode-select --install"
            exit 1
        fi
    elif [[ "$OS" == "linux" ]]; then
        if ! command_exists gcc; then
            log_error "GCC is required for Linux. Please install build-essential:"
            log_error "sudo apt-get install build-essential"
            exit 1
        fi
    fi

    log_success "Prerequisites check passed"
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment..."

    if [[ -d "venv" ]]; then
        log_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi

    python3 -m venv venv
    log_success "Virtual environment created"
}

# Activate virtual environment and install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    # Activate venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi

    # Deactivate venv
    deactivate
}

# Compile C libraries
compile_libraries() {
    log_info "Compiling C neural network libraries..."

    if [[ ! -f "Makefile" ]]; then
        log_error "Makefile not found. Cannot compile C libraries."
        exit 1
    fi

    # Check if libraries are already compiled
    if [[ -f "libsam_core.dylib" ]] || [[ -f "libsam_core.so" ]] || [[ -f "libsam_core.dll" ]]; then
        log_warning "C libraries already exist. Skipping compilation."
        return
    fi

    # Compile the libraries
    make shared

    if [[ $? -eq 0 ]]; then
        log_success "C libraries compiled successfully"
    else
        log_error "Failed to compile C libraries"
        exit 1
    fi
}

# Setup environment configuration
setup_environment() {
    log_info "Setting up environment configuration..."

    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# SAM 2.0 Environment Configuration

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=true

# SAM Configuration
SAM_LOG_LEVEL=INFO
SAM_ENABLE_MONITORING=true
SAM_ENABLE_METRICS=true

# LLM Configuration (Optional - uses local models)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

# Database Configuration (Optional)
# DATABASE_URL=sqlite:///sam.db

# Security (Change these!)
SECRET_KEY=your-secret-key-change-this
JWT_SECRET_KEY=your-jwt-secret-change-this

# Performance Tuning
MAX_WORKERS=4
REQUEST_TIMEOUT=30
MEMORY_LIMIT=1024

EOF
        log_success "Created .env configuration file"
    else
        log_warning ".env file already exists"
    fi

    # Make scripts executable
    chmod +x activate_sam.sh 2>/dev/null || true
    chmod +x launch_sam_system.py 2>/dev/null || true

    log_success "Environment setup complete"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Activate venv
    source venv/bin/activate

    # Test imports
    echo "Testing Python imports..."
    python3 -c "
import sys
sys.path.insert(0, '.')

# Test core imports
try:
    import numpy as np
    print('✓ numpy imported successfully')
except ImportError as e:
    print(f'✗ numpy import failed: {e}')
    sys.exit(1)

try:
    import torch
    print('✓ torch imported successfully')
except ImportError as e:
    print(f'✗ torch import failed: {e}')
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('✓ transformers imported successfully')
except ImportError as e:
    print(f'✗ transformers import failed: {e}')
    sys.exit(1)

try:
    from flask import Flask
    print('✓ flask imported successfully')
except ImportError as e:
    print(f'✗ flask import failed: {e}')
    sys.exit(1)

# Test custom imports
try:
    from sam_neural_core import create_sam_core
    print('✓ sam_neural_core imported successfully')
except ImportError as e:
    print(f'⚠ sam_neural_core import failed: {e}')

try:
    from local_llm import local_llm
    print('✓ local_llm imported successfully')
except ImportError as e:
    print(f'⚠ local_llm import failed: {e}')

try:
    from custom_consciousness import CustomConsciousnessModule
    print('✓ custom_consciousness imported successfully')
except ImportError as e:
    print(f'⚠ custom_consciousness import failed: {e}')

print('Python import verification complete')
"

    if [[ $? -eq 0 ]]; then
        log_success "Python imports verified"
    else
        log_warning "Some Python imports failed - system may have limited functionality"
    fi

    # Test C library
    if [[ -f "libsam_core.dylib" ]] || [[ -f "libsam_core.so" ]] || [[ -f "libsam_core.dll" ]]; then
        python3 -c "
try:
    from sam_neural_core import create_sam_core
    core, manager = create_sam_core()
    print('✓ C library loaded successfully')
    core.cleanup()
except Exception as e:
    print(f'⚠ C library test failed: {e}')
"
        log_success "C library verification complete"
    else
        log_warning "C library not found - some features may not work"
    fi

    # Deactivate venv
    deactivate
}

# Create startup scripts
create_startup_scripts() {
    log_info "Creating startup scripts..."

    # Create comprehensive startup script
    cat > start_sam.sh << 'EOF'
#!/bin/bash
# SAM 2.0 Startup Script

echo "🚀 Starting SAM 2.0 AGI System"
echo "================================"

# Check if setup has been run
if [[ ! -d "venv" ]]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if C library exists
if [[ ! -f "libsam_core.dylib" ]] && [[ ! -f "libsam_core.so" ]] && [[ ! -f "libsam_core.dll" ]]; then
    echo "⚠️  C library not found. Some features may not work."
fi

# Start the system
echo "🧠 Starting SAM 2.0..."
python3 complete_sam_system.py

EOF

    chmod +x start_sam.sh
    log_success "Created start_sam.sh startup script"

    # Create development startup script
    cat > dev_start.sh << 'EOF'
#!/bin/bash
# SAM 2.0 Development Startup Script

echo "🚀 Starting SAM 2.0 in Development Mode"
echo "========================================"

# Activate environment
source venv/bin/activate

# Set development environment
export FLASK_ENV=development
export FLASK_DEBUG=true

# Start with development settings
echo "🔧 Development mode enabled"
python3 complete_sam_system.py

EOF

    chmod +x dev_start.sh
    log_success "Created dev_start.sh development script"
}

# Print final instructions
print_instructions() {
    log_success "SAM 2.0 setup complete!"
    echo ""
    echo "🎯 Getting Started:"
    echo "==================="
    echo ""
    echo "1. Start the system:"
    echo "   ./start_sam.sh"
    echo ""
    echo "2. Or start in development mode:"
    echo "   ./dev_start.sh"
    echo ""
    echo "3. Access the web interface:"
    echo "   http://127.0.0.1:8080"
    echo ""
    echo "4. View system status:"
    echo "   http://127.0.0.1:8080/api/system/status"
    echo ""
    echo "📚 Documentation:"
    echo "================="
    echo "- README.md - Main documentation"
    echo "- CONSCIOUSNESS_INTEGRATION.md - Consciousness architecture"
    echo "- FINAL_INTEGRATION_SUMMARY.md - System overview"
    echo ""
    echo "🆘 Troubleshooting:"
    echo "==================="
    echo "- Check logs in the terminal output"
    echo "- Ensure all prerequisites are installed"
    echo "- Try running: source venv/bin/activate && python3 -c \"import torch; print('OK')\""
    echo ""
    echo "🎉 Welcome to SAM 2.0 AGI System!"
    echo "=================================="
}

# Main setup function
main() {
    echo "🎯 SAM 2.0 Complete Setup"
    echo "========================="
    echo ""

    # Run all setup steps
    check_prerequisites
    echo ""
    create_venv
    echo ""
    install_dependencies
    echo ""
    compile_libraries
    echo ""
    setup_environment
    echo ""
    verify_installation
    echo ""
    create_startup_scripts
    echo ""

    print_instructions
}

# Handle command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "SAM 2.0 Setup Script"
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --verify       Only run verification (skip setup)"
        echo "  --clean         Clean and reinstall everything"
        echo ""
        exit 0
        ;;
    "--verify")
        check_prerequisites
        verify_installation
        exit 0
        ;;
    "--clean")
        log_info "Cleaning existing installation..."
        rm -rf venv
        rm -f libsam_core.*
        log_success "Clean complete. Run setup again."
        exit 0
        ;;
    *)
        main
        ;;
esac
