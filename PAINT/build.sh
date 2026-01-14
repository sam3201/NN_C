#!/bin/bash
# 3D Paint with Neural Networks - Comprehensive Build Script
# =====================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project info
PROJECT_NAME="paint3d"
VERSION="1.0.0"

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

# Help function
show_help() {
    cat << EOF
${PROJECT_NAME} v${VERSION} - Build Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build           Build all components (default)
    paint           Build paint application only
    viewer          Build 3D viewer only
    test            Build and run mesh test
    train           Train neural networks
    demo            Run complete demo
    clean           Clean build files
    deep-clean      Remove all generated files
    install         Install to system
    uninstall       Remove from system
    dev-setup       Setup development environment
    status          Show build status
    help            Show this help

Options:
    --verbose       Verbose output
    --debug         Debug build
    --release       Release build (default)
    --check         Run tests after build
    --no-sdl        Skip SDL dependencies check

Examples:
    $0              # Build everything
    $0 build        # Build everything
    $0 train        # Train networks
    $0 demo         # Run demo
    $0 clean        # Clean build files

EOF
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check for make
    if ! command -v make &> /dev/null; then
        log_error "make is required but not installed"
        exit 1
    fi
    
    # Check for gcc/clang
    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        log_error "gcc or clang is required but not installed"
        exit 1
    fi
    
    # Check for SDL3 (unless skipped)
    if [[ "${SKIP_SDL:-}" != "true" ]]; then
        if ! pkg-config --exists sdl3 2>/dev/null; then
            log_warning "SDL3 not found via pkg-config"
            log_warning "Build may fail without SDL3 development libraries"
            log_warning "Install SDL3 development packages or use --no-sdl"
        fi
    fi
    
    log_success "Dependencies check complete"
}

# Setup development environment
setup_dev_env() {
    log_info "Setting up development environment..."
    
    # Create directories
    mkdir -p src/{core,viewer,utils} bin lib build assets
    mkdir -p build/src/{core,viewer,utils}
    mkdir -p build/utils/{NN,SDL3}
    
    # Create .gitignore if not exists
    if [[ ! -f .gitignore ]]; then
        cat > .gitignore << EOF
# Build files
build/
bin/
lib/
*.o
*.exe
*.out

# Generated files
*.mesh
*.net
trained_convolution.net
generated_3d_object.mesh
test_terrain.mesh

# System files
.DS_Store
Thumbs.db
*.log
*.bak

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Temporary files
tmp/
temp/
EOF
        log_success "Created .gitignore"
    fi
    
    # Create README if not exists
    if [[ ! -f README.md ]]; then
        cat > README.md << EOF
# $(PROJECT_NAME) v${VERSION}

3D Paint with Neural Networks - Generate 3D objects from 2D paintings using AI.

## Quick Start

\`\`\`bash
# Build everything
make

# Train neural networks
make train

# Run demo
make demo
\`\`\`

## Features

- Neural network-powered depth generation
- Real-time 3D object generation
- Interactive 3D viewer
- Training system with multiple patterns

## Documentation

See the project documentation for detailed usage instructions.

## Build Requirements

- GCC or Clang
- SDL3 development libraries
- Make
- pthread support

## License

MIT License
EOF
        log_success "Created README.md"
    fi
    
    log_success "Development environment setup complete"
}

# Build status
show_status() {
    log_info "Build status:"
    
    echo "Directories:"
    for dir in src bin lib build assets; do
        if [[ -d "$dir" ]]; then
            echo "  ✓ $dir"
        else
            echo "  ✗ $dir (missing)"
        fi
    done
    
    echo ""
    echo "Source files:"
    find src -name "*.c" 2>/dev/null | while read -r file; do
        echo "  ✓ $file"
    done || echo "  No source files found"
    
    echo ""
    echo "Executables:"
    if [[ -d "bin" ]]; then
        find bin -type f -executable 2>/dev/null | while read -r file; do
            echo "  ✓ $file"
        done || echo "  No executables found"
    else
        echo "  ✗ bin directory missing"
    fi
    
    echo ""
    echo "Generated files:"
    for file in *.mesh *.net trained_convolution.net; do
        if [[ -f "$file" ]]; then
            echo "  ✓ $file ($(stat -f%z "$file" 2>/dev/null || echo "unknown") bytes)"
        fi
    done || echo "  No generated files found"
}

# Clean function
clean_build() {
    local deep=${1:-false}
    
    if [[ "$deep" == "true" ]]; then
        log_info "Deep cleaning..."
        make deep-clean
    else
        log_info "Cleaning build files..."
        make clean
    fi
    
    log_success "Clean complete"
}

# Main build function
build_project() {
    local target=${1:-all}
    
    log_info "Building $(PROJECT_NAME) v$(VERSION)..."
    
    # Check dependencies
    check_dependencies
    
    # Run make
    if [[ "$VERBOSE" == "true" ]]; then
        make "$target" V=1
    else
        make "$target"
    fi
    
    if [[ $? -eq 0 ]]; then
        log_success "Build successful!"
        
        # Show what was built
        if [[ -d "bin" ]]; then
            echo ""
            log_info "Built executables:"
            find bin -type f -executable 2>/dev/null | while read -r file; do
                echo "  ✓ $file"
            done
        fi
    else
        log_error "Build failed!"
        exit 1
    fi
}

# Demo function
run_demo() {
    log_info "Running $(PROJECT_NAME) demo..."
    
    # Build first
    build_project all
    
    if [[ ! -f "bin/paint" ]]; then
        log_error "Paint executable not found"
        exit 1
    fi
    
    log_info "Starting paint application..."
    log_info "Demo instructions:"
    log_info "1. Paint something on the canvas"
    log_info "2. Press 'G' to generate 3D object"
    log_info "3. Press 'V' to view 3D object"
    log_info "4. Press 'ESC' to exit"
    
    ./bin/paint
}

# Training function
train_networks() {
    log_info "Training neural networks..."
    
    # Build first
    build_project paint
    
    if [[ ! -f "bin/paint" ]]; then
        log_error "Paint executable not found"
        exit 1
    fi
    
    log_info "Starting training session..."
    log_info "Training instructions:"
    log_info "1. Press 'T' to enter training mode"
    log_info "2. Press 'R' to train networks"
    log_info "3. Press 'S' to save trained networks"
    log_info "4. Press 'ESC' to exit"
    
    ./bin/paint
}

# Test function
run_tests() {
    log_info "Running tests..."
    
    # Build test
    build_project test
    
    if [[ ! -f "bin/test_mesh" ]]; then
        log_error "Test executable not found"
        exit 1
    fi
    
    log_info "Running mesh generation test..."
    ./bin/test_mesh
    
    if [[ $? -eq 0 ]]; then
        log_success "Tests passed!"
        
        # Check for generated mesh
        if [[ -f "test_terrain.mesh" ]]; then
            log_info "Test mesh generated: test_terrain.mesh"
        fi
    else
        log_error "Tests failed!"
        exit 1
    fi
}

# Install function
install_project() {
    log_info "Installing $(PROJECT_NAME)..."
    
    # Build first
    build_project all
    
    # Run make install
    make install
    
    if [[ $? -eq 0 ]]; then
        log_success "Installation complete!"
        log_info "Installed commands:"
        log_info "  - $(PROJECT_NAME)"
        log_info "  - $(PROJECT_NAME)_viewer"
    else
        log_error "Installation failed!"
        exit 1
    fi
}

# Uninstall function
uninstall_project() {
    log_info "Uninstalling $(PROJECT_NAME)..."
    
    make uninstall
    
    if [[ $? -eq 0 ]]; then
        log_success "Uninstall complete!"
    else
        log_error "Uninstall failed!"
        exit 1
    fi
}

# Main script logic
main() {
    local command=${1:-build}
    local target=${2:-all}
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose)
                VERBOSE=true
                shift
                ;;
            --debug)
                CFLAGS="-g -O0"
                shift
                ;;
            --release)
                CFLAGS="-O2"
                shift
                ;;
            --check)
                RUN_TESTS=true
                shift
                ;;
            --no-sdl)
                SKIP_SDL=true
                shift
                ;;
            help|--help|-h)
                show_help
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Export variables for make
    export CFLAGS="${CFLAGS:-}"
    export VERBOSE="${VERBOSE:-false}"
    export SKIP_SDL="${SKIP_SDL:-}"
    
    # Execute command
    case $command in
        build|all)
            build_project "$target"
            ;;
        paint)
            build_project paint
            ;;
        viewer)
            build_project viewer
            ;;
        test)
            run_tests
            ;;
        train)
            train_networks
            ;;
        demo)
            run_demo
            ;;
        clean)
            clean_build false
            ;;
        deep-clean)
            clean_build true
            ;;
        install)
            install_project
            ;;
        uninstall)
            uninstall_project
            ;;
        dev-setup)
            setup_dev_env
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
    
    # Run tests if requested
    if [[ "${RUN_TESTS:-}" == "true" && "$command" != "test" ]]; then
        run_tests
    fi
}

# Run main function with all arguments
main "$@"
