#!/bin/bash
# Deployment Script for Automation Framework
# Usage: ./deploy.sh [environment]
# Environments: development, staging, production

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Automation Framework Deployment"
echo "=================================="
echo "Environment: $ENVIRONMENT"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Pre-deployment checks
echo "📋 Step 1: Pre-deployment Checks"
echo "-----------------------------------"

# Check Rust version
if ! command -v rustc &> /dev/null; then
    log_error "Rust not found. Please install Rust."
    exit 1
fi

RUST_VERSION=$(rustc --version | awk '{print $2}')
log_info "Rust version: $RUST_VERSION"

# Check if version meets minimum requirement
if ! rustc --version | grep -E "1\.(7[0-9]|[8-9][0-9])" &> /dev/null; then
    log_warn "Rust version should be 1.70 or higher"
fi

# Step 2: Run tests
echo ""
echo "🧪 Step 2: Running Test Suite"
echo "-----------------------------------"

cd "$PROJECT_ROOT/automation_framework"

log_info "Running cargo test..."
if cargo test --release 2>&1 | tee /tmp/test_output.log; then
    TEST_COUNT=$(grep -c "^test " /tmp/test_output.log || echo "0")
    PASSED_COUNT=$(grep -c "test result: ok" /tmp/test_output.log || echo "0")
    log_info "Tests completed: $TEST_COUNT tests"
    log_info "All tests passed ✅"
else
    log_error "Tests failed! Aborting deployment."
    exit 1
fi

# Step 3: Build release binary
echo ""
echo "🔨 Step 3: Building Release Binary"
echo "-----------------------------------"

log_info "Building release binary..."
if cargo build --release; then
    log_info "Build successful ✅"
else
    log_error "Build failed!"
    exit 1
fi

# Check binary exists
if [ ! -f "$PROJECT_ROOT/automation_framework/target/release/automation_cli" ]; then
    log_error "Binary not found at expected location"
    exit 1
fi

BINARY_SIZE=$(du -h "$PROJECT_ROOT/automation_framework/target/release/automation_cli" | cut -f1)
log_info "Binary size: $BINARY_SIZE"

# Step 4: Environment setup
echo ""
echo "⚙️  Step 4: Environment Configuration"
echo "-----------------------------------"

# Create config directory
CONFIG_DIR="$HOME/.config/automation_framework"
mkdir -p "$CONFIG_DIR"

# Copy default config if doesn't exist
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    log_info "Creating default configuration..."
    cat > "$CONFIG_DIR/config.toml" << 'EOF'
[general]
environment = "development"
log_level = "info"

[quotas]
api_calls_per_minute = 1000
tokens_per_hour = 1000000
compute_seconds_per_day = 3600
storage_mb = 1024

[circuit_breaker]
failure_threshold = 5
success_threshold = 3
timeout_seconds = 60

[retry]
max_attempts = 3
initial_delay_ms = 100
max_delay_ms = 30000
exponential_base = 2.0

[rate_limiter]
max_tokens = 100
refill_rate = 10

[health_check]
interval_seconds = 30
EOF
    log_info "Configuration created at $CONFIG_DIR/config.toml"
fi

# Step 5: Install binary
echo ""
echo "📦 Step 5: Installing Binary"
echo "-----------------------------------"

INSTALL_DIR="/usr/local/bin"
if [ "$EUID" -ne 0 ]; then
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
    log_warn "Installing to user directory: $INSTALL_DIR"
fi

# Backup existing binary if present
if [ -f "$INSTALL_DIR/automation_cli" ]; then
    BACKUP_NAME="automation_cli.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$INSTALL_DIR/automation_cli" "$INSTALL_DIR/$BACKUP_NAME"
    log_info "Existing binary backed up as $BACKUP_NAME"
fi

# Copy new binary
cp "$PROJECT_ROOT/automation_framework/target/release/automation_cli" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/automation_cli"
log_info "Binary installed to $INSTALL_DIR/automation_cli ✅"

# Step 6: Environment-specific setup
echo ""
echo "🌍 Step 6: Environment-Specific Setup ($ENVIRONMENT)"
echo "-----------------------------------"

case $ENVIRONMENT in
    development)
        log_info "Development environment setup"
        export RUST_LOG=debug
        ;;
    staging)
        log_info "Staging environment setup"
        export RUST_LOG=info
        # Additional staging config
        ;;
    production)
        log_info "Production environment setup"
        export RUST_LOG=warn
        
        # Production safety checks
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            log_warn "ANTHROPIC_API_KEY not set"
        fi
        
        if [ -z "$AUTOMATION_ALERT_WEBHOOK" ]; then
            log_warn "AUTOMATION_ALERT_WEBHOOK not set (alerts will only be logged)"
        fi
        
        # Check system resources
        AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$AVAILABLE_MEM" -lt 512 ]; then
            log_warn "Low available memory: ${AVAILABLE_MEM}MB (recommend 512MB+)"
        fi
        ;;
    *)
        log_error "Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Step 7: Create systemd service (Linux only)
echo ""
echo "🔧 Step 7: Service Configuration"
echo "-----------------------------------"

if command -v systemctl &> /dev/null && [ "$EUID" -eq 0 ]; then
    log_info "Creating systemd service..."
    
    cat > /etc/systemd/system/automation-framework.service << EOF
[Unit]
Description=Automation Framework Service
After=network.target

[Service]
Type=simple
User=automation
Group=automation
WorkingDirectory=$PROJECT_ROOT
Environment=RUST_LOG=info
Environment=CONFIG_DIR=$CONFIG_DIR
ExecStart=$INSTALL_DIR/automation_cli
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_info "Systemd service created ✅"
    log_info "Start with: sudo systemctl start automation-framework"
    log_info "Enable auto-start: sudo systemctl enable automation-framework"
else
    log_warn "Systemd not available or not root. Skipping service creation."
    log_info "You can run the binary directly: $INSTALL_DIR/automation_cli"
fi

# Step 8: Health check
echo ""
echo "🏥 Step 8: Health Check"
echo "-----------------------------------"

# Try to run a simple health check
if $INSTALL_DIR/automation_cli --version &> /dev/null; then
    VERSION=$($INSTALL_DIR/automation_cli --version 2>&1)
    log_info "Binary is executable: $VERSION ✅"
else
    log_warn "Could not verify binary version"
fi

# Step 9: Post-deployment summary
echo ""
echo "✅ Deployment Complete!"
echo "=================================="
echo ""
echo "📊 Summary:"
echo "  • Environment: $ENVIRONMENT"
echo "  • Binary: $INSTALL_DIR/automation_cli"
echo "  • Config: $CONFIG_DIR/config.toml"
echo "  • Tests: All passed ✅"
echo ""
echo "🚀 Next Steps:"
echo "  1. Update configuration in $CONFIG_DIR/config.toml"
echo "  2. Set environment variables (ANTHROPIC_API_KEY, etc.)"
echo "  3. Start the service:"
echo "     sudo systemctl start automation-framework"
echo ""
echo "📖 Documentation:"
echo "  • README: $PROJECT_ROOT/README.md"
echo "  • Production Guide: $PROJECT_ROOT/DOCS/PRODUCTION_READINESS_REPORT.md"
echo ""
echo "🔍 Monitoring:"
echo "  • Logs: journalctl -u automation-framework -f"
echo "  • Config: $CONFIG_DIR/config.toml"
echo ""

# Add to PATH if installed to user directory
if [ "$INSTALL_DIR" = "$HOME/.local/bin" ]; then
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "⚠️  Note: $HOME/.local/bin is not in your PATH"
        echo "    Add this to your ~/.bashrc or ~/.zshrc:"
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

log_info "Deployment successful! 🎉"
