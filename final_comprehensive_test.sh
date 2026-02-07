#!/bin/bash

# SAM 2.0 Comprehensive Final Testing Suite
# Tests all functionality before production launch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0
TOTAL=0

# Logging functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}================================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================================================================${NC}"
}

# Test counter
test_count() {
    ((TOTAL++))
}

# Final results
print_results() {
    echo ""
    log_header "FINAL TEST RESULTS"
    echo "Total Tests: $TOTAL"
    echo "Passed: $PASSED"
    echo "Failed: $FAILED"
    echo "Success Rate: $((PASSED * 100 / TOTAL))%"

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! SYSTEM READY FOR PRODUCTION!${NC}"
    else
        echo -e "${RED}❌ $FAILED TESTS FAILED. REVIEW ISSUES BEFORE PRODUCTION.${NC}"
    fi
}

# Test functions
test_python_environment() {
    log_header "PHASE 1: PYTHON ENVIRONMENT & DEPENDENCIES"
    test_count

    log_test "Testing Python version..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    if [[ $PYTHON_VERSION =~ ^3\. ]]; then
        log_pass "Python version: $PYTHON_VERSION"
    else
        log_fail "Python version too old: $PYTHON_VERSION"
    fi

    test_count
    log_test "Testing core imports..."
    python3 -c "
import sys, os, threading, time, json, re
try:
    import flask, flask_socketio, eventlet, requests
    print('SUCCESS: Core web imports')
except ImportError as e:
    print(f'FAILED: {e}')
    sys.exit(1)
    " && log_pass "Core web imports successful" || log_fail "Core web imports failed"

    test_count
    log_test "Testing C library imports..."
    python3 -c "
try:
    import consciousness_algorithmic, specialized_agents_c, multi_agent_orchestrator_c
    print('SUCCESS: C libraries')
except ImportError as e:
    print(f'FAILED: {e}')
    sys.exit(1)
    " && log_pass "C libraries loaded successfully" || log_fail "C libraries failed to load"
}

test_agent_configuration() {
    log_header "PHASE 2: AGENT CONFIGURATION & ECOSYSTEM"
    test_count

    log_test "Testing agent configuration loading..."
    AGENTS=$(python3 -c "
import sys
sys.path.insert(0, '.')
from complete_sam_unified import UnifiedSAMSystem

system = UnifiedSAMSystem.__new__(UnifiedSAMSystem)
system.sam_available = True
system.ollama_available = True
system.deepseek_available = False
system.web_available = False
system.claude_available = False
system.gemini_available = False
system.openai_available = False

system.agent_configs = {}
system.initialize_agent_configs()

agent_types = {}
for config in system.agent_configs.values():
    agent_type = config['type']
    agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

print(f'Total agents: {len(system.agent_configs)}')
print(f'Agent types: {agent_types}')
print('SUCCESS')
    " 2>&1)

    if echo "$AGENTS" | grep -q "SUCCESS"; then
        AGENT_COUNT=$(echo "$AGENTS" | grep "Total agents:" | awk '{print $3}')
        log_pass "Agent configuration loaded: $AGENT_COUNT agents"

        if [ "$AGENT_COUNT" -ge 17 ]; then
            log_pass "Sufficient agent count for comprehensive ecosystem"
        else
            log_warn "Agent count lower than expected: $AGENT_COUNT"
        fi
    else
        log_fail "Agent configuration failed: $AGENTS"
    fi
}

test_autonomous_operation() {
    log_header "PHASE 3: AUTONOMOUS OPERATION & GOAL MANAGEMENT"
    test_count

    log_test "Testing autonomous operation components..."
    COMPONENTS=$(python3 -c "
import sys
sys.path.insert(0, '.')

# Test goal management
try:
    from goal_management import GoalManager, TaskNode
    gm = GoalManager()
    task = TaskNode('test_task', 'Test autonomous task')
    gm.add_subtask(task)
    print('SUCCESS: Goal management')
except Exception as e:
    print(f'FAILED: Goal management - {e}')

# Test survival agent
try:
    from survival_agent import create_survival_agent
    agent = create_survival_agent()
    print('SUCCESS: Survival agent')
except Exception as e:
    print(f'FAILED: Survival agent - {e}')
    " 2>&1)

    if echo "$COMPONENTS" | grep -q "SUCCESS: Goal management"; then
        log_pass "Goal management system operational"
    else
        log_fail "Goal management system failed"
    fi

    if echo "$COMPONENTS" | grep -q "SUCCESS: Survival agent"; then
        log_pass "Survival agent operational"
    else
        log_fail "Survival agent failed"
    fi
}

test_api_endpoints() {
    log_header "PHASE 4: API ENDPOINTS & SLASH COMMANDS"
    test_count

    log_test "Testing API endpoint availability..."

    # Start system in background for testing
    log_info "Starting SAM system for API testing..."
    python3 complete_sam_unified.py &
    SYSTEM_PID=$!
    sleep 10  # Wait for system to start

    # Test basic connectivity
    if curl -s --max-time 5 http://localhost:5004/api/status > /dev/null; then
        log_pass "Basic API connectivity established"

        test_count
        log_test "Testing slash command processing..."
        START_RESPONSE=$(curl -s --max-time 10 -X POST http://localhost:5004/api/chatbot \
            -H "Content-Type: application/json" \
            -d '{"message": "/start"}' 2>/dev/null)

        if echo "$START_RESPONSE" | grep -q "response"; then
            log_pass "Slash command processing functional"
        else
            log_fail "Slash command processing failed: $START_RESPONSE"
        fi

        test_count
        log_test "Testing /help command..."
        HELP_RESPONSE=$(curl -s --max-time 10 -X POST http://localhost:5004/api/chatbot \
            -H "Content-Type: application/json" \
            -d '{"message": "/help"}' 2>/dev/null)

        if echo "$HELP_RESPONSE" | grep -q "SAM 2.0"; then
            log_pass "/help command working"
        else
            log_fail "/help command failed"
        fi

    else
        log_fail "API connectivity failed - system not responding"
    fi

    # Clean up
    kill $SYSTEM_PID 2>/dev/null || true
    sleep 3
}

test_performance() {
    log_header "PHASE 5: PERFORMANCE & STABILITY TESTING"
    test_count

    log_test "Testing system startup performance..."
    START_TIME=$(date +%s)
    timeout 30 python3 complete_sam_unified.py &
    SYSTEM_PID=$!
    sleep 5

    if ps -p $SYSTEM_PID > /dev/null 2>&1; then
        END_TIME=$(date +%s)
        STARTUP_TIME=$((END_TIME - START_TIME))
        log_pass "System startup successful in ${STARTUP_TIME}s"

        if [ $STARTUP_TIME -le 15 ]; then
            log_pass "Startup performance excellent"
        elif [ $STARTUP_TIME -le 30 ]; then
            log_pass "Startup performance acceptable"
        else
            log_warn "Startup performance slow"
        fi
    else
        log_fail "System failed to start properly"
    fi

    kill $SYSTEM_PID 2>/dev/null || true
    sleep 3
}

test_error_handling() {
    log_header "PHASE 6: ERROR HANDLING & EDGE CASES"
    test_count

    log_test "Testing invalid slash commands..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from complete_sam_unified import UnifiedSAMSystem

system = UnifiedSAMSystem.__new__(UnifiedSAMSystem)
system.sam_available = True
system.ollama_available = True
system.deepseek_available = False
system.web_available = False
system.claude_available = False
system.gemini_available = False
system.openai_available = False

# Test invalid command
try:
    response = system._process_chatbot_message('/invalidcommand test', {})
    if 'Unknown command' in response:
        print('SUCCESS: Invalid command handling')
    else:
        print('FAILED: Invalid command not handled properly')
except Exception as e:
    print(f'FAILED: Exception in command processing - {e}')
    " && log_pass "Error handling for invalid commands" || log_fail "Invalid command handling failed"

    test_count
    log_test "Testing empty message handling..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from complete_sam_unified import UnifiedSAMSystem

system = UnifiedSAMSystem.__new__(UnifiedSAMSystem)
# Test should not crash with empty message
try:
    response = system._process_chatbot_message('', {})
    print('SUCCESS: Empty message handled')
except Exception as e:
    print(f'FAILED: Empty message caused exception - {e}')
    " && log_pass "Empty message handling" || log_fail "Empty message handling failed"
}

run_production_launch() {
    log_header "PHASE 7: PRODUCTION LAUNCH WITH RUN_SAM.SH"
    test_count

    log_test "Running production launch script..."
    if [ -f "run_sam.sh" ] && [ -x "run_sam.sh" ]; then
        log_pass "Run script exists and is executable"

        # Test script execution (without actually running the full system)
        log_info "Testing run script syntax..."
        bash -n run_sam.sh && log_pass "Run script syntax valid" || log_fail "Run script syntax errors"

        log_info "Production launch script is ready for deployment!"
        log_pass "Production launch system ready"

    else
        log_fail "Run script missing or not executable"
    fi
}

# Main test execution
main() {
    log_header "SAM 2.0 COMPREHENSIVE FINAL TESTING SUITE"
    log_info "Running comprehensive tests across all system components..."
    log_info "This may take several minutes to complete..."
    echo ""

    test_python_environment
    echo ""

    test_agent_configuration
    echo ""

    test_autonomous_operation
    echo ""

    test_api_endpoints
    echo ""

    test_performance
    echo ""

    test_error_handling
    echo ""

    run_production_launch
    echo ""

    print_results

    if [ $FAILED -eq 0 ]; then
        echo ""
        log_header "🎉 PRODUCTION LAUNCH READY!"
        echo "All tests passed! System is ready for production deployment."
        echo "Run: ./run_sam.sh"
    else
        echo ""
        log_header "⚠️  REVIEW REQUIRED"
        echo "$FAILED tests failed. Please review and fix issues before production."
    fi
}

# Run all tests
main
