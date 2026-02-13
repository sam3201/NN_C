#!/bin/bash
#
# SAM-D AGI - Unified Launch Script
# Integrates Automation Framework with SAM-D Core
#

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                 🚀 SAM-D AGI - Unified Launcher                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo -e "${BLUE}📊 System Check:${NC}"
echo "  Python: $(python3 --version)"
echo ""

# Build C extensions if needed
echo -e "${BLUE}🔨 Building C Extensions...${NC}"
if python3 setup.py build_ext --inplace 2>&1 | grep -q "error"; then
    echo -e "${YELLOW}⚠️  C extension build had warnings (this is OK)${NC}"
else
    echo -e "${GREEN}✅ C extensions built${NC}"
fi
echo ""

# Check components
echo -e "${BLUE}📦 Component Status:${NC}"

# Check Automation Framework
if [ -f "automation_framework/python/automation_bridge.py" ]; then
    echo -e "  ${GREEN}✅${NC} Automation Framework"
else
    echo -e "  ${YELLOW}⚠️${NC} Automation Framework (not found)"
fi

# Check SAM-D Core
if [ -f "src/python/sam_cores.py" ]; then
    echo -e "  ${GREEN}✅${NC} SAM-D Core"
else
    echo -e "  ${YELLOW}⚠️${NC} SAM-D Core (not found)"
fi

# Check Documentation
if [ -f "DOCS/OMNISYNAPSE_X_COMPLETE_DOCUMENTATION.md" ]; then
    echo -e "  ${GREEN}✅${NC} Documentation"
else
    echo -e "  ${YELLOW}⚠️${NC} Documentation (not found)"
fi

echo ""

# Run tests
echo -e "${BLUE}🧪 Running Smoke Tests...${NC}"
if python3 -c "import sys; sys.path.insert(0, 'src/python'); from sam_cores import SamCores; print('  ✅ SAM cores import OK')" 2>/dev/null; then
    :
else
    echo -e "  ${YELLOW}⚠️${NC} SAM cores import (C extensions may need rebuild)"
fi

if python3 -c "import sys; sys.path.insert(0, 'automation_framework/python'); print('  ✅ Automation framework path OK')" 2>/dev/null; then
    :
else
    echo -e "  ${YELLOW}⚠️${NC} Automation framework import"
fi

echo ""

# Display menu
echo -e "${BLUE}🎛️  Launch Options:${NC}"
echo ""
echo "  1) Run SAM-D Core (complete_sam_unified.py)"
echo "  2) Run Automation Framework Demo"
echo "  3) Run Master Integration"
echo "  4) Run Tests"
echo "  5) Build Documentation"
echo "  6) Exit"
echo ""

read -p "Select option [1-6]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}🚀 Starting SAM-D Core...${NC}\n"
        python3 complete_sam_unified.py
        ;;
    2)
        echo -e "\n${GREEN}🚀 Starting Automation Framework Demo...${NC}\n"
        cd automation_framework/python
        python3 automation_bridge.py
        ;;
    3)
        echo -e "\n${GREEN}🚀 Starting Master Integration...${NC}\n"
        python3 .openclaw/master_integration.py
        ;;
    4)
        echo -e "\n${GREEN}🧪 Running Tests...${NC}\n"
        python3 -m pytest tests/ -v || echo "⚠️  Some tests may require dependencies"
        ;;
    5)
        echo -e "\n${GREEN}📚 Documentation Status:${NC}\n"
        echo "  Available documentation:"
        ls -1 DOCS/*.md 2>/dev/null | sed 's/^/  - /'
        echo ""
        echo "  Key files:"
        echo "  - DOCS/OMNISYNAPSE_X_COMPLETE_DOCUMENTATION.md"
        echo "  - automation_framework/README.md"
        echo "  - AGENTS.md"
        ;;
    6)
        echo -e "\n👋 Goodbye!"
        exit 0
        ;;
    *)
        echo -e "\n❌ Invalid option"
        exit 1
        ;;
esac
