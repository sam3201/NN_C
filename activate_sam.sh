#!/bin/bash
echo "Activating SAM 2.0 Virtual Environment..."
source venv/bin/activate
echo "Virtual Environment Activated!"
echo ""
echo "You can now run:"
echo "  python3 correct_sam_hub.py"
echo "  python3 system_test_suite.py"
echo ""
exec "$SHELL"
