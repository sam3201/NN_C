#!/usr/bin/env python3
"""
Deprecated: use test_meta_agent.py as the unified meta-agent test suite.
This wrapper keeps old entrypoints working while avoiding duplicated logic.
"""

from test_meta_agent import MetaAgentTestSuite


if __name__ == "__main__":
    MetaAgentTestSuite().run_comprehensive_tests()
