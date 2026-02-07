#!/usr/bin/env python3
"""
SAM 2.0 System Launcher
Starts the complete system with all capabilities
"""

import sys
import os
import time
from pathlib import Path

def main():
    """Main launcher function"""
    print("🚀 SAM 2.0 System Launcher")
    print("=" * 60)
    
    # Check virtual environment
    if not Path("venv").exists():
        print("❌ Virtual environment not found")
        print("📋 Run: python3 setup_environment.py")
        return False
    
    # Check if port is available
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8080))
    sock.close()
    
    if result == 0:
        print("❌ Port 8080 is already in use")
        print("📋 Run: lsof -ti:8080 | xargs kill -9")
        return False
    
    print("✅ Environment checks passed")
    print("🚀 Starting SAM 2.0 Complete System...")
    print("=" * 60)
    
    # Start the system
    try:
        import subprocess
        process = subprocess.Popen([sys.executable, 'complete_sam_system.py'])
        
        print("📊 System started successfully!")
        print("🌐 Access at: http://127.0.0.1:8080")
        print("📈 Metrics at: http://127.0.0.1:8080/api/system/status")
        print("🛑 Press Ctrl+C to stop gracefully")
        print("=" * 60)
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✅ System stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
