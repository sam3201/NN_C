#!/usr/bin/env python3
"""
Setup Virtual Environment and Install Dependencies for SAM 2.0
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr}")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("🚀 Setting up Virtual Environment for SAM 2.0")
    print("=" * 60)
    
    # Check if venv already exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("📁 Virtual environment already exists")
    else:
        # Create virtual environment
        if not run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
            return False
    
    # Determine activation script
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"\n📋 To activate virtual environment, run:")
    print(f"   {activate_cmd}")
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    return True, pip_cmd

def install_dependencies(pip_cmd):
    """Install all required dependencies"""
    print("\n📦 Installing Dependencies...")
    print("=" * 60)
    
    # Core dependencies
    core_deps = [
        "numpy>=1.21.0",
        "flask>=2.0.0",
        "flask-socketio>=5.0.0",
        "eventlet>=0.33.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0"
    ]
    
    # Async and optimization dependencies
    async_deps = [
        "aiohttp>=3.8.0",
        "asyncio",
        "psutil>=5.8.0"
    ]
    
    # Monitoring dependencies
    monitoring_deps = [
        "watchdog>=2.1.0"
    ]
    
    # ML/AI dependencies (optional)
    ml_deps = [
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "sentence-transformers>=2.0.0"
    ]
    
    # Testing dependencies
    test_deps = [
        "pytest>=6.0.0",
        "pytest-asyncio>=0.18.0"
    ]
    
    # Install core dependencies first
    print("\n🔧 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    # Install async dependencies
    print("\n⚡ Installing async dependencies...")
    for dep in async_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, using fallback...")
    
    # Install monitoring dependencies
    print("\n👁️ Installing monitoring dependencies...")
    for dep in monitoring_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, using fallback...")
    
    # Install ML dependencies (optional)
    print("\n🧠 Installing ML dependencies (optional)...")
    for dep in ml_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, using fallback...")
    
    # Install testing dependencies
    print("\n🧪 Installing testing dependencies...")
    for dep in test_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    return True

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """# SAM 2.0 Dependencies

# Core
numpy>=1.21.0
flask>=2.0.0
flask-socketio>=5.0.0
eventlet>=0.33.0
requests>=2.25.0
python-dotenv>=0.19.0

# Async & Performance
aiohttp>=3.8.0
psutil>=5.8.0

# Monitoring
watchdog>=2.1.0

# ML/AI (Optional)
torch>=1.9.0
transformers>=4.0.0
sentence-transformers>=2.0.0

# Testing
pytest>=6.0.0
pytest-asyncio>=0.18.0

# Local LLM (Optional)
llama-cpp-python>=0.1.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("📄 Created requirements.txt")

def create_activation_script():
    """Create activation script for easy setup"""
    if sys.platform == "win32":
        script = """@echo off
echo Activating SAM 2.0 Virtual Environment...
call venv\\Scripts\\activate
echo Virtual Environment Activated!
echo.
echo You can now run:
echo   python3 correct_sam_hub.py
echo   python3 system_test_suite.py
echo.
cmd /k
"""
        script_name = "activate_sam.bat"
    else:
        script = """#!/bin/bash
echo "Activating SAM 2.0 Virtual Environment..."
source venv/bin/activate
echo "Virtual Environment Activated!"
echo ""
echo "You can now run:"
echo "  python3 correct_sam_hub.py"
echo "  python3 system_test_suite.py"
echo ""
exec "$SHELL"
"""
        script_name = "activate_sam.sh"
    
    with open(script_name, "w") as f:
        f.write(script)
    
    if sys.platform != "win32":
        os.chmod(script_name, 0o755)
    
    print(f"🚀 Created {script_name} for easy activation")

def main():
    """Main setup function"""
    print("🏗️ SAM 2.0 Environment Setup")
    print("=" * 60)
    
    # Setup virtual environment
    result = setup_virtual_environment()
    if not result:
        print("❌ Failed to setup virtual environment")
        return False
    
    success, pip_cmd = result
    
    # Install dependencies
    if not install_dependencies(pip_cmd):
        print("⚠️ Some dependencies failed to install, but continuing...")
    
    # Create requirements file
    create_requirements_file()
    
    # Create activation script
    create_activation_script()
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   .\\activate_sam.bat")
        print("   OR: venv\\Scripts\\activate")
    else:
        print("   source activate_sam.sh")
        print("   OR: source venv/bin/activate")
    
    print("\n2. Run the system:")
    print("   python3 correct_sam_hub.py")
    print("   python3 system_test_suite.py")
    
    print("\n3. Install additional dependencies if needed:")
    print("   pip install -r requirements.txt")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
