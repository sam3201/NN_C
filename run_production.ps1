$ErrorActionPreference = "Stop"

Write-Host "🚀 Initializing SAM-D Production Launcher..."

$venv = "venv"

if (!(Test-Path $venv)) {
  python -m venv $venv
}

& "$venv\Scripts\python.exe" -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,14), sys.version"
& "$venv\Scripts\python.exe" -m pip install -U pip setuptools wheel
& "$venv\Scripts\python.exe" -m pip install -r requirements.txt
& "$venv\Scripts\python.exe" setup.py build_ext --inplace | Out-Null

New-Item -ItemType Directory -Force -Path "logs","sam_data\backups" | Out-Null

$env:PYTHONPATH="src/python;."
$env:SAM_PROFILE="full"
$env:SAM_AUTONOMOUS_ENABLED="1"
$env:SAM_UNBOUNDED_MODE="1"
$env:SAM_RESTART_ENABLED="1"
$env:SAM_STRICT_LOCAL_ONLY="1"
$env:SAM_HOT_RELOAD="1"

while ($true) {
  & "$venv\Scripts\python.exe" "src\python\complete_sam_unified.py" "--port" "5005"
  Start-Sleep -Seconds 3
}

