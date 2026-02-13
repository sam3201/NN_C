$ErrorActionPreference = "Stop"

Write-Host "🚀 Initializing SAM-D Production Launcher (Windows)..."

Set-Location $PSScriptRoot

if (-not (Test-Path "venv")) {
  Write-Host "🧪 Creating venv..."
  py -3 -m venv venv
}

$py = Join-Path "venv" "Scripts\python.exe"

Write-Host "🐍 Using Python: " (& $py -V)
Write-Host "📍 Python path: " (& $py -c "import sys; print(sys.executable)")

& $py -m pip install -U pip setuptools wheel
& $py -m pip install -r requirements.txt

Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
& $py setup.py build_ext --inplace

New-Item -ItemType Directory -Force -Path "logs","sam_data\backups" | Out-Null

$env:PYTHONPATH = "src/python;."
$env:SAM_PROFILE = $env:SAM_PROFILE ?? "full"
$env:SAM_AUTONOMOUS_ENABLED = "1"
$env:SAM_STRICT_LOCAL_ONLY = "1"

if (-not $env:PORT) { $env:PORT = "5005" }

Write-Host "========================================================"
Write-Host "🤖 Starting SAM-D"
Write-Host "📊 Dashboard: http://localhost:$($env:PORT)"
Write-Host "========================================================"

while ($true) {
  Write-Host "🎯 Launching..."
  & $py "src/python/complete_sam_unified.py" "--port" $env:PORT
  Write-Host "🔄 Restarting in 2s..."
  Start-Sleep -Seconds 2
}

