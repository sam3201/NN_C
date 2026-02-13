#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def die(msg: str, code: int = 1) -> int:
    print(f"❌ {msg}", file=sys.stderr)
    return code


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, cwd=str(cwd), env=env)
    except FileNotFoundError:
        return die(f"Command not found: {cmd[0]}")


def venv_python(root: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def ensure_venv(root: Path, base_python: str) -> int:
    vpy = venv_python(root)
    if vpy.exists():
        return 0
    print("📦 Creating virtual environment in .venv ...")
    return run([base_python, "-m", "venv", ".venv"], cwd=root)


def pip_install(root: Path, pkgs: list[str]) -> int:
    vpy = str(venv_python(root))
    # Upgrade pip tooling
    rc = run(
        [vpy, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=root,
    )
    if rc != 0:
        return rc
    # Install deps
    return run([vpy, "-m", "pip", "install", *pkgs], cwd=root)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Universal SAM runner (no just/bash/powershell needed)."
    )
    ap.add_argument("--profile", default=os.environ.get("SAM_PROFILE", "full"))
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Base python used to create venv (default: current python)",
    )
    ap.add_argument(
        "--no-build", action="store_true", help="Skip building C extensions"
    )
    ap.add_argument("--no-deps", action="store_true", help="Skip installing deps")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)

    # Sanity checks
    if not (root / "setup.py").exists():
        return die("setup.py not found. Run this from the repo root.")
    if not (root / "src/python/run_sam.py").exists():
        return die("src/python/run_sam.py not found.")

    # Create venv if missing
    rc = ensure_venv(root, args.python)
    if rc != 0:
        return rc

    vpy = venv_python(root)
    if not vpy.exists():
        return die("Virtualenv python missing after venv creation.")

    # Install deps
    if not args.no_deps:
        req = root / "requirements.txt"
        if req.exists():
            print("📚 Installing dependencies from requirements.txt ...")
            rc = pip_install(root, ["-r", "requirements.txt"])
        else:
            print("⚠️ requirements.txt not found; installing minimal deps ...")
            rc = pip_install(root, ["requests", "requests-oauthlib", "numpy"])
        if rc != 0:
            return rc
    else:
        print("⏭️  Skipping dependency install (--no-deps).")

    # Build extensions
    if not args.no_build:
        print("🧩 Building C extensions (setup.py build_ext --inplace) ...")
        rc = run([str(vpy), "setup.py", "build_ext", "--inplace"], cwd=root)
        if rc != 0:
            return rc
    else:
        print("⏭️  Skipping extension build (--no-build).")

    # Run SAM
    print(f"🚀 Starting SAM (profile={args.profile}) ...")
    env = os.environ.copy()
    env["SAM_PROFILE"] = args.profile
    # Ensure root and src/python are in PYTHONPATH
    python_path = [str(root), str(root / "src" / "python")]
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = ":".join(python_path + [env["PYTHONPATH"]])
    else:
        env["PYTHONPATH"] = ":".join(python_path)
    
    rc = run([str(vpy), "src/python/run_sam.py", "--profile", args.profile], cwd=root, env=env)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
