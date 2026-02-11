#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

def load_env_file(p: Path) -> None:
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("SAM_PROFILE", "full"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)

    load_env_file(root / ".env.local")
    load_env_file(root / "profiles" / f"{args.profile}.env")
    os.environ["SAM_PROFILE"] = args.profile

    # Two-phase defaults (can be overridden by env files)
    os.environ.setdefault("SAM_TWO_PHASE_BOOT", "1")
    os.environ.setdefault("SAM_META_ONLY_BOOT", "0")
    os.environ.setdefault("SAM_REQUIRE_META_AGENT", "1")
    os.environ.setdefault("SAM_AUTONOMOUS_ENABLED", "1")
    os.environ.setdefault("SAM_REQUIRE_SELF_MOD", "1")
    os.environ.setdefault("SAM_STRICT_LOCAL_ONLY", "1")

    os.environ.setdefault("SAM_PROVIDER_AUTO_SWITCH", "1")
    strict_local_only = os.environ.get("SAM_STRICT_LOCAL_ONLY", "0") == "1"
    if strict_local_only:
        local_spec = "local:rules"
        os.environ["SAM_POLICY_PROVIDER_PRIMARY"] = local_spec
        os.environ["SAM_POLICY_PROVIDER_FALLBACK"] = local_spec
        os.environ["SAM_TEACHER_POOL_PRIMARY"] = local_spec
        os.environ["SAM_TEACHER_POOL_FALLBACK"] = local_spec
        os.environ["SAM_CHAT_PROVIDER"] = ""
    else:
        default_hf_dir = os.environ.get(
            "SAM_HF_MODEL_DIR",
            str(root / "training" / "output_lora_qwen2.5_1.5b_fp16_v2"),
        )
        os.environ.setdefault("SAM_POLICY_PROVIDER_PRIMARY", f"hf:Qwen/Qwen2.5-1.5B@{default_hf_dir}")
        os.environ.setdefault("SAM_POLICY_PROVIDER_FALLBACK", "ollama:qwen2.5-coder:7b")
        os.environ.setdefault("SAM_TEACHER_POOL_PRIMARY", f"hf:Qwen/Qwen2.5-1.5B@{default_hf_dir}")
        os.environ.setdefault("SAM_TEACHER_POOL_FALLBACK", "ollama:mistral:latest")
        os.environ.setdefault("SAM_CHAT_PROVIDER", "ollama:qwen2.5-coder:7b")
        os.environ.setdefault("SAM_HF_DEVICE_MAP", "cpu")
        os.environ.setdefault("SAM_HF_DTYPE", "float16")
        os.environ.setdefault("SAM_HF_FORCE_GREEDY", "1")

    cmd = [sys.executable, str(root / "complete_sam_unified.py")]
    return subprocess.call(cmd, env=os.environ.copy())

if __name__ == "__main__":
    raise SystemExit(main())
