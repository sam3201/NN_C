#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BACKUP_SUFFIX = ".bak_portable"


def backup(path: Path) -> None:
    b = path.with_name(path.name + BACKUP_SUFFIX)
    if not b.exists():
        b.write_bytes(path.read_bytes())


def write_text(path: Path, text: str) -> None:
    backup(path)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def replace_all(text: str, replacements: list[tuple[str, str]]) -> tuple[str, int]:
    n = 0
    for pat, rep in replacements:
        new, k = re.subn(pat, rep, text, flags=re.MULTILINE)
        text = new
        n += k
    return text, n


def ensure_shebang_bash(text: str) -> str:
    lines = text.splitlines(True)
    if lines and lines[0].startswith("#!"):
        lines[0] = "#!/usr/bin/env bash\n"
    else:
        lines.insert(0, "#!/usr/bin/env bash\n")
    return "".join(lines)


def patch_run_sam_simple(root: Path) -> None:
    p = root / "run_sam_simple.sh"
    if not p.exists():
        return
    t = read_text(p)

    # Fully replace with a portable, robust version.
    new = """#!/usr/bin/env bash
set -euo pipefail

# Repo root = directory containing this script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source "venv/bin/activate"
fi

# Load env files (optional)
set -a
if [ -f ".env.local" ]; then
  source ".env.local"
fi
PROFILE_NAME="${SAM_PROFILE:-full}"
PROFILE_FILE="$ROOT/profiles/${PROFILE_NAME}.env"
if [ -f "$PROFILE_FILE" ]; then
  source "$PROFILE_FILE"
fi
set +a

exec "$ROOT/tools/run_sam_two_phase.sh"
"""
    write_text(p, new)
    print(f"patched: {p}")


def patch_profile_wrappers(root: Path) -> None:
    for name, profile in [
        ("run_sam_full.sh", "full"),
        ("run_sam_experimental.sh", "experimental"),
    ]:
        p = root / name
        if not p.exists():
            continue
        new = f"""#!/usr/bin/env bash
set -euo pipefail
export SAM_PROFILE={profile}
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
exec "$ROOT/run_sam_simple.sh"
"""
        write_text(p, new)
        print(f"patched: {p}")


def patch_tools_two_phase(root: Path) -> None:
    p = root / "tools" / "run_sam_two_phase.sh"
    if not p.exists():
        return
    t = read_text(p)
    t = ensure_shebang_bash(t)

    # Insert ROOT logic at top + replace cd and python paths + model dir
    # If script already has ROOT, avoid duplicating
    if 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' not in t:
        t = re.sub(
            r"(?m)^set\s+-e.*\n",
            'set -euo pipefail\n\nROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\ncd "$ROOT"\n\n',
            t,
            count=1,
        )
        # If there wasn't set -e line, just prepend
        if 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' not in t:
            t = (
                'set -euo pipefail\n\nROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\ncd "$ROOT"\n\n'
                + t
            )

    # Remove any hardcoded cd /Users/... lines
    t, _ = replace_all(
        t,
        [
            (r"(?m)^\s*cd\s+/Users/samueldasari/Personal/NN_C\s*$", r'cd "$ROOT"'),
            (
                r"(?m)^\s*source\s+\.venv/bin/activate\s*$",
                r'if [ -f ".venv/bin/activate" ]; then\n  source ".venv/bin/activate"\nelif [ -f "venv/bin/activate" ]; then\n  source "venv/bin/activate"\nfi',
            ),
            (
                r"(?m)^\s*source\s+\.env\.local\s*$",
                r'if [ -f ".env.local" ]; then\n  source ".env.local"\nfi',
            ),
            (
                r"(?m)PYTHONUNBUFFERED=1\s+python\s+/Users/samueldasari/Personal/NN_C/complete_sam_unified\.py",
                r'PYTHONUNBUFFERED=1 python "$ROOT/complete_sam_unified.py"',
            ),
        ],
    )

    # Replace hardcoded model dir exports if present
    # We add a DEFAULT_HF_MODEL_DIR block + use it for PRIMARY + TEACHER.
    if "DEFAULT_HF_MODEL_DIR" not in t:
        t = t.replace(
            "export SAM_POLICY_PROVIDER_PRIMARY=",
            'DEFAULT_HF_MODEL_DIR="${SAM_HF_MODEL_DIR:-$ROOT/training/output_lora_qwen2.5_1.5b_fp16_v2}"\n'
            'export SAM_POLICY_PROVIDER_PRIMARY="${SAM_POLICY_PROVIDER_PRIMARY:-',
        )
        t = t.replace(
            "export SAM_TEACHER_POOL_PRIMARY=",
            'export SAM_TEACHER_POOL_PRIMARY="${SAM_TEACHER_POOL_PRIMARY:-',
        )
        # Close the injected quotes for the primary if we injected above
        t = re.sub(
            r'export SAM_POLICY_PROVIDER_PRIMARY="\$\{SAM_POLICY_PROVIDER_PRIMARY:-([^"]+)"',
            r'export SAM_POLICY_PROVIDER_PRIMARY="${SAM_POLICY_PROVIDER_PRIMARY:-\1}',
            t,
        )

    # Also rewrite any occurrences of /Users/.../training/... into ${DEFAULT_HF_MODEL_DIR}
    t, _ = replace_all(
        t,
        [
            (
                r"/Users/samueldasari/Personal/NN_C/training/output_lora_qwen2\.5_1\.5b_fp16_v2",
                r"${DEFAULT_HF_MODEL_DIR}",
            ),
            (r"/Users/samueldasari/Personal/NN_C", r"$ROOT"),
        ],
    )

    write_text(p, t)
    print(f"patched: {p}")


def patch_run_sam(root: Path) -> None:
    p = root / "run_sam.sh"
    if not p.exists():
        return
    t = read_text(p)
    t = ensure_shebang_bash(t)

    # Ensure ROOT + cd at top
    if 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' not in t:
        if "set -e" in t:
            t = re.sub(
                r"(?m)^set\s+-e.*\n",
                'set -euo pipefail\n\nROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\ncd "$ROOT"\n\n',
                t,
                count=1,
            )
        else:
            t = (
                'set -euo pipefail\n\nROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\ncd "$ROOT"\n\n'
                + t
            )

    t = t.replace(
        "/Users/samueldasari/Personal/NN_C/tools/run_sam_two_phase.sh",
        '"$ROOT/tools/run_sam_two_phase.sh"',
    )
    write_text(p, t)
    print(f"patched: {p}")


def patch_recursive_checks(root: Path) -> None:
    p = root / "tools" / "run_recursive_checks.sh"
    if not p.exists():
        return
    t = read_text(p)
    t = ensure_shebang_bash(t)

    # Replace hardcoded cd with ROOT cd
    if 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' not in t:
        t = 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\ncd "$ROOT"\n' + t

    t, _ = replace_all(
        t,
        [
            (r"(?m)^\s*cd\s+/Users/samueldasari/Personal/NN_C\s*$", r'cd "$ROOT"'),
            (r"/Users/samueldasari/Personal/NN_C", r"$ROOT"),
        ],
    )
    write_text(p, t)
    print(f"patched: {p}")


def patch_complete_sam(root: Path) -> None:
    p = root / "complete_sam_unified.py"
    if not p.exists():
        return
    t = read_text(p)
    # Only replace the exact strings you had
    t = t.replace(
        "let currentDirectory = '/Users/samueldasari/Personal/NN_C';",
        "let currentDirectory = '/NN_C';",
    )
    t = t.replace('return "/Users/samueldasari/Personal/NN_C"', 'return "/NN_C"')
    # Also sweep for remaining absolute repo path in this file
    t = t.replace("/Users/samueldasari/Personal/NN_C", "")
    write_text(p, t)
    print(f"patched: {p}")


def patch_setup_py(root: Path) -> None:
    p = root / "setup.py"
    if not p.exists():
        return
    t = read_text(p)

    # If already patched, skip
    if "SAM_NATIVE" in t and "COMMON_ARGS" in t:
        print("setup.py already portable-patched; skipping.")
        return

    # Insert portability header + COMMON_ARGS and replace occurrences of extra_compile_args
    # This is safe even if layout differs; we do minimal textual transformation.
    if "from setuptools import setup, Extension" in t:
        t = t.replace(
            "from setuptools import setup, Extension",
            "from __future__ import annotations\n\nimport os\nimport sys\nfrom setuptools import setup, Extension\n\n"
            "def _is_msvc() -> bool:\n"
            "    return sys.platform.startswith('win')\n\n"
            "def _common_compile_args() -> list[str]:\n"
            "    if _is_msvc():\n"
            "        return ['/O2']\n"
            "    args = ['-O3']\n"
            "    if os.environ.get('SAM_NATIVE') == '1':\n"
            "        args.append('-march=native')\n"
            "    return args\n\n"
            "COMMON_ARGS = _common_compile_args()\n",
        )
    else:
        # If import line differs, prepend the helper block
        t = (
            "from __future__ import annotations\n\nimport os\nimport sys\n\n"
            "def _is_msvc() -> bool:\n"
            "    return sys.platform.startswith('win')\n\n"
            "def _common_compile_args() -> list[str]:\n"
            "    if _is_msvc():\n"
            "        return ['/O2']\n"
            "    args = ['-O3']\n"
            "    if os.environ.get('SAM_NATIVE') == '1':\n"
            "        args.append('-march=native')\n"
            "    return args\n\n"
            "COMMON_ARGS = _common_compile_args()\n\n" + t
        )

    # Replace typical patterns
    t = re.sub(
        r"extra_compile_args=\s*\[[^\]]*?-march=native[^\]]*\]",
        "extra_compile_args=COMMON_ARGS",
        t,
        flags=re.DOTALL,
    )
    t = re.sub(
        r"extra_compile_args=\s*\[[^\]]*?\-O3[^\]]*\]",
        "extra_compile_args=COMMON_ARGS",
        t,
        flags=re.DOTALL,
    )

    write_text(p, t)
    print(f"patched: {p}")


def create_run_sam_py(root: Path) -> None:
    p = root / "run_sam.py"
    if p.exists():
        print("run_sam.py already exists; leaving it.")
        return
    content = """#!/usr/bin/env python3
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

    os.environ.setdefault("SAM_PROVIDER_AUTO_SWITCH", "1")
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
"""
    p.write_text(content, encoding="utf-8")
    try:
        os.chmod(p, 0o755)
    except Exception:
        pass
    print(f"created: {p}")


def main() -> int:
    root = Path.cwd()

    # sanity check: must look like repo root
    must_exist = ["setup.py", "complete_sam_unified.py", "tools"]
    for m in must_exist:
        if not (root / m).exists():
            print(f"ERROR: run this from the repo root. Missing: {m}")
            return 2

    patch_run_sam_simple(root)
    patch_profile_wrappers(root)
    patch_tools_two_phase(root)
    patch_run_sam(root)
    patch_recursive_checks(root)
    patch_complete_sam(root)
    patch_setup_py(root)
    create_run_sam_py(root)

    print("\nDone.")
    print(f"Backups were saved as *{BACKUP_SUFFIX}")
    print(
        "Tip: portable launcher works on Windows too:  python run_sam.py --profile full"
    )
    print("Optional: native CPU tuning (not portable) by setting: SAM_NATIVE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
