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

    # Fully replace wit
