from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

SAM_CODE_MODIFIER_AVAILABLE = True

_project_root: Path | None = None
_backup_dir: Path | None = None
_history_path: Path | None = None


def initialize_sam_code_modifier(project_root: str) -> None:
    global _project_root, _backup_dir, _history_path
    _project_root = Path(project_root).resolve()
    _backup_dir = _project_root / "SAM_Code_Backups"
    _backup_dir.mkdir(parents=True, exist_ok=True)
    _history_path = _backup_dir / "modification_history.json"
    if not _history_path.exists():
        _history_path.write_text("[]", encoding="utf-8")


def _load_history() -> list[Dict[str, Any]]:
    if not _history_path or not _history_path.exists():
        return []
    try:
        return json.loads(_history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Preserve corrupt history for inspection
        corrupt_path = _history_path.with_suffix(".corrupt.json")
        corrupt_path.write_text(_history_path.read_text(encoding="utf-8"), encoding="utf-8")
        _history_path.write_text("[]", encoding="utf-8")
        return []


def _save_history(entries: list[Dict[str, Any]]) -> None:
    if not _history_path:
        return
    _history_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def analyze_codebase() -> Dict[str, Any]:
    if not _project_root:
        raise RuntimeError("Code modifier not initialized")
    py_files = list(_project_root.rglob("*.py"))
    history = _load_history()
    return {
        "file_count": len(py_files),
        "modification_history": history,
        "improvements": [],
    }


def modify_code_safely(filepath: str, old_code: str, new_code: str, description: str) -> Dict[str, Any]:
    if not _project_root or not _backup_dir:
        raise RuntimeError("Code modifier not initialized")

    path = Path(filepath)
    if not path.is_absolute():
        path = _project_root / path
    path = path.resolve()
    if path != _project_root and _project_root not in path.parents:
        return {"success": False, "message": "Path escapes project root", "file": str(path)}

    if not path.exists():
        return {"success": False, "message": "File not found", "file": str(path)}

    content = path.read_text(encoding="utf-8")
    if old_code not in content:
        return {"success": False, "message": "Old code not found in file", "file": str(path)}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.name}.{timestamp}.bak"
    backup_path = _backup_dir / backup_name
    backup_path.write_text(content, encoding="utf-8")

    meta = {
        "original": str(path),
        "backup": str(backup_path),
        "timestamp": timestamp,
        "description": description,
    }
    meta_path = _backup_dir / f"{backup_name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    updated = content.replace(old_code, new_code, 1)
    path.write_text(updated, encoding="utf-8")

    history = _load_history()
    history.append(meta)
    _save_history(history)

    return {
        "success": True,
        "message": "Modification applied",
        "file": str(path),
        "backup": str(backup_path),
        "meta": meta,
    }


def rollback_modification(backup_path: str) -> Dict[str, Any]:
    if not _project_root:
        raise RuntimeError("Code modifier not initialized")

    backup_file = Path(backup_path)
    if _backup_dir:
        backup_dir = _backup_dir.resolve()
        backup_file = backup_file.resolve()
        if backup_file != backup_dir and backup_dir not in backup_file.parents:
            return {"success": False, "message": "Backup path escapes backup directory"}
    if not backup_file.exists():
        return {"success": False, "message": "Backup file not found"}

    meta_path = backup_file.with_suffix(backup_file.suffix + ".meta.json")
    if not meta_path.exists():
        return {"success": False, "message": "Backup metadata missing"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    original_path = Path(meta["original"])
    if not original_path.is_absolute():
        original_path = _project_root / original_path
    original_path = original_path.resolve()
    if original_path != _project_root and _project_root not in original_path.parents:
        return {"success": False, "message": "Original path escapes project root"}

    original_path.write_text(backup_file.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "success": True,
        "rolled_back_file": str(original_path),
        "current_backup": str(backup_file),
    }


# attach rollback to function for existing callers
modify_code_safely.rollback_modification = rollback_modification  # type: ignore[attr-defined]
