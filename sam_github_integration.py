from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

SAM_GITHUB_AVAILABLE = True

_repo_root: Optional[Path] = None
sam_github_instance: Optional["SAMGitHub"] = None


def _run_git(*args: str) -> subprocess.CompletedProcess:
    if _repo_root is None:
        raise RuntimeError("Git repository not initialized")
    timeout_s = int(os.getenv("SAM_GIT_TIMEOUT_S", "60"))
    try:
        return subprocess.run(
            ["git", "-C", str(_repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            exc.cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr="git command timed out",
        )


def initialize_sam_github(repo_path: Optional[str] = None) -> "SAMGitHub":
    global _repo_root
    _repo_root = Path(repo_path or Path(__file__).parent).resolve()
    proc = _run_git("rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise RuntimeError(f"Not a git repository: {_repo_root}")
    global sam_github_instance
    sam_github_instance = SAMGitHub(_repo_root)
    return sam_github_instance


def test_github_connection(remote: str = "origin") -> Dict[str, object]:
    try:
        proc = _run_git("remote", "get-url", remote)
        if proc.returncode != 0:
            return {"success": False, "error": proc.stderr.strip() or "remote not found"}
        remote_url = proc.stdout.strip()

        ls_proc = _run_git("ls-remote", remote, "HEAD")
        success = ls_proc.returncode == 0
        write_access = False
        if success:
            push_proc = _run_git("push", "--dry-run", remote, "HEAD")
            write_access = push_proc.returncode == 0
        return {
            "success": success,
            "remote": remote,
            "remote_url": remote_url,
            "write_access": write_access,
            "error": None if success else (ls_proc.stderr.strip() or "ls-remote failed"),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def save_to_github(commit_message: Optional[str] = None, remote: str = "origin") -> Dict[str, object]:
    if commit_message is None:
        commit_message = "SAM system save"

    add_proc = _run_git("add", "-A")
    if add_proc.returncode != 0:
        return {"success": False, "error": add_proc.stderr.strip() or "git add failed"}

    commit_proc = _run_git("commit", "-m", commit_message)
    if commit_proc.returncode != 0:
        output = (commit_proc.stdout + commit_proc.stderr).lower()
        if "nothing to commit" not in output:
            return {"success": False, "error": commit_proc.stderr.strip() or "git commit failed"}

    branch_proc = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch_proc.returncode != 0:
        return {"success": False, "error": branch_proc.stderr.strip() or "branch detection failed"}
    branch = branch_proc.stdout.strip()

    push_proc = _run_git("push", remote, branch)
    if push_proc.returncode != 0:
        return {"success": False, "error": push_proc.stderr.strip() or "git push failed"}

    return {"success": True, "remote": remote, "branch": branch}


def get_recent_commits(limit: int = 5) -> List[Dict[str, str]]:
    if _repo_root is None:
        raise RuntimeError("Git repository not initialized")
    proc = _run_git("log", f"-n{limit}", "--pretty=format:%H|%s|%an|%ad")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git log failed")
    commits: List[Dict[str, str]] = []
    for line in proc.stdout.strip().splitlines():
        sha, subject, author, date = (line.split("|", 3) + ["", "", "", ""])[:4]
        commits.append({
            "sha": sha,
            "message": subject,
            "subject": subject,
            "author": author,
            "date": date,
        })
    return commits


class SAMGitHub:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def get_recent_commits(self, limit: int = 5) -> List[Dict[str, str]]:
        return get_recent_commits(limit)
