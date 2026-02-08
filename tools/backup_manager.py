from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class BackupResult:
    success: bool
    message: str
    details: Dict[str, Any]


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_output(proc: subprocess.CompletedProcess) -> str:
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if out and err:
        return f"{out}\n{err}"
    return out or err


class BackupManager:
    def __init__(
        self,
        repo_path: str | Path,
        enabled: bool = False,
        interval_s: int = 3600,
        primary_remote: str = "origin",
        secondary_remote: Optional[str] = None,
        branch: Optional[str] = None,
        auto_commit: bool = True,
        commit_prefix: str = "auto-backup",
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
        require_success: bool = False,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.enabled = enabled
        self.interval_s = max(60, int(interval_s))
        self.primary_remote = primary_remote
        self.secondary_remote = secondary_remote
        self.branch = branch
        self.auto_commit = auto_commit
        self.commit_prefix = commit_prefix
        self.author_name = author_name
        self.author_email = author_email
        self.require_success = require_success
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            result = self.run_once()
            if not result.success and self.require_success:
                raise RuntimeError(result.message)
            self._stop_event.wait(self.interval_s)

    def _ensure_repo(self) -> None:
        proc = _run_git(self.repo_path, "rev-parse", "--is-inside-work-tree")
        if proc.returncode != 0 or proc.stdout.strip() != "true":
            raise RuntimeError(f"Not a git repository: {self.repo_path}")

    def _current_branch(self) -> str:
        if self.branch:
            return self.branch
        proc = _run_git(self.repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        if proc.returncode != 0:
            raise RuntimeError("Unable to resolve current git branch")
        return proc.stdout.strip()

    def _status_clean(self) -> bool:
        proc = _run_git(self.repo_path, "status", "--porcelain")
        if proc.returncode != 0:
            raise RuntimeError("Unable to read git status")
        return proc.stdout.strip() == ""

    def _commit_changes(self, message: str) -> None:
        proc = _run_git(self.repo_path, "add", "-A")
        if proc.returncode != 0:
            raise RuntimeError(f"git add failed: {_git_output(proc)}")
        env = os.environ.copy()
        if self.author_name:
            env["GIT_AUTHOR_NAME"] = self.author_name
            env["GIT_COMMITTER_NAME"] = self.author_name
        if self.author_email:
            env["GIT_AUTHOR_EMAIL"] = self.author_email
            env["GIT_COMMITTER_EMAIL"] = self.author_email
        proc = subprocess.run(
            ["git", "-C", str(self.repo_path), "commit", "-m", message],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            output = _git_output(proc)
            if "nothing to commit" in output.lower():
                return
            raise RuntimeError(f"git commit failed: {output}")

    def _push(self, remote: str, branch: str) -> None:
        proc = _run_git(self.repo_path, "push", remote, branch)
        if proc.returncode != 0:
            raise RuntimeError(f"git push {remote} failed: {_git_output(proc)}")

    def run_once(self) -> BackupResult:
        details: Dict[str, Any] = {
            "repo": str(self.repo_path),
            "primary_remote": self.primary_remote,
            "secondary_remote": self.secondary_remote,
        }
        try:
            self._ensure_repo()
            branch = self._current_branch()
            details["branch"] = branch

            changed = not self._status_clean()
            details["changed"] = changed
            if changed and self.auto_commit:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                message = f"{self.commit_prefix}: {timestamp}"
                self._commit_changes(message)

            # Always attempt push to keep remotes in sync
            self._push(self.primary_remote, branch)
            if self.secondary_remote:
                self._push(self.secondary_remote, branch)

            return BackupResult(True, "Backup completed", details)
        except Exception as exc:
            details["error"] = str(exc)
            return BackupResult(False, f"Backup failed: {exc}", details)
