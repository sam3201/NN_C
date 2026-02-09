#!/usr/bin/env python3
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
import requests


def run_server(env: dict):
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    os.environ.update(env)
    from complete_sam_unified import UnifiedSAMSystem
    system = UnifiedSAMSystem()
    system.run()


def wait_for_health(url: str, timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


BASE_URL = os.getenv("SAM_BASE_URL", "http://localhost:5004")
TOKEN = os.getenv("SAM_ADMIN_TOKEN") or os.getenv("SAM_CODE_MODIFY_TOKEN")


def _headers():
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["X-SAM-ADMIN-TOKEN"] = TOKEN
    return headers


def main():
    env = os.environ.copy()
    env["SAM_AUTONOMOUS_ENABLED"] = "0"
    env["SAM_TEACHER_POOL_ENABLED"] = "0"
    env["SAM_ADMIN_TOKEN"] = env.get("SAM_ADMIN_TOKEN", "smoke_test_token")
    global TOKEN
    TOKEN = env["SAM_ADMIN_TOKEN"]

    server_proc = mp.Process(target=run_server, args=(env,))
    server_proc.start()

    try:
        if not wait_for_health(f"{BASE_URL}/api/health", timeout_s=90):
            raise RuntimeError("Server did not become healthy")

        account_payload = {"name": "Sandbox Ops", "initial_balance": 1000.0, "currency": "USD"}
        resp = requests.post(f"{BASE_URL}/api/banking/account", json=account_payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        account = resp.json()["account"]
        print("✅ Created account:", account["account_id"])

        spend_payload = {
            "account_id": account["account_id"],
            "amount": 125.0,
            "memo": "Sandbox purchase test",
            "requested_by": "smoke_test",
        }
        resp = requests.post(f"{BASE_URL}/api/banking/spend", json=spend_payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        req = resp.json()["request"]
        print("✅ Created spend request:", req["request_id"])

        approve_payload = {"request_id": req["request_id"], "approver": "smoke_test", "auto_execute": True}
        resp = requests.post(f"{BASE_URL}/api/banking/approve", json=approve_payload, headers=_headers(), timeout=10)
        resp.raise_for_status()
        result = resp.json()["result"]
        print("✅ Approved & executed:", result.get("status"))

        snapshot = requests.get(f"{BASE_URL}/api/banking/status", timeout=10).json()
        print("✅ Snapshot:", snapshot)
        return 0
    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
