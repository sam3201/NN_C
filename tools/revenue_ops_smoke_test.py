import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

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


def main():
    env = os.environ.copy()
    env["SAM_AUTONOMOUS_ENABLED"] = "0"
    env["SAM_TEACHER_POOL_ENABLED"] = "0"

    server_proc = mp.Process(target=run_server, args=(env,))
    server_proc.start()

    try:
        if not wait_for_health("http://localhost:5004/api/health", timeout_s=90):
            raise RuntimeError("Server did not become healthy")

        # UI validation
        ui = requests.get("http://localhost:5004", timeout=10)
        assert "Revenue Ops Queue" in ui.text
        assert "Invoices" in ui.text
        assert "Playbook Templates" in ui.text

        # API smoke tests
        queue = requests.get("http://localhost:5004/api/revenue/queue?status=PENDING", timeout=10).json()
        assert "actions" in queue

        playbooks = requests.get("http://localhost:5004/api/revenue/playbooks", timeout=10).json()
        assert "sequence_templates" in playbooks

        # Submit a lead action
        lead_action = requests.post(
            "http://localhost:5004/api/revenue/action",
            json={
                "action_type": "create_lead",
                "payload": {"name": "Smoke Lead", "email": "smoke@example.com", "company": "SmokeCo"},
                "requested_by": "smoke_test",
            },
            timeout=10,
        ).json()
        action_id = lead_action["action"]["action_id"]

        # Approve and execute
        approved = requests.post(
            "http://localhost:5004/api/revenue/approve",
            json={"action_id": action_id, "approver": "smoke_test"},
            timeout=10,
        ).json()
        assert approved.get("result")

        # Export CRM CSV
        csv_resp = requests.get("http://localhost:5004/api/revenue/crm/export", timeout=10)
        assert csv_resp.status_code == 200

        # Invoices list
        invoices = requests.get("http://localhost:5004/api/revenue/invoices", timeout=10).json()
        assert "invoices" in invoices

        print(json.dumps({
            "ui": "ok",
            "api": "ok",
            "actions_in_queue": len(queue.get("actions", [])),
            "invoices": len(invoices.get("invoices", [])),
        }, indent=2))
    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=10)


if __name__ == "__main__":
    main()
