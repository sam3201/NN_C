from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        corrupt_path = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
        corrupt_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        raise ValueError(f"Corrupt JSON in {path}") from exc


def _safe_json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class SpendRequest:
    request_id: str
    account_id: str
    amount: float
    currency: str
    memo: str
    status: str
    created_at: str
    requested_by: str
    requires_approval: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    executed_at: Optional[str] = None
    failure_reason: Optional[str] = None


class BankingAuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: dict, actor: str) -> None:
        entry = {
            "timestamp": _now_iso(),
            "event": event,
            "actor": actor,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class BankingLedger:
    """Sandbox banking ledger with explicit approval gating.

    This does NOT connect to real financial systems. It only records
    simulated balances and requires approvals for any spend.
    """

    def __init__(
        self,
        data_dir: Path,
        ledger_path: Path,
        requests_path: Path,
        audit_log: Path,
    ):
        self.data_dir = data_dir
        self.ledger_path = ledger_path
        self.requests_path = requests_path
        self.audit = BankingAuditLog(audit_log)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        self.ledger = _safe_json_load(self.ledger_path, {"accounts": {}, "transactions": []})
        self.requests = _safe_json_load(self.requests_path, {"requests": []})

    def _save(self) -> None:
        _safe_json_write(self.ledger_path, self.ledger)
        _safe_json_write(self.requests_path, self.requests)

    def create_account(self, name: str, initial_balance: float = 0.0, currency: str = "USD") -> dict:
        if not name:
            raise ValueError("Account name required")
        if initial_balance < 0:
            raise ValueError("Initial balance must be >= 0")
        account_id = f"acct_{uuid.uuid4().hex[:10]}"
        account = {
            "account_id": account_id,
            "name": name,
            "balance": float(initial_balance),
            "currency": currency,
            "created_at": _now_iso(),
        }
        self.ledger["accounts"][account_id] = account
        self._save()
        self.audit.append("account_created", account, actor="system")
        return account

    def list_accounts(self) -> List[dict]:
        return list(self.ledger.get("accounts", {}).values())

    def request_spend(
        self,
        account_id: str,
        amount: float,
        memo: str,
        requested_by: str,
        requires_approval: bool = True,
    ) -> SpendRequest:
        if account_id not in self.ledger.get("accounts", {}):
            raise ValueError("Unknown account_id")
        if amount <= 0:
            raise ValueError("Amount must be > 0")
        req = SpendRequest(
            request_id=f"req_{uuid.uuid4().hex[:10]}",
            account_id=account_id,
            amount=float(amount),
            currency=self.ledger["accounts"][account_id]["currency"],
            memo=memo or "",
            status="PENDING",
            created_at=_now_iso(),
            requested_by=requested_by,
            requires_approval=requires_approval,
        )
        self.requests["requests"].append(asdict(req))
        self._save()
        self.audit.append("spend_requested", asdict(req), actor=requested_by)
        return req

    def list_requests(self, status: Optional[str] = None) -> List[dict]:
        requests = self.requests.get("requests", [])
        if status:
            return [r for r in requests if r.get("status") == status]
        return requests

    def approve_request(self, request_id: str, approver: str, auto_execute: bool = True) -> dict:
        req = self._get_request(request_id)
        if req["status"] != "PENDING":
            raise ValueError("Request not pending")
        req["status"] = "APPROVED"
        req["approved_by"] = approver
        req["approved_at"] = _now_iso()
        self._save()
        self.audit.append("spend_approved", req, actor=approver)
        if auto_execute:
            return self.execute_request(request_id, actor=approver)
        return req

    def reject_request(self, request_id: str, approver: str, reason: Optional[str] = None) -> dict:
        req = self._get_request(request_id)
        if req["status"] != "PENDING":
            raise ValueError("Request not pending")
        req["status"] = "REJECTED"
        req["approved_by"] = approver
        req["approved_at"] = _now_iso()
        req["failure_reason"] = reason
        self._save()
        self.audit.append("spend_rejected", req, actor=approver)
        return req

    def execute_request(self, request_id: str, actor: str) -> dict:
        req = self._get_request(request_id)
        if req["requires_approval"] and req["status"] != "APPROVED":
            raise ValueError("Request requires approval")
        if req["status"] not in {"APPROVED", "PENDING"}:
            raise ValueError("Request not executable")
        account = self.ledger["accounts"].get(req["account_id"])
        if not account:
            raise ValueError("Account not found")
        if account["balance"] < req["amount"]:
            req["status"] = "FAILED"
            req["failure_reason"] = "insufficient_funds"
            self._save()
            self.audit.append("spend_failed", req, actor=actor)
            return req
        account["balance"] = float(account["balance"]) - float(req["amount"])
        transaction = {
            "transaction_id": f"txn_{uuid.uuid4().hex[:10]}",
            "account_id": req["account_id"],
            "amount": -float(req["amount"]),
            "currency": req["currency"],
            "memo": req.get("memo", ""),
            "created_at": _now_iso(),
            "request_id": req["request_id"],
        }
        self.ledger["transactions"].append(transaction)
        req["status"] = "EXECUTED"
        req["executed_at"] = _now_iso()
        self._save()
        self.audit.append("spend_executed", {**req, "transaction": transaction}, actor=actor)
        return req

    def get_snapshot(self) -> dict:
        accounts = self.list_accounts()
        requests = self.list_requests()
        pending = [r for r in requests if r.get("status") == "PENDING"]
        executed = [r for r in requests if r.get("status") == "EXECUTED"]
        return {
            "accounts": accounts,
            "requests": requests,
            "pending_requests": len(pending),
            "executed_requests": len(executed),
            "total_balance": sum(a.get("balance", 0.0) for a in accounts),
        }

    def get_metrics(self) -> dict:
        accounts = self.list_accounts()
        transactions = self.ledger.get("transactions", [])
        balance_by_currency: Dict[str, float] = {}
        for acct in accounts:
            currency = acct.get("currency", "USD")
            balance_by_currency[currency] = balance_by_currency.get(currency, 0.0) + float(acct.get("balance", 0.0))

        total_spent = sum(
            abs(float(txn.get("amount", 0.0)))
            for txn in transactions
            if float(txn.get("amount", 0.0)) < 0
        )
        total_incoming = sum(
            float(txn.get("amount", 0.0))
            for txn in transactions
            if float(txn.get("amount", 0.0)) > 0
        )

        return {
            "total_balance": sum(balance_by_currency.values()),
            "balances_by_currency": balance_by_currency,
            "total_spent": round(total_spent, 2),
            "total_incoming": round(total_incoming, 2),
            "transaction_count": len(transactions),
            "account_count": len(accounts),
        }

    def _get_request(self, request_id: str) -> dict:
        for req in self.requests.get("requests", []):
            if req.get("request_id") == request_id:
                return req
        raise ValueError("Request not found")
