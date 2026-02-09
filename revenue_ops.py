from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

try:
    from fpdf import FPDF  # type: ignore
    FPDF_AVAILABLE = True
except Exception:
    FPDF = None
    FPDF_AVAILABLE = False


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class RevenueAction:
    action_id: str
    action_type: str
    payload: Dict[str, Any]
    requested_by: str
    created_at: str
    status: str = "PENDING"
    requires_approval: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    executed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class RevenueAuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event: str, action: RevenueAction, actor: Optional[str] = None) -> None:
        entry = {
            "event": event,
            "timestamp": _utc_now(),
            "actor": actor,
            "action": asdict(action),
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")


class RevenueDataStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.leads_path = self.base_dir / "crm_leads.json"
        self.sequences_path = self.base_dir / "email_sequences.json"
        self.sequence_runs_path = self.base_dir / "email_sequence_runs.json"
        self.invoices_path = self.base_dir / "invoices.json"
        self._lock = Lock()

    def _load(self, path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            # Preserve the corrupt payload and fail fast to avoid silent data loss
            try:
                raw = path.read_text()
                corrupt_path = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
                corrupt_path.write_text(raw, encoding="utf-8")
                print(
                    f"[revenue_ops] Corrupt JSON detected in {path}. "
                    f"Backup written to {corrupt_path}.",
                    file=sys.stderr,
                )
            except Exception as backup_exc:
                print(
                    f"[revenue_ops] Corrupt JSON detected in {path}, "
                    f"but failed to write backup: {backup_exc}",
                    file=sys.stderr,
                )
            raise RuntimeError(f"Corrupt JSON in {path}") from exc

    def _save(self, path: Path, data) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def create_lead(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            leads = self._load(self.leads_path, [])
            lead_id = payload.get("lead_id") or f"lead_{uuid.uuid4().hex[:10]}"
            record = {**payload, "lead_id": lead_id, "created_at": _utc_now()}
            leads.append(record)
            self._save(self.leads_path, leads)
        return {"lead_id": lead_id, "record": record}

    def update_lead(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        lead_id = payload.get("lead_id")
        if not lead_id:
            raise ValueError("lead_id required")
        with self._lock:
            leads = self._load(self.leads_path, [])
            updated = None
            for lead in leads:
                if lead.get("lead_id") == lead_id:
                    lead.update(payload)
                    lead["updated_at"] = _utc_now()
                    updated = lead
                    break
            if updated is None:
                raise ValueError(f"lead_id not found: {lead_id}")
            self._save(self.leads_path, leads)
        return {"lead_id": lead_id, "record": updated}

    def list_leads(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(self.leads_path, [])

    def create_email_sequence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            sequences = self._load(self.sequences_path, [])
            sequence_id = payload.get("sequence_id") or f"seq_{uuid.uuid4().hex[:10]}"
            record = {**payload, "sequence_id": sequence_id, "created_at": _utc_now()}
            sequences.append(record)
            self._save(self.sequences_path, sequences)
        return {"sequence_id": sequence_id, "record": record}

    def list_sequences(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(self.sequences_path, [])

    def schedule_sequence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            runs = self._load(self.sequence_runs_path, [])
            run_id = payload.get("run_id") or f"run_{uuid.uuid4().hex[:10]}"
            record = {
                **payload,
                "run_id": run_id,
                "created_at": _utc_now(),
                "status": "scheduled",
                "current_step": payload.get("current_step", 0),
                "sent_steps": payload.get("sent_steps", []),
                "next_send_at": payload.get("next_send_at") or payload.get("schedule_at"),
            }
            runs.append(record)
            self._save(self.sequence_runs_path, runs)
        return {"run_id": run_id, "record": record}

    def list_sequence_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(self.sequence_runs_path, [])

    def save_sequence_runs(self, runs: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._save(self.sequence_runs_path, runs)

    def list_sequences(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(self.sequences_path, [])

    def create_invoice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            invoices = self._load(self.invoices_path, [])
            invoice_id = payload.get("invoice_id") or f"inv_{uuid.uuid4().hex[:10]}"
            record = {**payload, "invoice_id": invoice_id, "created_at": _utc_now(), "status": "open"}
            invoices.append(record)
            self._save(self.invoices_path, invoices)
        return {"invoice_id": invoice_id, "record": record}

    def record_payment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        invoice_id = payload.get("invoice_id")
        if not invoice_id:
            raise ValueError("invoice_id required")
        with self._lock:
            invoices = self._load(self.invoices_path, [])
            updated = None
            for inv in invoices:
                if inv.get("invoice_id") == invoice_id:
                    inv["status"] = "paid"
                    inv["paid_at"] = _utc_now()
                    inv["payment_details"] = payload.get("payment_details", {})
                    updated = inv
                    break
            if updated is None:
                raise ValueError(f"invoice_id not found: {invoice_id}")
            self._save(self.invoices_path, invoices)
        return {"invoice_id": invoice_id, "record": updated}

    def list_invoices(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(self.invoices_path, [])

    def export_leads_csv(self) -> str:
        leads = self.list_leads()
        if not leads:
            return "lead_id,name,email,company,stage,source,created_at\n"
        fieldnames = []
        for lead in leads:
            for key in lead.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for lead in leads:
            writer.writerow(lead)
        return output.getvalue()

    def import_leads_csv(self, csv_text: str) -> Dict[str, Any]:
        reader = csv.DictReader(io.StringIO(csv_text))
        imported = 0
        failures: List[Dict[str, Any]] = []
        for row in reader:
            try:
                payload = {k: v for k, v in row.items() if k}
                if not payload.get("email") and not payload.get("name"):
                    raise ValueError("lead must include name or email")
                self.create_lead(payload)
                imported += 1
            except Exception as exc:
                failures.append({"row": row, "error": str(exc)})
        return {"imported": imported, "failed": len(failures), "failures": failures}


class RevenueOpsEngine:
    def __init__(
        self,
        data_dir: Path,
        queue_path: Path,
        audit_log: Path,
        send_email: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
        schedule_email: Optional[Callable[[str, str, str, str], Dict[str, Any]]] = None,
    ):
        self.data_store = RevenueDataStore(data_dir)
        self.queue_path = queue_path
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = RevenueAuditLog(audit_log)
        self.send_email = send_email
        self.schedule_email = schedule_email
        self.render_dir = self.data_store.base_dir / "rendered_invoices"
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._handlers = {
            "create_lead": self._handle_create_lead,
            "update_lead": self._handle_update_lead,
            "create_email_sequence": self._handle_create_sequence,
            "schedule_sequence": self._handle_schedule_sequence,
            "send_email": self._handle_send_email,
            "schedule_email": self._handle_schedule_email,
            "create_invoice": self._handle_create_invoice,
            "record_payment": self._handle_record_payment,
        }

    def submit_action(
        self,
        action_type: str,
        payload: Dict[str, Any],
        requested_by: str,
        requires_approval: bool = True,
    ) -> RevenueAction:
        if action_type not in self._handlers:
            raise ValueError(f"Unknown action type: {action_type}")
        action = RevenueAction(
            action_id=f"rev_{uuid.uuid4().hex[:12]}",
            action_type=action_type,
            payload=payload,
            requested_by=requested_by,
            created_at=_utc_now(),
            requires_approval=requires_approval,
        )
        with self._lock:
            queue = self._load_queue()
            queue[action.action_id] = asdict(action)
            self._save_queue(queue)
        self.audit.append("submitted", action, actor=requested_by)
        return action

    def list_actions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        queue = self._load_queue()
        actions = list(queue.values())
        if status:
            actions = [a for a in actions if a.get("status") == status]
        return actions

    def export_leads_csv(self) -> str:
        return self.data_store.export_leads_csv()

    def import_leads_csv(self, csv_text: str) -> Dict[str, Any]:
        return self.data_store.import_leads_csv(csv_text)

    def generate_invoice_html(self, invoice_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        invoice = payload or self._find_invoice(invoice_id)
        if not invoice:
            raise ValueError("invoice not found")
        invoice_id = self._safe_invoice_id(invoice.get("invoice_id"))
        items = invoice.get("items", [])
        total = invoice.get("amount") or sum((item.get("qty", 1) * item.get("unit_price", 0)) for item in items)
        currency = self._escape(invoice.get("currency", "USD"))
        client_name = self._escape(invoice.get("client", "Client"))
        client_email = self._escape(invoice.get("client_email", ""))
        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
body {{ font-family: Arial, sans-serif; background:#f7f7f7; padding:40px; }}
.invoice {{ background:#fff; padding:32px; border-radius:12px; max-width:720px; margin:auto; }}
.header {{ display:flex; justify-content:space-between; align-items:flex-start; }}
.title {{ font-size:28px; font-weight:bold; }}
.meta {{ color:#666; font-size:12px; }}
table {{ width:100%; border-collapse:collapse; margin-top:24px; }}
th, td {{ border-bottom:1px solid #eee; padding:10px 6px; text-align:left; }}
.total {{ text-align:right; font-size:18px; font-weight:bold; margin-top:16px; }}
</style>
</head>
<body>
  <div class="invoice">
    <div class="header">
      <div>
        <div class="title">Invoice</div>
        <div class="meta">Invoice ID: {self._escape(invoice_id)}</div>
        <div class="meta">Date: {_utc_now()}</div>
      </div>
      <div>
        <div class="meta">Bill To</div>
        <div>{client_name}</div>
        <div class="meta">{client_email}</div>
      </div>
    </div>
    <table>
      <thead>
        <tr><th>Item</th><th>Qty</th><th>Unit</th><th>Total</th></tr>
      </thead>
      <tbody>
        {''.join([
            f"<tr><td>{self._escape(item.get('name','Service'))}</td>"
            f"<td>{item.get('qty',1)}</td>"
            f"<td>{item.get('unit_price',0):.2f}</td>"
            f"<td>{(item.get('qty',1)*item.get('unit_price',0)):.2f}</td></tr>"
            for item in items
        ])}
      </tbody>
    </table>
    <div class="total">Total: {currency} {total:.2f}</div>
  </div>
</body>
</html>
"""
        html_path = self.render_dir / f"{invoice_id}.html"
        html_path.write_text(html, encoding="utf-8")
        return {"invoice_id": invoice_id, "html": html, "html_path": str(html_path)}

    def generate_invoice_pdf(self, invoice_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not FPDF_AVAILABLE:
            raise RuntimeError("fpdf2 is required for PDF generation")
        invoice = payload or self._find_invoice(invoice_id)
        if not invoice:
            raise ValueError("invoice not found")
        invoice_id = self._safe_invoice_id(invoice.get("invoice_id"))
        items = invoice.get("items", [])
        total = invoice.get("amount") or sum((item.get("qty", 1) * item.get("unit_price", 0)) for item in items)
        currency = invoice.get("currency", "USD")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=16)
        pdf.cell(0, 10, "Invoice", ln=1)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, f"Invoice ID: {invoice_id}", ln=1)
        pdf.cell(0, 6, f"Date: {_utc_now()}", ln=1)
        pdf.cell(0, 6, f"Client: {invoice.get('client', 'Client')}", ln=1)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(80, 6, "Item", border=1)
        pdf.cell(20, 6, "Qty", border=1)
        pdf.cell(30, 6, "Unit", border=1)
        pdf.cell(30, 6, "Total", border=1, ln=1)
        for item in items:
            qty = item.get("qty", 1)
            unit = item.get("unit_price", 0)
            pdf.cell(80, 6, str(item.get("name", "Service")), border=1)
            pdf.cell(20, 6, str(qty), border=1)
            pdf.cell(30, 6, f"{unit:.2f}", border=1)
            pdf.cell(30, 6, f"{qty * unit:.2f}", border=1, ln=1)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 8, f"Total: {currency} {total:.2f}", ln=1)

        pdf_path = self.render_dir / f"{invoice_id}.pdf"
        pdf.output(str(pdf_path))
        return {"invoice_id": invoice_id, "pdf_path": str(pdf_path)}

    @staticmethod
    def _escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def _safe_invoice_id(value: Optional[str]) -> str:
        if not value:
            return f"inv_{uuid.uuid4().hex[:10]}"
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(value))
        safe = safe.strip("._-")
        return safe or f"inv_{uuid.uuid4().hex[:10]}"

    def autoplan(
        self,
        max_actions: int = 3,
        default_sequence_id: Optional[str] = None,
        requested_by: str = "revenue_autoplanner",
    ) -> List[RevenueAction]:
        pending = self.list_actions(status="PENDING")
        if len(pending) >= max_actions:
            return []
        pending_keys = {(a.get("action_type"), a.get("payload", {}).get("lead_id")) for a in pending}
        actions: List[RevenueAction] = []

        sequences = self.data_store.list_sequences()
        if not default_sequence_id and sequences:
            default_sequence_id = sequences[0].get("sequence_id")

        leads = self.data_store.list_leads()
        for lead in leads:
            if len(actions) + len(pending) >= max_actions:
                break
            lead_id = lead.get("lead_id")
            stage = str(lead.get("stage", "new")).lower()

            if not sequences and lead.get("email") and ("create_email_sequence", None) not in pending_keys:
                payload = {
                    "name": "Outbound Revenue Sequence",
                    "steps": [
                        {"subject": "Intro + value proposition", "body": "Quick intro and value proposition.", "delay_days": 0},
                        {"subject": "Case study + proof", "body": "Share proof and results.", "delay_days": 2},
                        {"subject": "Last touch + CTA", "body": "Invite next step.", "delay_days": 4},
                    ],
                }
                action = self.submit_action("create_email_sequence", payload, requested_by=requested_by, requires_approval=True)
                actions.append(action)
                pending_keys.add(("create_email_sequence", None))
                continue

            if stage in ("new", "prospect", "outreach") and lead.get("email") and default_sequence_id:
                if ("schedule_sequence", lead_id) not in pending_keys:
                    payload = {
                        "lead_id": lead_id,
                        "sequence_id": default_sequence_id,
                        "to_email": lead.get("email"),
                        "schedule_at": lead.get("next_contact_at") or _utc_now(),
                    }
                    action = self.submit_action("schedule_sequence", payload, requested_by=requested_by, requires_approval=True)
                    actions.append(action)
                    pending_keys.add(("schedule_sequence", lead_id))

            if stage in ("proposal_sent", "negotiation", "contract"):
                invoice_amount = lead.get("invoice_amount")
                if stage == "contract" and not invoice_amount:
                    try:
                        invoice_amount = float(os.getenv("SAM_REVENUE_DEFAULT_INVOICE_AMOUNT", "0")) or None
                    except Exception:
                        invoice_amount = None
                if invoice_amount and ("create_invoice", lead_id) not in pending_keys:
                    payload = {
                        "client": lead.get("company") or lead.get("name") or "Client",
                        "client_email": lead.get("email"),
                        "amount": invoice_amount,
                        "currency": lead.get("currency", "USD"),
                        "items": lead.get("items") or [{"name": "Services", "qty": 1, "unit_price": invoice_amount}],
                        "lead_id": lead_id,
                    }
                    action = self.submit_action("create_invoice", payload, requested_by=requested_by, requires_approval=True)
                    actions.append(action)
                    pending_keys.add(("create_invoice", lead_id))
        return actions

    def approve_action(self, action_id: str, approver: str, auto_execute: bool = True) -> Dict[str, Any]:
        with self._lock:
            queue = self._load_queue()
            action_data = queue.get(action_id)
            if not action_data:
                raise ValueError("Action not found")
            if action_data["status"] != "PENDING":
                raise ValueError(f"Action already {action_data['status']}")
            action_data["status"] = "APPROVED"
            action_data["approved_by"] = approver
            action_data["approved_at"] = _utc_now()
            queue[action_id] = action_data
            self._save_queue(queue)
        action = RevenueAction(**action_data)
        self.audit.append("approved", action, actor=approver)
        if auto_execute:
            return self.execute_action(action_id, approver)
        return action_data

    def reject_action(self, action_id: str, approver: str, reason: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            queue = self._load_queue()
            action_data = queue.get(action_id)
            if not action_data:
                raise ValueError("Action not found")
            if action_data["status"] != "PENDING":
                raise ValueError(f"Action already {action_data['status']}")
            action_data["status"] = "REJECTED"
            action_data["approved_by"] = approver
            action_data["approved_at"] = _utc_now()
            if reason:
                action_data["error"] = reason
            queue[action_id] = action_data
            self._save_queue(queue)
        action = RevenueAction(**action_data)
        self.audit.append("rejected", action, actor=approver)
        return action_data

    def execute_action(self, action_id: str, actor: str) -> Dict[str, Any]:
        with self._lock:
            queue = self._load_queue()
            action_data = queue.get(action_id)
            if not action_data:
                raise ValueError("Action not found")
            if action_data["status"] not in ("APPROVED", "PENDING"):
                raise ValueError(f"Action not executable (status={action_data['status']})")
            if action_data["requires_approval"] and action_data["status"] != "APPROVED":
                raise ValueError("Action requires approval")
            action_data["status"] = "EXECUTING"
            queue[action_id] = action_data
            self._save_queue(queue)

        action = RevenueAction(**action_data)
        handler = self._handlers.get(action.action_type)
        if not handler:
            raise ValueError(f"No handler for {action.action_type}")

        try:
            result = handler(action.payload)
            action.status = "EXECUTED"
            action.executed_at = _utc_now()
            action.result = result
            action.error = None
            self.audit.append("executed", action, actor=actor)
        except Exception as exc:
            action.status = "FAILED"
            action.executed_at = _utc_now()
            action.error = str(exc)
            self.audit.append("failed", action, actor=actor)

        with self._lock:
            queue = self._load_queue()
            queue[action.action_id] = asdict(action)
            self._save_queue(queue)

        return asdict(action)

    def _load_queue(self) -> Dict[str, Dict[str, Any]]:
        if not self.queue_path.exists():
            return {}
        try:
            return json.loads(self.queue_path.read_text())
        except Exception:
            return {}

    def _save_queue(self, queue: Dict[str, Dict[str, Any]]) -> None:
        self.queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")

    def _find_invoice(self, invoice_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not invoice_id:
            return None
        invoices = self.data_store.list_invoices()
        for inv in invoices:
            if inv.get("invoice_id") == invoice_id:
                return inv
        return None

    def _handle_create_lead(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.data_store.create_lead(payload)

    def _handle_update_lead(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.data_store.update_lead(payload)

    def _handle_create_sequence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.data_store.create_email_sequence(payload)

    def _handle_schedule_sequence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.data_store.schedule_sequence(payload)

    def _handle_send_email(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.send_email:
            raise RuntimeError("Email sender not configured")
        to_email = payload.get("to_email")
        subject = payload.get("subject")
        body = payload.get("body")
        if not to_email or not subject or not body:
            raise ValueError("to_email, subject, and body required")
        return self.send_email(to_email, subject, body)

    def _handle_schedule_email(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.schedule_email:
            raise RuntimeError("Email scheduler not configured")
        to_email = payload.get("to_email")
        subject = payload.get("subject")
        body = payload.get("body")
        send_time = payload.get("send_time")
        if not to_email or not subject or not body or not send_time:
            raise ValueError("to_email, subject, body, send_time required")
        return self.schedule_email(to_email, subject, body, send_time)

    def _handle_create_invoice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.data_store.create_invoice(payload)

    def _handle_record_payment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.data_store.record_payment(payload)

    def get_crm_snapshot(self) -> Dict[str, Any]:
        return {"leads": self.data_store.list_leads()}

    def get_sequence_snapshot(self) -> Dict[str, Any]:
        return {
            "sequences": self.data_store.list_sequences(),
            "runs": self.data_store.list_sequence_runs(),
        }

    def get_invoice_snapshot(self) -> Dict[str, Any]:
        return {"invoices": self.data_store.list_invoices()}

    def process_sequence_runs(self, now_ts: Optional[float] = None) -> Dict[str, int]:
        """Send scheduled sequence emails that are due."""
        if not self.send_email:
            return {"sent": 0, "skipped": 0, "failed": 0, "blocked": 1}

        now_ts = now_ts or time.time()
        sequences = {s.get("sequence_id"): s for s in self.data_store.list_sequences()}
        leads = {l.get("lead_id"): l for l in self.data_store.list_leads()}
        runs = self.data_store.list_sequence_runs()
        sent = skipped = failed = 0

        for run in runs:
            status = run.get("status", "scheduled")
            if status not in ("scheduled", "active"):
                continue

            sequence_id = run.get("sequence_id")
            sequence = sequences.get(sequence_id)
            if not sequence:
                run["status"] = "failed"
                run["error"] = "sequence_not_found"
                failed += 1
                continue

            steps = sequence.get("steps", [])
            if not steps:
                run["status"] = "failed"
                run["error"] = "sequence_has_no_steps"
                failed += 1
                continue

            current_step = int(run.get("current_step", 0))
            if current_step >= len(steps):
                run["status"] = "completed"
                continue

            next_send_at = run.get("next_send_at") or run.get("schedule_at")
            if not next_send_at:
                run["next_send_at"] = _utc_now()
                next_send_at = run["next_send_at"]

            try:
                next_send_ts = time.mktime(time.strptime(next_send_at, "%Y-%m-%dT%H:%M:%SZ"))
            except Exception:
                next_send_ts = now_ts

            if now_ts < next_send_ts:
                skipped += 1
                continue

            step = steps[current_step]
            lead = leads.get(run.get("lead_id"), {})
            subject = self._render_template(step.get("subject", "Follow up"), lead)
            body = self._render_template(step.get("body", ""), lead)

            try:
                self.send_email(run.get("to_email"), subject, body)
                run.setdefault("sent_steps", []).append({
                    "step": current_step,
                    "sent_at": _utc_now(),
                    "subject": subject,
                })
                run["last_sent_at"] = _utc_now()
                current_step += 1
                run["current_step"] = current_step
                if current_step >= len(steps):
                    run["status"] = "completed"
                    run["next_send_at"] = None
                else:
                    delay_days = float(steps[current_step - 1].get("delay_days", 1))
                    run["next_send_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(now_ts + delay_days * 86400),
                    )
                    run["status"] = "active"
                sent += 1
            except Exception as exc:
                run["status"] = "failed"
                run["error"] = str(exc)
                failed += 1

        self.data_store.save_sequence_runs(runs)
        return {"sent": sent, "skipped": skipped, "failed": failed, "blocked": 0}

    def get_playbook_templates(self) -> Dict[str, Any]:
        return {
            "crm_fields": [
                "lead_id", "name", "email", "company", "stage", "source",
                "next_contact_at", "invoice_amount", "currency"
            ],
            "stages": ["new", "prospect", "outreach", "proposal_sent", "negotiation", "contract", "won", "lost"],
            "lead_templates": [
                {"name": "Sample Lead", "email": "lead@example.com", "company": "Example Co", "stage": "prospect", "source": "playbook"}
            ],
            "sequence_templates": [
                {
                    "name": "Outbound Intro Sequence",
                    "steps": [
                        {"subject": "Quick intro", "body": "Hi {{name}}, quick intro...", "delay_days": 0},
                        {"subject": "Case study", "body": "Sharing results that map to {{company}}...", "delay_days": 2},
                        {"subject": "Final nudge", "body": "Should we schedule a 15‑min call?", "delay_days": 4},
                    ],
                },
                {
                    "name": "Post‑demo Sequence",
                    "steps": [
                        {"subject": "Recap + next steps", "body": "Thanks for the time, {{name}}...", "delay_days": 0},
                        {"subject": "ROI summary", "body": "Estimated ROI for {{company}}...", "delay_days": 3},
                    ],
                },
            ],
            "invoice_template": {
                "client": "Client Name",
                "client_email": "client@example.com",
                "currency": "USD",
                "items": [
                    {"name": "Implementation", "qty": 1, "unit_price": 2500},
                    {"name": "Retainer", "qty": 1, "unit_price": 1500},
                ],
            },
            "action_templates": [
                {"action_type": "create_lead", "payload": {"name": "Lead", "email": "lead@company.com", "company": "Company"}},
                {"action_type": "create_email_sequence", "payload": {"name": "Outbound Sequence", "steps": []}},
                {"action_type": "schedule_sequence", "payload": {"lead_id": "lead_x", "sequence_id": "seq_x", "to_email": "lead@company.com"}},
                {"action_type": "create_invoice", "payload": {"client": "Client", "amount": 2500, "currency": "USD"}},
            ],
        }

    @staticmethod
    def _render_template(text: str, lead: Dict[str, Any]) -> str:
        rendered = text
        for key, value in lead.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    def import_playbooks(
        self,
        create_sequences: bool = True,
        create_leads: bool = True,
        limit_leads: int = 1,
    ) -> Dict[str, Any]:
        templates = self.get_playbook_templates()
        sequences_created = 0
        leads_created = 0

        existing_sequences = {s.get("name") for s in self.data_store.list_sequences()}
        existing_leads = {l.get("email") for l in self.data_store.list_leads()}

        if create_sequences:
            for seq in templates.get("sequence_templates", []):
                if seq.get("name") in existing_sequences:
                    continue
                self.data_store.create_email_sequence(seq)
                sequences_created += 1

        if create_leads:
            for lead in templates.get("lead_templates", [])[:limit_leads]:
                if lead.get("email") in existing_leads:
                    continue
                self.data_store.create_lead(lead)
                leads_created += 1

        return {
            "sequences_created": sequences_created,
            "leads_created": leads_created,
        }
