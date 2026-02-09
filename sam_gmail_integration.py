from __future__ import annotations

import base64
import datetime as dt
import json
import os
import threading
import time
import mimetypes
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    SAM_GMAIL_AVAILABLE = True
except Exception:
    Credentials = None
    InstalledAppFlow = None
    Request = None
    build = None
    SAM_GMAIL_AVAILABLE = False

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


class SAMGmail:
    def __init__(self, credentials_path: Path, token_path: Path):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = self._load_credentials()
        self.service = build("gmail", "v1", credentials=self.creds)
        self._scheduled: List[Dict[str, Any]] = []
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def _load_credentials(self):
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def send_email(self, to_email: str, subject: str, body: str, attachments: Optional[List[str]] = None):
        message = EmailMessage()
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        if attachments:
            for attachment in attachments:
                path = Path(attachment)
                if not path.exists():
                    raise RuntimeError(f"Attachment not found: {path}")
                mime_type, _ = mimetypes.guess_type(str(path))
                if mime_type:
                    maintype, subtype = mime_type.split("/", 1)
                else:
                    maintype, subtype = "application", "octet-stream"
                message.add_attachment(
                    path.read_bytes(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=path.name,
                )

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = self.service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"success": True, "message_id": sent.get("id")}

    def schedule_email(self, to_email: str, subject: str, body: str, send_time: str):
        send_at = dt.datetime.fromisoformat(send_time)
        entry = {
            "to": to_email,
            "subject": subject,
            "body": body,
            "send_time": send_at,
            "status": "scheduled",
        }
        self._scheduled.append(entry)
        return {"success": True, "scheduled": send_time}

    def get_scheduled_emails(self):
        return [
            {
                "to": e["to"],
                "subject": e["subject"],
                "send_time": e["send_time"].isoformat(),
                "status": e["status"],
            }
            for e in self._scheduled
        ]

    def send_system_report(self, recipient: str, report_type: str):
        body = f"SAM System Report\nType: {report_type}\nTime: {dt.datetime.utcnow().isoformat()}Z\n"
        return self.send_email(recipient, f"SAM System Report ({report_type})", body)

    def _scheduler_loop(self):
        while True:
            now = dt.datetime.now()
            for entry in list(self._scheduled):
                if entry["status"] != "scheduled":
                    continue
                if entry["send_time"] <= now:
                    try:
                        self.send_email(entry["to"], entry["subject"], entry["body"])
                        entry["status"] = "sent"
                    except Exception:
                        entry["status"] = "failed"
            time.sleep(30)


sam_gmail_instance: Optional[SAMGmail] = None


def initialize_sam_gmail(google_drive: Optional[Any] = None) -> SAMGmail:
    if not SAM_GMAIL_AVAILABLE:
        raise RuntimeError("Gmail dependencies not installed")

    credentials_path = Path(os.getenv("SAM_GMAIL_CREDENTIALS", "secrets/gmail_credentials.json"))
    token_path = Path(os.getenv("SAM_GMAIL_TOKEN", "secrets/gmail_token.json"))
    if not credentials_path.exists():
        raise RuntimeError(f"Gmail credentials file not found: {credentials_path}")

    global sam_gmail_instance
    sam_gmail_instance = SAMGmail(credentials_path, token_path)
    return sam_gmail_instance


def send_sam_email(to_email: str, subject: str, body: str, attachments: Optional[List[str]] = None):
    if not sam_gmail_instance:
        raise RuntimeError("Gmail not initialized")
    return sam_gmail_instance.send_email(to_email, subject, body, attachments)


def schedule_sam_email(to_email: str, subject: str, body: str, send_time: str):
    if not sam_gmail_instance:
        raise RuntimeError("Gmail not initialized")
    return sam_gmail_instance.schedule_email(to_email, subject, body, send_time)
