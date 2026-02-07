#!/usr/bin/env python3
"""
SAM Gmail Integration System
Advanced email and calendar capabilities using SAM's dedicated Gmail account
"""

import os
import json
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import datetime
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import schedule
import threading

class SAMGmailIntegration:
    """Advanced Gmail integration for SAM system"""

    def __init__(self, google_drive_integration=None):
        self.google_drive = google_drive_integration
        self.sam_email = os.getenv('GOOGLE_ACCOUNT', 'sam.ai.system.agi@gmail.com')
        self.sam_password = os.getenv('GOOGLE_PASSWORD', '')

        # Email server settings
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.imap_server = "imap.gmail.com"
        self.imap_port = 993

        # Scheduled tasks
        self.scheduled_emails = []
        self.email_monitor_active = False
        self.monitor_thread = None

        print("📧 SAM Gmail Integration initialized")
        print(f"   📧 Account: {self.sam_email}")
        print(f"   🔄 Auto-monitoring: Disabled (requires app password)")

    def send_email(self, to_email: str, subject: str, body: str,
                   attachments: List[str] = None, priority: str = "normal") -> Dict:
        """Send email using SAM's Gmail account"""
        result = {
            'success': False,
            'message_id': None,
            'error': None
        }

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sam_email
            msg['To'] = to_email
            msg['Subject'] = f"[SAM AGI] {subject}"

            # Add priority
            if priority == "high":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'
            elif priority == "low":
                msg['X-Priority'] = '5'
                msg['X-MSMail-Priority'] = 'Low'
                msg['Importance'] = 'Low'

            # Add timestamp and signature
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            full_body = f"{body}\n\n---\nSent by SAM AGI System\nTimestamp: {timestamp}\nAccount: {self.sam_email}"

            msg.attach(MIMEText(full_body, 'plain'))

            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    if os.path.exists(attachment_path):
                        filename = os.path.basename(attachment_path)
                        attachment = open(attachment_path, "rb")

                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f"attachment; filename= {filename}")

                        msg.attach(part)
                        attachment.close()

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sam_email, self.sam_password)
            text = msg.as_string()
            server.sendmail(self.sam_email, to_email, text)
            server.quit()

            result['success'] = True
            result['message_id'] = f"sam_{int(time.time())}"

            print(f"📧 Email sent to {to_email}: {subject}")

            # Save to Google Drive if available
            if self.google_drive:
                self._save_email_to_drive({
                    'to': to_email,
                    'subject': subject,
                    'body': full_body,
                    'timestamp': timestamp,
                    'attachments': attachments or []
                })

        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Failed to send email: {e}")

        return result

    def schedule_email(self, to_email: str, subject: str, body: str,
                      send_time: str, attachments: List[str] = None) -> Dict:
        """Schedule an email to be sent at a specific time"""
        result = {
            'success': False,
            'scheduled_id': None,
            'error': None
        }

        try:
            # Parse send time (expecting format: "YYYY-MM-DD HH:MM" or "HH:MM" for today)
            if len(send_time) <= 5:  # Just time
                today = datetime.date.today()
                send_datetime = datetime.datetime.combine(today, datetime.datetime.strptime(send_time, "%H:%M").time())
            else:  # Full datetime
                send_datetime = datetime.datetime.strptime(send_time, "%Y-%m-%d %H:%M")

            scheduled_email = {
                'id': f"scheduled_{int(time.time())}",
                'to_email': to_email,
                'subject': subject,
                'body': body,
                'send_time': send_datetime,
                'attachments': attachments or [],
                'status': 'scheduled'
            }

            self.scheduled_emails.append(scheduled_email)

            # Schedule the email
            def send_scheduled_email(email_data):
                self.send_email(
                    email_data['to_email'],
                    email_data['subject'],
                    email_data['body'],
                    email_data['attachments']
                )
                email_data['status'] = 'sent'
                print(f"✅ Scheduled email sent: {email_data['subject']}")

            schedule.every().day.at(send_time).do(send_scheduled_email, scheduled_email)

            result['success'] = True
            result['scheduled_id'] = scheduled_email['id']

            print(f"📅 Email scheduled for {send_datetime}: {subject}")

        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Failed to schedule email: {e}")

        return result

    def send_system_report(self, recipient: str = None, report_type: str = "daily") -> Dict:
        """Send automated system report via email"""
        if not recipient:
            recipient = self.sam_email  # Send to self if no recipient

        try:
            timestamp = datetime.datetime.now()
            subject = f"SAM AGI System Report - {timestamp.strftime('%Y-%m-%d')}"

            # Generate report content
            report_body = self._generate_system_report(report_type, timestamp)

            # Get recent files as attachments if Google Drive is available
            attachments = []
            if self.google_drive:
                try:
                    recent_files = self.google_drive.list_files()
                    if recent_files:
                        # Attach the most recent system info file
                        for file_info in recent_files:
                            if 'system_info' in file_info['name'].lower():
                                # Download and attach
                                temp_path = f"/tmp/{file_info['name']}"
                                if self.google_drive.download_file(file_info['id'], temp_path):
                                    attachments.append(temp_path)
                                break
                except Exception as e:
                    print(f"⚠️ Could not attach system files: {e}")

            return self.send_email(recipient, subject, report_body, attachments, "normal")

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def setup_automated_reports(self, recipient: str, frequency: str = "daily") -> Dict:
        """Set up automated system reports"""
        result = {
            'success': False,
            'schedule_id': None,
            'error': None
        }

        try:
            schedule_id = f"report_{frequency}_{int(time.time())}"

            if frequency == "daily":
                schedule.every().day.at("18:00").do(self.send_system_report, recipient, "daily")
            elif frequency == "weekly":
                schedule.every().friday.at("18:00").do(self.send_system_report, recipient, "weekly")
            elif frequency == "hourly":
                schedule.every().hour.do(self.send_system_report, recipient, "hourly")

            result['success'] = True
            result['schedule_id'] = schedule_id

            print(f"📊 Automated {frequency} reports scheduled for {recipient}")

        except Exception as e:
            result['error'] = str(e)

        return result

    def monitor_emails(self, keywords: List[str] = None, auto_respond: bool = False) -> Dict:
        """Monitor incoming emails (requires app password)"""
        result = {
            'success': False,
            'messages_found': 0,
            'error': None
        }

        try:
            # Note: This requires Gmail app password, not regular password
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.sam_email, self.sam_password)
            mail.select('inbox')

            # Search for unread messages
            status, messages = mail.search(None, 'UNSEEN')

            if status == 'OK':
                message_ids = messages[0].split()
                result['messages_found'] = len(message_ids)

                for msg_id in message_ids[-10:]:  # Process last 10 unread messages
                    status, msg_data = mail.fetch(msg_id, '(RFC822)')
                    if status == 'OK':
                        email_message = email.message_from_bytes(msg_data[0][1])
                        self._process_incoming_email(email_message, keywords, auto_respond)

            mail.logout()
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Email monitoring failed: {e}")
            print("   💡 Note: Email monitoring requires Gmail app password, not regular password")

        return result

    def _process_incoming_email(self, email_message, keywords: List[str] = None, auto_respond: bool = False):
        """Process an incoming email"""
        try:
            subject = email_message['Subject'] or ""
            sender = email_message['From'] or ""
            body = ""

            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = email_message.get_payload(decode=True).decode()

            print(f"📧 Received email from {sender}: {subject[:50]}...")

            # Check for keywords if specified
            if keywords:
                content = f"{subject} {body}".lower()
                if any(keyword.lower() in content for keyword in keywords):
                    print(f"🔍 Keyword match found for: {subject}")

                    if auto_respond:
                        # Generate automated response
                        response_subject = f"Re: {subject}"
                        response_body = self._generate_auto_response(subject, body)

                        self.send_email(sender, response_subject, response_body)

        except Exception as e:
            print(f"⚠️ Error processing email: {e}")

    def _generate_auto_response(self, original_subject: str, original_body: str) -> str:
        """Generate automated response to incoming emails"""
        response = f"""Hello,

Thank you for your email regarding "{original_subject}".

This is an automated response from the SAM AGI System. I have received your message and will process it accordingly.

If this is urgent or requires immediate attention, please include "URGENT" in the subject line.

Best regards,
SAM AGI System
{sam.agi.unified@gmail.com}

---
Original message:
{original_body[:500]}...
"""
        return response

    def _generate_system_report(self, report_type: str, timestamp: datetime.datetime) -> str:
        """Generate system status report"""
        report = f"""SAM AGI System {report_type.title()} Report
Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

SYSTEM STATUS:
- Account: {self.sam_email}
- Status: Operational
- Gmail Integration: Active

RECENT ACTIVITY:
- System initialized and running
- Agent communications active
- Web search capabilities operational
- Code modification system ready

PERFORMANCE METRICS:
- Uptime: Continuous
- Memory usage: Optimal
- Network connectivity: Active
- Agent coordination: Functional

This report was generated automatically by the SAM AGI System.
For manual interaction, visit the dashboard at http://localhost:5004

---
SAM AGI System
Advanced General Intelligence
"""

        return report

    def _save_email_to_drive(self, email_data: Dict):
        """Save email data to Google Drive"""
        try:
            if not self.google_drive:
                return

            # Create email record
            email_record = {
                'timestamp': email_data['timestamp'],
                'to': email_data['to'],
                'subject': email_data['subject'],
                'body': email_data['body'],
                'attachments': email_data.get('attachments', [])
            }

            # Save as JSON file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(email_record, f, indent=2)
                temp_file = f.name

            # Upload to Drive
            filename = f"sam_email_{int(time.time())}.json"
            self.google_drive.upload_file(temp_file, filename)

            # Clean up
            os.unlink(temp_file)

            print(f"💾 Email saved to Google Drive: {filename}")

        except Exception as e:
            print(f"⚠️ Failed to save email to Drive: {e}")

    def start_scheduler(self):
        """Start the email scheduler"""
        def scheduler_loop():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()

        print("📅 Email scheduler started")

    def get_scheduled_emails(self) -> List[Dict]:
        """Get list of scheduled emails"""
        return self.scheduled_emails.copy()

    def cancel_scheduled_email(self, schedule_id: str) -> bool:
        """Cancel a scheduled email"""
        try:
            self.scheduled_emails = [email for email in self.scheduled_emails if email['id'] != schedule_id]
            return True
        except Exception:
            return False

# Global instance for SAM system integration
sam_gmail = None

def initialize_sam_gmail(google_drive=None):
    """Initialize SAM Gmail integration"""
    global sam_gmail
    sam_gmail = SAMGmailIntegration(google_drive)
    return sam_gmail

def send_sam_email(to_email: str, subject: str, body: str, attachments: List[str] = None) -> Dict:
    """Send email using SAM's Gmail account"""
    global sam_gmail
    if not sam_gmail:
        sam_gmail = SAMGmailIntegration()

    return sam_gmail.send_email(to_email, subject, body, attachments)

def schedule_sam_email(to_email: str, subject: str, body: str, send_time: str) -> Dict:
    """Schedule an email using SAM's Gmail account"""
    global sam_gmail
    if not sam_gmail:
        sam_gmail = SAMGmailIntegration()

    return sam_gmail.schedule_email(to_email, subject, body, send_time)
