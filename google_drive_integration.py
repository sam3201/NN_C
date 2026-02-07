#!/usr/bin/env python3
"""
Google Drive Integration for SAM System
Provides cloud storage, backup, and file synchronization capabilities
"""

import os
import io
import json
import pickle
from datetime import datetime
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

class GoogleDriveIntegration:
    """Google Drive integration for SAM system data persistence and backup"""

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    CREDENTIALS_FILE = 'credentials.json'
    TOKEN_FILE = 'token.pickle'

    def __init__(self, credentials_path=None, token_path=None):
        self.service = None
        self.credentials_path = credentials_path or self.CREDENTIALS_FILE
        self.token_path = token_path or self.TOKEN_FILE
        self.sam_folder_id = None
        self._authenticated = False

    def authenticate(self):
        """Authenticate with Google Drive API using OAuth 2.0"""
        creds = None

        # Check if token exists
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)

        # If credentials are invalid or don't exist, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    print("❌ Google Drive credentials.json not found")
                    print("   Please download credentials from Google Cloud Console")
                    print("   Go to: https://console.cloud.google.com/")
                    print("   Create a project -> APIs & Services -> Credentials -> OAuth 2.0 Client ID")
                    print("   Download the credentials.json file and place it in the project root")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials for next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)

        try:
            self.service = build('drive', 'v3', credentials=creds)
            self._authenticated = True
            print("✅ Google Drive authentication successful")

            # Create/find SAM folder
            self._ensure_sam_folder()
            return True

        except Exception as e:
            print(f"❌ Google Drive authentication failed: {e}")
            return False

    def _ensure_sam_folder(self):
        """Create or find the SAM system folder in Google Drive"""
        try:
            # Search for existing SAM folder
            query = "name='SAM_System_Data' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces='drive').execute()
            items = results.get('files', [])

            if items:
                self.sam_folder_id = items[0]['id']
                print(f"📁 Found existing SAM folder: {self.sam_folder_id}")
            else:
                # Create new SAM folder
                file_metadata = {
                    'name': 'SAM_System_Data',
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                file = self.service.files().create(body=file_metadata, fields='id').execute()
                self.sam_folder_id = file.get('id')
                print(f"📁 Created new SAM folder: {self.sam_folder_id}")

        except Exception as e:
            print(f"❌ Failed to create/find SAM folder: {e}")

    def upload_file(self, local_path, remote_name=None, folder_id=None):
        """Upload a file to Google Drive"""
        if not self._authenticated:
            if not self.authenticate():
                return None

        try:
            file_path = Path(local_path)
            if not file_path.exists():
                print(f"❌ Local file not found: {local_path}")
                return None

            file_name = remote_name or file_path.name
            folder_id = folder_id or self.sam_folder_id

            # Check if file already exists
            existing_file = self._find_file(file_name, folder_id)
            if existing_file:
                # Update existing file
                media = MediaFileUpload(local_path, resumable=True)
                file = self.service.files().update(
                    fileId=existing_file['id'],
                    media_body=media
                ).execute()
                print(f"📤 Updated file in Drive: {file_name}")
            else:
                # Create new file
                file_metadata = {
                    'name': file_name,
                    'parents': [folder_id] if folder_id else []
                }
                media = MediaFileUpload(local_path, resumable=True)
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"📤 Uploaded file to Drive: {file_name}")

            return file.get('id')

        except Exception as e:
            print(f"❌ Failed to upload file {local_path}: {e}")
            return None

    def download_file(self, file_id, local_path):
        """Download a file from Google Drive"""
        if not self._authenticated:
            if not self.authenticate():
                return False

        try:
            request = self.service.files().get_media(fileId=file_id)
            with io.FileIO(local_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"📥 Download progress: {int(status.progress() * 100)}%")

            print(f"📥 Downloaded file to: {local_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to download file {file_id}: {e}")
            return False

    def _find_file(self, name, folder_id=None):
        """Find a file by name in the specified folder"""
        try:
            query = f"name='{name}' and trashed=false"
            if folder_id:
                query += f" and '{folder_id}' in parents"

            results = self.service.files().list(q=query, spaces='drive').execute()
            items = results.get('files', [])

            return items[0] if items else None

        except Exception as e:
            print(f"❌ Failed to find file {name}: {e}")
            return None

    def list_files(self, folder_id=None):
        """List files in the specified folder"""
        if not self._authenticated:
            if not self.authenticate():
                return []

        try:
            folder_id = folder_id or self.sam_folder_id
            query = f"'{folder_id}' in parents and trashed=false"

            results = self.service.files().list(q=query, spaces='drive').execute()
            items = results.get('files', [])

            return items

        except Exception as e:
            print(f"❌ Failed to list files: {e}")
            return []

    def backup_sam_data(self, data_dir="SAM_Backup"):
        """Backup SAM system data to Google Drive - EXCLUDING SENSITIVE FILES"""
        if not self._authenticated:
            if not self.authenticate():
                return False

        try:
            # EXCLUDE sensitive files that contain private information
            safe_backup_files = [
                "README.md",  # Generally safe
                # EXCLUDE: complete_sam_unified.py (contains API keys, credentials)
                # EXCLUDE: requirements.txt (dependency info, not sensitive)
                # EXCLUDE: run_sam.sh (contains system paths, not sensitive for backup)
            ]

            # Add additional safe files if they exist
            additional_safe_files = [
                "COMPLETION_REPORT.md",
                "ARCHITECTURE.md",
                "CHANGELOG.md"
            ]

            uploaded_count = 0
            for filename in safe_backup_files + additional_safe_files:
                if os.path.exists(filename):
                    # Create sanitized backup filename
                    safe_filename = f"backup_{filename}"
                    self.upload_file(filename, safe_filename)
                    uploaded_count += 1

            # Create a sanitized system info file
            system_info = self._create_system_info_backup()
            if system_info:
                self.upload_file(system_info, "backup_system_info.txt")
                uploaded_count += 1

            print(f"✅ Backed up {uploaded_count} SAFE SAM files to Google Drive")
            print("   🔒 Private credentials and sensitive data excluded from backup")
            return True

        except Exception as e:
            print(f"❌ Failed to backup SAM data: {e}")
            return False

    def sync_chat_logs(self, logs_dir="CHAT_LOGS"):
        """Sync chat logs to Google Drive"""
        if not self._authenticated:
            if not self.authenticate():
                return False

        try:
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)

            # Create chat logs subfolder
            chat_folder = self._create_folder("Chat_Logs", self.sam_folder_id)
            if not chat_folder:
                return False

            uploaded_count = 0
            for filename in os.listdir(logs_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(logs_dir, filename)
                    self.upload_file(filepath, filename, chat_folder)
                    uploaded_count += 1

            print(f"✅ Synced {uploaded_count} chat logs to Google Drive")
            return True

        except Exception as e:
            print(f"❌ Failed to sync chat logs: {e}")
            return False

    def _create_folder(self, name, parent_id=None):
        """Create a folder in Google Drive"""
        try:
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id] if parent_id else []
            }

            file = self.service.files().create(
                body=file_metadata, fields='id'
            ).execute()

            return file.get('id')

        except Exception as e:
            print(f"❌ Failed to create folder {name}: {e}")
            return None

    def get_drive_info(self):
        """Get information about the Google Drive account"""
        if not self._authenticated:
            if not self.authenticate():
                return None

        try:
            about = self.service.about().get(fields="user,storageQuota").execute()
            return {
                'email': about.get('user', {}).get('emailAddress'),
                'quota': about.get('storageQuota', {})
            }

        except Exception as e:
            print(f"❌ Failed to get drive info: {e}")
            return None

    def _create_system_info_backup(self):
        """Create a sanitized system information file for backup"""
        try:
            import platform
            import datetime

            system_info = f"""SAM System Backup Information
Generated: {datetime.datetime.now().isoformat()}

SYSTEM OVERVIEW:
- Operating System: {platform.system()} {platform.release()}
- Python Version: {platform.python_version()}
- Architecture: {platform.machine()}

SAM SYSTEM STATUS:
- Google Drive Integration: Active (Dedicated Account)
- Account: sam.ai.system.agi@gmail.com (sanitized)
- Backup Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMPORTANT SECURITY NOTES:
- Private API keys and credentials are NOT included in this backup
- Personal user data is excluded from all backups
- This backup contains only system architecture and general information

BACKUP CONTENTS:
- System documentation (README, reports)
- Architecture information (if available)
- General system status

For security, sensitive files are automatically excluded from cloud backups.
"""

            # Write to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(system_info)
                return f.name

        except Exception as e:
            print(f"⚠️ Failed to create system info backup: {e}")
            return None

def test_google_drive_connection():
    """Test Google Drive integration"""
    drive = GoogleDriveIntegration()

    if drive.authenticate():
        info = drive.get_drive_info()
        if info:
            print(f"📧 Account: {info['email']}")
            quota = info['quota']
            used = int(quota.get('usage', 0)) / (1024**3)  # GB
            total = int(quota.get('limit', 0)) / (1024**3)  # GB
            print(".2f")

        # Test backup
        drive.backup_sam_data()

        return True
    else:
        print("❌ Google Drive connection failed")
        return False

if __name__ == "__main__":
    test_google_drive_connection()
