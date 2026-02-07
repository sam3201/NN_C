#!/usr/bin/env python3
"""
SAM GitHub Integration System
Allows SAM to save itself and manage its own codebase on GitHub
"""

import os
import json
import requests
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tempfile
import subprocess
import shutil

class SAMGitHubIntegration:
    """GitHub integration for SAM system self-management"""

    def __init__(self, repo_owner: str = None, repo_name: str = None, token: str = None):
        # GitHub configuration
        self.repo_owner = repo_owner or os.getenv('GITHUB_REPO_OWNER', 'samaisystemagi')
        self.repo_name = repo_name or os.getenv('GITHUB_REPO_NAME', 'NN_C')
        self.token = token or os.getenv('GITHUB_TOKEN', '')
        self.base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"

        # Headers for API requests
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }

        # Local repository path
        self.local_repo_path = Path.cwd()

        print("🐙 SAM GitHub Integration initialized")
        print(f"   📁 Repository: {self.repo_owner}/{self.repo_name}")
        print(f"   🔑 Token: {'Configured' if self.token else 'Not configured'}")

    def test_connection(self) -> Dict:
        """Test GitHub API connection and repository access"""
        result = {
            'success': False,
            'repo_exists': False,
            'write_access': False,
            'error': None
        }

        try:
            # Test basic API access
            response = requests.get(self.base_url, headers=self.headers)
            if response.status_code == 401:
                result['error'] = "Invalid GitHub token"
                return result
            elif response.status_code == 404:
                result['error'] = f"Repository {self.repo_owner}/{self.repo_name} not found"
                return result
            elif response.status_code != 200:
                result['error'] = f"API error: {response.status_code}"
                return result

            repo_data = response.json()
            result['repo_exists'] = True

            # Test write access by checking permissions
            permissions = repo_data.get('permissions', {})
            if permissions.get('push', False):
                result['write_access'] = True
            else:
                result['error'] = "No write access to repository"

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        return result

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content from GitHub repository"""
        try:
            url = f"{self.base_url}/contents/{file_path}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                if 'content' in data:
                    # Decode base64 content
                    import base64
                    return base64.b64decode(data['content']).decode('utf-8')
            elif response.status_code == 404:
                return None  # File doesn't exist
            else:
                print(f"❌ Failed to get file {file_path}: {response.status_code}")

        except Exception as e:
            print(f"❌ Error getting file {file_path}: {e}")

        return None

    def update_file(self, file_path: str, content: str, commit_message: str) -> Dict:
        """Update a file in the GitHub repository"""
        result = {
            'success': False,
            'commit_sha': None,
            'error': None
        }

        try:
            url = f"{self.base_url}/contents/{file_path}"

            # First, get current file info (if it exists)
            current_file = None
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                current_file = response.json()

            # Prepare the update payload
            payload = {
                'message': commit_message,
                'content': base64.b64encode(content.encode('utf-8')).decode('utf-8')
            }

            if current_file:
                # File exists, include SHA for update
                payload['sha'] = current_file['sha']

            # Make the API request
            response = requests.put(url, headers=self.headers, data=json.dumps(payload))

            if response.status_code in [200, 201]:
                data = response.json()
                result['success'] = True
                result['commit_sha'] = data['commit']['sha']
                print(f"✅ File updated: {file_path} (commit: {result['commit_sha'][:8]})")
            else:
                result['error'] = f"GitHub API error: {response.status_code} - {response.text}"

        except Exception as e:
            result['error'] = str(e)

        return result

    def create_commit(self, files_to_update: Dict[str, str], commit_message: str) -> Dict:
        """Create a commit with multiple file updates"""
        result = {
            'success': False,
            'commit_sha': None,
            'files_updated': 0,
            'errors': []
        }

        try:
            # Get the latest commit SHA
            ref_url = f"{self.base_url}/git/refs/heads/main"
            response = requests.get(ref_url, headers=self.headers)

            if response.status_code != 200:
                # Try 'master' branch
                ref_url = f"{self.base_url}/git/refs/heads/master"
                response = requests.get(ref_url, headers=self.headers)

            if response.status_code != 200:
                result['error'] = "Could not find main/master branch"
                return result

            ref_data = response.json()
            latest_commit_sha = ref_data['object']['sha']

            # Get the commit tree
            commit_url = f"{self.base_url}/git/commits/{latest_commit_sha}"
            response = requests.get(commit_url, headers=self.headers)
            if response.status_code != 200:
                result['error'] = "Could not get latest commit"
                return result

            commit_data = response.json()
            tree_sha = commit_data['tree']['sha']

            # Create blobs for each file
            new_tree_items = []

            for file_path, content in files_to_update.items():
                # Create blob
                blob_payload = {
                    'content': content,
                    'encoding': 'utf-8'
                }

                blob_url = f"{self.base_url}/git/blobs"
                response = requests.post(blob_url, headers=self.headers, data=json.dumps(blob_payload))

                if response.status_code == 201:
                    blob_data = response.json()
                    new_tree_items.append({
                        'path': file_path,
                        'mode': '100644',  # Regular file
                        'type': 'blob',
                        'sha': blob_data['sha']
                    })
                else:
                    result['errors'].append(f"Failed to create blob for {file_path}")

            if not new_tree_items:
                result['error'] = "No files could be processed"
                return result

            # Create new tree
            tree_payload = {
                'base_tree': tree_sha,
                'tree': new_tree_items
            }

            tree_url = f"{self.base_url}/git/trees"
            response = requests.post(tree_url, headers=self.headers, data=json.dumps(tree_payload))

            if response.status_code != 201:
                result['error'] = "Failed to create tree"
                return result

            tree_data = response.json()
            new_tree_sha = tree_data['sha']

            # Create commit
            commit_payload = {
                'message': commit_message,
                'parents': [latest_commit_sha],
                'tree': new_tree_sha
            }

            commit_url = f"{self.base_url}/git/commits"
            response = requests.post(commit_url, headers=self.headers, data=json.dumps(commit_payload))

            if response.status_code != 201:
                result['error'] = "Failed to create commit"
                return result

            commit_data = response.json()
            new_commit_sha = commit_data['sha']

            # Update branch reference
            update_ref_payload = {
                'sha': new_commit_sha,
                'force': False
            }

            response = requests.patch(ref_url, headers=self.headers, data=json.dumps(update_ref_payload))

            if response.status_code == 200:
                result['success'] = True
                result['commit_sha'] = new_commit_sha
                result['files_updated'] = len(new_tree_items)
                print(f"✅ Commit created: {new_commit_sha[:8]} ({len(new_tree_items)} files)")
            else:
                result['error'] = f"Failed to update branch: {response.status_code}"

        except Exception as e:
            result['error'] = str(e)

        return result

    def save_sam_system(self, commit_message: str = None) -> Dict:
        """Save the entire SAM system to GitHub"""
        result = {
            'success': False,
            'files_saved': 0,
            'commit_sha': None,
            'error': None
        }

        try:
            # Define safe files to save (exclude sensitive files)
            safe_files = [
                'complete_sam_unified.py',
                'google_drive_integration.py',
                'sam_web_search.py',
                'sam_code_modifier.py',
                'sam_gmail_integration.py',
                'sam_github_integration.py',
                'run_sam.sh',
                'README.md',
                'ARCHITECTURE.md',
                'CHANGELOG.md',
                'coherence_dashboard.html',
                'multi_user_conversation_server.py'
            ]

            files_to_update = {}

            for filename in safe_files:
                filepath = self.local_repo_path / filename
                if filepath.exists():
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Sanitize content (remove any hardcoded secrets)
                        content = self._sanitize_content(content)

                        files_to_update[filename] = content
                        result['files_saved'] += 1

                    except Exception as e:
                        print(f"⚠️ Could not read {filename}: {e}")
                        continue

            if not files_to_update:
                result['error'] = "No files to save"
                return result

            # Create commit message
            if not commit_message:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"SAM System Self-Save - {timestamp}"

            # Create the commit
            commit_result = self.create_commit(files_to_update, commit_message)

            if commit_result['success']:
                result['success'] = True
                result['commit_sha'] = commit_result['commit_sha']
                result['files_saved'] = commit_result['files_updated']
                print(f"💾 SAM system saved to GitHub: {result['commit_sha'][:8]}")
            else:
                result['error'] = commit_result.get('error', 'Commit failed')

        except Exception as e:
            result['error'] = str(e)

        return result

    def _sanitize_content(self, content: str) -> str:
        """Sanitize content to remove sensitive information"""
        import re

        # Remove potential API keys and passwords
        content = re.sub(r'(?i)(api_key|apikey|token|password|secret).*?[\'\"](.*?)[\'\"]', r'\1": "[REDACTED]"', content)
        content = re.sub(r'(?i)(GOOGLE_ACCOUNT|GOOGLE_PASSWORD).*?=.*', r'# \1=REDACTED', content)

        # Remove any hardcoded email addresses that might contain sensitive info
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', content)

        return content

    def get_recent_commits(self, limit: int = 10) -> List[Dict]:
        """Get recent commits from the repository"""
        try:
            url = f"{self.base_url}/commits?per_page={limit}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                commits = response.json()
                return [{
                    'sha': commit['sha'][:8],
                    'message': commit['commit']['message'],
                    'author': commit['commit']['author']['name'],
                    'date': commit['commit']['author']['date']
                } for commit in commits]
            else:
                print(f"❌ Failed to get commits: {response.status_code}")

        except Exception as e:
            print(f"❌ Error getting commits: {e}")

        return []

    def create_pull_request(self, title: str, body: str, head_branch: str = None) -> Dict:
        """Create a pull request (for major changes)"""
        result = {
            'success': False,
            'pr_number': None,
            'pr_url': None,
            'error': None
        }

        try:
            # Create a new branch for the PR
            if not head_branch:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                head_branch = f"sam-update-{timestamp}"

            # Create branch from current main/master
            self._create_branch(head_branch)

            # The PR would be created after pushing changes to this branch
            # For now, just prepare the structure

            result['success'] = True
            result['branch_created'] = head_branch

        except Exception as e:
            result['error'] = str(e)

        return result

    def _create_branch(self, branch_name: str) -> bool:
        """Create a new branch from main/master"""
        try:
            # Get main/master ref
            for branch in ['main', 'master']:
                ref_url = f"{self.base_url}/git/refs/heads/{branch}"
                response = requests.get(ref_url, headers=self.headers)
                if response.status_code == 200:
                    ref_data = response.json()
                    sha = ref_data['object']['sha']

                    # Create new branch
                    new_ref_url = f"{self.base_url}/git/refs"
                    payload = {
                        'ref': f'refs/heads/{branch_name}',
                        'sha': sha
                    }

                    response = requests.post(new_ref_url, headers=self.headers, data=json.dumps(payload))
                    return response.status_code == 201

            return False

        except Exception as e:
            print(f"❌ Error creating branch: {e}")
            return False

# Global instance for SAM system integration
sam_github = None

def initialize_sam_github(repo_owner: str = None, repo_name: str = None, token: str = None):
    """Initialize SAM GitHub integration"""
    global sam_github
    sam_github = SAMGitHubIntegration(repo_owner, repo_name, token)
    return sam_github

def save_sam_to_github(commit_message: str = None) -> Dict:
    """Save SAM system to GitHub"""
    global sam_github
    if not sam_github:
        sam_github = SAMGitHubIntegration()

    return sam_github.save_sam_system(commit_message)

def test_github_connection() -> Dict:
    """Test GitHub connection"""
    global sam_github
    if not sam_github:
        sam_github = SAMGitHubIntegration()

    return sam_github.test_connection()
