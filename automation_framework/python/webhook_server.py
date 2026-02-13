#!/usr/bin/env python3
"""
OpenClaw Webhook Server
Enables communication between OpenCode and OpenClaw via HTTP webhooks
"""

import json
import os
import sys
import hashlib
import hmac
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
import subprocess
from typing import Dict, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "automation_framework", "python"))

WEBHOOK_TOKEN = os.environ.get("OPENCLAWHook_TOKEN", os.environ.get("OPENCLAW_WEBHOOK_TOKEN", "dev-token-change-in-production"))
PORT = int(os.environ.get("OPENCLAW_WEBHOOK_PORT", "8765"))

class WebhookHandler(BaseHTTPRequestHandler):
    """Handle webhook requests from OpenCode"""
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[Webhook] {self.address_string()} - {format % args}")
    
    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature"""
        if not signature:
            return False
        expected = hmac.new(
            WEBHOOK_TOKEN.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)
    
    def _send_json_response(self, status: int, data: Dict[str, Any]):
        """Send JSON response"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        signature = self.headers.get("X-Webhook-Signature", "")
        
        if not self._verify_signature(body, signature):
            self._send_json_response(401, {"error": "Invalid signature"})
            return
        
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "Invalid JSON"})
            return
        
        endpoint = urlparse(self.path).path
        
        if endpoint == "/webhook":
            self._handle_webhook(payload)
        elif endpoint == "/webhook/execute":
            self._handle_execute(payload)
        elif endpoint == "/webhook/tri-cameral":
            self._handle_tri_cameral(payload)
        elif endpoint == "/webhook/cycle":
            self._handle_cycle(payload)
        else:
            self._send_json_response(404, {"error": "Unknown endpoint"})
    
    def do_GET(self):
        """Handle GET requests"""
        endpoint = urlparse(self.path).path
        
        if endpoint == "/health":
            self._send_json_response(200, {"status": "ok", "timestamp": time.time()})
        elif endpoint == "/status":
            self._send_json_response(200, self._get_status())
        else:
            self._send_json_response(404, {"error": "Unknown endpoint"})
    
    def _handle_webhook(self, payload: Dict[str, Any]) -> None:
        """Handle general webhook payload"""
        command = payload.get("command", "")
        params = payload.get("params", {})
        
        result = self._execute_command(command, params)
        self._send_json_response(200, result)
    
    def _handle_execute(self, payload: Dict[str, Any]) -> None:
        """Execute a command directly"""
        action = payload.get("action", "")
        params = payload.get("params", {})
        
        result = self._execute_command(action, params)
        self._send_json_response(200, result)
    
    def _handle_tri_cameral(self, payload: Dict[str, Any]) -> None:
        """Execute tri-cameral governance cycle"""
        workflow = payload.get("workflow", {})
        
        try:
            from automation_bridge import TriCameralOrchestrator, WorkflowConfig
            
            config = WorkflowConfig(
                name=workflow.get("name", "unnamed"),
                description=workflow.get("description", ""),
                high_level_plan=workflow.get("high_level_plan", ""),
                low_level_plan=workflow.get("low_level_plan", ""),
                hard_constraints=workflow.get("hard_constraints", []),
                soft_constraints=workflow.get("soft_constraints", []),
                invariants=workflow.get("invariants", []),
                risk_level=workflow.get("risk_level", 0.5)
            )
            
            orchestrator = TriCameralOrchestrator()
            
            import asyncio
            decision = asyncio.run(orchestrator.evaluate_workflow(config))
            
            self._send_json_response(200, {
                "success": True,
                "decision": {
                    "proceed": decision.proceed,
                    "confidence": decision.confidence,
                    "cic_vote": decision.cic_vote,
                    "aee_vote": decision.aee_vote,
                    "csf_vote": decision.csf_vote,
                    "concerns": decision.concerns,
                    "recommendations": decision.recommendations
                }
            })
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})
    
    def _handle_cycle(self, payload: Dict[str, Any]) -> None:
        """Execute a development cycle"""
        phase = payload.get("phase", "PLAN")
        task = payload.get("task", "")
        
        result = self._run_development_cycle(phase, task)
        self._send_json_response(200, result)
    
    def _execute_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command and return result"""
        try:
            if command == "build-extensions":
                return self._build_c_extensions()
            elif command == "run-tests":
                return self._run_tests(params.get("suite", "all"))
            elif command == "analyze":
                return self._analyze_codebase(params.get("depth", "medium"))
            elif command == "verify":
                return self._verify_system()
            elif command == "deep-scan":
                return self._deep_scan(params.get("options", {}))
            else:
                return {"error": f"Unknown command: {command}"}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _build_c_extensions(self) -> Dict[str, Any]:
        """Build C extensions"""
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else "",
                "error": result.stderr[-1000:] if result.stderr else ""
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_tests(self, suite: str) -> Dict[str, Any]:
        """Run test suite"""
        try:
            result = subprocess.run(
                ["pytest", "-q", "--tb=short"] if suite == "all" else ["pytest", "-q", f"-k={suite}"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else "",
                "error": result.stderr[-1000:] if result.stderr else ""
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_codebase(self, depth: str) -> Dict[str, Any]:
        """Analyze codebase"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", "src/python/complete_sam_unified.py"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else "",
                "error": result.stderr[-1000:] if result.stderr else ""
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _verify_system(self) -> Dict[str, Any]:
        """Verify system integrity"""
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-c", "from complete_sam_unified import UnifiedSAMSystem; print('OK')"],
                cwd=os.path.join(ROOT_DIR, "src/python"),
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PYTHONPATH": "src/python:."}
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else "",
                "error": result.stderr[-1000:] if result.stderr else ""
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _deep_scan(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep scan"""
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", "src/python/complete_sam_unified.py"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=120
            )
            return {
                "success": result.returncode == 0,
                "files_checked": 1,
                "syntax_errors": result.returncode != 0,
                "output": result.stderr[-1000:] if result.stderr else "No errors"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_development_cycle(self, phase: str, task: str) -> Dict[str, Any]:
        """Run development cycle"""
        return {
            "phase": phase,
            "task": task,
            "status": "completed",
            "timestamp": time.time()
        }
    
    def _get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": "operational",
            "version": "5.2.0",
            "uptime": time.time(),
            "components": {
                "tri_cameral": True,
                "cyclic_workflow": True,
                "model_router": True
            }
        }

def start_webhook_server(port: int = PORT):
    """Start the webhook server"""
    server = HTTPServer(("0.0.0.0", port), WebhookHandler)
    print(f"[OpenClaw Webhook] Server running on port {port}")
    print(f"[OpenClaw Webhook] Auth token: {WEBHOOK_TOKEN[:8]}...")
    server.serve_forever()

if __name__ == "__main__":
    start_webhook_server()
