import { z } from "zod";

const openclawWebhookUrl = "http://localhost:8765/webhook";
const openclawToken = process.env.OPENCLAW_WEBHOOK_TOKEN || "dev-token-change-in-production";

function generateSignature(payload: string): string {
  const crypto = require("crypto");
  const hmac = crypto.createHmac("sha256", openclawToken);
  return "sha256=" + hmac.update(payload).digest("hex");
}

export const openclawExecuteSchema = {
  name: "openclaw_execute",
  description: "Execute commands via OpenClaw webhook server for SAM-D automation",
  inputSchema: {
    type: "object" as const,
    properties: {
      command: {
        type: "string",
        enum: ["build-extensions", "run-tests", "analyze", "verify", "deep-scan"],
        description: "Command to execute"
      },
      params: {
        type: "object",
        description: "Command parameters"
      }
    },
    required: ["command"]
  }
};

export async function openclawExecute(args: any) {
  const { command, params = {} } = args;
  
  const payload = JSON.stringify({ command, params });
  const signature = generateSignature(payload);
  
  const response = await fetch(openclawWebhookUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Webhook-Signature": signature
    },
    body: payload
  });
  
  if (!response.ok) {
    return { error: `Webhook error: ${response.status}` };
  }
  
  return await response.json();
}

export const openclawTriCameralSchema = {
  name: "openclaw_tri_cameral",
  description: "Execute tri-cameral governance cycle via OpenClaw",
  inputSchema: {
    type: "object" as const,
    properties: {
      workflow: {
        type: "object",
        properties: {
          name: { type: "string" },
          description: { type: "string" },
          high_level_plan: { type: "string" },
          low_level_plan: { type: "string" },
          hard_constraints: { type: "array", items: { type: "string" } },
          soft_constraints: { type: "array", items: { type: "string" } },
          invariants: { type: "array", items: { type: "string" } },
          risk_level: { type: "number" }
        },
        required: ["name"]
      }
    },
    required: ["workflow"]
  }
};

export async function openclawTriCameral(args: any) {
  const { workflow } = args;
  
  const payload = JSON.stringify({ workflow });
  const signature = generateSignature(payload);
  
  const response = await fetch("http://localhost:8765/webhook/tri-cameral", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Webhook-Signature": signature
    },
    body: payload
  });
  
  if (!response.ok) {
    return { error: `Webhook error: ${response.status}` };
  }
  
  return await response.json();
}

export const openclawStatusSchema = {
  name: "openclaw_status",
  description: "Get OpenClaw system status",
  inputSchema: {
    type: "object" as const,
    properties: {}
  }
};

export async function openclawStatus() {
  const response = await fetch("http://localhost:8765/status");
  return await response.json();
}
