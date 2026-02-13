import { tool } from "@opencode-ai/plugin";
import path from "path";

export default tool({
  description: "Orchestrate multiple subagents for parallel task execution",
  args: {
    task: tool.schema.string().describe("The main task to accomplish"),
    num_agents: tool.schema.number().default(3).describe("Number of subagents to deploy"),
    mode: tool.schema.enum(["parallel", "reader-processor-writer", "verification"]).default("parallel").describe("Execution mode for subagents")
  },
  async execute(args, context) {
    const script = path.join(context.worktree, ".opencode/tools/subagent_orchestrator.py");
    const result = await Bun.$`python3 ${script} ${args.task} ${args.num_agents} ${args.mode}`.text();
    return result;
  },
});
