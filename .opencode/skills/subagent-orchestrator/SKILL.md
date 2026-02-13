---
name: subagent-orchestrator
description: Coordinate multiple subagents for parallel task execution
license: MIT
compatibility: opencode
metadata:
  audience: architects
  workflow: coordination
---

## What I do
- Split complex tasks into parallel subtasks
- Coordinate multiple subagents working simultaneously
- Aggregate results from distributed processing
- Manage task dependencies and synchronization
- Ensure comprehensive coverage of large tasks

## When to use me
Use this skill when:
- Processing large files that need parallel reading
- Conducting comprehensive codebase scans
- Extracting information from multiple sources simultaneously
- Building documentation from scattered information
- Verifying work across multiple dimensions

## Orchestration Strategy
1. **Task Decomposition**: Break large tasks into independent subtasks
2. **Parallel Dispatch**: Launch subagents simultaneously with different focuses
3. **Reader-Processor-Writer Pattern**: 
   - Reader Agent: Read ahead to gather context
   - Processor Agent: Extract current chunk
   - Writer Agent: Document findings
4. **Result Aggregation**: Combine outputs from all subagents
5. **Verification**: Cross-check results for completeness

## Subagent Types
- **Explorer Agent**: Scans codebase structure
- **Reader Agent**: Reads and summarizes file contents
- **Extractor Agent**: Pulls specific technical details
- **Verifier Agent**: Checks completeness and accuracy
- **Writer Agent**: Documents findings

## Example Usage
"Orchestrate 3 subagents to simultaneously analyze the C modules, Python modules, and documentation"
"Use parallel subagents to extract equations from different sections of this document"
"Coordinate subagents for comprehensive test coverage analysis"
