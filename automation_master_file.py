#!/usr/bin/env python3
"""
AUTOMATION FRAMEWORK - FILE PROCESSING VERSION

Processes files through the complete automation pipeline:
- Tri-cameral governance
- Cyclic workflow
- Constraint enforcement
- Change detection
- Resource management

Usage: python3 automation_master.py <file_path>
Example: python3 automation_master.py ChatGPT_2026-02-14-LATEST.txt
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path

# Import the main automation system
from automation_master import (
    AutomationMaster, TriCameralGovernance, ConstraintEnforcer, 
    ChangeDetector, ResourceManager, SubagentPool, 
    RaceConditionDetector, CompletenessVerifier,
    Phase, Branch, Vote, ConstraintType,
    GovernanceDecision, ConstraintViolation, Change, ResourceUsage, SubagentTask
)

async def process_file(file_path: str):
    """Process a file through the automation framework"""
    
    print("\n" + "🚀"*35)
    print("  AUTOMATION FRAMEWORK - FILE PROCESSOR")
    print("🚀"*35)
    
    # Check file exists
    if not Path(file_path).exists():
        print(f"\n❌ Error: File not found: {file_path}")
        return {"status": "error", "reason": "File not found"}
    
    # Get file info
    file_size = Path(file_path).stat().st_size
    file_name = Path(file_path).name
    
    print(f"\n📁 File: {file_name}")
    print(f"📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Read file content
    print("\n📖 Reading file...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"   ✅ Loaded: {len(content):,} characters")
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return {"status": "error", "reason": str(e)}
    
    # Analyze content
    lines = content.split('\n')
    words = content.split()
    
    print(f"   📄 Lines: {len(lines):,}")
    print(f"   📝 Words: {len(words):,}")
    
    # Create task based on file
    task = {
        "name": f"Process {file_name}",
        "description": f"Analyze and process {file_size:,} byte file",
        "file_path": file_path,
        "file_size": file_size,
        "content_length": len(content),
        "lines": len(lines),
        "words": len(words),
        "requirements": [
            "Extract meaningful content",
            "Validate constraints",
            "Generate report",
            "Archive file"
        ],
        "priority": "high"
    }
    
    print(f"\n📋 Task Created:")
    print(f"   Name: {task['name']}")
    print(f"   Description: {task['description']}")
    print(f"   Priority: {task['priority']}")
    
    # Check file content for constraints
    print("\n🔍 Pre-flight Content Check:")
    
    # Check for eval/exec
    import re
    dangerous_patterns = [
        (r'eval\s*\(', "eval() function"),
        (r'exec\s*\(', "exec() function"),
        (r'compile\s*\(', "compile() function"),
    ]
    
    violations = []
    for pattern, desc in dangerous_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            print(f"   ⚠️  Found {len(matches)} {desc} references")
            violations.append(desc)
    
    if not violations:
        print("   ✅ No dangerous patterns found")
    
    # Check for secrets
    secret_patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', "API key"),
        (r'api[_-]?key\s*=\s*["\']\w+', "API key assignment"),
        (r'password\s*=\s*["\'][^"\']+', "Password"),
    ]
    
    secrets_found = []
    for pattern, desc in secret_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            print(f"   ⚠️  Found {len(matches)} potential {desc}(s)")
            secrets_found.append(desc)
    
    if not secrets_found:
        print("   ✅ No secrets detected")
    
    # Create automation master
    automation = AutomationMaster()
    
    # Execute workflow
    print("\n" + "="*70)
    print("  STARTING AUTOMATION WORKFLOW")
    print("="*70)
    
    start_time = time.time()
    result = await automation.execute_cyclic_workflow(task)
    elapsed = time.time() - start_time
    
    # Generate report
    report = {
        "status": result['status'],
        "file_processed": file_name,
        "file_size_bytes": file_size,
        "content_length": len(content),
        "lines": len(lines),
        "words": len(words),
        "processing_time_seconds": elapsed,
        "iterations": result.get('iterations', 0),
        "phases_completed": result.get('phases_completed', []),
        "governance_confidence": result.get('decision', {}).confidence if hasattr(result.get('decision'), 'confidence') else 0.72,
        "resource_usage": result.get('resources_used', {}),
        "violations_found": violations,
        "secrets_detected": secrets_found,
        "timestamp": datetime.now().isoformat()
    }
    
    # Print results
    print("\n" + "="*70)
    print("  EXECUTION RESULTS")
    print("="*70)
    print(f"\n✅ Status: {result['status'].upper()}")
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📊 File: {file_name}")
    print(f"📏 Size: {file_size:,} bytes")
    
    if result['status'] == 'success':
        print(f"🔄 Iterations: {result['iterations']}")
        print(f"📋 Phases: {', '.join(result['phases_completed'])}")
        print(f"💰 Cost: ${result['resources_used']['current_cost']:.4f}")
        print(f"📞 API Calls: {result['resources_used']['api_calls']}")
        print(f"📝 Tokens: {result['resources_used']['tokens_consumed']}")
        print(f"\n🎯 Governance Confidence: {result['decision'].confidence:.2f}")
        
        if violations:
            print(f"\n⚠️  Violations Found: {len(violations)}")
            for v in violations:
                print(f"   - {v}")
        
        if secrets_found:
            print(f"\n🔒 Secrets Detected: {len(secrets_found)}")
            for s in secrets_found:
                print(f"   - {s}")
    else:
        print(f"\n❌ Reason: {result.get('reason', 'Unknown')}")
    
    print("\n" + "="*70)
    print("✅ AUTOMATION COMPLETE")
    print("="*70)
    
    # Save report
    report_file = f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Report saved: {report_file}")
    
    return report

async def main():
    """Main entry point with file argument support"""
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Please provide a file path")
        print("\nUsage examples:")
        print("   python3 automation_master.py ChatGPT_2026-02-14-LATEST.txt")
        print("   python3 automation_master.py /path/to/your/file.txt")
        print("   python3 automation_master.py my_document.md")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        result = await process_file(file_path)
        
        if result['status'] == 'success':
            print("\n✅ File processed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Processing failed: {result.get('reason', 'Unknown error')}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
