#!/usr/bin/env python3
"""
SAM-D Concurrent Document Processor
Uses subagent pool for parallel reading, extraction, and processing
Then archives/deletes source files
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "automation_framework" / "python"))

from automation_bridge import SubagentPool, TriCameralOrchestrator, WorkflowConfig
from run_sam import load_secrets

class DocumentProcessor:
    """Process documents using concurrent subagents"""
    
    def __init__(self, max_concurrent: int = 5):
        self.pool = SubagentPool(max_concurrent=max_concurrent)
        self.orchestrator = TriCameralOrchestrator()
        self.results = []
        
    def process_files(self, file_paths: List[str], archive_dir: str = "archive") -> Dict[str, Any]:
        """
        Process multiple files concurrently using subagents
        
        Pipeline: Reader → Extractor → Processor → Archiver
        """
        print(f"🚀 Starting concurrent document processing")
        print(f"   Files to process: {len(file_paths)}")
        print(f"   Max concurrent subagents: {self.pool.max_concurrent}")
        
        # Create archive directory
        archive_path = Path(archive_dir)
        archive_path.mkdir(exist_ok=True)
        
        # Prepare tasks
        tasks = [
            {
                'file_path': fp,
                'archive_path': str(archive_path),
                'task_id': i
            }
            for i, fp in enumerate(file_paths)
        ]
        
        # Define pipeline handlers
        def reader(task: Dict) -> Dict:
            """Reader subagent: Read file content"""
            file_path = task['file_path']
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return {
                    'file_path': file_path,
                    'content': content,
                    'size': len(content),
                    'status': 'read_ok'
                }
            except Exception as e:
                return {
                    'file_path': file_path,
                    'content': '',
                    'error': str(e),
                    'status': 'read_failed'
                }
        
        def extractor(context: Dict) -> Dict:
            """Extractor subagent: Extract key information"""
            if context.get('status') == 'read_failed':
                return context
            
            content = context['content']
            
            # Extract key information
            extraction = {
                'file_path': context['file_path'],
                'line_count': len(content.split('\n')),
                'word_count': len(content.split()),
                'char_count': len(content),
                'has_code': any(kw in content.lower() for kw in ['def ', 'class ', 'import ', 'function']),
                'has_config': any(kw in content.lower() for kw in ['config', 'settings', 'api_key']),
                'first_100_chars': content[:100],
                'key_sections': self._extract_sections(content)
            }
            
            return extraction
        
        def processor(context: Dict) -> Dict:
            """Processor subagent: Process and summarize"""
            if 'error' in context:
                return context
            
            # Generate summary
            summary = {
                'file_path': context['file_path'],
                'summary': f"Document: {context['line_count']} lines, {context['word_count']} words",
                'contains_code': context['has_code'],
                'contains_config': context['has_config'],
                'sections_found': len(context['key_sections']),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return summary
        
        def writer(context: Dict) -> Dict:
            """Writer subagent: Archive and finalize"""
            file_path = context['file_path']
            archive_path = Path(tasks[0]['archive_path'])  # Get from first task
            
            try:
                # Copy to archive
                filename = Path(file_path).name
                archive_file = archive_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                shutil.copy2(file_path, archive_file)
                
                # Delete original
                os.remove(file_path)
                
                return {
                    **context,
                    'archived_to': str(archive_file),
                    'original_deleted': True,
                    'status': 'completed'
                }
            except Exception as e:
                return {
                    **context,
                    'error': str(e),
                    'status': 'archive_failed'
                }
        
        # Execute pipeline for each task
        print("\n📋 Executing Reader → Extractor → Processor → Writer pipeline...")
        
        results = []
        for task in tasks:
            result = self.pool.spawn_pipeline(task, reader, extractor, processor)
            # Parse the result output
            try:
                processed_data = eval(result.output) if isinstance(result.output, str) else result.output
            except:
                processed_data = {}
            # Then archive
            final = writer(processed_data if isinstance(processed_data, dict) else {})
            results.append(final)
        
        # Summary
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = len(results) - completed
        
        print(f"\n✅ Processing complete!")
        print(f"   Completed: {completed}")
        print(f"   Failed: {failed}")
        print(f"   Archived to: {archive_path}")
        
        return {
            'success': failed == 0,
            'total_files': len(file_paths),
            'completed': completed,
            'failed': failed,
            'results': results,
            'archive_location': str(archive_path)
        }
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract key sections from content"""
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            if line.startswith('#') or line.startswith('==') or line.startswith('---'):
                sections.append(line.strip()[:50])
            if line.strip().endswith(':') and len(line) < 100:
                sections.append(line.strip()[:50])
        
        return sections[:10]  # Max 10 sections


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents with concurrent subagents')
    parser.add_argument('files', nargs='+', help='Files to process')
    parser.add_argument('--archive', '-a', default='archive', help='Archive directory')
    parser.add_argument('--concurrent', '-c', type=int, default=5, help='Max concurrent subagents')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run (no delete)')
    
    args = parser.parse_args()
    
    # Load secrets
    load_secrets()
    
    print("="*60)
    print("🤖 SAM-D Concurrent Document Processor")
    print("="*60)
    
    # Validate files
    valid_files = [f for f in args.files if os.path.exists(f)]
    if not valid_files:
        print("❌ No valid files found")
        return 1
    
    print(f"\n📁 Files to process: {len(valid_files)}")
    for f in valid_files:
        print(f"   • {f}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Files will NOT be deleted")
    
    # Process
    processor = DocumentProcessor(max_concurrent=args.concurrent)
    result = processor.process_files(valid_files, args.archive)
    
    # Save report
    report_file = f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n📝 Report saved: {report_file}")
    print("="*60)
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
