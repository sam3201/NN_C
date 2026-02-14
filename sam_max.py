#!/usr/bin/env python3
"""
SAM-D MAX Utility - Default/Max Setup
One-command automation for processing LATEST logs
Auto-detects, processes with max subagents, archives, deletes
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "automation_framework" / "python"))

from automation_bridge import SubagentPool
from run_sam import load_secrets

class SAMMaxProcessor:
    """MAX utility - everything enabled by default"""
    
    def __init__(self):
        self.pool = SubagentPool(max_concurrent=10)  # MAX concurrent
        self.archive_dir = ROOT / "DOCS" / "archive" / "chatlogs"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        load_secrets()
        
    def find_latest_log(self, pattern="ChatGPT_*_LATEST.txt") -> Path:
        """Auto-find latest log file"""
        # Search in common locations
        search_paths = [
            ROOT,
            ROOT / "DOCS",
            ROOT / "logs",
            Path.home() / "Downloads",
            Path.cwd()
        ]
        
        latest_file = None
        latest_time = 0
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            files = list(search_path.glob(pattern))
            for f in files:
                mtime = f.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = f
        
        return latest_file
    
    def process_latest(self, file_path: str = None) -> dict:
        """Process with MAX settings - everything enabled"""
        
        # Auto-find if not provided
        if not file_path:
            latest = self.find_latest_log()
            if not latest:
                print("❌ No LATEST log file found")
                print("   Searched for: ChatGPT_*_LATEST.txt")
                return {"success": False, "error": "No file found"}
            file_path = str(latest)
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return {"success": False, "error": "File not found"}
        
        print("="*70)
        print("🚀 SAM-D MAX UTILITY - Full Automation")
        print("="*70)
        print(f"\n📁 Target: {file_path.name}")
        print(f"📊 Size: {file_path.stat().st_size:,} bytes")
        print(f"⚡ Max Concurrent Subagents: 10")
        print(f"📦 Archive: {self.archive_dir}")
        print(f"🗑️  Auto-delete: ENABLED")
        print(f"🔍 Deep Extraction: ENABLED")
        
        # Read file
        print(f"\n📖 Reading file...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"   ✅ Loaded: {len(content):,} characters")
        
        # Process with subagents
        print(f"\n🤖 Processing with concurrent subagents...")
        
        # Split into chunks for parallel processing
        chunk_size = max(len(content) // 10, 10000)  # ~10 chunks or min 10k
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        print(f"   Split into {len(chunks)} chunks for parallel processing")
        
        # Create tasks
        tasks = [
            {
                'chunk_id': i,
                'content': chunk[:50000],  # Max 50k per chunk
                'file_name': file_path.name
            }
            for i, chunk in enumerate(chunks[:10])  # Max 10 chunks
        ]
        
        # Process handler
        def process_chunk(task: dict) -> dict:
            chunk_id = task['chunk_id']
            content = task['content']
            
            # Extract information
            lines = content.split('\n')
            
            extraction = {
                'chunk_id': chunk_id,
                'line_count': len(lines),
                'word_count': len(content.split()),
                'char_count': len(content),
                'key_sections': [],
                'code_blocks': [],
                'urls': [],
                'timestamps': []
            }
            
            # Extract key info
            for line in lines[:100]:  # Check first 100 lines
                # Headers/sections
                if line.startswith('#') or line.startswith('==') or '---' in line:
                    extraction['key_sections'].append(line.strip()[:100])
                
                # Code blocks
                if '```' in line or line.strip().startswith('def ') or line.strip().startswith('class '):
                    extraction['code_blocks'].append(line.strip()[:100])
                
                # URLs
                if 'http' in line:
                    urls = [w for w in line.split() if w.startswith('http')]
                    extraction['urls'].extend(urls[:5])
                
                # Timestamps
                if ':' in line and any(c.isdigit() for c in line):
                    if len(line) < 50:
                        extraction['timestamps'].append(line.strip()[:50])
            
            return extraction
        
        # Run parallel processing
        results = self.pool.spawn_parallel(tasks, process_chunk)
        
        # Aggregate results
        print(f"   ✅ Processed {len(results)} chunks")
        
        total_lines = sum(r.output.get('line_count', 0) for r in results if isinstance(r.output, dict))
        total_words = sum(r.output.get('word_count', 0) for r in results if isinstance(r.output, dict))
        all_sections = []
        all_urls = []
        
        for r in results:
            if isinstance(r.output, dict):
                all_sections.extend(r.output.get('key_sections', []))
                all_urls.extend(r.output.get('urls', []))
        
        # Archive
        print(f"\n📦 Archiving...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f"{timestamp}_{file_path.name}"
        archive_path = self.archive_dir / archive_name
        
        shutil.copy2(file_path, archive_path)
        print(f"   ✅ Archived to: {archive_path}")
        
        # Delete original
        print(f"\n🗑️  Deleting original...")
        os.remove(file_path)
        print(f"   ✅ Deleted: {file_path}")
        
        # Generate summary
        summary = {
            'success': True,
            'file_processed': file_path.name,
            'archive_location': str(archive_path),
            'processing_timestamp': datetime.now().isoformat(),
            'stats': {
                'total_characters': len(content),
                'total_lines': total_lines,
                'total_words': total_words,
                'chunks_processed': len(results),
                'key_sections_found': len(set(all_sections)),
                'urls_found': len(set(all_urls))
            },
            'extracted_sections': list(set(all_sections))[:20],  # Top 20
            'extracted_urls': list(set(all_urls))[:10]  # Top 10
        }
        
        # Save report
        report_file = self.archive_dir / f"{timestamp}_report.json"
        with open(report_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"\n📝 Summary Report:")
        print(f"   📊 Characters: {summary['stats']['total_characters']:,}")
        print(f"   📄 Lines: {summary['stats']['total_lines']:,}")
        print(f"   📝 Words: {summary['stats']['total_words']:,}")
        print(f"   🔍 Sections: {summary['stats']['key_sections_found']}")
        print(f"   🌐 URLs: {summary['stats']['urls_found']}")
        print(f"   📦 Archive: {archive_path.name}")
        print(f"   📄 Report: {report_file.name}")
        
        print("\n" + "="*70)
        print("✅ SAM-D MAX PROCESSING COMPLETE")
        print("="*70)
        
        return summary


def main():
    """Main entry - simple command"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SAM-D MAX - Auto-process LATEST logs with everything enabled',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find and process latest log
  python sam_max.py
  
  # Process specific file
  python sam_max.py /path/to/ChatGPT_2026-02-13_12-00-08_LATEST.txt
  
  # Process with custom archive
  python sam_max.py file.txt --archive /custom/archive/path
        """
    )
    
    parser.add_argument('file', nargs='?', help='File to process (auto-detects if not provided)')
    parser.add_argument('--archive', '-a', help='Custom archive directory')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run (no deletion)')
    
    args = parser.parse_args()
    
    processor = SAMMaxProcessor()
    
    if args.archive:
        processor.archive_dir = Path(args.archive)
        processor.archive_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - Files will not be deleted")
    
    result = processor.process_latest(args.file)
    
    if result['success']:
        print("\n🎉 All done! File processed, archived, and deleted.")
        return 0
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
