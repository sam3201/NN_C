#!/usr/bin/env python3
"""
Synthetic Training Data Generator for SAM 2.0
Generates Q&A pairs from story text files using LLM (Ollama)
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict
import re


class SyntheticTrainingGenerator:
    """Generates training data from story text files"""
    
    def __init__(self, story_dirs=None, output_dir='SAM_TRAINING_DATA'):
        """Initialize the training generator
        
        Args:
            story_dirs: List of directories containing .txt story files
            output_dir: Where to save generated training data
        """
        if story_dirs is None:
            # Default locations based on scan
            story_dirs = [
                '/Users/samueldasari/Personal/NN_C/ORGANIZED/TRAINING/training_data/raw_texts',
                '/Users/samueldasari/Personal/NN_C/ORGANIZED/UTILS/utils/DATASETS',
                '/Users/samueldasari/Personal/NN_C/TEXT_DATA/TRAINING/training_data/raw_texts',  # Found new story here
                '/Users/samueldasari/Personal/NN_C/TEXT_DATA'
            ]
        
        self.story_dirs = [Path(d) for d in story_dirs if os.path.exists(d)]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check Ollama availability
        self.ollama_available = self._check_ollama()
        
        # Statistics
        self.stats = {
            'stories_processed': 0,
            'qa_pairs_generated': 0,
            'conversations_created': 0,
            'errors': 0
        }
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ['which', 'ollama'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def find_story_files(self) -> List[Path]:
        """Find all .txt files in story directories"""
        story_files = []
        
        for story_dir in self.story_dirs:
            if story_dir.exists():
                for txt_file in story_dir.glob('*.txt'):
                    # Skip non-story files (config, README, etc.)
                    if self._is_story_file(txt_file):
                        story_files.append(txt_file)
        
        # Remove duplicates (same filename)
        seen = set()
        unique_files = []
        for f in story_files:
            if f.name not in seen:
                seen.add(f.name)
                unique_files.append(f)
        
        return unique_files
    
    def _is_story_file(self, filepath: Path) -> bool:
        """Check if file is likely a story (not config/code)"""
        # Skip files with certain patterns
        skip_patterns = [
            'CMake', 'LICENSE', 'README', 'requirements', 
            'config', 'words.txt', 'input.txt', 'OFL.txt',
            'link.txt', 'Makefile', 'stage', 'collocations',
            'phrases', 'vocabulary', 'prompt', 'npes_saved'
        ]
        
        name_lower = filepath.name.lower()
        for pattern in skip_patterns:
            if pattern.lower() in name_lower:
                return False
        
        # Check if file is reasonable size for a story (1KB - 1MB)
        size = filepath.stat().st_size
        return 1000 < size < 1_000_000
    
    def read_story(self, filepath: Path) -> str:
        """Read story text from file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")
            return ""
    
    def generate_qa_pairs(self, story_text: str, num_questions: int = 5) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from story using LLM
        
        Args:
            story_text: The story content
            num_questions: How many Q&A pairs to generate
            
        Returns:
            List of (question, answer) tuples
        """
        if not self.ollama_available:
            print("⚠️ Ollama not available, using fallback Q&A generation")
            return self._fallback_qa_generation(story_text)
        
        # Split story into chunks if too long (OLLAMA context limit ~4k tokens)
        max_chars = 2000  # Reduced from 3000 for better processing
        chunks = self._chunk_story(story_text, max_chars)
        
        all_qa_pairs = []
        
        # Process more chunks for more comprehensive Q&A (up to 5 chunks)
        chunks_to_process = min(len(chunks), 5)
        for i, chunk in enumerate(chunks[:chunks_to_process]):
            print(f"   📝 Processing chunk {i+1}/{chunks_to_process}...")
            
            # Generate 2 Q&A per chunk = up to 10 total per story
            chunk_qa = self._generate_qa_for_chunk(chunk, num_questions=2)
            all_qa_pairs.extend(chunk_qa)
            
            # Delay between chunks to prevent overwhelming Ollama
            if i < chunks_to_process - 1:
                time.sleep(2)
        
        # If we got good Q&A from chunks, use them (target 10)
        if len(all_qa_pairs) >= 5:
            return all_qa_pairs[:10]  # Return up to 10 Q&A pairs
        
        # Fallback if chunks didn't work
        print("⚠️ Chunk processing failed, using fallback")
        return self._fallback_qa_generation(story_text)
    
    def _chunk_story(self, story_text: str, chunk_size: int) -> List[str]:
        """Split story into manageable chunks at sentence boundaries"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', story_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_qa_for_chunk(self, chunk: str, num_questions: int = 2) -> List[Tuple[str, str]]:
        """Generate Q&A for a single story chunk with longer timeout"""
        prompt = f"""Read this story excerpt and generate {num_questions} factual questions that can be answered directly from the text. Then provide the answers.

Story excerpt:
{chunk[:1500]}

Format each Q&A pair exactly as:
Q: [question]
A: [answer from the story]

Generate {num_questions} pairs."""
        
        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama2', prompt],
                capture_output=True,
                text=True,
                timeout=120  # Increased from 60 to 120 seconds
            )
            
            if result.returncode != 0:
                print(f"⚠️ Ollama chunk error: {result.stderr[:100]}")
                return []
            
            return self._parse_qa_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            print("⚠️ Ollama chunk timeout (120s)")
            return []
        except Exception as e:
            print(f"⚠️ Chunk error: {e}")
            return []
    
    def _parse_qa_output(self, output: str) -> List[Tuple[str, str]]:
        """Parse Q&A pairs from LLM output"""
        qa_pairs = []
        
        # Split by lines and look for Q: and A: patterns
        lines = output.strip().split('\n')
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('Question:'):
                # Save previous pair if exists
                if current_q and current_a:
                    qa_pairs.append((current_q, current_a))
                # Start new question
                current_q = line.split(':', 1)[1].strip()
                current_a = None
            elif line.startswith('A:') or line.startswith('Answer:'):
                current_a = line.split(':', 1)[1].strip()
        
        # Don't forget the last pair
        if current_q and current_a:
            qa_pairs.append((current_q, current_a))
        
        return qa_pairs
    
    def _fallback_qa_generation(self, story_text: str) -> List[Tuple[str, str]]:
        """Fallback: Generate simple Q&A without LLM"""
        qa_pairs = []
        
        # Get first few sentences
        sentences = re.split(r'[.!?]+', story_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) >= 3:
            # Q1: First sentence
            qa_pairs.append((
                "What is the opening of the story?",
                sentences[0][:200] + ("..." if len(sentences[0]) > 200 else "")
            ))
            
            # Q2: Main characters
            qa_pairs.append((
                "What happens at the beginning?",
                sentences[1][:200] + ("..." if len(sentences[1]) > 200 else "")
            ))
            
            # Q3: Setting
            qa_pairs.append((
                "What is described in the story?",
                sentences[2][:200] + ("..." if len(sentences[2]) > 200 else "")
            ))
        
        return qa_pairs
    
    def create_conversation(self, qa_pairs: List[Tuple[str, str]], source: str) -> List[Dict]:
        """Convert Q&A pairs into conversation format
        
        Args:
            qa_pairs: List of (question, answer) tuples
            source: Source file name
            
        Returns:
            List of conversation turns
        """
        conversation = []
        timestamp = time.time()
        
        for i, (q, a) in enumerate(qa_pairs):
            # User asks question
            conversation.append({
                'id': f'{source}_{i}_q',
                'sender': 'You',
                'message': q,
                'timestamp': time.strftime('%H:%M:%S'),
                'ts': timestamp + i * 2,
                'agent_id': 'user',
                'color': '#888888'
            })
            
            # SAM responds
            conversation.append({
                'id': f'{source}_{i}_a',
                'sender': 'SAM',
                'message': a,
                'timestamp': time.strftime('%H:%M:%S'),
                'ts': timestamp + i * 2 + 1,
                'agent_id': 'sam_conversation',
                'color': '#3498db',
                'verified': True,
                'source': source
            })
        
        return conversation
    
    def save_training_data(self, conversation: List[Dict], source: str):
        """Save conversation to training data file"""
        output_file = self.output_dir / f'synthetic_{source}.jsonl'
        
        with open(output_file, 'a') as f:
            for turn in conversation:
                f.write(json.dumps(turn) + '\n')
        
        print(f"💾 Saved {len(conversation)} turns to {output_file}")
    
    def process_all_stories(self):
        """Process all story files and generate training data"""
        story_files = self.find_story_files()
        
        print(f"\n{'='*70}")
        print(f"🚀 Synthetic Training Data Generator")
        print(f"{'='*70}")
        print(f"📚 Found {len(story_files)} story files")
        print(f"🤖 Ollama available: {self.ollama_available}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        for i, story_file in enumerate(story_files, 1):
            print(f"\n[{i}/{len(story_files)}] Processing: {story_file.name}")
            
            # Read story
            story_text = self.read_story(story_file)
            if not story_text:
                self.stats['errors'] += 1
                continue
            
            print(f"   📖 Story length: {len(story_text):,} characters")
            
            # Generate Q&A - INCREASED to 10 per story for more training data
            qa_pairs = self.generate_qa_pairs(story_text, num_questions=10)
            print(f"   ❓ Generated {len(qa_pairs)} Q&A pairs")
            
            if len(qa_pairs) == 0:
                self.stats['errors'] += 1
                continue
            
            # Create conversation
            conversation = self.create_conversation(qa_pairs, story_file.stem)
            print(f"   💬 Created {len(conversation)} conversation turns")
            
            # Save training data
            self.save_training_data(conversation, story_file.stem)
            
            # Update stats
            self.stats['stories_processed'] += 1
            self.stats['qa_pairs_generated'] += len(qa_pairs)
            self.stats['conversations_created'] += 1
            
            # Small delay between stories
            time.sleep(0.5)
        
        # Print final stats
        print(f"\n{'='*70}")
        print("📊 Training Generation Complete")
        print(f"{'='*70}")
        print(f"✅ Stories processed: {self.stats['stories_processed']}")
        print(f"✅ Q&A pairs generated: {self.stats['qa_pairs_generated']}")
        print(f"✅ Conversations created: {self.stats['conversations_created']}")
        print(f"⚠️ Errors: {self.stats['errors']}")
        print(f"{'='*70}\n")
        
        # Create summary file
        self._create_summary()
    
    def _create_summary(self):
        """Create summary of generated training data"""
        summary_file = self.output_dir / 'synthetic_training_summary.json'
        
        summary = {
            'timestamp': time.time(),
            'stats': self.stats,
            'story_dirs': [str(d) for d in self.story_dirs],
            'ollama_available': self.ollama_available,
            'generated_files': [
                f.name for f in self.output_dir.glob('synthetic_*.jsonl')
            ]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved to: {summary_file}")


def main():
    """Main entry point"""
    generator = SyntheticTrainingGenerator()
    generator.process_all_stories()
    
    print("\n🎉 Training data generation complete!")
    print(f"📁 Check {generator.output_dir}/ for generated files")
    print("\nTo use for training:")
    print("  1. Load synthetic_*.jsonl files")
    print("  2. Run: python3 correct_sam_hub.py")
    print("  3. Training will use this data automatically")


if __name__ == "__main__":
    main()
