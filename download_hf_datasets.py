#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader for SAM 2.0 Training
Downloads and processes story datasets from HuggingFace
"""

import os
import json
import subprocess
from pathlib import Path


class HuggingFaceDatasetDownloader:
    """Downloads story datasets from HuggingFace for training"""
    
    DATASETS = {
        'tinystories': {
            'name': 'roneneldan/TinyStories',
            'split': 'train',
            'text_field': 'text',
            'max_samples': 1000,  # Limit for processing time
            'output_file': 'TinyStories_sample.txt'
        },
        'gutenberg': {
            'name': 'storytracer/US-PD-Books',
            'split': 'train',
            'text_field': 'full_text',
            'max_samples': 100,
            'output_file': 'US_PD_Books_sample.txt'
        }
    }
    
    def __init__(self, output_dir='TEXT_DATA/HF_DATASETS'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_datasets_library(self):
        """Check if datasets library is installed"""
        try:
            import datasets
            return True
        except ImportError:
            print("⚠️ HuggingFace datasets library not installed")
            print("   Install with: pip install datasets")
            return False
    
    def download_dataset(self, dataset_key):
        """Download a specific dataset"""
        if not self.check_datasets_library():
            return False
        
        from datasets import load_dataset
        
        config = self.DATASETS[dataset_key]
        print(f"\n📥 Downloading {config['name']}...")
        
        try:
            # Load dataset
            ds = load_dataset(config['name'], split=config['split'], streaming=True)
            
            # Collect samples
            samples = []
            for i, item in enumerate(ds):
                if i >= config['max_samples']:
                    break
                
                text = item.get(config['text_field'], '')
                if len(text) > 500:  # Only substantial stories
                    samples.append(text)
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i+1} samples...")
            
            # Save to file
            output_file = self.output_dir / config['output_file']
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n\n===STORY_SEPARATOR===\n\n'.join(samples))
            
            print(f"✅ Saved {len(samples)} stories to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_key}: {e}")
            return False
    
    def download_all(self):
        """Download all configured datasets"""
        print("="*70)
        print("🤗 HuggingFace Dataset Downloader")
        print("="*70)
        
        if not self.check_datasets_library():
            print("\n💡 To install: pip install datasets")
            return False
        
        success_count = 0
        for dataset_key in self.DATASETS:
            if self.download_dataset(dataset_key):
                success_count += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Downloaded {success_count}/{len(self.DATASETS)} datasets")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*70)
        
        return success_count > 0
    
    def list_downloaded(self):
        """List downloaded datasets"""
        files = list(self.output_dir.glob('*.txt'))
        print(f"\n📚 Downloaded datasets in {self.output_dir}:")
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"   - {f.name}: {size:.1f} KB")
        return files


def main():
    """Main entry point"""
    downloader = HuggingFaceDatasetDownloader()
    
    # Download all datasets
    if downloader.download_all():
        downloader.list_downloaded()
        print("\n🎉 Datasets ready for training!")
        print("   Run: python3 synthetic_training_generator.py")
    else:
        print("\n⚠️ Some datasets failed to download")
        print("   You can still use local story files")


if __name__ == "__main__":
    main()
