#!/bin/bash

echo "📊 TEXT DATA STATISTICS"
echo "======================"

echo ""
echo "📝 Vocabulary:"
echo "  Words: $(wc -l < TEXT_DATA/VOCABULARY/stage2_vocabulary.txt)"
echo "  Size: $(du -h TEXT_DATA/VOCABULARY/stage2_vocabulary.txt | cut -f1)"
echo ""

echo "📖 Phrases:"
echo "  Phrases: $(wc -l < TEXT_DATA/PHRASES/stage3_phrases.txt)"
echo "  Size: $(du -h TEXT_DATA/PHRASES/stage3_phrases.txt | cut -f1)"
echo ""

echo "🔗 Collocations:"
echo "  Lines: $(wc -l < TEXT_DATA/COLLOCATIONS/stage3_collocations.txt)"
echo "  Size: $(du -h TEXT_DATA/COLLOCATIONS/stage3_collocations.txt | cut -f1)"
echo ""

echo "🎯 Training:"
echo "  Files: $(find TEXT_DATA/TRAINING -name '*.txt' -o -name '*.csv' | wc -l)"
echo "  Size: $(du -sh TEXT_DATA/TRAINING | cut -f1)"
echo ""

echo "🤖 Models:"
echo "  Files: $(find TEXT_DATA/MODELS -name '*.json' -o -name '*.txt' | wc -l)"
echo "  Size: $(du -sh TEXT_DATA/MODELS | cut -f1)"
echo ""

echo "📊 Total:"
echo "  Size: $(du -sh TEXT_DATA | cut -f1)"
echo "  Files: $(find TEXT_DATA -type f | wc -l)"
echo "  Directories: $(find TEXT_DATA -type d | wc -l)"

echo ""
echo "🔍 Sample content:"
echo ""
echo "📝 Vocabulary sample:"
head -5 TEXT_DATA/VOCABULARY/stage2_vocabulary.txt
echo ""
echo "📖 Phrases sample:"
head -3 TEXT_DATA/PHRASES/stage3_phrases.txt
echo ""
echo "🔗 Collocations sample:"
head -3 TEXT_DATA/COLLOCATIONS/stage3_collocations.txt
echo ""
echo "🎯 Training data sample:"
if [ -f "TEXT_DATA/TRAINING/training_data.csv" ]; then
    head -3 TEXT_DATA/TRAINING/training_data.csv
else
    echo "No training data.csv found"
fi
