#!/bin/bash

echo "🔍 VALIDATING TEXT DATA"
echo "=================="

# Validate vocabulary format
echo ""
echo "📝 Validating vocabulary..."
python3 -c "
import sys
issues = []
with open('TEXT_DATA/VOCABULARY/stage2_vocabulary.txt', 'r') as f:
    for i, line in enumerate(f, 1):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            issues.append(f'Line {i}: Invalid format - {line.strip()}')
        elif not parts[0].isalpha():
            issues.append(f'Line {i}: Non-alphabetic word - {parts[0]}')
        elif not parts[1].isdigit():
            issues.append(f'Line {i}: Invalid frequency - {parts[1]}')

if issues:
    print(f'❌ Found {len(issues)} vocabulary issues:')
    for issue in issues[:10]:
        print(f'  {issue}')
    if len(issues) > 10:
        print(f'  ... and {len(issues) - 10} more issues')
else:
    print('✅ Vocabulary format is valid')

# Validate phrases
echo ""
echo "📖 Validating phrases..."
issues = []
with open('TEXT_DATA/PHRASES/stage3_phrases.txt', 'r') as f:
    for i, line in enumerate(f, 1):
        if len(line.strip()) < 3:
            issues.append(f'Line {i}: Too short phrase')
        elif len(line.strip()) > 500:
            issues.append(f'Line {i}: Too long phrase')

if issues:
    print(f'❌ Found {len(issues)} phrase issues:')
    for issue in issues[:10]:
        print(f'  {issue}')
    if len(issues) > 10:
        print(f'  ... and {len(issues) - 10} more issues')
else:
    print('✅ Phrases are valid')

# Validate collocations
echo ""
echo "🔗 Validating collocations..."
issues = []
with open('TEXT_DATA/COLLOCATIONS/stage3_collocations.txt', 'r') as f:
    for i, line in enumerate(f, 1):
        parts = line.strip().split()
        if len(parts) < 2:
            issues.append(f'Line {i}: Invalid format - {line.strip()}')
        elif not all(part.isalpha() for part in parts):
            issues.append(f'Line {i}: Non-alphabetic words - {line.strip()}')

if issues:
    print(f'❌ Found {len(issues)} collocation issues:')
    for issue in issues[:10]:
        print(f'  {issue}')
    if len(issues) > 10:
        print(f'  ... and {len(issues) - 10} more issues')
else:
    print('✅ Collocations are valid')

# Validate training data
echo ""
echo "🎯 Validating training data..."
if [ -f "TEXT_DATA/TRAINING/training_data.csv" ]; then
    issues = []
    with open('TEXT_DATA/TRAINING/training_data.csv', 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('"'):
                issues.append(f'Line {i}: Invalid CSV format - {line}')
            elif line.count('"') != 2:
                issues.append(f'Line {i}: Missing quotes - {line}')
            elif line.count(',') != 1:
                issues.append(f'Line {i}: Extra commas - {line}')

    if issues:
        print(f'❌ Found {len(issues)} training data issues:')
        for issue in issues[:10]:
            print(f'  {issue}')
        if len(issues) > 10:
            print(f'  ... and {len(issues) - 10} more issues')
    else:
        print('✅ Training data is valid')
else:
    print('⚠️  Training data file not found')

# Validate model data
echo ""
echo "🤖 Validating model data..."
if [ -f "TEXT_DATA/MODELS/hf_training_data.json" ]; then
    python3 -c "
import json
try:
    with open('TEXT_DATA/MODELS/hf_training_data.json', 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            print(f'✅ Model data is valid ({len(data)} entries)')
        else:
            print('❌ Model data is not a list')
except Exception as e:
    print(f'❌ Model data validation failed: {e}')
else:
    print('⚠️  Model data file not found')

echo ""
echo "📊 File sizes:"
ls -lh TEXT_DATA/*

echo ""
echo "📈 Total statistics:"
echo "  Total size: $(du -sh TEXT_DATA | cut -f1)"
echo "  Total files: $(find TEXT_DATA -type f | wc -l)"
echo "  Directories: $(find TEXT_DATA -type d | wc -l)"

echo ""
echo "✅ Validation complete!"
