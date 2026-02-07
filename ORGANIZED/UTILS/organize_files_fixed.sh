#!/bin/bash

echo "=== ORGANIZING NN_C DIRECTORY STRUCTURE ==="

# Create organized directory structure
mkdir -p ORGANIZED/{CHATBOT,MODELS,TRAINING,UTILS,DOCS,WEB,TESTS,CODE}
mkdir -p ORGANIZED/MODELS/{STAGE1,STAGE2,STAGE3,STAGE4,STAGE5}
mkdir -p ORGANIZED/CHATBOT/{TERMINAL,WEB,DEMO}
mkdir -p ORGANIZED/UTILS/{SAM,NEAT,TRANSFORMER,NN}
mkdir -p ORGANIZED/DOCS/{REPORTS,PLANS,ARCHITECTURE}

echo "✅ Directory structure created"

# Move documentation files
echo "📚 Moving documentation files..."
find . -maxdepth 1 -name "*.md" -exec mv {} ORGANIZED/DOCS/ \; 2>/dev/null || true

# Move chatbot files
echo "🤖 Moving chatbot files..."
find . -maxdepth 1 -name "full_llm_chatbot*" -exec mv {} ORGANIZED/CHATBOT/TERMINAL/ \; 2>/dev/null || true
find . -maxdepth 1 -name "web_chatbot.html" -exec mv {} ORGANIZED/CHATBOT/WEB/ \; 2>/dev/null || true
find . -maxdepth 1 -name "web_server*" -exec mv {} ORGANIZED/CHATBOT/WEB/ \; 2>/dev/null || true
find . -maxdepth 1 -name "demo_chatbot_system*" -exec mv {} ORGANIZED/CHATBOT/DEMO/ \; 2>/dev/null || true

# Move stage 1 files
echo "📝 Moving Stage 1 files..."
find . -maxdepth 1 -name "stage1_*" -exec mv {} ORGANIZED/MODELS/STAGE1/ \; 2>/dev/null || true
find . -maxdepth 1 -name "stage1_basic*" -exec mv {} ORGANIZED/MODELS/STAGE1/ \; 2>/dev/null || true
find . -maxdepth 1 -name "stage1_conservative*" -exec mv {} ORGANIZED/MODELS/STAGE1/ \; 2>/dev/null || true
find . -maxdepth 1 -name "stage1_continuous*" -exec mv {} ORGANIZED/MODELS/STAGE1/ \; 2>/dev/null || true
find . -maxdepth 1 -name "stage1_fixed*" -exec mv {} ORGANIZED/MODELS/STAGE1/ \; 2>/dev/null || true

# Move stage 2 files
echo "📝 Moving Stage 2 files..."
find . -maxdepth 1 -name "stage2_*" -exec mv {} ORGANIZED/MODELS/STAGE2/ \; 2>/dev/null || true

# Move stage 3 files
echo "📝 Moving Stage 3 files..."
find . -maxdepth 1 -name "stage3_*" -exec mv {} ORGANIZED/MODELS/STAGE3/ \; 2>/dev/null || true

# Move stage 4 files
echo "📝 Moving Stage 4 files..."
find . -maxdepth 1 -name "stage4_*" -exec mv {} ORGANIZED/MODELS/STAGE4/ \; 2>/dev/null || true

# Move stage 5 files
echo "📝 Moving Stage 5 files..."
find . -maxdepth 1 -name "stage5_*" -exec mv {} ORGANIZED/MODELS/STAGE5/ \; 2>/dev/null || true

# Move stage 6 files
echo "📝 Moving Stage 6 files..."
find . -maxdepth 1 -name "stage6_*" -exec mv {} ORGANIZED/MODELS/STAGE5/ \; 2>/dev/null || true

# Move training files
echo "🎯 Moving training files..."
find . -maxdepth 1 -name "train*.sh" -exec mv {} ORGANIZED/TRAINING/ \; 2>/dev/null || true
find . -maxdepth 1 -name "*training*.c" -exec mv {} ORGANIZED/TRAINING/ \; 2>/dev/null || true
find . -maxdepth 1 -name "prepare_training_data.sh" -exec mv {} ORGANIZED/TRAINING/ \; 2>/dev/null || true
find . -maxdepth 1 -name "monitor_training.sh" -exec mv {} ORGANIZED/TRAINING/ \; 2>/dev/null || true

# Move test files
echo "🧪 Moving test files..."
find . -maxdepth 1 -name "test_*" -exec mv {} ORGANIZED/TESTS/ \; 2>/dev/null || true
find . -maxdepth 1 -name "*test*.c" -exec mv {} ORGANIZED/TESTS/ \; 2>/dev/null || true

# Move utility files
echo "🔧 Moving utility files..."
if [ -d "SAM" ]; then
    mv SAM ORGANIZED/UTILS/SAM/ 2>/dev/null || true
fi
if [ -d "utils" ]; then
    mv utils ORGANIZED/UTILS/ 2>/dev/null || true
fi

# Move miscellaneous files
echo "📦 Moving miscellaneous files..."
find . -maxdepth 1 -name "debug_*" -exec mv {} ORGANIZED/UTILS/ \; 2>/dev/null || true
find . -maxdepth 1 -name "demo_*" -not -name "demo_chatbot_system*" -exec mv {} ORGANIZED/UTILS/ \; 2>/dev/null || true
find . -maxdepth 1 -name "sam_*" -not -name "sam_agi" -not -name "sam_chatbot_*" -not -name "sam_checkpoint_*" -not -name "sam_hf_*" -not -name "sam_production_*" -not -name "sam_text_*" -not -name "sam_trained_*" -exec mv {} ORGANIZED/UTILS/ \; 2>/dev/null || true

# Keep main executables and important files in root
echo "🚀 Keeping important files in root..."

# Create organized README
cat > ORGANIZED/README.md << 'EOF'
# NN_C - Organized Directory Structure

## 📁 Directory Structure

### 🤖 CHATBOT/
- **TERMINAL/** - Terminal-based chatbot interface
- **WEB/** - Web-based chatbot interface and server
- **DEMO/** - Demo and testing system

### 📝 MODELS/
- **STAGE1/** - Character-level learning models
- **STAGE2/** - Word-level learning models
- **STAGE3/** - Phrase-level learning models
- **STAGE4/** - Response generation models
- **STAGE5/** - Advanced AGI components

### 🎯 TRAINING/
- Training scripts and programs
- Data preparation tools
- Monitoring utilities

### 🔧 UTILS/
- **SAM/** - SAM framework
- **NEAT/** - NEAT neural evolution
- **TRANSFORMER/** - Transformer models
- **NN/** - Neural network utilities

### 📚 DOCS/
- **REPORTS/** - Completion reports and documentation
- **PLANS/** - Development plans and roadmaps
- **ARCHITECTURE/** - System architecture documentation

### 🧪 TESTS/
- Test programs and utilities
- Validation scripts

### 🌐 WEB/
- Web interface files
- HTML, CSS, JavaScript

### 💻 CODE/
- Source code files
- Development utilities

## 🚀 Quick Start

### Terminal Chatbot
```bash
cd ORGANIZED/CHATBOT/TERMINAL
./full_llm_chatbot
```

### Web Chatbot
```bash
cd ORGANIZED/CHATBOT/WEB
./web_server
# Open browser to http://localhost:8080
```

### Demo System
```bash
cd ORGANIZED/CHATBOT/DEMO
./demo_chatbot_system
```

## 📊 Model Status
- ✅ Stage 1: Character Model - LOADED
- ✅ Stage 2: Word Model - LOADED
- ✅ Stage 3: Phrase Model - LOADED
- ✅ Stage 4: Response Model - LOADED
- ✅ Stage 5: Advanced AGI - LOADED

## 🎯 Progressive Learning
Characters → Words → Phrases → Responses → Advanced AGI
EOF

echo "✅ Organization complete!"
echo "📁 Organized structure created in ORGANIZED/"
echo "📚 Documentation moved to ORGANIZED/DOCS/"
echo "🤖 Chatbot files moved to ORGANIZED/CHATBOT/"
echo "📝 Model files moved to ORGANIZED/MODELS/"
echo "🎯 Training files moved to ORGANIZED/TRAINING/"
echo "🔧 Utilities moved to ORGANIZED/UTILS/"
echo "🧪 Test files moved to ORGANIZED/TESTS/"
