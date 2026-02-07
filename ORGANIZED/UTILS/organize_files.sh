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
mv *.md ORGANIZED/DOCS/ 2>/dev/null || true

# Move chatbot files
echo "🤖 Moving chatbot files..."
mv full_llm_chatbot* ORGANIZED/CHATBOT/TERMINAL/ 2>/dev/null || true
mv web_chatbot.html ORGANIZED/CHATBOT/WEB/ 2>/dev/null || true
mv web_server* ORGANIZED/CHATBOT/WEB/ 2:/dev/null || true
mv demo_chatbot_system* ORGANIZED/CHATBOT/DEMO/ 2>/dev/null || true

# Move stage 1 files
echo "📝 Moving Stage 1 files..."
mv stage1_* ORGANIZED/MODELS/STAGE1/ 2>/dev/null || true
mv stage1_basic* ORGANIZED/MODELS/STAGE1/ 2>/dev/null || true
mv stage1_conservative* ORGANIZED/MODELS/STAGE1/ 2>/dev/null || true
mv stage1_continuous* ORGANIZED/MODELS/STAGE1/ 2>/dev/null || true
mv stage1_fixed* ORGANIZED/MODELS/STAGE1/ 2>/dev/null || true

# Move stage 2 files
echo "📝 Moving Stage 2 files..."
mv stage2_* ORGANIZED/MODELS/STAGE2/ 2>/dev/null || true

# Move stage 3 files
echo "📝 Moving Stage 3 files..."
mv stage3_* ORGANIZED/MODELS/STAGE3/ 2>/dev/null || true

# Move stage 4 files
echo "📝 Moving Stage 4 files..."
mv stage4_* ORGANIZED/MODELS/STAGE4/ 2>/dev/null || true

# Move stage 5 files
echo "📝 Moving Stage 5 files..."
mv stage5_* ORGANIZED/MODELS/STAGE5/ 2>/dev/null || true

# Move stage 6 files
echo "📝 Moving Stage 6 files..."
mv stage6_* ORGANIZED/MODELS/STAGE5/ 2>/dev/null || true

# Move training files
echo "🎯 Moving training files..."
mv train*.sh ORGANIZED/TRAINING/ 2>/dev/null || true
mv *training*.c ORGANIZED/TRAINING/ 2>/dev/null || true
mv prepare_training_data.sh ORGANIZED/TRAINING/ 2>/dev/null || true
mv monitor_training.sh ORGANIZED/TRAINING/ 2>/dev/null || true

# Move test files
echo "🧪 Moving test files..."
mv test_* ORGANIZED/TESTS/ 2>/dev/null || true
mv *test*.c ORGANIZED/TESTS/ 2>/dev/null || true

# Move utility files
echo "🔧 Moving utility files..."
mv SAM/ ORGANIZED/UTILS/SAM/ 2>/dev/null || true
mv utils/ ORGANIZED/UTILS/ 2>/dev/null || true

# Move miscellaneous files
echo "📦 Moving miscellaneous files..."
mv *.sh ORGANIZED/UTILS/ 2>/dev/null || true
mv debug_* ORGANIZED/UTILS/ 2>/dev/null || true
mv demo_* ORGANIZED/UTILS/ 2>/dev/null || true
mv sam_* ORGANIZED/UTILS/ 2>/dev/null || true

# Keep main executables in root
echo "🚀 Keeping main executables in root..."

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
