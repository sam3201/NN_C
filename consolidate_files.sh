#!/bin/bash

echo "=== NN_C FILE CONSOLIDATION AND CLEANUP ==="
echo "Removing duplicates and unnecessary files..."

# Create backup directory
mkdir -p BACKUP_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="BACKUP_$(date +%Y%m%d_%H%M%S)"

echo "📁 Created backup directory: $BACKUP_DIR"

# Function to move file to backup
backup_file() {
    if [ -f "$1" ]; then
        mv "$1" "$BACKUP_DIR/"
        echo "  📦 Backed up: $1"
    fi
}

# Function to remove executable and keep source
remove_executable_keep_source() {
    local source="$1"
    local executable="$2"
    
    if [ -f "$executable" ] && [ -f "$source" ]; then
        backup_file "$executable"
        echo "  🗑️  Removed executable: $executable (keeping source: $source)"
    fi
}

echo ""
echo "🤖 Consolidating Chatbot Files..."

# Move LLM chatbot to GUI directory if it's different
mkdir -p ORGANIZED/CHATBOT/GUI
if [ -f "ORGANIZED/PROJECTS/LLM/chatbot.c" ]; then
    echo "  📋 Analyzing LLM chatbot vs main chatbot..."
    
    # Check if it's significantly different
    if grep -q "raylib" "ORGANIZED/PROJECTS/LLM/chatbot.c" 2>/dev/null; then
        echo "  ✅ LLM chatbot uses raylib GUI - keeping as separate GUI version"
        mv ORGANIZED/PROJECTS/LLM/chatbot.c ORGANIZED/CHATBOT/GUI/
        mv ORGANIZED/PROJECTS/LLM/Makefile ORGANIZED/CHATBOT/GUI/
        mv ORGANIZED/PROJECTS/LLM/build.sh ORGANIZED/CHATBOT/GUI/
        mv ORGANIZED/PROJECTS/LLM/train.sh ORGANIZED/CHATBOT/GUI/
        mv ORGANIZED/PROJECTS/LLM/train_chatbot.c ORGANIZED/CHATBOT/GUI/
        mv ORGANIZED/PROJECTS/LLM/TRAINING.md ORGANIZED/CHATBOT/GUI/
        mv ORGANIZED/PROJECTS/LLM/README.md ORGANIZED/CHATBOT/GUI/LLM_README.md
        echo "  📁 Moved LLM chatbot to GUI directory"
    else
        echo "  ❌ LLM chatbot is redundant - backing up"
        backup_file "ORGANIZED/PROJECTS/LLM/chatbot.c"
    fi
fi

echo ""
echo "📝 Consolidating Model Files..."

# Remove duplicate executables, keep sources
echo "  🗑️  Removing duplicate executables (keeping sources)..."

# Stage 1
remove_executable_keep_source "ORGANIZED/MODELS/STAGE1/stage1_basic_training.c" "ORGANIZED/MODELS/STAGE1/stage1_basic"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE1/stage1_conservative.c" "ORGANIZED/MODELS/STAGE1/stage1_conservative"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE1/stage1_fixed_training.c" "ORGANIZED/MODELS/STAGE1/stage1_fixed"

# Stage 2
remove_executable_keep_source "ORGANIZED/MODELS/STAGE2/stage2_word_extraction.c" "ORGANIZED/MODELS/STAGE2/stage2_word_extraction"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE2/stage2_word_training.c" "ORGANIZED/MODELS/STAGE2/stage2_word_training"

# Stage 3
remove_executable_keep_source "ORGANIZED/MODELS/STAGE3/stage3_phrase_extraction.c" "ORGANIZED/MODELS/STAGE3/stage3_phrase_extraction"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE3/stage3_phrase_training.c" "ORGANIZED/MODELS/STAGE3/stage3_phrase_training"

# Stage 4
remove_executable_keep_source "ORGANIZED/MODELS/STAGE4/stage4_hybrid_actions.c" "ORGANIZED/MODELS/STAGE4/stage4_hybrid_actions"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE4/stage4_hybrid_actions_fixed.c" "ORGANIZED/MODELS/STAGE4/stage4_hybrid_actions_fixed"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE4/stage4_hybrid_simple.c" "ORGANIZED/MODELS/STAGE4/stage4_hybrid_simple"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE4/stage4_response_generation.c" "ORGANIZED/MODELS/STAGE4/stage4_response_generation"

# Stage 5/6
remove_executable_keep_source "ORGANIZED/MODELS/STAGE5/stage5_complete.c" "ORGANIZED/MODELS/STAGE5/stage5_complete"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE5/stage5_mcts_planner.c" "ORGANIZED/MODELS/STAGE5/stage5_mcts_planner"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE5/stage5_mcts_fixed.c" "ORGANIZED/MODELS/STAGE5/stage5_mcts_fixed"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE5/stage6_final_integration.c" "ORGANIZED/MODELS/STAGE5/stage6_final_integration"
remove_executable_keep_source "ORGANIZED/MODELS/STAGE5/stage6_integration_simple.c" "ORGANIZED/MODELS/STAGE5/stage6_integration_simple"

echo ""
echo "🚀 Consolidating Executable Files..."

# Move test executables to TESTS directory
echo "  📁 Moving test executables to TESTS directory..."
for test_exe in test_* word_prediction_test verify_organized_system analyze_duplicates; do
    if [ -f "$test_exe" ]; then
        mv "$test_exe" ORGANIZED/TESTS/
        echo "    📄 Moved: $test_exe"
    fi
done

# Move training executables to TRAINING directory
echo "  📁 Moving training executables to TRAINING directory..."
for train_exe in train_sam_*; do
    if [ -f "$train_exe" ]; then
        mv "$train_exe" ORGANIZED/TRAINING/
        echo "    📄 Moved: $train_exe"
    fi
done

echo ""
echo "📚 Consolidating Documentation Files..."

# Move all documentation to DOCS directory
echo "  📁 Moving documentation to DOCS directory..."
for doc_file in *.md; do
    if [ -f "$doc_file" ] && [ "$doc_file" != "README_ORGANIZED.md" ]; then
        mv "$doc_file" ORGANIZED/DOCS/
        echo "    📚 Moved: $doc_file"
    fi
done

echo ""
echo "🔧 Consolidating Utility Files..."

# Move analysis and organization scripts to UTILS
echo "  📁 Moving utility scripts to UTILS directory..."
if [ -f "organize_files_fixed.sh" ]; then
    mv organize_files_fixed.sh ORGANIZED/UTILS/
    echo "    🔧 Moved: organize_files_fixed.sh"
fi

if [ -f "consolidate_files.sh" ]; then
    echo "    🔧 Keeping: consolidate_files.sh (in root)"
fi

echo ""
echo "🗑️  Removing Unnecessary Files..."

# Remove old/unused files from root
echo "  🗑️  Removing old files from root directory..."
for old_file in debug_* demo_* sam_*; do
    if [ -f "$old_file" ] && [[ ! "$old_file" =~ sam_agi$ ]] && [[ ! "$old_file" =~ sam_checkpoint_ ]] && [[ ! "$old_file" =~ sam_hf_ ]] && [[ ! "$old_file" =~ sam_production_ ]] && [[ ! "$old_file" =~ sam_text_ ]] && [[ ! "$old_file" =~ sam_trained_ ]]; then
        backup_file "$old_file"
    fi
done

# Remove empty LLM directory if it's now empty
if [ -d "ORGANIZED/PROJECTS/LLM" ] && [ ! "$(ls -A ORGANIZED/PROJECTS/LLM)" ]; then
    rmdir ORGANIZED/PROJECTS/LLM
    echo "  🗑️  Removed empty LLM directory"
fi

echo ""
echo "📊 Final Statistics..."

# Count remaining files
remaining_files=$(find . -maxdepth 1 -type f ! -name ".*" ! -name "consolidate_files.sh" | wc -l)
backup_files=$(find "$BACKUP_DIR" -type f | wc -l)

echo "  📄 Files remaining in root: $remaining_files"
echo "  📦 Files backed up: $backup_files"
echo "  📁 Backup directory: $BACKUP_DIR"

echo ""
echo "✅ Consolidation Complete!"
echo ""
echo "📋 Summary of Changes:"
echo "  • Moved LLM chatbot to GUI directory (raylib version)"
echo "  • Removed duplicate executables (kept source files)"
echo "  • Moved test executables to TESTS directory"
echo "  • Moved training executables to TRAINING directory"
echo "  • Moved documentation to DOCS directory"
echo "  • Moved utility scripts to UTILS directory"
echo "  • Backed up old/unused files"
echo "  • Removed empty directories"
echo ""
echo "🚀 System is now clean and organized!"
echo "📁 Main chatbot: ORGANIZED/CHATBOT/"
echo "🌐 Web interface: ORGANIZED/CHATBOT/WEB/"
echo "🎮 GUI version: ORGANIZED/CHATBOT/GUI/"
echo "📝 Models: ORGANIZED/MODELS/"
echo "🧪 Tests: ORGANIZED/TESTS/"
echo "📚 Documentation: ORGANIZED/DOCS/"
