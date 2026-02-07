#!/bin/bash

echo "=== THREADED CONTINUOUS TRAINING WITH NCURSES ==="
echo "Real-time monitoring with threaded training and Ollama teaching"
echo "=========================================================="

# Check if Ollama is installed
echo "🔍 Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed or not in PATH"
    echo "💡 Please install Ollama: https://ollama.ai/"
    echo ""
    echo "Installation commands:"
    echo "  # macOS/Linux"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "  # Start Ollama service"
    echo "  ollama serve"
    echo ""
    echo "  # Pull a model"
    echo "  ollama pull llama2"
    echo ""
    exit 1
fi

echo "✅ Ollama is installed"

# Check if ncurses is available for Python
echo ""
echo "🔍 Checking ncurses availability..."
if python3 -c "import curses" 2>/dev/null; then
    echo "✅ Python ncurses is available"
else
    echo "❌ Python ncurses not available"
    echo "💡 Install with: pip install windows-curses (on Windows) or use system package manager"
    exit 1
fi

# Check if required Python packages are available
echo ""
echo "🔍 Checking Python packages..."
python3 -c "import numpy" 2>/dev/null || {
    echo "❌ Required Python packages not found"
    echo "💡 Install with: pip install numpy"
    exit 1
}

echo "✅ Required Python packages are available"

# Check if SAM model exists
echo ""
echo "🔍 Checking SAM model..."
if [ -f "ORGANIZED/MODELS/STAGE4/stage4_response_final.bin" ]; then
    echo "✅ SAM model found: ORGANIZED/MODELS/STAGE4/stage4_response_final.bin"
elif [ -f "stage4_response_final.bin" ]; then
    echo "✅ SAM model found: stage4_response_final.bin"
else
    echo "⚠️  SAM model not found"
    echo "💡 Make sure you have trained the SAM model first"
    echo "💡 Run: cd ORGANIZED/CHATBOT/TERMINAL && ./full_llm_chatbot"
    echo ""
fi

# Default model
DEFAULT_MODEL="llama2"

# Parse command line arguments
MODEL="$DEFAULT_MODEL"
INTERVAL=30

if [ "$1" != "" ]; then
    MODEL="$1"
fi

if [ "$2" != "" ]; then
    INTERVAL="$2"
fi

echo ""
echo "🎯 Configuration:"
echo "  Model: $MODEL"
echo "  Training interval: $INTERVAL seconds"
echo "  Interface: NCurses (real-time monitoring)"
echo ""

# Choose implementation
echo "🚀 Choose implementation:"
echo "1) Python version with NCurses (recommended, real-time monitoring)"
echo "2) C version with NCurses (lighter, real-time monitoring)"
echo "3) Python version without NCurses (simple terminal output)"
echo ""
read -p "Enter choice (1-3) [1]: " choice

choice=${choice:-1}

case $choice in
    1)
        echo "🐍 Starting Python threaded training with NCurses..."
        echo "💡 Press Ctrl+C to stop gracefully"
        echo "💡 In NCurses: Q-Quit, S-Status, C-Clear log, H-Help"
        echo ""
        
        # Install required packages if needed
        python3 -c "import numpy" 2>/dev/null || {
            echo "📦 Installing required Python packages..."
            pip install numpy
        }
        
        # Run Python version
        python3 continuous_training_threaded.py "$MODEL" "$INTERVAL"
        ;;
    2)
        echo "💻 Starting C threaded training with NCurses..."
        echo "💡 Press Ctrl+C to stop gracefully"
        echo "💡 In NCurses: Q-Quit, S-Status, C-Clear log, H-Help"
        echo ""
        
        # Compile C version if needed
        if [ ! -f "continuous_training_threaded" ]; then
            echo "🔨 Compiling C version..."
            gcc -o continuous_training_threaded continuous_training_threaded.c \
                ORGANIZED/UTILS/SAM/SAM.c \
                ORGANIZED/UTILS/utils/NN/NEAT/NEAT.c \
                ORGANIZED/UTILS/utils/NN/TRANSFORMER/TRANSFORMER.c \
                ORGANIZED/UTILS/utils/NN/NN/NN.c \
                -lncurses -lm -lpthread
            
            if [ $? -ne 0 ]; then
                echo "❌ Compilation failed"
                echo "💡 Make sure ncurses development libraries are installed:"
                echo "   Ubuntu/Debian: sudo apt-get install libncurses5-dev"
                echo "   macOS: brew install ncurses"
                echo "   CentOS/RHEL: sudo yum install ncurses-devel"
                exit 1
            fi
            
            echo "✅ Compilation successful"
        fi
        
        # Run C version
        ./continuous_training_threaded "$MODEL"
        ;;
    3)
        echo "🐍 Starting Python training without NCurses..."
        echo "💡 Press Ctrl+C to stop gracefully"
        echo ""
        
        # Install required packages if needed
        python3 -c "import numpy" 2>/dev/null || {
            echo "📦 Installing required Python packages..."
            pip install numpy
        }
        
        # Run Python version without ncurses
        python3 continuous_training_ollama.py "$MODEL" "$INTERVAL"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Continuous training session completed!"
echo ""
echo "📊 Check the log files for training details:"
echo "  📝 continuous_training_threaded_*.log"
echo "  💾 continuous_training_epoch_*.json (checkpoints)"
echo ""
echo "🎯 Features used:"
echo "  🧵 Multi-threaded operation"
echo "  📊 Real-time NCurses monitoring"
echo "  🤖 Ollama teaching integration"
echo "  💾 Automatic checkpointing"
echo "  🛑 Graceful shutdown"
echo ""
echo "🚀 To start another session:"
echo "  ./start_threaded_training.sh [model] [interval]"
echo ""
echo "💡 Examples:"
echo "  ./start_threaded_training.sh llama2 30"
echo "  ./start_threaded_training.sh mistral 60"
echo "  ./start_threaded_training.sh gemma 15"
