#!/bin/bash

echo "=== CONTINUOUS TRAINING WITH OLLAMA LAUNCHER ==="
echo "Starting continuous training system with Ollama integration"
echo "========================================================"

# Check if Ollama is installed
echo "🔍 Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed or not in PATH"
    echo "💡 Please install Ollama: https://ollama.ai/"
    echo ""
    echo "Installation commands:"
    echo "  # macOS"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "  # Linux"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ Ollama is installed"

# Check if required models are available
echo ""
echo "🔍 Checking available Ollama models..."
ollama list 2>/dev/null | grep -E "(NAME|llama|mistral|gemma)" || echo "⚠️  No models found. You may need to pull a model first."

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
echo ""

# Check if Python version is available
echo "🔍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed"
    echo "💡 Please install Python 3.6 or higher"
    exit 1
fi

echo "✅ Using Python: $PYTHON_CMD"

# Check if required Python packages are available
echo ""
echo "🔍 Checking Python packages..."
$PYTHON_CMD -c "import numpy" 2>/dev/null || {
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

# Choose implementation
echo ""
echo "🚀 Choose implementation:"
echo "1) Python version (recommended, more robust)"
echo "2) C version (lighter, but more basic)"
echo ""
read -p "Enter choice (1-2) [1]: " choice

choice=${choice:-1}

case $choice in
    1)
        echo "🐍 Starting Python continuous training..."
        echo "💡 Press Ctrl+C to stop gracefully"
        echo ""
        
        # Install required packages if needed
        $PYTHON_CMD -c "import numpy" 2>/dev/null || {
            echo "📦 Installing required Python packages..."
            pip install numpy
        }
        
        # Run Python version
        $PYTHON_CMD continuous_training_ollama.py "$MODEL" "$INTERVAL"
        ;;
    2)
        echo "💻 Starting C continuous training..."
        echo "💡 Press Ctrl+C to stop gracefully"
        echo ""
        
        # Compile C version if needed
        if [ ! -f "continuous_training_ollama" ]; then
            echo "🔨 Compiling C version..."
            gcc -o continuous_training_ollama continuous_training_ollama.c \
                ORGANIZED/UTILS/SAM/SAM.c \
                ORGANIZED/UTILS/utils/NN/NEAT/NEAT.c \
                ORGANIZED/UTILS/utils/NN/TRANSFORMER/TRANSFORMER.c \
                ORGANIZED/UTILS/utils/NN/NN/NN.c \
                -lm
            
            if [ $? -ne 0 ]; then
                echo "❌ Compilation failed"
                exit 1
            fi
            
            echo "✅ Compilation successful"
        fi
        
        # Run C version
        ./continuous_training_ollama "$MODEL"
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
echo "  📝 continuous_training_*.log"
echo "  💾 continuous_training_epoch_*.json (checkpoints)"
echo ""
echo "🚀 To start another session:"
echo "  ./start_continuous_training.sh [model] [interval]"
echo ""
echo "💡 Examples:"
echo "  ./start_continuous_training.sh llama2 30"
echo "  ./start_continuous_training.sh mistral 60"
echo "  ./start_continuous_training.sh gemma 15"
