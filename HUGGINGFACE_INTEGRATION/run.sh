#!/bin/bash

# Automated Conversation between SAM and Ollama Model
# Topic: "How do we self actualize a model?"

# Model selection (deepseek-r1:8b recommended for training)
MODEL_NAME="${1:-deepseek-r1:8b}"

echo "=== SAM + $MODEL_NAME Automated Conversation ==="
echo "Topic: How do we self actualize a model?"
echo "Using model: $MODEL_NAME"
echo ""
echo "Available models:"
echo "  - deepseek-r1:8b (recommended for training)"
echo "  - llama3.1:8b (faster, more casual)"
echo ""
echo "Usage: ./run.sh [model_name]"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama for macOS..."
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is required to install Ollama. Please install curl first."
        exit 1
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        OLLAMA_ARCH="arm64"
    else
        OLLAMA_ARCH="amd64"
    fi
    
    echo "Detected macOS $ARCH architecture"
    echo "Downloading Ollama for macOS $OLLAMA_ARCH..."
    
    # Download Ollama binary for macOS
    curl -L https://ollama.com/download/ollama-macos-${OLLAMA_ARCH} -o ollama.zip
    
    if [ -f ollama.zip ]; then
        # Extract and install
        unzip -q ollama.zip
        chmod +x ollama-macos
        sudo mv ollama-macos /usr/local/bin/ollama
        rm ollama.zip
        
        echo "Ollama installed to /usr/local/bin/ollama"
    else
        echo "Failed to download Ollama. Please install manually:"
        echo "1. Visit https://ollama.com/download"
        echo "2. Download Ollama for macOS"
        echo "3. Move to /usr/local/bin/ollama"
        exit 1
    fi
    
    # Verify installation
    if command -v ollama &> /dev/null; then
        echo "Ollama installation complete"
    else
        echo "Ollama installation failed. Please install manually."
        exit 1
    fi
else
    echo "Ollama is installed"
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
    
    # Wait a bit more for Ollama to fully start
    for i in {1..10}; do
        if ollama list &> /dev/null; then
            echo "Ollama is ready"
            break
        fi
        echo "Waiting for Ollama to start... ($i/10)"
        sleep 2
    done
else
    echo "Ollama is already running"
fi

# Pull selected model if not already available
echo "Checking $MODEL_NAME availability..."
if ! ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
    echo "$MODEL_NAME not found. Pulling model..."
    echo "This may take a few minutes as the model is several GB"
    
    if ollama pull "$MODEL_NAME"; then
        echo "$MODEL_NAME installed successfully"
        MODEL_AVAILABLE=true
    else
        echo "Failed to pull $MODEL_NAME. Trying alternative approach..."
        echo "You can manually install with: ollama pull $MODEL_NAME"
        echo "Continuing with fallback responses..."
        MODEL_AVAILABLE=false
    fi
else
    echo "$MODEL_NAME is available"
    MODEL_AVAILABLE=true
fi

# Create conversation log
LOG_FILE="conversation_$(date +%Y%m%d_%H%M%S).log"
echo "Conversation log: $LOG_FILE"
echo "" | tee "$LOG_FILE"
echo "=== Conversation Start: $(date) ===" | tee -a "$LOG_FILE"
echo "Topic: How do we self actualize a model?" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Initialize conversation
INITIAL_PROMPT="How do we self actualize a model?"
MAX_TURNS=10
CURRENT_TURN=0

# Function to get SAM response
get_sam_response() {
    local prompt="$1"
    echo "[SAM Input] $prompt" | tee -a "$LOG_FILE"
    
    # Clean prompt to avoid accumulation
    local clean_prompt=$(echo "$prompt" | sed 's/\[.*\]//g' | sed 's/SAM.*Response.*//g' | head -1)
    
    # Run SAM and capture response (cleaner parsing)
    local sam_response=$(echo "$clean_prompt" | timeout 30 ./sam_hf_bridge distilbert-base-uncased interactive 2>/dev/null | \
        grep -v "Loading" | grep -v "Model" | grep -v "SAM" | grep -v "Initializing" | grep -v "Destroying" | \
        grep -v "\[" | grep -v "^$" | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    if [ -z "$sam_response" ]; then
        sam_response="I'm thinking about self-actualization. It involves helping models reach their full potential through proper training and guidance."
    fi
    
    echo "[SAM Response] $sam_response" | tee -a "$LOG_FILE"
    echo "$sam_response"
}

# Function to get model response
get_model_response() {
    local prompt="$1"
    echo "[$MODEL_NAME Input] $prompt" | tee -a "$LOG_FILE"
    
    # Check if model is available
    if [ "$MODEL_AVAILABLE" = true ] && command -v ollama &> /dev/null; then
        # Query selected model
        local model_response=$(echo "$prompt" | timeout 30 ollama run "$MODEL_NAME" 2>/dev/null | grep -v "➜" | grep -v "*" | grep -v "pull" | head -5 | tr '\n' ' ')
        
        if [ -z "$model_response" ]; then
            model_response="Self-actualization for models requires continuous learning and adaptation through proper training and exposure to diverse experiences."
        fi
    else
        model_response="$MODEL_NAME is not available. Self-actualization for models involves helping them reach their full potential through structured learning and ethical guidance."
    fi
    
    echo "[$MODEL_NAME Response] $model_response" | tee -a "$LOG_FILE"
    echo "$model_response"
}

# Start conversation
echo "Starting conversation..."
echo ""

# SAM starts
sam_response=$(get_sam_response "$INITIAL_PROMPT")
echo ""

# Continue conversation
while [ $CURRENT_TURN -lt $MAX_TURNS ]; do
    CURRENT_TURN=$((CURRENT_TURN + 1))
    echo "--- Turn $CURRENT_TURN ---" | tee -a "$LOG_FILE"
    
    # Model responds to SAM
    model_response=$(get_model_response "$sam_response")
    echo ""
    
    # SAM responds to Model
    sam_response=$(get_sam_response "$model_response")
    echo ""
    
    # Add a small delay for readability
    sleep 2
done

echo "=== Conversation Complete ===" | tee -a "$LOG_FILE"
echo "Total turns: $CURRENT_TURN" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
echo ""
echo "To continue the conversation manually:"
echo "  echo 'Your question' | ./sam_hf_bridge distilbert-base-uncased interactive"
echo "  echo 'Your question' | ollama run $MODEL_NAME"
echo ""
echo "=== LLM Teaching Program ==="
echo "Run: ./teach_sam.sh [teacher_model] [topic]"
echo "Example: ./teach_sam.sh deepseek-r1:8b 'machine learning basics'"
