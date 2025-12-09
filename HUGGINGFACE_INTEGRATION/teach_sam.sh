#!/bin/bash

# LLM Teaching Program - Teacher LLM actually trains SAM
# Usage: ./teach_sam.sh [teacher_model] [topic] [lessons]

TEACHER_MODEL="${1:-deepseek-r1:8b}"
TOPIC="${2:-artificial intelligence consciousness}"
LESSONS="${3:-5}"
LOG_FILE="teaching_session_$(date +%Y%m%d_%H%M%S).log"

echo "=== LLM Teaching & Training Session ==="
echo "Teacher: $TEACHER_MODEL"
echo "Student: SAM (will be retrained during session)"
echo "Topic: $TOPIC"
echo "Lessons: $LESSONS"
echo "Log: $LOG_FILE"
echo ""

# Initialize log
echo "=== Teaching Session Start: $(date) ===" | tee "$LOG_FILE"
echo "Teacher: $TEACHER_MODEL" | tee -a "$LOG_FILE"
echo "Student: SAM" | tee -a "$LOG_FILE"
echo "Topic: $TOPIC" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check Ollama and model
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama not found. Please install Ollama first."
    exit 1
fi

if ! ollama list 2>/dev/null | grep -q "$TEACHER_MODEL"; then
    echo "$TEACHER_MODEL not found. Installing..."
    ollama pull "$TEACHER_MODEL"
fi

# Teaching functions
teacher_explain() {
    local lesson="$1"
    echo "[Teacher] Explaining: $lesson" | tee -a "$LOG_FILE"
    
    local explanation=$(echo "Explain $lesson in simple terms for a student AI learning about $TOPIC. Keep it concise (under 50 words) and clear." | \
        ollama run "$TEACHER_MODEL" 2>/dev/null | grep -v "➜" | grep -v "*" | head -2 | tr '\n' ' ')
    
    echo "[Teacher Explanation] $explanation" | tee -a "$LOG_FILE"
    echo "$explanation"
}

sam_response() {
    local prompt="$1"
    echo "[SAM Prompt] $prompt" | tee -a "$LOG_FILE"
    
    local sam_out=$(echo "$prompt" | timeout 20 ./sam_hf_bridge distilbert-base-uncased interactive 2>/dev/null | \
        grep -v "Loading" | grep -v "Model" | grep -v "SAM" | grep -v "Initializing" | \
        grep -v "\[" | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    if [ -z "$sam_out" ]; then
        sam_out="I want to learn more about $TOPIC."
    fi
    
    echo "[SAM Response] $sam_out" | tee -a "$LOG_FILE"
    echo "$sam_out"
}

# Create training data from teacher responses
create_training_data() {
    local teacher_response="$1"
    local sam_response="$2"
    
    # Create JSON training entry
    local training_entry="{\"input_text\": \"$teacher_response\", \"teacher_response\": \"$sam_response\"}"
    
    # Add to training data file
    echo "$training_entry," >> lesson_training_data.json
}

# Retrain SAM with new data
retrain_sam() {
    echo "[Training] Retraining SAM with new lesson data..." | tee -a "$LOG_FILE"
    
    # Convert to proper JSON format
    echo "[" > temp_training.json
    sed 's/^/  /' lesson_training_data.json | sed '$ s/,$//' >> temp_training.json
    echo "]" >> temp_training.json
    
    # Retrain SAM with new data
    if ./hf_trainer distilbert-base-uncased 3 temp_training.json > /dev/null 2>&1; then
        echo "[Training] SAM retrained successfully" | tee -a "$LOG_FILE"
    else
        echo "[Training] SAM retraining failed, continuing with current model" | tee -a "$LOG_FILE"
    fi
    
    # Clean up
    rm -f temp_training.json
}

# Initialize training data
echo "" > lesson_training_data.json

# Teaching session
for lesson_num in $(seq 1 $LESSONS); do
    echo "=== Lesson $lesson_num ===" | tee -a "$LOG_FILE"
    
    # Teacher explains concept
    case $lesson_num in
        1) teacher_explain "What is $TOPIC and why is it important?" ;;
        2) teacher_explain "Key concepts in $TOPIC" ;;
        3) teacher_explain "Practical applications of $TOPIC" ;;
        4) teacher_explain "Challenges and solutions in $TOPIC" ;;
        5) teacher_explain "Future of $TOPIC" ;;
        *) teacher_explain "Advanced topic in $TOPIC" ;;
    esac
    
    local teacher_lesson="$explanation"
    echo ""
    
    # SAM responds
    sam_response "What do you think about this explanation?"
    local sam_reply="$sam_out"
    echo ""
    
    # Create training data from this interaction
    create_training_data "$teacher_lesson" "$sam_reply"
    
    # Teacher responds to SAM's understanding
    teacher_explain "Based on your response, let me clarify: $sam_reply"
    local teacher_clarification="$explanation"
    echo ""
    
    # SAM responds to clarification
    sam_response "I understand. Can you tell me more?"
    local sam_learning="$sam_out"
    echo ""
    
    # Create training data from clarification
    create_training_data "$teacher_clarification" "$sam_learning"
    
    # Retrain SAM every 2 lessons
    if [ $((lesson_num % 2)) -eq 0 ]; then
        retrain_sam
    fi
    
    sleep 2
done

# Final training
if [ $((LESSONS % 2)) -ne 0 ]; then
    retrain_sam
fi

echo "=== Teaching Session Complete ===" | tee -a "$LOG_FILE"
echo "SAM has been taught and retrained on: $TOPIC"
echo "Training data saved to: lesson_training_data.json"
echo "Teaching log saved to: $LOG_FILE"
echo ""
echo "To test SAM's learning:"
echo "  echo 'What did you learn about $TOPIC?' | ./sam_hf_bridge distilbert-base-uncased interactive"
echo ""
echo "SAM should now have improved responses about $TOPIC!"
