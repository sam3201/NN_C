#!/bin/bash

echo "=== Starting Stage 1 Training ==="

# Compile the training program
echo "Compiling Stage 1 trainer..."
gcc -o stage1_trainer stage1_raw_training.c SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
    
    # Check for existing model
    if [ -f "stage1_final_model.bin" ]; then
        echo "Found existing Stage 1 model"
        echo "Options:"
        echo "1. Continue training existing model"
        echo "2. Start fresh training"
        echo "3. Test existing model"
        read -p "Choose option (1-3): " choice
        
        case $choice in
            2)
                echo "Removing existing model..."
                rm -f stage1_final_model.bin
                rm -f stage1_checkpoint_*.bin
                ;;
            3)
                echo "Testing existing model..."
                ./stage1_trainer test
                exit 0
                ;;
            *)
                echo "Continuing with existing model..."
                ;;
        esac
    fi
    
    # Start training
    echo "Starting Stage 1 training..."
    echo "This will take several hours. Monitor progress in logs/stage1/"
    
    # Create log file
    log_file="logs/stage1/training_$(date +%Y%m%d_%H%M%S).log"
    
    # Run training with logging
    ./stage1_trainer training_data/raw_texts 2>&1 | tee "$log_file"
    
    echo "Training completed. Check logs in: $log_file"
    
else
    echo "✗ Compilation failed"
    exit 1
fi
