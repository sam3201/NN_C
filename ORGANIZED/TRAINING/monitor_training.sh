#!/bin/bash

echo "=== SAM AGI Training Monitor ==="

# Check if training is running
if pgrep -f "stage1_trainer" > /dev/null; then
    echo "✓ Training is running"
    
    # Show recent log entries
    echo ""
    echo "=== Recent Training Logs ==="
    if [ -d "logs/stage1" ]; then
        latest_log=$(ls -t logs/stage1/*.log 2>/dev/null | head -1)
        if [ -f "$latest_log" ]; then
            echo "From: $latest_log"
            tail -20 "$latest_log"
        fi
    fi
    
    # Check checkpoint files
    echo ""
    echo "=== Checkpoint Files ==="
    ls -lh stage1_checkpoint_*.bin 2>/dev/null | tail -5
    
    # Check model file
    if [ -f "stage1_final_model.bin" ]; then
        echo ""
        echo "✓ Final model exists: $(ls -lh stage1_final_model.bin)"
    fi
    
else
    echo "Training is not currently running"
    
    echo ""
    echo "Available actions:"
    echo "1. Start training: ./run_stage1_training.sh"
    echo "2. Check logs: ls logs/stage1/"
    echo "3. Check models: ls stage1_*.bin"
fi
