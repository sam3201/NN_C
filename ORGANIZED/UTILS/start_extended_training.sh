#!/bin/bash

echo "🚀 SAM AGI Extended Raw Training Launch"
echo "========================================"
echo ""

# Check if basic model exists
if [ ! -f "stage1_basic_final.bin" ]; then
    echo "❌ No basic model found. Run basic training first:"
    echo "   ./stage1_basic training_data/raw_texts/Frankenstein.txt"
    exit 1
fi

echo "✅ Found basic model: stage1_basic_final.bin"
echo ""

# Training options
echo "Training Options:"
echo "1. Extended Frankenstein training (100 epochs)"
echo "2. Multi-text training (all available files)"
echo "3. Continuous training loop"
echo "4. Custom training"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "🎯 Starting extended Frankenstein training..."
        echo "   Data: Frankenstein.txt"
        echo "   Epochs: 100"
        echo "   This will take approximately 10-20 minutes"
        echo ""
        
        # Create extended version
        gcc -o stage1_extended stage1_basic_training.c -DEPOCHS=100 -DSAMPLES_PER_EPOCH=20 SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm
        
        if [ $? -eq 0 ]; then
            echo "✅ Compilation successful"
            echo "🚀 Starting extended training..."
            ./stage1_extended training_data/raw_texts/Frankenstein.txt
        else
            echo "❌ Compilation failed"
            exit 1
        fi
        ;;
        
    2)
        echo "🎯 Starting multi-text training..."
        echo "   Data: All available text files"
        echo "   Epochs per file: 50"
        echo ""
        
        for file in training_data/raw_texts/*.txt; do
            if [ -f "$file" ]; then
                echo "📚 Training on: $(basename "$file")"
                gcc -o stage1_multi stage1_basic_training.c -DEPOCHS=50 -DSAMPLES_PER_EPOCH=10 SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm
                
                if [ $? -eq 0 ]; then
                    ./stage1_multi "$file"
                    
                    # Save with filename
                    cp stage1_basic_final.bin "stage1_$(basename "$file" .txt).bin"
                    echo "✅ Saved: stage1_$(basename "$file" .txt).bin"
                fi
            fi
        done
        ;;
        
    3)
        echo "🔄 Starting continuous training loop..."
        echo "   Press Ctrl+C to stop"
        echo ""
        
        while true; do
            echo "🔄 Training cycle: $(date)"
            
            # Random file selection
            files=(training_data/raw_texts/*.txt)
            random_file=${files[$RANDOM % ${#files[@]}]}
            
            echo "📚 Training on: $(basename "$random_file")"
            gcc -o stage1_continuous stage1_basic_training.c -DEPOCHS=20 -DSAMPLES_PER_EPOCH=5 SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm
            
            if [ $? -eq 0 ]; then
                ./stage1_continuous "$random_file"
                
                # Save with timestamp
                timestamp=$(date +%Y%m%d_%H%M%S)
                cp stage1_basic_final.bin "continuous_models/stage1_${timestamp}.bin"
                echo "✅ Saved checkpoint: ${timestamp}"
            fi
            
            echo "⏱️ Resting 60 seconds..."
            sleep 60
        done
        ;;
        
    4)
        echo "🎯 Custom training configuration"
        read -p "Enter data file path: " data_file
        read -p "Enter number of epochs: " epochs
        read -p "Enter samples per epoch: " samples
        
        echo "🚀 Starting custom training..."
        echo "   Data: $data_file"
        echo "   Epochs: $epochs"
        echo "   Samples per epoch: $samples"
        
        gcc -o stage1_custom stage1_basic_training.c -DEPOCHS=$epochs -DSAMPLES_PER_EPOCH=$samples SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm
        
        if [ $? -eq 0 ]; then
            ./stage1_custom "$data_file"
        else
            echo "❌ Compilation failed"
            exit 1
        fi
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Training completed!"
echo ""
echo "Generated models:"
ls -la stage1_*.bin
echo ""
echo "Next steps:"
echo "1. Test the model: ./test_trained_model.sh"
echo "2. Generate samples: ./generate_text.sh"
echo "3. Move to Stage 2: ./prepare_stage2.sh"
echo ""
echo "🧠 Remember: This is RAW training with no constraints!"
echo "   Monitor outputs carefully and expect experimental behavior."
