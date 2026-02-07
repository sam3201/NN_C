#!/bin/bash

# SAM AGI Training Data Preparation Script
# Prepares raw text data for Stage 1 training

echo "=== SAM AGI Training Data Preparation ==="

# Create necessary directories
mkdir -p training_data/raw_texts
mkdir -p training_data/processed
mkdir -p checkpoints/stage1
mkdir -p logs/stage1

echo "✓ Directories created"

# Check for existing text files
if [ -d "utils/DATASETS" ]; then
    echo "Found existing datasets in utils/DATASETS"
    
    # Copy existing text files
    find utils/DATASETS -name "*.txt" -exec cp {} training_data/raw_texts/ \;
    echo "✓ Copied existing .txt files"
    
    # List available files
    echo "Available text files:"
    ls -la training_data/raw_texts/
else
    echo "No existing datasets found. Creating sample data..."
    
    # Create sample training data
    cat > training_data/raw_texts/sample_philosophy.txt << 'EOF'
The nature of consciousness remains one of the greatest mysteries in science and philosophy. 
What we experience as subjective awareness emerges from complex neural processes that we are only beginning to understand. 
The relationship between mind and matter, between thought and physical reality, challenges our fundamental assumptions about existence itself.

Artificial intelligence presents new questions about the nature of thought and awareness. 
Can machines truly think, or do they merely simulate thinking? 
What distinguishes genuine consciousness from sophisticated information processing?

These questions touch upon deep philosophical issues that have puzzled thinkers for centuries. 
The hard problem of consciousness, as formulated by David Chalmers, asks why and how physical processes give rise to subjective experience. 
This remains an open question that bridges neuroscience, philosophy, and computer science.

The future of AI research may hold answers to some of these fundamental questions. 
As we develop more sophisticated models of intelligence, we may gain insights into the nature of our own minds. 
The pursuit of artificial general intelligence may ultimately illuminate the nature of natural intelligence itself.
EOF

    cat > training_data/raw_texts/sample_science.txt << 'EOF'
Machine learning represents a paradigm shift in how we approach complex problems. 
Unlike traditional programming where explicit rules are coded, machine learning systems learn patterns from data. 
This approach has revolutionized fields from computer vision to natural language processing.

Neural networks, inspired by the structure of biological brains, form the foundation of deep learning. 
These networks consist of interconnected layers of nodes that process information in increasingly abstract ways. 
Through training on vast datasets, these systems can recognize patterns and make predictions with remarkable accuracy.

The mathematics behind neural networks involves optimization algorithms that minimize prediction errors. 
Gradient descent and backpropagation enable these networks to learn from examples. 
The process involves adjusting millions or billions of parameters to improve performance on specific tasks.

Recent advances in transformer architectures have transformed natural language processing. 
Models like GPT and BERT demonstrate unprecedented capabilities in understanding and generating human language. 
These systems learn statistical patterns from enormous text corpora, enabling them to engage in coherent dialogue and reasoning.

The future of machine learning lies in developing more efficient and interpretable models. 
As these systems become more integrated into society, questions of ethics, bias, and transparency become increasingly important. 
The responsible development of AI technology requires careful consideration of its societal impact.
EOF

    cat > training_data/raw_texts/sample_creativity.txt << 'EOF'
Creativity emerges from the intersection of knowledge and imagination. 
The human mind possesses the remarkable ability to combine existing concepts in novel ways, generating ideas that have never before existed. 
This creative process draws upon our experiences, emotions, and understanding of the world.

Artificial systems are beginning to exhibit creative behaviors as well. 
Generative models can produce music, art, and text that rival human creations in quality and originality. 
These systems learn patterns from vast datasets and recombine them in innovative ways, suggesting that creativity may be fundamentally a pattern-based process.

The relationship between creativity and intelligence remains complex and poorly understood. 
Some of the most creative individuals in history displayed unconventional thinking patterns that challenged established norms. 
This suggests that creativity may involve breaking free from conventional patterns rather than simply following them.

Technology is changing how we express and experience creativity. 
Digital tools enable new forms of artistic expression and collaboration across geographical boundaries. 
The internet has democratized creative production, allowing anyone to share their creations with global audiences.

The future of creativity may involve collaboration between humans and AI systems. 
These partnerships could lead to new forms of art and innovation that neither could achieve alone. 
The boundaries between human and machine creativity may become increasingly blurred as technology advances.
EOF

    echo "✓ Created sample training data files"
fi

# Create training configuration
cat > training_data/stage1_config.txt << 'EOF'
STAGE1_TRAINING_CONFIG
=====================

Model Configuration:
- Input Dimension: 512
- Output Dimension: 128
- Attention Heads: 12
- Sequence Length: 128
- Batch Size: 16
- Learning Rate: 0.01
- Epochs: 50

Data Sources:
- Raw text files from training_data/raw_texts/
- Minimal preprocessing for pattern learning
- Character-level encoding
- Sliding window sequences

Training Goals:
- Basic pattern recognition
- Statistical language modeling
- Character prediction
- Text structure learning

Expected Outcomes:
- 70% pattern accuracy
- Coherent character sequences
- Basic text generation
- Foundation for Stage 2
EOF

echo "✓ Created training configuration"

# Check data quality
echo ""
echo "=== Data Quality Check ==="
total_chars=0
total_files=0

for file in training_data/raw_texts/*.txt; do
    if [ -f "$file" ]; then
        file_size=$(wc -c < "$file")
        char_count=$(wc -m < "$file")
        total_chars=$((total_chars + char_count))
        total_files=$((total_files + 1))
        
        echo "File: $(basename "$file")"
        echo "  Size: $file_size bytes"
        echo "  Characters: $char_count"
        echo ""
    fi
done

echo "Total files: $total_files"
echo "Total characters: $total_chars"

if [ $total_chars -lt 10000 ]; then
    echo "⚠️  Warning: Limited training data. Consider adding more text files."
else
    echo "✓ Sufficient training data for Stage 1"
fi

# Create training script
cat > run_stage1_training.sh << 'EOF'
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
EOF

chmod +x run_stage1_training.sh

echo "✓ Created training script: run_stage1_training.sh"

# Create monitoring script
cat > monitor_training.sh << 'EOF'
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
EOF

chmod +x monitor_training.sh

echo "✓ Created monitoring script: monitor_training.sh"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Add more text files to training_data/raw_texts/ (optional)"
echo "2. Run training: ./run_stage1_training.sh"
echo "3. Monitor progress: ./monitor_training.sh"
echo ""
echo "Training will take several hours and can be interrupted/resumed."
echo "Checkpoints are saved every 5 epochs."
