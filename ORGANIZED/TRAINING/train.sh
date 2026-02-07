#!/bin/bash

# Set error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Compiler settings
CC="gcc"
CFLAGS="-O3 -march=native -flto -Wall -Wextra -ffast-math"
INCLUDES="-I."
LIBS="-lm"

# Source files
SRC_FILES="sam_agi.c SAM/SAM.c utils/NN/transformer.c utils/NN/NN.c utils/NN/NEAT.c"

# Output executable
OUTPUT="sam_agi"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ -f "$OUTPUT" ]; then
        rm $OUTPUT
    fi
}

# Set trap for cleanup
trap cleanup EXIT

echo -e "${YELLOW}Starting SAM AGI build and training process...${NC}"

# Clean previous build
rm -f $OUTPUT

# Compile
echo -e "${YELLOW}Compiling SAM AGI with optimizations...${NC}"
$CC $CFLAGS $INCLUDES $SRC_FILES -o $OUTPUT $LIBS

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Compilation successful!${NC}"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Run the training with output logged
    echo -e "${YELLOW}Starting training...${NC}"
    ./$OUTPUT 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
    
    # Check if training was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}Training completed successfully!${NC}"
        
        # Check if model files were created
        if [ -f "sam_final.model_head.model" ] && [ -f "sam_final.model_sub.model" ]; then
            echo -e "${GREEN}Model files saved successfully!${NC}"
        else
            echo -e "${RED}Warning: Model files were not created${NC}"
        fi
    else
        echo -e "${RED}Training failed!${NC}"
        exit 1
    fi
else
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi
