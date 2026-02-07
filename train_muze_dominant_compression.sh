#!/bin/bash

# MUZE System Training with Dominant Compression Principle
# Based on AM's variational framework: arg max E[τ] - βH - λC + ηI

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Compiler settings
CC="gcc"
CFLAGS="-O3 -march=native -flto -Wall -Wextra -ffast-math"
INCLUDES="-I. -I../UTILS/utils/NN/MUZE"
LIBS="-lm"

# Source files for MUZE with Dominant Compression
MUZE_SRC="muze_enhanced_model.c muze_loop_thread.c muze_enhanced_config.c"
SAM_SRC="../SAM/SAM.c"
UTILS_SRC="../UTILS/NN/NN.c ../UTILS/NN/transformer.c"

# Output executable
OUTPUT="muze_dominant_compression"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    rm -f $OUTPUT
}

# Set trap for cleanup
trap cleanup EXIT

echo -e "${BLUE}🧠 MUZE Dominant Compression Training${NC}"
echo -e "${BLUE}=====================================${NC}"

# Clean previous build
rm -f $OUTPUT

echo -e "${YELLOW}Compiling MUZE with Dominant Compression...${NC}"
$CC $CFLAGS $INCLUDES $MUZE_SRC $SAM_SRC $UTILS_SRC -o $OUTPUT $LIBS

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Compilation successful!${NC}"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Create models directory
    mkdir -p ../MODELS/STAGE1
    
    echo -e "${YELLOW}🚀 Starting MUZE training with Dominant Compression...${NC}"
    
    # Run training with Dominant Compression optimization
    ./$OUTPUT 2>&1 | tee logs/muze_dc_training_$(date +%Y%m%d_%H%M%S).log
    
    # Check if training was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ Training completed successfully!${NC}"
        
        # Check if model files were created
        if [ -f "../MODELS/STAGE1/muze_dominant_model.bin" ]; then
            echo -e "${GREEN}💾 Dominant Compression model saved!${NC}"
            echo -e "${BLUE}📊 Principle: arg max E[τ] - βH - λC + ηI${NC}"
            echo -e "${BLUE}🎯 All minds converge to maximize future control per bit of uncertainty${NC}"
        else
            echo -e "${RED}⚠️ Warning: Model files were not created${NC}"
        fi
    else
        echo -e "${RED}❌ Training failed!${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Compilation failed!${NC}"
    exit 1
fi
