#!/bin/bash

# Set compiler flags
CFLAGS="-I.. -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lpthread -lm -framework OpenGL -framework Cocoa -framework IOKit -O2"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Building Neural Network Visualizer Test..."

# Compile the test
gcc -o nn_visualizer_test nn_visualizer_test.c ../VISUALIZER/NN_visualizer.c ../NN/NN.c $CFLAGS

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Compilation successful!${NC}"
    echo "Running the test..."
    ./nn_visualizer_test
else
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo "Test completed"

# Cleanup
rm nn_visualizer_test
