#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/build_${TIMESTAMP}.log"

echo "Building game... Logging to $LOG_FILE"

# Compile with all errors and warnings redirected to log file
cc -g -O0 game.c generated/impl.c ../utils/NN/NN/NN.c ../utils/NN/TRANSFORMER/TRANSFORMER.c ../utils/NN/NEAT/NEAT.c ../utils/NN/CONVOLUTION/CONVOLUTION.c $(find ../RL_AGENT -type f -name '*.c' -print | grep -v test | sort -u | tr '\n' ' ') $(find ../utils/NN/MUZE -type f -name '*.c' -print | sort -u | tr '\n' ' ') $(find ../SAM -type f -name '*.c' -print | sort -u | tr '\n' ' ') -I../utils/NN/NN -I../utils/NN/CONVOLUTION -I../utils/NN/MUZE -I../SAM -I../RL_AGENT -I../utils/NN/TRANSFORMER -I../utils/NN/NEAT -I../utils/Raylib/src -I../utils/SDL/include -I../utils/SDL_ttf/include -Igenerated -F../utils/SDL/build -framework SDL3 -L../utils/SDL_ttf/build -lSDL3_ttf -Wl,-rpath,../utils/SDL/build -Wl,-rpath,../utils/SDL_ttf/build ../utils/Raylib/src/libraylib.a -framework Cocoa -framework IOKit -framework CoreVideo -pthread -lm -framework OpenGL -arch arm64 -o game 2>&1 | tee "$LOG_FILE"

# Check if compilation was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Build successful!"
    
    # Test run with lldb to check for runtime issues
    echo "Testing game with lldb..."
    timeout 10s lldb --batch -o run -o bt -- ./game 2>&1 | tee -a "$LOG_FILE"
    
    echo "Build and test completed. Log saved to $LOG_FILE"
else
    echo "Build failed! Check $LOG_FILE for details."
fi

echo "Cleaning up log directory..."
# Keep only the last 5 log files
cd logs
ls -t | tail -n +6 | xargs -r rm -f
cd ..

echo "Done."
