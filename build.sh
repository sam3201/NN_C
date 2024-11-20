#!/bin/bash

gcc -o main main.c utils/VISUALIZER/NN_visualizer.c utils/NN/NN.c utils/NN/NEAT.c -I. -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lpthread -lm -framework OpenGL -framework Cocoa -framework IOKit -O2

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./main
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ran Successfully"

rm main 
