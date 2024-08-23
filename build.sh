#!/bin/bash

# Check if pkg-config is installed
if ! command -v pkg-config &> /dev/null
then
    echo "pkg-config could not be found. Please install it."
    exit 1
fi

# Compile the MNIST Raylib application
gcc -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL utils/Raylib/libraylib.a utils/NN/NN.c MNIST.c -o MNIST 

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the application..."
    lldb ./MNIST
    rm MNIST 
else
    echo "Compilation failed. Please check for errors."
fi
