#!/bin/bash

if ! command -v pkg-config &> /dev/null
then
    echo "pkg-config could not be found. Please install it."
    exit 1
fi

gcc -w -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL utils/Raylib/libraylib.a utils/NN/NN.c MNIST.c -o MNIST 

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the application..."
    lldb ./MNIST
    rm MNIST 
    clear
else
    echo "Compilation failed. Please check for errors."
fi
