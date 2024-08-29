#!/bin/bash

if ! command -v pkg-config &> /dev/null
then
    echo "pkg-config could not be found. Please install it."
    exit 1
fi

gcc -w -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL utils/Raylib/libraylib.a utils/NN/NN.c You_Vs_Ai.c -o You_Vs_Ai 

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the application..."
    lldb ./You_Vs_Ai
    rm You_Vs_Ai 
    clear
else
    echo "Compilation failed. Please check for errors."
fi

