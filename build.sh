#!/bin/bash

if ! command -v pkg-config &> /dev/null
then
    echo "pkg-config could not be found. Please install it."
    exit 1
fi

read -p "Enter: MNIST / Game / Invaders: " file 

if [ -z "$file" ]
then
  echo "No input provided. Please enter either 'MNIST' or 'Game'."
  exit 1
fi

case "$file" in
  "MNIST")
  gcc -w -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL utils/Raylib/libraylib.a utils/NN/NN.c MNIST.c -o MNIST 
  if [ $? -eq 0 ]; then
      echo "Compilation successful. Running the application with AddressSanitizer..."
      lldb ./MNIST
      rm MNIST 
      clear
  else
      echo "Compilation failed. Please check for errors."
  fi
  ;;
"Game")
  gcc -w -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL utils/Raylib/libraylib.a utils/NN/NN.c You_Vs_Ai.c -o You_Vs_Ai 

  if [ $? -eq 0 ]; then
      echo "Compilation successful. Running the application with AddressSanitizer..."
      lldb ./You_Vs_Ai
      rm You_Vs_Ai 
      clear
  else
      echo "Compilation failed. Please check for errors."
  fi
  ;;
"Invaders")
  gcc -w -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL utils/Raylib/libraylib.a utils/NN/NN.c Invaders.c -o Invaders
  if [ $? -eq 0 ]; then
      echo "Compilation successful. Running the application with AddressSanitizer..."
      lldb ./Invaders
      rm Invaders 
      clear
  else
      echo "Compilation failed. Please check for errors."
  fi
  ;;
esac

