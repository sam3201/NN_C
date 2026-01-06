#!/bin/bash

# Build script for SAM LLM Chatbot
# Simple approach: compile raylib sources directly

CC="gcc"
CFLAGS="-Wall -O2 -std=c99"
INCLUDES="-I../utils/Raylib/src -I../utils/Raylib/src/external/glfw/include -I../SAM -I../utils/NN"
DEFINES="-DPLATFORM_DESKTOP -D_GLFW_COCOA"

# Check if we can use pre-built raylib
if [ -f "../utils/Raylib/src/libraylib.a" ]; then
    echo "Using pre-built raylib library..."
    RAYLIB_LIB="../utils/Raylib/src/libraylib.a"
    USE_PREBUILT=true
else
    echo "Building raylib from source..."
    USE_PREBUILT=false
    
    # GLFW source files (embedded in raylib)
    GLFW_SRC="../utils/Raylib/src/external/glfw/src/context.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/init.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/input.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/monitor.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/platform.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/vulkan.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/window.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/egl_context.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/osmesa_context.c"
    
    # macOS specific GLFW files
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/cocoa_init.m"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/cocoa_joystick.m"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/cocoa_monitor.m"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/cocoa_time.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/cocoa_window.m"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/nsgl_context.m"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/posix_module.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/posix_thread.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/posix_time.c"
    GLFW_SRC="$GLFW_SRC ../utils/Raylib/src/external/glfw/src/posix_poll.c"
    
    # Raylib source files
    RAYLIB_SRC="../utils/Raylib/src/rcore.c"
    RAYLIB_SRC="$RAYLIB_SRC ../utils/Raylib/src/rshapes.c"
    RAYLIB_SRC="$RAYLIB_SRC ../utils/Raylib/src/rtext.c"
    RAYLIB_SRC="$RAYLIB_SRC ../utils/Raylib/src/utils.c"
    RAYLIB_SRC="$RAYLIB_SRC ../utils/Raylib/src/rglfw.c"
fi

# SAM and NN source files
SAM_SRC="../SAM/SAM.c"
SAM_SRC="$SAM_SRC ../utils/NN/transformer.c"
SAM_SRC="$SAM_SRC ../utils/NN/NEAT.c"
SAM_SRC="$SAM_SRC ../utils/NN/NN.c"

# Libraries and frameworks for macOS
LIBS="-lm -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo"

echo "Building SAM LLM Chatbot..."

if [ "$USE_PREBUILT" = true ]; then
    $CC $CFLAGS $DEFINES $INCLUDES chatbot.c $SAM_SRC $RAYLIB_LIB -o chatbot $LIBS
else
    $CC $CFLAGS $DEFINES $INCLUDES chatbot.c $GLFW_SRC $RAYLIB_SRC $SAM_SRC -o chatbot $LIBS
fi

if [ $? -eq 0 ]; then
    echo "✓ Build successful! Run ./chatbot to start the chatbot."
else
    echo "✗ Build failed!"
    exit 1
fi

