#!/bin/bash
# Simple Build Script for Paint3D with Enhanced Neural Networks

echo "=== Paint3D Simple Build with Enhanced NN Framework ==="

# Build paint program
echo "Building paint program with enhanced neural networks..."
gcc -o paint \
    src/core/paint.c \
    ../utils/SDL3/SDL3_compat.c \
    ../utils/NN/NN.c \
    ../utils/NN/TRANSFORMER.c \
    -Isrc -I../utils -I../utils/SDL3 -I../utils/NN \
    $(pkg-config --cflags --libs sdl3 sdl3-ttf 2>/dev/null) \
    -lm -pthread \
    -framework OpenGL \
    -arch arm64

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Enhanced features:"
    echo "  - Complete weight initialization methods:"
    echo "    * Zero/Constant (bad due to symmetry)"
    echo "    * Random Uniform (basic, risks gradient issues)"
    echo "    * Random Normal (basic, risks gradient issues)"
    echo "    * Xavier/Glorot (for sigmoid/tanh)"
    echo "    * He (for ReLU - optimal)"
    echo "    * LeCun (for deeper models)"
    echo "    * Orthogonal (for complex models)"
    echo "  - Adam optimizer with learning rate scheduling"
    echo "  - Stochastic gradient descent with batch processing"
    echo "  - Activation-aware variance adjustment"
    echo "  - Advanced optimizers (SGD, Adam, RMSProp, AdaGrad, NAG)"
    echo ""
    echo "Run with: ./paint"
    echo "Controls:"
    echo "  - T: Training mode"
    echo "  - 1-7: Select weight initialization method"
    echo "  - R: Train networks (stops at 0% loss or ESC)"
    echo "  - G: Generate 3D object"
    echo "  - V: View 3D object"
    echo ""
    echo "Weight initialization details:"
    echo "  - Press 1: Zero/Constant (demonstrates symmetry problem)"
    echo "  - Press 2: Random Uniform (basic uniform distribution)"
    echo "  - Press 3: Random Normal (basic normal distribution)"
    echo "  - Press 4: Xavier/Glorot (optimal for sigmoid/tanh)"
    echo "  - Press 5: He (optimal for ReLU - default)"
    echo "  - Press 6: LeCun (for deeper models)"
    echo "  - Press 7: Orthogonal (for complex models)"
else
    echo "✗ Build failed!"
    exit 1
fi
