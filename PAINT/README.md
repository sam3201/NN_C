# Paint3D v1.0.0

**3D Paint with Neural Networks** - Generate 3D objects from 2D paintings using AI

## 🎨 Overview

Paint3D is a revolutionary application that combines neural networks with creative painting. Draw on a 2D canvas and watch as artificial intelligence transforms your artwork into fully-realized 3D objects with depth, texture, and structure.

## ✨ Features

### 🧠 **Neural Network Integration**
- **Convolutional Networks**: Extract spatial features from painted patterns
- **Transformer Networks**: Apply attention-based depth generation
- **Training System**: Learn from synthetic patterns to create diverse 3D shapes
- **Real-time Processing**: Instant neural network inference during painting

### 🎮 **Interactive Painting**
- **Smart Canvas**: 64x64 pixel canvas with neural network depth visualization
- **3D Mode**: Toggle between 2D painting and 3D depth preview
- **Multiple Training Patterns**: Grass, architecture, characters, objects, and more
- **Intuitive Controls**: Mouse-based painting with keyboard shortcuts

### 🏗️ **3D Generation**
- **Vertex Generation**: Convert painted canvas to 4096 vertices with proper triangulation
- **Normal Calculation**: Automatic lighting-ready surface normals
- **Mesh Export**: Save generated objects in custom mesh format
- **Real-time Viewer**: Interactive 3D visualization with rotation and zoom

### 🛠️ **Professional Build System**
- **Make-based**: Industry-standard build system with dependency management
- **Modular Architecture**: Clean separation of core, viewer, and utility components
- **Cross-platform**: macOS ARM64 support with SDL3 and OpenGL
- **Development Tools**: Comprehensive build scripts and project configuration

## 🚀 Quick Start

### Prerequisites
- GCC or Clang compiler
- SDL3 development libraries
- Make build system
- pthread support

### Installation & Build

```bash
# Clone and setup
git clone <repository-url>
cd paint3d

# Build everything
./build.sh

# Or use make directly
make
```

### First Run

```bash
# Train neural networks (one-time setup)
./build.sh train

# Run the main application
./bin/paint

# Or use the demo
./build.sh demo
```

## 🎮 Usage Guide

### **Painting Controls**
- **Left Mouse**: Paint on canvas
- **Right Mouse**: Erase from canvas
- **SPACE**: Toggle 3D depth visualization
- **ENTER**: Process with neural networks

### **Training Controls**
- **T**: Toggle training mode
- **R**: Train networks (in training mode)
- **S**: Save trained networks (in training mode)
- **O**: Load pre-trained networks

### **3D Generation**
- **G**: Generate 3D object from current canvas
- **V**: View generated 3D object in interactive viewer

### **System**
- **ESC**: Exit application

## 🏗️ Project Structure

```
paint3d/
├── src/                    # Source code
│   ├── core/              # Main paint application
│   │   └── paint.c
│   ├── viewer/            # 3D mesh viewer
│   │   └── view_3d.c
│   └── utils/             # Utility functions
│       ├── mesh_loader.c
│       └── mesh_loader.h
├── bin/                    # Built executables
│   ├── paint              # Main application
│   ├── view_3d            # 3D viewer
│   └── test_mesh          # Mesh generation test
├── build/                  # Build artifacts
├── lib/                    # Libraries
├── assets/                 # Game assets
├── Makefile               # Build system
├── build.sh               # Build script
├── project.config         # Project configuration
└── README.md              # This file
```

## 🔧 Build System

### **Build Script Commands**

```bash
./build.sh [COMMAND] [OPTIONS]

Commands:
    build           Build all components (default)
    paint           Build paint application only
    viewer          Build 3D viewer only
    test            Build and run mesh test
    train           Train neural networks
    demo            Run complete demo
    clean           Clean build files
    deep-clean      Remove all generated files
    install         Install to system
    uninstall       Remove from system
    dev-setup       Setup development environment
    status          Show build status
    help            Show help

Options:
    --verbose       Verbose output
    --debug         Debug build
    --release       Release build (default)
    --check         Run tests after build
    --no-sdl        Skip SDL dependencies check
```

### **Makefile Targets**

```bash
make [TARGET]

Targets:
    all        - Build all components
    paint      - Build paint app
    viewer     - Build 3D viewer
    test       - Build and run tests
    train      - Train networks
    demo       - Run demo
    clean      - Clean build files
    install    - Install to system
    help       - Show help
```

## 🧠 Neural Network Architecture

### **Training Configuration**
- **Epochs**: 50 training iterations
- **Batch Size**: 4 samples per batch
- **Learning Rate**: 0.01 adaptive learning
- **Early Stopping**: 10 epochs patience
- **Convergence**: 0.001 loss threshold

### **Network Structure**
- **Convolution**: 16-channel 3x3 kernels
- **Transformer**: 4 attention heads, 2 layers
- **Model Dimension**: 128 hidden units
- **Input**: 64x64 RGB canvas (12,288 features)
- **Output**: 64x64 depth map (4,096 values)

### **Training Patterns**
1. **Grass**: Organic, flowing terrain patterns
2. **Architecture**: Geometric, structured shapes
3. **Trees**: Vertical, organic growth patterns
4. **Characters**: Centered, important features
5. **Objects**: Scattered, organic forms

## 📊 Technical Specifications

### **Canvas System**
- **Resolution**: 64x64 pixels
- **Color**: RGB 24-bit color depth
- **Depth**: 32-bit floating point precision
- **3D Mapping**: Canvas coordinates to 3D space

### **3D Generation**
- **Vertices**: 4,096 points (64x64 grid)
- **Triangles**: 7,938 faces (2 per grid cell)
- **Normals**: Calculated surface normals for lighting
- **File Size**: ~100KB mesh files

### **Performance**
- **Build Time**: ~30 seconds
- **Training Time**: ~60 seconds (one-time)
- **Generation Time**: <1 second
- **Memory Usage**: ~50MB runtime

## 🎨 Creative Applications

### **Terrain Generation**
- Paint landscapes and generate realistic 3D terrain
- Create mountains, valleys, and organic surfaces
- Apply neural network learned terrain patterns

### **Architecture Design**
- Draw buildings and create 3D structures
- Generate architectural forms with learned patterns
- Create complex geometric compositions

### **Character Design**
- Sketch characters and generate 3D models
- Apply character-specific depth patterns
- Create unique 3D character sculptures

### **Abstract Art**
- Create abstract paintings and generate unique 3D sculptures
- Explore the intersection of 2D art and 3D form
- Generate unexpected and creative 3D shapes

## 🔧 Development

### **Setting Up Development Environment**

```bash
# Setup directories and configuration
./build.sh dev-setup

# Check build status
./build.sh status

# Build with debug information
./build.sh --debug

# Run tests after build
./build.sh build --check
```

### **Code Organization**
- **Modular Design**: Clear separation of concerns
- **Header Files**: Proper interface definitions
- **Error Handling**: Comprehensive error checking
- **Memory Management**: Safe allocation and cleanup

### **Adding New Features**
1. Add source files to appropriate `src/` subdirectory
2. Update `Makefile` with new targets
3. Update `build.sh` with new commands
4. Add documentation to README

## 📈 Future Enhancements

### **Planned Features**
- **GLB Export**: Proper GLB file format support
- **Texture Mapping**: Apply painted colors as textures
- **Real-time Preview**: Live 3D preview while painting
- **Animation**: Support for animated 3D objects
- **Multiple Materials**: Different materials for regions
- **Advanced Lighting**: Better normal calculation and shading

### **Technical Improvements**
- **GPU Acceleration**: CUDA/OpenCL neural network processing
- **Higher Resolution**: Support for larger canvases
- **More Networks**: Additional neural network architectures
- **Better Training**: Improved training algorithms and datasets

## 🐛 Troubleshooting

### **Build Issues**
- **SDL3 Missing**: Install SDL3 development packages
- **Compiler Errors**: Check GCC/Clang version compatibility
- **Linker Errors**: Verify library paths and frameworks

### **Runtime Issues**
- **Network Loading**: Ensure trained networks exist
- **Memory Issues**: Check system memory availability
- **Display Issues**: Verify SDL3 and OpenGL support

### **Performance**
- **Slow Training**: Reduce epochs or canvas size
- **Memory Usage**: Lower canvas resolution
- **Generation Speed**: Optimize neural network architecture

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Paint3D v1.0.0** - Where 2D Art Meets 3D Reality 🎨🏗️
