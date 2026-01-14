# Paint3D - Simple 3D Paint with Neural Networks

## 🎨 What It Does
Paint on a 2D canvas and watch AI transform it into a 3D object!

## 🚀 Quick Start

### Build
```bash
# Simple build
./simple_build.sh
```

### Run
```bash
# Start the program
./paint
```

## 🎮 Controls

### Painting
- **Left Mouse**: Paint
- **Right Mouse**: Erase
- **SPACE**: Toggle 3D depth view
- **ENTER**: Process with neural networks

### Training
- **T**: Enter training mode
- **R**: Train networks (stops at 0% loss or ESC)
- **S**: Save trained networks
- **O**: Load trained networks

### 3D Generation
- **G**: Generate 3D object from canvas
- **V**: View 3D object

### System
- **ESC**: Exit (also stops training)

## 🧠 Training

The neural networks train until:
- **0% loss** (perfect training) OR
- **ESC key** pressed

Training progress shows every 10 epochs with loss values.

## 🎨 Workflow

1. **Train once**: Press 'T' → 'R' → wait for 0% loss or press ESC
2. **Paint**: Draw on the canvas
3. **Generate**: Press 'G' to create 3D object
4. **View**: Press 'V' to see your 3D creation

## 📁 Files

- `paint` - Main executable
- `trained_convolution.net` - Trained neural network (auto-saved)
- `generated_3d_object.mesh` - Your 3D creations

## 🎯 Features

- **Neural Networks**: Convolution + Transformer for depth generation
- **Real-time 3D**: Instant 3D object generation
- **Interactive Viewer**: Rotate and zoom 3D objects
- **Smart Training**: Stops automatically at optimal performance

## 🛠️ Requirements

- GCC compiler
- SDL3 development libraries
- macOS ARM64

---

**Simple. Fast. Creative.** 🎨🏗️
