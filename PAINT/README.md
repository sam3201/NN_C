# Paint3D - Enhanced 3D Paint with Advanced Neural Networks

## 🎨 What It Does
Paint on a 2D canvas and watch AI transform it into a 3D object using state-of-the-art neural networks!

## 🧠 Enhanced Neural Network Features

### **Complete Weight Initialization Methods**
- **Zero/Constant**: Demonstrates symmetry problem (all weights = 0)
- **Random Uniform**: Basic uniform distribution [-1, 1] (risks gradient issues)
- **Random Normal**: Basic normal distribution (risks gradient issues)
- **Xavier/Glorot**: Optimized for sigmoid/tanh with variance = 2/(fan_in + fan_out)
- **He**: Optimized for ReLU with variance = 2/fan_in (DEFAULT)
- **LeCun**: Optimized for deeper models with variance = 1/fan_in
- **Orthogonal**: For complex models using Gram-Schmidt orthogonalization

### **State-of-the-Art Optimizers**
- **Adam**: Adaptive Moment Estimation with learning rate scheduling
- **SGD**: Stochastic Gradient Descent with momentum
- **RMSProp**: Root Mean Square Propagation
- **AdaGrad**: Adaptive Gradient Algorithm
- **NAG**: Nesterov Accelerated Gradient

### **Advanced Training Features**
- **Stochastic Gradient Descent**: Random batch shuffling for better convergence
- **Batch Processing**: Larger batches for stable gradient estimation
- **Learning Rate Scheduling**: Automatic learning rate decay
- **Early Stopping**: Stop training at optimal performance
- **ESC Key Control**: Stop training anytime during epoch

## 🚀 Quick Start

### Build
```bash
# Simple build with enhanced NN framework
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
- **1-7**: Select weight initialization method
  - **1**: Zero/Constant (demonstrates symmetry problem)
  - **2**: Random Uniform (basic uniform distribution)
  - **3**: Random Normal (basic normal distribution)
  - **4**: Xavier/Glorot (optimal for sigmoid/tanh)
  - **5**: He (optimal for ReLU - default)
  - **6**: LeCun (for deeper models)
  - **7**: Orthogonal (for complex models)
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

Training progress shows every 10 epochs with:
- **Loss values** for convergence monitoring
- **Learning rate** if scheduling is active
- **Best loss tracking** for model selection

## 🎨 Workflow

1. **Select initialization**: Press 'T' → '1-7' to choose weight initialization method
2. **Train once**: Press 'R' → wait for 0% loss or press ESC
3. **Paint**: Draw on the canvas
4. **Generate**: Press 'G' to create 3D object
5. **View**: Press 'V' to see your 3D creation

### **Weight Initialization Comparison**
- **Press 1**: See why zero initialization fails (symmetry problem)
- **Press 2**: Try basic uniform (may have gradient issues)
- **Press 3**: Try basic normal (may have gradient issues)
- **Press 4**: Xavier/Glorot (good for sigmoid/tanh)
- **Press 5**: He (optimal for ReLU - recommended)
- **Press 6**: LeCun (good for deeper models)
- **Press 7**: Orthogonal (best for complex models)

## 📁 Files

- `paint` - Main executable
- `trained_neural_network.nn` - Trained neural network (auto-saved)
- `generated_3d_object.mesh` - Your 3D creations

## 🎯 Technical Details

### **Network Architecture**
- **Input Layer**: 12,288 neurons (64x64x3 RGB canvas)
- **Hidden Layers**: 128 → 64 neurons with ReLU activation
- **Output Layer**: 4,096 neurons (64x64 depth map)
- **Total Parameters**: ~1.7 million trainable weights

### **Training Configuration**
- **Optimizer**: Adam with learning rate scheduling (0.001 → 0.0001)
- **Batch Size**: 8 samples for stable gradients
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: L2 regularization
- **Weight Init**: He distribution (variance = 2/fan_in)

### **Weight Initialization Variance Formulas**
- **Zero**: variance = 0 (demonstrates symmetry problem)
- **Random Uniform**: variance = 1.0 (basic, no adjustment)
- **Random Normal**: variance = 1.0 (basic, no adjustment)
- **Xavier/Glorot**: variance = 2/(fan_in + fan_out)
- **He**: variance = 2/fan_in (optimal for ReLU)
- **LeCun**: variance = 1/fan_in (for deeper models)
- **Orthogonal**: orthogonal matrix (Gram-Schmidt process)

### **Performance**
- **Convergence**: Typically reaches <0.1% loss within 50-100 epochs
- **Training Time**: ~2-5 minutes for full convergence
- **Generation Time**: <1 second for 3D object creation
- **Memory Usage**: ~100MB runtime

## 🎨 Creative Applications

### **Enhanced by Advanced Training**
- **Better Depth Generation**: More accurate 3D depth from 2D paintings
- **Faster Convergence**: Reaches optimal performance quicker
- **Stable Training**: Less sensitive to learning rate selection
- **Robust Performance**: Works consistently across different painting styles

### **Training Patterns**
- **Grass**: Organic, flowing terrain with Gaussian noise
- **Architecture**: Geometric structures with precise edges
- **Characters**: Centered features with attention to detail
- **Objects**: Scattered patterns with natural variation

## 🛠️ Requirements

- GCC compiler
- SDL3 development libraries
- macOS ARM64

---

**Enhanced with State-of-the-Art Neural Networks!** 🧠🏗️
