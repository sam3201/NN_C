# 🧠 Neural Network Models Directory

This directory contains all neural network models and frameworks organized by type.

## 📁 Directory Structure

### **Core Models**
- **MLP/** - Multi-Layer Perceptron (Standard Neural Network)
- **RNN/** - Recurrent Neural Networks (Simple RNN, LSTM, GRU)
- **GNN/** - Graph Neural Networks (Message Passing, Graph Processing)
- **SNN/** - Spiking Neural Networks (Energy-efficient, Brain-like)
- **KAN/** - Kolmogorov-Arnold Networks (Interpretable, Fewer Parameters)
- **GAN/** - Generative Adversarial Networks (Data Generation)

### **Specialized Models**
- **CONVOLUTION/** - Convolutional Neural Networks (Image Processing)
- **TRANSFORMER/** - Transformer Networks (Attention-based)
- **NEAT/** - NeuroEvolution of Augmenting Topologies (Evolutionary)
- **TOKENIZER/** - Text Tokenization and Processing

## 🚀 Quick Start

### **Basic MLP**
```c
#include "MLP/NN.h"

// Create a simple neural network
size_t layers[] = {784, 128, 64, 10};
NN_t *nn = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L2, ADAM, 0.001L);
```

### **Recurrent Networks**
```c
#include "RNN/RNN.h"

// Create LSTM network
RNN_t *rnn = RNN_create(input_size, hidden_size, output_size, num_layers, RNN_LSTM);
long double *output = RNN_forward(rnn, inputs, sequence_length);
```

### **Graph Neural Networks**
```c
#include "GNN/GNN.h"

// Create graph neural network
GNN_t *gnn = GNN_create(num_nodes, feature_dim, hidden_dim, num_layers);
GNN_add_edge(gnn, node1, node2, weight);
long double *output = GNN_forward(gnn, node_features);
```

### **Spiking Neural Networks**
```c
#include "SNN/SNN.h"

// Create spiking neural network
SNN_t *snn = SNN_create(num_neurons, input_size, output_size);
long double *spikes = SNN_forward(snn, inputs, duration);
```

### **Kolmogorov-Arnold Networks**
```c
#include "KAN/KAN.h"

// Create interpretable network
KAN_t *kan = KAN_create(input_dim, hidden_dim, output_dim, num_layers);
KAN_enable_symbolic(kan, true);
char *formula = kan_get_formula(kan, layer_idx, output_idx);
```

### **Generative Adversarial Networks**
```c
#include "GAN/GAN.h"

// Create GAN for data generation
GAN_t *gan = GAN_create(input_dim, hidden_dim, output_dim, latent_dim);
long double *generated = GAN_generate(gan, noise);
GAN_train_step(gan, real_data, batch_size);
```

## 🧪 Testing

Each model type has its own test suite in `utils/TESTS/`:

```bash
cd utils/TESTS
make test-rnn        # Test RNN/LSTM
make test-advanced    # Test advanced network concepts
make test-all          # Run all tests
```

## 📊 Model Comparison

| Model Type | Best For | Complexity | Interpretability | Energy Efficiency |
|------------|----------|------------|------------------|------------------|
| **MLP** | General purpose | Low | Medium | Medium |
| **RNN** | Sequential data | Medium | Low | Medium |
| **GNN** | Graph data | High | Medium | Medium |
| **SNN** | Real-time, low-power | High | Low | **Very High** |
| **KAN** | Scientific computing | Medium | **High** | Medium |
| **GAN** | Data generation | High | Low | Medium |
| **CNN** | Image processing | High | Low | Medium |
| **Transformer** | Text/sequence | **Very High** | Low | Medium |

## 🔧 Usage Guidelines

### **Choose MLP when:**
- Simple classification/regression tasks
- Tabular data
- Baseline model needed

### **Choose RNN when:**
- Time series data
- Sequential dependencies
- Text processing (without attention)

### **Choose GNN when:**
- Social networks
- Molecular structures
- Recommendation systems

### **Choose SNN when:**
- Neuromorphic hardware
- Energy-efficient applications
- Real-time processing

### **Choose KAN when:**
- Scientific modeling
- Interpretable AI needed
- Fewer parameters desired

### **Choose GAN when:**
- Data generation
- Image synthesis
- Creative applications

## 📚 Documentation

Each model directory contains:
- `*.h` - Header file with API
- `*.c` - Implementation file
- `README.md` - Specific documentation
- Examples and usage patterns

## 🤝 Integration

All models share common interfaces:
- **Activation functions**: ReLU, Sigmoid, Tanh, Linear
- **Optimizers**: SGD, Adam, RMSprop, AdaGrad
- **Loss functions**: MSE, Cross-Entropy, Custom
- **Regularization**: L1, L2, Dropout

## 🎯 Applications

### **Game Development**
- **RNN**: Game state prediction, sequence modeling
- **SNN**: Real-time AI, low-power NPCs
- **GAN**: Procedural content generation

### **Scientific Computing**
- **KAN**: Symbolic discovery, interpretable models
- **GNN**: Molecular modeling, network analysis
- **MLP**: Baseline scientific models

### **Computer Vision**
- **CNN**: Image classification, object detection
- **GAN**: Image generation, style transfer
- **Transformer**: Vision transformers

### **Natural Language**
- **RNN**: Text generation, language modeling
- **Transformer**: Machine translation, text classification
- **TOKENIZER**: Text preprocessing

## 🔄 Migration from Old Structure

Old structure:
```
utils/NN/
├── NN.h/c
├── RNN.h/c
├── GNN.h/c
├── SNN.h/c
├── KAN.h/c
├── GAN.h/c
└── ...
```

New structure:
```
models/
├── MLP/          # Standard neural networks
├── RNN/          # Recurrent networks
├── GNN/          # Graph networks
├── SNN/          # Spiking networks
├── KAN/          # Kolmogorov-Arnold networks
├── GAN/          # Generative adversarial networks
├── CONVOLUTION/  # Convolutional networks
├── TRANSFORMER/ # Transformer networks
├── NEAT/         # Neuroevolution
└── TOKENIZER/    # Text processing
```

## 🚀 Future Development

- **Hybrid Models**: Combining multiple model types
- **AutoML**: Automatic model selection and optimization
- **Distributed Training**: Multi-GPU and multi-node training
- **Model Compression**: Pruning, quantization, distillation
- **Explainability**: Feature importance, attribution methods

## 📞 Support

For questions about specific models:
- Check the model-specific README in each directory
- Review the test files for usage examples
- Consult the API documentation in header files

---

**🎯 Ready to build cutting-edge neural networks!**
