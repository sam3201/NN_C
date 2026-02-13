# Neural Network Core Library (Legacy but Core)

This directory contains the foundational neural network implementations that form the core of SAM-D's learning capabilities.

## Directory Structure

```
NN/
├── CNN/            # Convolutional Neural Networks
├── RNN/            # Recurrent Neural Networks  
├── LSTM/           # Long Short-Term Memory networks
├── TRANSFORMER/    # Transformer architectures
├── NEAT/           # NeuroEvolution of Augmenting Topologies
├── GAN/            # Generative Adversarial Networks
├── GNN/            # Graph Neural Networks
├── SNN/            # Spiking Neural Networks
├── KAN/            # Kolmogorov-Arnold Networks
├── MEMORY/         # Memory-augmented networks
└── UTILS/          # NN utilities and helpers
```

## Purpose

While these implementations are considered "legacy" in the sense that they predate the current ΨΔ•Ω-Core architecture, they remain the **core operational foundation** of SAM-D's neural computation capabilities. These modules provide:

- **Fundamental Learning Primitives**: Basic NN operations that higher-level systems build upon
- **Architectural Diversity**: Multiple network types for different cognitive tasks
- **Evolutionary Path**: Historical implementations showing the progression to current systems
- **Reference Implementations**: Clean, educational implementations of key algorithms

## Integration with ΨΔ•Ω-Core

The NN directory works in concert with the main SAM-D system:

- C extensions in `src/c_modules/` may reference these implementations
- Python orchestration layers use these as computational substrates
- Training pipeline (`training/`) can utilize these architectures
- Meta-controller can spawn specialized NN instances from these templates

## Status

- **Maintenance**: Legacy (stable, not actively extended)
- **Usage**: Core (actively used by production systems)
- **Priority**: High (foundational infrastructure)

## Note

This directory preserves the original neural network research and implementations that led to the current SAM-D AGI architecture. While the system has evolved to use more sophisticated C-accelerated cores and hybrid Python/C orchestration, these NN modules remain essential for understanding and extending the system's capabilities.
