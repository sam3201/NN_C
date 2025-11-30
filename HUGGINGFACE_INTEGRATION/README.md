# Hugging Face Integration for SAM Model

This module enables the SAM model to communicate with and learn from Hugging Face transformer models.

## Overview

The integration allows:
- **Knowledge Transfer**: Extract knowledge from pre-trained Hugging Face models
- **Distillation**: Use HF models as teachers to train SAM
- **Fine-tuning**: Leverage HF embeddings and representations
- **Hybrid Training**: Combine HF model outputs with SAM's adaptive learning

## Structure

```
HUGGINGFACE_INTEGRATION/
├── README.md              # This file
├── hf_interface.h         # C interface for Hugging Face models
├── hf_interface.c        # Implementation (Python bridge)
├── hf_trainer.c          # Training script using HF models
├── hf_trainer.py          # Python wrapper for HF model access
├── requirements.txt       # Python dependencies
└── build.sh              # Build script
```

## Requirements

- Python 3.8+
- transformers library
- torch (PyTorch)
- numpy

## Usage

1. Install Python dependencies:
   ```bash
   cd HUGGINGFACE_INTEGRATION
   pip install -r requirements.txt
   ```

2. Build the integration:
   ```bash
   ./build.sh
   ```

3. Train SAM using a Hugging Face model:
   ```bash
   ./hf_trainer [model_name] [epochs] [data_file]
   ```

   Examples:
   ```bash
   # Train with GPT-2 on Romeo & Juliet
   ./hf_trainer gpt2 10 ../utils/DATASETS/RomeoAndJuliet.txt
   
   # Train with DistilBERT on Frankenstein
   ./hf_trainer distilbert-base-uncased 20 ../utils/DATASETS/Frankenstein.txt
   
   # Train with BERT
   ./hf_trainer bert-base-uncased 15 ../utils/DATASETS/RomeoAndJuliet.txt
   ```

## Supported Models

- GPT-2 (gpt2)
- BERT (bert-base-uncased)
- DistilBERT (distilbert-base-uncased)
- T5 (t5-small)
- And more...

## Communication Mode

The integration also supports direct communication between SAM and HF models:

```bash
# Single query about self-actualization (default)
./sam_hf_bridge gpt2

# Interactive dialogue
./sam_hf_bridge gpt2 interactive
```

This allows SAM to:
- Query HF models with prompts
- Process HF responses
- Learn from HF model outputs
- Have interactive conversations

### Default Query

By default, `sam_hf_bridge` asks HF models:
**"How can we create a model that self actualizes?"**

SAM then processes the response and can learn from it.

