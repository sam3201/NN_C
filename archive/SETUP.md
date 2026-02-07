# Setup Guide for Hugging Face Integration

## Prerequisites

Before using the SAM-Hugging Face communication bridge, you need to install Python dependencies.

### Install Python Dependencies

```bash
cd HUGGINGFACE_INTEGRATION
pip install -r requirements.txt
```

This will install:
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model execution
- `numpy` - Numerical operations
- `tokenizers` - Tokenization utilities
- `accelerate` - Model acceleration

### First Run

The first time you run the communicator, Hugging Face will download the model (e.g., GPT-2). This may take a few minutes depending on your internet connection.

### Troubleshooting

**Error: "ModuleNotFoundError: No module named 'numpy'"**
- Solution: Run `pip install -r requirements.txt`

**Error: "Failed to get response from HF model"**
- Check that Python dependencies are installed
- Check your internet connection (first run downloads models)
- Try a different model: `./sam_hf_bridge distilbert-base-uncased`

**Conversation closes immediately**
- Make sure you're waiting for the "Choice:" prompt
- Type your choice and press Enter
- The conversation continues until you type 'stop' or 'quit'

