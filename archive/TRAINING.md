# SAM Chatbot Training Guide

This guide explains how to train the SAM chatbot model using all available text files from the TESTS directory.

## Quick Start

1. **Build the training program:**
   ```bash
   cd SAM_LLM
   ./train.sh
   ```

2. **Run training:**
   ```bash
   ./train_chatbot [directory] [epochs]
   ```

   Examples:
   ```bash
   # Train on DATASETS (Romeo & Juliet, Frankenstein) + TESTS with 10 epochs (default)
   ./train_chatbot
   
   # Train with custom epochs
   ./train_chatbot ../utils/DATASETS 20
   
   # Train on a different directory
   ./train_chatbot /path/to/text/files 15
   ```

## How It Works

The training script:

1. **Scans the DATASETS directory** (default: `../utils/DATASETS`) for all text files:
   - **RomeoAndJuliet.txt** (159K) - Shakespeare's classic play
   - **Frankenstein.txt** (430K) - Mary Shelley's novel
   - Other `.txt`, `.c`, `.h`, `.md` files
2. **Also includes TESTS directory** (`../utils/TESTS`) for additional training data
3. **Extracts word sequences** from each file
4. **Creates training samples** by:
   - Taking sequences of 10 words
   - Using the next word as the target
5. **Trains the SAM model** using:
   - Word-level encoding (character-based)
   - Sequence-to-sequence learning
   - Multiple epochs of training
6. **Saves the trained model** to:
   - `../sam_trained_model.bin` (default, used by chatbot)
   - `../sam_chatbot_YYYYMMDD_HHMMSS.bin` (timestamped backup)

**Expected training data:**
- ~75,000+ training samples from Romeo & Juliet and Frankenstein
- ~200 additional samples from test files

## Training Parameters

- **Sequence Length**: 10 words (fixed)
- **Model Dimensions**: 256 input, 256 output
- **Number of Heads**: 8 (transformer attention heads)
- **Learning Rate**: 0.001 (fixed)
- **Epochs**: Configurable (default: 10)

## Output

The training script will:
- Show progress for each epoch
- Display average loss per epoch
- Save the model automatically when complete
- Print confirmation messages

## Using the Trained Model

After training, the chatbot will automatically load the model from `../sam_trained_model.bin` when you run:

```bash
./chatbot
```

## Tips

- **More data = better results**: Include more text files in your training directory
- **More epochs**: Increase epochs for better learning (but takes longer)
- **Monitor loss**: Lower loss = better model performance
- **Training time**: Depends on number of samples and epochs (can take several minutes)

## Troubleshooting

- **"No training samples found"**: Make sure the directory contains text files
- **"Failed to initialize SAM model"**: Check that all dependencies are built
- **Slow training**: Normal for large datasets; reduce epochs for faster testing

