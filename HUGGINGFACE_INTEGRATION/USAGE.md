# SAM-Hugging Face Communication Usage Guide

## Setting Up Your Starting Prompt

### Method 1: Edit prompt.txt (Recommended)
Edit the file `prompt.txt` in the HUGGINGFACE_INTEGRATION directory:

```bash
cd HUGGINGFACE_INTEGRATION
nano prompt.txt  # or use your preferred editor
```

Put your starting prompt in the file, for example:
```
How can we create a model that self actualizes?
```

### Method 2: Provide Prompt as Command Line Argument
```bash
./sam_hf_bridge gpt2 "What is the meaning of consciousness in AI?"
```

### Method 3: Use Custom Prompt File
```bash
./sam_hf_bridge gpt2 interactive my_custom_prompt.txt
```

## Running the Communication Bridge

### Continuous Dialogue Mode (Default)
The conversation will continue automatically, using each HF response as the next prompt:

```bash
./sam_hf_bridge gpt2
```

**Controls:**
- `[Enter]` - Continue conversation (uses HF response as next prompt)
- `new` - Enter a new prompt
- `stop` or `quit` - End conversation

### Interactive Mode
You control each exchange:

```bash
./sam_hf_bridge gpt2 interactive
```

**Controls:**
- Type your message and press Enter
- `new` - Start a new conversation thread
- `quit`, `stop`, or `q` - End dialogue

## Conversation Flow

### Continuous Mode Flow:
1. Starts with your prompt (from prompt.txt or command line)
2. SAM sends prompt to HF model
3. HF model generates response
4. SAM processes and learns from response
5. You choose: continue (use HF response as next prompt), enter new prompt, or stop

### Interactive Mode Flow:
1. You type a message
2. SAM sends it to HF model
3. HF model responds
4. SAM processes and learns
5. Repeat until you type 'quit'

## Examples

### Example 1: Self-Actualization Discussion
```bash
# Edit prompt.txt to contain:
# "How can we create a model that self actualizes?"

./sam_hf_bridge gpt2
# Conversation continues automatically, exploring the topic
```

### Example 2: Custom Topic
```bash
./sam_hf_bridge gpt2 "What are the philosophical implications of artificial general intelligence?"
```

### Example 3: Interactive Exploration
```bash
./sam_hf_bridge gpt2 interactive
# Then type your questions one by one
```

## Tips

1. **Long Conversations**: The continuous mode can run indefinitely - SAM learns from each exchange
2. **Changing Topics**: Use 'new' to start a new conversation thread
3. **Saving Progress**: SAM automatically saves after each session
4. **Model Selection**: Try different HF models:
   - `gpt2` - Fast, good for general conversation
   - `gpt2-medium` - Better quality, slower
   - `distilbert-base-uncased` - Different style
   - `bert-base-uncased` - Another perspective

## Ending Conversations

The conversation only ends when **you choose to stop it**:
- Type `stop`, `quit`, or `q` in continuous mode
- Type `quit`, `stop`, or `q` in interactive mode
- Press Ctrl+C to force exit

Otherwise, the conversation continues indefinitely, allowing SAM to learn from extended dialogues.

