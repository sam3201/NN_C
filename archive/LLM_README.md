# SAM LLM Chatbot

A simple, barebones chatbot GUI using raylib that interfaces with the SAM AGI model.

## Features

- Simple text-based chat interface
- Real-time message display
- Input box with text entry
- Scrollable chat history
- Integration with SAM model for responses
- Automatic model loading

## Building

Simply run the build script:

```bash
./build.sh
```

Or use make:

```bash
make
```

## Running

```bash
./chatbot
```

## Usage

1. The chatbot will attempt to load a trained SAM model from:
   - `../sam_trained_model.bin`
   - `../sam_hello_world.bin`

2. Type your message in the input box at the bottom

3. Press Enter or click the "Send" button to send your message

4. The SAM model will generate a response

5. Use the mouse wheel to scroll through chat history

## Requirements

- Trained SAM model (run training scripts first)
- macOS (for the current build configuration)
- raylib library (included in utils/Raylib)

## Notes

- This is a barebones implementation
- Model responses are character-level encoded/decoded
- The GUI is simple and functional, not polished
- For best results, train the SAM model first using the training scripts

