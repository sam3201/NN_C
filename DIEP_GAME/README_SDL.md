# Diep.io Tank Game - SDL Version with One-Hot Vector AI

A graphics-based tank battle game using SDL2 with advanced AI observation system using one-hot vector encoding.

## Features

### Graphics & Rendering
- **SDL2-based graphics** with smooth 60 FPS rendering
- **Visual tank representations** with rotating cannons
- **Bullet physics** and collision detection
- **Arena boundaries** and UI overlay
- **Grid visualization** (press P to pause and see AI observation grid)

### AI System with One-Hot Encoding
- **40x30 observation grid** for AI perception
- **5-channel one-hot encoding**:
  - Channel 0: Empty space (0.0)
  - Channel 1: Self tank (1.0) 
  - Channel 2: Enemy tank (2.0)
  - Channel 3: Self bullets (3.0)
  - Channel 4: Enemy bullets (4.0)
- **6000-dimensional input vector** (40×30×5) ready for neural networks
- **Real-time observation updates** every frame

### Game Mechanics
- **Player tank** (green) vs **AI opponent** (red)
- **Health system**: 100 HP, 20 damage per bullet hit
- **Respawn system**: 3-second respawn timer
- **Score tracking** for competitive gameplay
- **Smooth physics** with friction and momentum

## Controls

- **W/S** - Move forward/backward
- **A/D** - Rotate left/right
- **Space** - Shoot bullets
- **P** - Pause game (shows AI observation grid)
- **Q/Escape** - Quit game

## Building and Running

```bash
# Install dependencies (macOS)
make -f Makefile.sdl install-deps

# Compile the game
make -f Makefile.sdl

# Run the game
make -f Makefile.sdl run
# or directly:
./diep_sdl
```

## AI Observation System

The game uses a sophisticated one-hot vector encoding system:

### Grid Structure
- **40×30 cells** covering the game arena
- **5 channels per cell** for different entity types
- **Total input size**: 6000 float values

### Entity Encoding
```c
#define ENTITY_EMPTY 0.0f
#define ENTITY_SELF 1.0f
#define ENTITY_ENEMY 2.0f
#define ENTITY_BULLET_SELF 3.0f
#define ENTITY_BULLET_ENEMY 4.0f
```

### Usage for Neural Networks
The `get_ai_input_vector()` function provides the complete observation grid as a flat array, perfect for:
- Convolutional neural networks (reshape to 40×30×5)
- Feed-forward networks (6000 input neurons)
- Reinforcement learning agents

## Future Enhancements

### MUZE Integration
- Replace rule-based AI with MUZE learning agent
- Train AI to improve through self-play
- Implement experience replay and model updates

### Advanced Features
- Multiple tank types and upgrades
- Power-ups and special abilities
- Network multiplayer support
- Particle effects and sound

## Files

- `diep_sdl.c` - Main SDL game with one-hot encoding
- `Makefile.sdl` - Build configuration for SDL2 version
- `diep_simple.c` - Console version (fallback)
- `README_SDL.md` - This documentation

## Technical Details

### Dependencies
- SDL2 (graphics and events)
- SDL2_ttf (text rendering)
- Standard C libraries (math, stdio, etc.)

### Performance
- 60 FPS target with VSync
- Efficient collision detection
- Optimized rendering with SDL2 hardware acceleration

### AI Integration Points
The code is structured for easy AI integration:
- `update_observation_grid()` - Creates one-hot encoded state
- `get_ai_input_vector()` - Extracts flat array for neural networks
- `update_ai_decision()` - Placeholder for MUZE agent calls
