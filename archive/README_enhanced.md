# Enhanced Diep.io Tank Game

## Overview
This is an enhanced version of the Diep.io tank game with advanced features including large world exploration, NPC AI, leveling system, tank evolution, and training mode for reinforcement learning.

## Features

### 🌍 Large World System
- **World Size**: 4000x3000 units
- **Camera Following**: Smooth camera that follows the player
- **Zoom Controls**: Use +/- keys to zoom in/out
- **Minimap**: Real-time minimap showing all entities and camera view

### 🤖 Advanced AI System
- **20 NPC Tanks**: Scattered across the world with wandering behavior
- **Smart AI**: Enhanced AI with continuous aiming and movement strategies
- **Training Mode**: Toggle with 'T' key for self-play training
- **One-Hot Encoding**: 60x45 grid with 8 channels for AI observation

### ⚡ Leveling & Evolution
- **Experience System**: Gain XP from destroying shapes and tanks
- **5 Tank Types**: Basic → Twin → Sniper → Machine → Destroyer
- **Stat Progression**: Speed, damage, fire rate improve with levels
- **Visual Evolution**: Different cannon configurations per tank type

### 🎮 Controls
- **WASD/Arrow Keys**: Move tank
- **Space**: Fire bullets
- **P**: Pause (shows AI observation grid)
- **T**: Toggle training mode
- **+/-**: Zoom in/out
- **ESC/Q**: Quit game

### 🎯 Tank Types
1. **Basic** (Level 1): Standard single cannon
2. **Twin** (Level 5): Dual cannons with spread
3. **Sniper** (Level 10): Long range, high damage
4. **Machine** (Level 15): Rapid fire, low damage
5. **Destroyer** (Level 20): Massive cannon, high damage

### 🔧 Technical Features
- **SDL2 Graphics**: Hardware accelerated rendering
- **60 FPS**: Smooth gameplay with frame limiting
- **Collision Detection**: Accurate bullet-tank and bullet-shape collisions
- **Respawn System**: 3-second respawn with random positions
- **Shape Farming**: 50 static shapes for XP farming

## Building and Running

### Prerequisites
```bash
# Install dependencies (macOS with Homebrew)
brew install sdl2 sdl2_ttf

# Or on Ubuntu/Debian
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
```

### Compile
```bash
make                    # Build enhanced version
make clean             # Clean build artifacts
make run               # Run the game
make debug             # Debug build
```

### Manual Compilation
```bash
gcc -Wall -Wextra -std=c99 -O2 -g diep_game_enhanced_full.c \
    -o diep_game_enhanced_full \
    $(pkg-config --cflags sdl2 sdl2_ttf) \
    $(pkg-config --libs sdl2 sdl2_ttf) \
    -lm
```

## Game Mechanics

### Experience & Leveling
- **Shapes**: Give XP equal to their radius × 2
- **NPC Tanks**: 30 XP per kill
- **AI Tank**: 50 XP per kill
- **Level Formula**: 100 × current_level XP required

### Tank Stats by Type
| Tank Type | Speed | Bullet Speed | Damage | Fire Rate | Bullets |
|-----------|-------|--------------|--------|-----------|---------|
| Basic     | 3.0   | 8.0          | 20     | 300ms     | 1       |
| Twin      | 2.8   | 8.5          | 15     | 250ms     | 2       |
| Sniper    | 3.2   | 12.0         | 40     | 500ms     | 1       |
| Machine   | 2.5   | 7.0          | 10     | 100ms     | 1       |
| Destroyer | 2.2   | 6.0          | 60     | 800ms     | 1       |

### AI Behavior
- **Continuous Aiming**: Smooth rotation towards targets
- **Distance Management**: Different strategies per tank type
- **Target Selection**: Prioritizes nearest threats
- **Pattern Variation**: Random movement changes

## Training Mode
The training mode provides a framework for reinforcement learning:
- **Self-Play**: AI vs AI matches for training data
- **Observation Space**: 60×45×8 one-hot encoded grid
- **Action Space**: Continuous movement and aiming
- **State Tracking**: Full game state for learning algorithms

## File Structure
```
DIEP_GAME/
├── diep_game.c              # Original simple version
├── diep_game_enhanced_full.c # Enhanced version (this)
├── Makefile                  # Build system
├── .gitignore               # Git ignore rules
└── README_enhanced.md       # This file
```

## Future Enhancements
- [ ] MUZE RL integration
- [ ] Network multiplayer
- [ ] Additional tank types
- [ ] Power-ups and abilities
- [ ] Sound effects and music
- [ ] Save/load game state

## Troubleshooting

### Common Issues
1. **SDL Not Found**: Install SDL2 development packages
2. **Font Loading**: Game works without system fonts
3. **Performance**: Lower world size or reduce entities if lagging
4. **Compilation**: Ensure pkg-config is available

### Debug Mode
```bash
make debug              # Compile with debug symbols
lldb ./diep_game_enhanced_full  # Debug with LLDB
```

## License
This project is for educational and research purposes.
