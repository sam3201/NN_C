# Diep.io Tank Game

A console-based tank battle game inspired by diep.io with AI opponent.

## Features

- Player-controlled tank (green 'P')
- AI-controlled opponent tank (red 'A') 
- Real-time combat with bullets
- Health system and respawning
- Score tracking
- Simple but challenging AI opponent

## Controls

- **W/S** - Move forward/backward
- **A/D** - Rotate left/right  
- **Space** - Shoot
- **Q** - Quit game

## Building and Running

```bash
# Compile the game
gcc -o diep_simple diep_simple.c -lm

# Run the game
./diep_simple
```

## Game Mechanics

- Tanks have 100 HP and respawn after 3 seconds when destroyed
- Bullets deal 20 damage per hit
- AI opponent tracks and shoots at the player
- Score increases when you destroy the opponent
- Arena has boundaries that tanks and bullets cannot cross

## Future Enhancements

The full SDL3 version with MUZE AI integration is planned:
- Graphics-based rendering
- Advanced AI that learns from gameplay
- Power-ups and upgrades
- Multiple tank types
- Network multiplayer

## Files

- `diep_simple.c` - Console version (working)
- `diep_game.c` - Full SDL3 version (requires dependency fixes)
- `Makefile.diep` - Build configuration for full version
