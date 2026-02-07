# Diep.io Tank Simulation

A simulation environment where all NPCs are RL agents, with no human player. Watch as 20 AI-controlled tanks battle, evolve, and compete for dominance in a large world.

## Features

### Core Simulation
- **20 RL Agents**: All tanks are controlled by AI agents with observation grids and action spaces
- **No Human Player**: Pure observation mode - watch the simulation unfold
- **Large World**: 4000x3000 world with camera system
- **Auto-follow Camera**: Automatically follows the highest-scoring agent

### AI System
- **Grid-based Observation**: 60x45 grid with 8 channels (21600 input size)
- **Continuous Action Space**: Movement (x,y), aiming angle, and shooting
- **Rule-based AI**: Currently uses intelligent rule-based behavior (ready for neural network integration)
- **Tank-specific Strategies**: Different behavior patterns for each tank type

### Evolution System
- **5 Tank Types**: Basic → Twin → Sniper → Machine → Destroyer
- **Level Progression**: Agents gain XP from destroying shapes and other agents
- **Stat Scaling**: Speed, damage, and health improve with levels
- **Visual Evolution**: Tank appearance changes based on type

### Training Features
- **Fitness Scoring**: Based on score, level, and survival time
- **Performance Ranking**: Real-time display of top 5 agents
- **Generation Tracking**: Ready for evolutionary training
- **Respawn System**: Agents respawn after 2 seconds

## Controls

- **P**: Pause/Resume simulation (shows grid visualization when paused)
- **+/-**: Zoom in/out
- **Space**: Focus camera on random agent
- **ESC/Q**: Exit simulation

## Building and Running

### Prerequisites
```bash
# Install dependencies (macOS with Homebrew)
brew install sdl2 sdl2_ttf

# Or use the provided Makefile target
make -f Makefile.simulation install-deps
```

### Build
```bash
# Using the Makefile
make -f Makefile.simulation

# Or manually
gcc -Wall -Wextra -std=c99 -O2 -g diep_simulation_part3.c -o diep_simulation $(pkg-config --cflags sdl2 sdl2_ttf) $(pkg-config --libs sdl2 sdl2_ttf) -lm
```

### Run
```bash
# Using the Makefile
make -f Makefile.simulation run

# Or directly
./diep_simulation
```

## AI Architecture

### Observation System
The AI uses a one-hot encoded grid observation:
- Channel 0: Empty space
- Channel 1: Self agent
- Channel 2: Other agents
- Channel 3: Self bullets
- Channel 4: Enemy bullets
- Channel 5: Shapes
- Channels 6-7: Reserved for future features

### Action Space
Continuous 4-dimensional action vector:
- `action[0]`: Movement X velocity
- `action[1]`: Movement Y velocity
- `action[2]`: Aiming angle
- `action[3]`: Shoot trigger (0.0 or 1.0)

### Tank Behaviors
- **Basic**: Balanced offense and farming
- **Twin**: Dual cannons, moderate range
- **Sniper**: Long-range combat, high damage
- **Machine**: Rapid fire, low damage
- **Destroyer**: Aggressive, massive damage

## Fitness Calculation

```
Fitness = Score × 1.0 + Level × 50.0 + (Alive ? 100.0 : 0.0)
```

## Future Enhancements

### Neural Network Integration
The simulation is designed to integrate with MUZE or other RL frameworks:
- Replace `calculate_ai_action()` with neural network inference
- Use the existing observation/action interfaces
- Implement reinforcement learning training loops

### Advanced Features
- Multi-agent cooperation/competition scenarios
- Environmental obstacles and terrain
- Resource collection mechanics
- Team-based gameplay

## File Structure

- `diep_simulation.c`: Core game logic and AI system
- `diep_simulation_part2.c`: Game mechanics and updates
- `diep_simulation_part3.c`: Rendering and UI
- `Makefile.simulation`: Build configuration
- `README_simulation.md`: This documentation

## Performance Notes

- Runs at ~60 FPS with 20 agents
- Efficient grid-based collision detection
- Optimized rendering with viewport culling
- Memory-efficient entity management

## Troubleshooting

### Font Loading
If Arial.ttf is not found, the simulation uses a fallback UI with rectangles instead of text.

### Performance Issues
- Reduce `MAX_AGENTS` if performance is poor
- Lower screen resolution or zoom out to reduce rendering load
- Use `-O3` optimization for better performance

### Compilation Issues
Ensure SDL2 and SDL2_ttf are properly installed and pkg-config can find them.
