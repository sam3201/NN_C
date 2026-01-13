# MUZE Verbose Control

This system allows you to control the verbose output of MUZE print statements across all applications (GAME, SIMULATION, JUMP).

## Usage

All applications now support command line arguments to control MUZE verbosity:

### Enable verbose output
```bash
./game --verbose
# or
./game -v

./sim --verbose  
# or
./sim -v

./jump --verbose
# or  
./jump -v
```

### Disable verbose output (default)
```bash
./game --quiet
# or
./game -q

./sim --quiet
# or
./sim -q

./jump --quiet
# or
./jump -q
```

## What it controls

The verbose flag controls these MUZE print statements:
- `[reanalyze]` - Reanalysis progress and statistics
- `[eval]` - Evaluation results and metrics  
- `[loop]` - Training loop iterations and progress
- `[train]` - Training loss and statistics
- `[selfplay]` - Self-play episode progress and metrics

When verbose mode is disabled (`--quiet`), all these print statements are suppressed, making the console output much cleaner during runtime.

## Implementation

The system uses:
- `muze_verbose.h` - Header with macros and function declarations
- `muze_verbose.c` - Implementation with global flag `g_muze_verbose`
- `MUZE_PRINTF*` macros - Conditional printing macros that check the flag
- Command line parsing in each application's main function

Default behavior is quiet (disabled) to provide cleaner console output. Use `--verbose` to enable MUZE logging when needed.
