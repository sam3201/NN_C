# Makefile for Diep.io Tank Game
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
LIBS = -lm

# Source files
SIMPLE_SRC = diep_simple.c
SIMPLE_TARGET = diep_simple

FULL_SRC = diep_game.c
FULL_TARGET = diep_game

# Default target
all: $(SIMPLE_TARGET)

# Console version (working)
$(SIMPLE_TARGET): $(SIMPLE_SRC)
	$(CC) $(CFLAGS) $(SIMPLE_SRC) -o $(SIMPLE_TARGET) $(LIBS)

# Full SDL3 version (requires dependencies)
$(FULL_TARGET): $(FULL_SRC)
	@echo "Full version requires SDL3 dependencies - see Makefile.diep"

# Clean build files
clean:
	rm -f $(SIMPLE_TARGET) $(FULL_TARGET)

# Run console version
run: $(SIMPLE_TARGET)
	./$(SIMPLE_TARGET)

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: $(SIMPLE_TARGET)

.PHONY: all clean run debug
