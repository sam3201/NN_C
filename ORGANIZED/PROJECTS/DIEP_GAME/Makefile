# Makefile for Diep.io Tank Game
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g $(shell pkg-config --cflags sdl2 sdl2_ttf)
LIBS = $(shell pkg-config --libs sdl2 sdl2_ttf) -lm

# Source files
SOURCES = diep_game.c
TARGET = diep_game

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run game
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: $(TARGET)

# Install dependencies
install-deps:
	brew install sdl2 sdl2_ttf

.PHONY: all clean run debug install-deps
