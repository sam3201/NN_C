# Makefile for SAM LLM Chatbot

CC = gcc
CFLAGS = -Wall -Wextra -O2
INCLUDES = -I../utils/Raylib/src -I../SAM -I../utils/NN
LIBS = -lm -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo

# Raylib source files (minimal set for window and text)
RAYLIB_SRC = ../utils/Raylib/src/rcore.c \
             ../utils/Raylib/src/rshapes.c \
             ../utils/Raylib/src/rtext.c \
             ../utils/Raylib/src/utils.c

# SAM and NN source files
SAM_SRC = ../SAM/SAM.c \
          ../utils/NN/transformer.c \
          ../utils/NN/NEAT.c \
          ../utils/NN/NN.c

# All source files
ALL_SRC = chatbot.c $(RAYLIB_SRC) $(SAM_SRC)

# Output
TARGET = chatbot

all: $(TARGET)

$(TARGET): $(ALL_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(ALL_SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

