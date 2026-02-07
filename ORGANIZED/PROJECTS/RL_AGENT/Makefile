# Makefile for Scalable RL Architecture
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -I../../utils/NN/NN -I../../utils/NN/CONVOLUTION -I../../utils/NN/TRANSFORMER -I../../utils/NN/NEAT -I../../utils/NN/MUZE
LIBS = -lm -lpthread

# Source files
RL_AGENT_SOURCES = rl_agent.c \
                   encoder/observation_encoder.c \
                   policy/policy_value_network.c \
                   world_model/world_model.c \
                   training/training_infrastructure.c

# NN library sources
NN_SOURCES = ../../utils/NN/NN/NN.c \
              ../../utils/NN/CONVOLUTION/CONVOLUTION.c \
              ../../utils/NN/TRANSFORMER/TRANSFORMER.c \
              ../../utils/NN/NEAT/NEAT.c \
              ../../utils/NN/TOKENIZER/TOKENIZER.c

TEST_SOURCES = test_rl_agent.c

# Object files
RL_AGENT_OBJECTS = $(RL_AGENT_SOURCES:.c=.o)
NN_OBJECTS = $(NN_SOURCES:.c=.o)
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

# Executable
TARGET = test_rl_agent
LIBRARY = librl_agent.a

# Default target
all: $(TARGET)

# Build test executable
$(TARGET): $(TEST_OBJECTS) $(RL_AGENT_OBJECTS) $(NN_OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) -o $@ $(TEST_OBJECTS) $(RL_AGENT_OBJECTS) ../utils/NN/NN/NN.o ../utils/NN/CONVOLUTION/CONVOLUTION.o ../utils/NN/TRANSFORMER/TRANSFORMER.o ../utils/NN/NEAT/NEAT.o ../utils/NN/TOKENIZER/TOKENIZER.o

# Build static library
librl_agent.a: $(RL_AGENT_OBJECTS)
	ar rcs $@ $^

# Build all objects
$(RL_AGENT_OBJECTS):
	$(CC) $(CFLAGS) $(INCLUDES) -c rl_agent.c -o rl_agent.o
	$(CC) $(CFLAGS) $(INCLUDES) -c encoder/observation_encoder.c -o encoder/observation_encoder.o
	$(CC) $(CFLAGS) $(INCLUDES) -c policy/policy_value_network.c -o policy/policy_value_network.o
	$(CC) $(CFLAGS) $(INCLUDES) -c world_model/world_model.c -o world_model/world_model.o
	$(CC) $(CFLAGS) $(INCLUDES) -c training/training_infrastructure.c -o training/training_infrastructure.o

$(NN_OBJECTS):
	$(CC) $(CFLAGS) $(INCLUDES) -c ../utils/NN/NN/NN.c -o ../utils/NN/NN/NN.o
	$(CC) $(CFLAGS) $(INCLUDES) -c ../utils/NN/CONVOLUTION/CONVOLUTION.c -o ../utils/NN/CONVOLUTION/CONVOLUTION.o
	$(CC) $(CFLAGS) $(INCLUDES) -c ../utils/NN/TRANSFORMER/TRANSFORMER.c -o ../utils/NN/TRANSFORMER/TRANSFORMER.o
	$(CC) $(CFLAGS) $(INCLUDES) -c ../utils/NN/NEAT/NEAT.c -o ../utils/NN/NEAT/NEAT.o
	$(CC) $(CFLAGS) $(INCLUDES) -c ../utils/NN/TOKENIZER/TOKENIZER.c -o ../utils/NN/TOKENIZER/TOKENIZER.o

$(TEST_OBJECTS):
	$(CC) $(CFLAGS) $(INCLUDES) -c test_rl_agent.c -o test_rl_agent.o

# Clean
clean:
	rm -f $(TARGET) $(LIBRARY) *.o $(NN_OBJECTS)

# Install (copy headers to include directory)
install:
	mkdir -p /usr/local/include/rl_agent
	cp *.h /usr/local/include/rl_agent/

# Uninstall
uninstall:
	rm -rf /usr/local/include/rl_agent

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build test executable"
	@echo "  clean      - Remove built files"
	@echo "  install    - Install headers"
	@echo "  uninstall  - Remove installed files"
	@echo "  help       - Show this help message"

# Dependencies
.PHONY: all clean install uninstall help nn_deps
