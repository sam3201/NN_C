CC = gcc
CFLAGS = -O3 -march=native -flto -Wall -Wextra -ffast-math
INCLUDES = -I. -I../UTILS/NN/MUZE -I../SAM
LIBS = -lm

# Source files for MUZE with Dominant Compression (Standalone)
MUZE_SRC = muze_dominant_conversation.c muze_enhanced_model.c muze_enhanced_config.c muze_loop_thread.c muze_dominant_compression_standalone.c
SAM_SRC = ../SAM/SAM.c
UTILS_SRC = ../UTILS/NN/NN.c ../UTILS/NN/transformer.c

# Output executable
OUTPUT = muze_dominant_compression

all: $(OUTPUT)

$(OUTPUT): $(MUZE_SRC) $(SAM_SRC) $(UTILS_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(MUZE_SRC) $(SAM_SRC) $(UTILS_SRC) -o $(OUTPUT) $(LIBS)

clean:
	rm -f $(OUTPUT) *.o

test: $(OUTPUT)
	./$(OUTPUT)

.PHONY: all clean test
