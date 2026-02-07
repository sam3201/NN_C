# Makefile for SAM Neural Core
# Compiles C code into shared library for Python integration

CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC -std=c99
LDFLAGS = -shared -lm

# Directories
SAM_DIR = ORGANIZED/UTILS/SAM/SAM
NN_DIR = ORGANIZED/UTILS/models/MLP
MUZE_DIR = ORGANIZED/UTILS/utils/NN/MUZE
NEAT_DIR = ORGANIZED/UTILS/utils/NN/NEAT
TRANSFORMER_DIR = ORGANIZED/UTILS/utils/NN/TRANSFORMER

# Include paths
INCLUDES = -I$(SAM_DIR) -I$(NN_DIR) -I$(MUZE_DIR) -I$(NEAT_DIR) -I$(TRANSFORMER_DIR) -I.

# Source files - core morphogenesis and batch learning only
SAM_SOURCES = $(SAM_DIR)/sam_morphogenesis.c \
              $(SAM_DIR)/sam_full_context.c

NN_SOURCES = $(NN_DIR)/NN.c

ALL_SOURCES = $(SAM_SOURCES) $(NN_SOURCES)

# Output library (platform-specific)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_NAME = libsam_core.dylib
    LDFLAGS += -dynamiclib -install_name @rpath/libsam_core.dylib
else
    LIB_NAME = libsam_core.so
endif

# Targets
.PHONY: all clean shared test install

all: shared

shared: $(LIB_NAME)

$(LIB_NAME): $(ALL_SOURCES)
	@echo "🔧 Compiling SAM Neural Core..."
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^
	@echo "✅ Built: $(LIB_NAME)"

# Compile individual object files (for debugging)
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(LIB_NAME)
	rm -f *.o
	find . -name "*.o" -delete
	@echo "🧹 Cleaned build artifacts"

test: shared
	@echo "🧪 Testing SAM Neural Core..."
	python3 sam_neural_core.py

install: shared
	@echo "📦 Installing SAM Neural Core..."
	cp $(LIB_NAME) /usr/local/lib/ 2>/dev/null || cp $(LIB_NAME) .
	@echo "✅ Installation complete"

# Development targets
dev-setup:
	@echo "🔧 Setting up development environment..."
	@echo "Checking dependencies..."
	@which gcc > /dev/null || (echo "❌ gcc not found" && exit 1)
	@echo "✅ gcc found"
	@python3 --version > /dev/null 2>&1 || (echo "❌ Python3 not found" && exit 1)
	@echo "✅ Python3 found"
	@echo "✅ Development environment ready"

# Debug build
debug: CFLAGS = -Wall -Wextra -g -O0 -fPIC -std=c99 -DDEBUG
debug: $(LIB_NAME)
	@echo "🔧 Debug build complete"

# Static library (alternative)
static: $(ALL_SOURCES)
	@echo "🔧 Building static library..."
	$(CC) $(CFLAGS) $(INCLUDES) -c $^
	ar rcs libsam_core.a *.o
	@echo "✅ Static library: libsam_core.a"

help:
	@echo "SAM Neural Core Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  shared    - Build shared library (default)"
	@echo "  static    - Build static library"
	@echo "  debug     - Build debug version"
	@echo "  test      - Run tests"
	@echo "  install   - Install library"
	@echo "  clean     - Clean build artifacts"
	@echo "  dev-setup - Check development dependencies"
	@echo "  help      - Show this help"
