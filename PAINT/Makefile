# 3D Paint with Neural Networks - Build System
# ================================================

# Project configuration
PROJECT_NAME = paint3d
VERSION = 1.0.0

# Directories
SRC_DIR = src
BIN_DIR = bin
LIB_DIR = lib
UTILS_DIR = ../utils
BUILD_DIR = build

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
INCLUDES = -I$(SRC_DIR) -I$(UTILS_DIR) -I$(UTILS_DIR)/SDL3 -I$(UTILS_DIR)/NN
LIBS = -lm -pthread

# SDL3 configuration
SDL_CFLAGS = $(shell pkg-config --cflags sdl3 sdl3-ttf 2>/dev/null || echo "")
SDL_LIBS = $(shell pkg-config --libs sdl3 sdl3-ttf 2>/dev/null || echo "")
FRAMEWORKS = -framework OpenGL -arch arm64

# Neural network sources
NN_SOURCES = $(UTILS_DIR)/NN/CONVOLUTION.c $(UTILS_DIR)/NN/TRANSFORMER.c $(UTILS_DIR)/NN/NN.c
SDL_COMPAT = $(UTILS_DIR)/SDL3/SDL3_compat.c

# Source files
CORE_SOURCES = $(SRC_DIR)/core/paint.c
VIEWER_SOURCES = $(SRC_DIR)/viewer/view_3d.c $(SRC_DIR)/utils/mesh_loader.c
TEST_SOURCES = $(SRC_DIR)/utils/test_mesh_generation.c

# Object files
CORE_OBJECTS = $(CORE_SOURCES:%.c=$(BUILD_DIR)/%.o)
VIEWER_OBJECTS = $(VIEWER_SOURCES:%.c=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:%.c=$(BUILD_DIR)/%.o)
NN_OBJECTS = $(NN_SOURCES:%.c=$(BUILD_DIR)/%.o)
SDL_OBJECTS = $(SDL_COMPAT:%.c=$(BUILD_DIR)/%.o)

# Executables
PAINT_EXEC = $(BIN_DIR)/paint
VIEWER_EXEC = $(BIN_DIR)/view_3d
TEST_EXEC = $(BIN_DIR)/test_mesh

# Default target
.PHONY: all clean paint viewer test train install uninstall help

all: $(PAINT_EXEC) $(VIEWER_EXEC) $(TEST_EXEC)

# Main paint application
$(PAINT_EXEC): $(CORE_OBJECTS) $(NN_OBJECTS) $(SDL_OBJECTS) | $(BIN_DIR)
	@echo "Building paint application..."
	$(CC) $(CFLAGS) $(INCLUDES) $(SDL_CFLAGS) \
		-o $@ $(CORE_OBJECTS) $(NN_OBJECTS) $(SDL_OBJECTS) \
		$(LIBS) $(SDL_LIBS) $(FRAMEWORKS)
	@echo "✓ Paint application built: $@"

# 3D viewer
$(VIEWER_EXEC): $(VIEWER_OBJECTS) $(SDL_OBJECTS) | $(BIN_DIR)
	@echo "Building 3D viewer..."
	$(CC) $(CFLAGS) $(INCLUDES) $(SDL_CFLAGS) \
		-o $@ $(VIEWER_OBJECTS) $(SDL_OBJECTS) \
		$(LIBS) $(SDL_LIBS) $(FRAMEWORKS)
	@echo "✓ 3D viewer built: $@"

# Test mesh generation
$(TEST_EXEC): $(TEST_OBJECTS) | $(BIN_DIR)
	@echo "Building mesh test..."
	$(CC) $(CFLAGS) $(INCLUDES) \
		-o $@ $(TEST_OBJECTS) $(LIBS)
	@echo "✓ Mesh test built: $@"

# Object file compilation
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(INCLUDES) $(SDL_CFLAGS) -c $< -o $@

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/src/core
	mkdir -p $(BUILD_DIR)/src/viewer
	mkdir -p $(BUILD_DIR)/src/utils
	mkdir -p $(BUILD_DIR)/$(UTILS_DIR)/NN
	mkdir -p $(BUILD_DIR)/$(UTILS_DIR)/SDL3

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Training target
train: $(PAINT_EXEC)
	@echo "Training neural networks..."
	@echo "Starting paint in training mode..."
	@echo "Instructions:"
	@echo "1. Press 'T' to enter training mode"
	@echo "2. Press 'R' to train networks"
	@echo "3. Press 'S' to save trained networks"
	@echo "4. Press 'ESC' to exit"
	@$(PAINT_EXEC)

# Demo target
demo: $(PAINT_EXEC) $(VIEWER_EXEC)
	@echo "Running complete demo..."
	@echo "1. Paint something on the canvas"
	@echo "2. Press 'G' to generate 3D object"
	@echo "3. Press 'V' to view 3D object"
	@$(PAINT_EXEC)

# Test target
test: $(TEST_EXEC)
	@echo "Running mesh generation test..."
	@$(TEST_EXEC)
	@echo "Test complete! Check for test_terrain.mesh"

# Install target
install: all
	@echo "Installing $(PROJECT_NAME)..."
	mkdir -p /usr/local/bin
	cp $(PAINT_EXEC) /usr/local/bin/$(PROJECT_NAME)
	cp $(VIEWER_EXEC) /usr/local/bin/$(PROJECT_NAME)_viewer
	@echo "✓ Installed to /usr/local/bin"

# Uninstall target
uninstall:
	@echo "Uninstalling $(PROJECT_NAME)..."
	rm -f /usr/local/bin/$(PROJECT_NAME)
	rm -f /usr/local/bin/$(PROJECT_NAME)_viewer
	@echo "✓ Uninstalled"

# Clean target
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	rm -f *.mesh *.net
	rm -f trained_convolution.net
	rm -f generated_3d_object.mesh
	rm -f test_terrain.mesh
	@echo "✓ Clean complete"

# Deep clean (including backups)
deep-clean: clean
	@echo "Deep cleaning..."
	rm -f *.o *.exe *.out
	rm -f core core.*
	rm -f *.log *.bak
	@echo "✓ Deep clean complete"

# Help target
help:
	@echo "$(PROJECT_NAME) v$(VERSION) - 3D Paint with Neural Networks"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all components (default)"
	@echo "  paint      - Build paint application only"
	@echo "  viewer     - Build 3D viewer only"
	@echo "  test       - Build and run mesh test"
	@echo "  train      - Train neural networks"
	@echo "  demo       - Run complete demo"
	@echo "  install    - Install to system"
	@echo "  uninstall  - Remove from system"
	@echo "  clean      - Remove build files"
	@echo "  deep-clean - Remove all generated files"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build everything"
	@echo "  make paint        - Build paint app"
	@echo "  make viewer       - Build 3D viewer"
	@echo "  make train        - Train networks"
	@echo "  make demo         - Run demo"
	@echo "  make clean        - Clean build files"

# Individual targets for convenience
paint: $(PAINT_EXEC)
viewer: $(VIEWER_EXEC)
test: $(TEST_EXEC)

# Dependencies
$(CORE_OBJECTS): $(NN_OBJECTS) $(SDL_OBJECTS)
$(VIEWER_OBJECTS): $(SDL_OBJECTS)

# Prevent deletion of intermediate files
.PRECIOUS: $(BUILD_DIR)/%.o
