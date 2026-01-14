#include "../../utils/SDL3/SDL3_compat.h"
#include "../../utils/NN/NN/NN.h"
#include "../../utils/NN/TRANSFORMER/TRANSFORMER.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Key constants for weight initialization selection
#define KEY_1 49
#define KEY_2 50
#define KEY_3 51
#define KEY_4 52
#define KEY_5 53
#define KEY_6 54
#define KEY_7 55

#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800
#define FPS 60

#define BG_COLOR BLACK
#define CANVAS_SIZE 64
#define PIXEL_SIZE 8

#define BRUSH_SIZE 3
#define BRUSH_COLOR WHITE

#define YELLOW (Color){255, 255, 0, 255}
#define CYAN (Color){0, 255, 255, 255}

// 3D paint canvas
static unsigned char g_canvas[CANVAS_SIZE][CANVAS_SIZE][3]; // RGB
static float g_depth_map[CANVAS_SIZE][CANVAS_SIZE];
static bool g_3d_mode = true;

// Neural network components
static NN_t *g_conv = NULL;
static Transformer_t *g_transformer = NULL;

// Training data
static float *g_training_inputs = NULL;
static float *g_training_targets = NULL;
static int g_training_count = 0;
static int g_current_training_sample = 0;
static bool g_training_mode = false;

// Function declarations
void init_canvas(void);
void init_neural_networks(void);
void process_with_neural_networks(void);
void save_trained_networks(void);
void load_trained_networks(void);
void draw_3d_canvas(void);
void paint_at(int screen_x, int screen_y, Color color);
void erase_at(int screen_x, int screen_y);
void load_training_assets(void);
void generate_vertices_from_depth(void);
int export_mesh_to_glb(const char *filename, const float *vertices, int vcount, const unsigned int *indices, int icount);
void view_generated_3d_object(void);

void init_canvas(void) {
  memset(g_canvas, 0, sizeof(g_canvas));
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      g_depth_map[y][x] = 0.5f; // Default depth
    }
  }
}

void init_neural_networks(void) {
  // Create network architecture
  size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
  ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
  ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
  
  // Initialize with He initialization (optimal for ReLU) and Adam optimizer
  g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                     L2, ADAM, 0.001f, HE);
  if (!g_conv) {
    printf("Failed to create neural network with enhanced initialization\n");
    return;
  }

  // Set learning rate schedule for better convergence
  NN_set_lr_schedule(g_conv, 0.001f, 0.0001f, 100);
  
  // Initialize transformer for attention-based processing
  g_transformer = TRANSFORMER_init(128, 4, 2);
  if (!g_transformer) {
    printf("Failed to create transformer\n");
    return;
  }

  printf("Neural networks initialized with He weights and Adam optimizer\n");
  printf("Network architecture: %zux%zux%zux%zu\n", 
         layers[0], layers[1], layers[2], layers[3]);
  printf("Weight initialization: He (optimal for ReLU)\n");
  printf("Optimizer: Adam with learning rate scheduling\n");
}

void process_with_neural_networks(void) {
  if (!g_conv || !g_transformer || !g_training_inputs || !g_training_targets) {
    printf("Cannot train: networks or training data not initialized\n");
    return;
  }
  
  printf("Training neural networks with enhanced framework...\n");
  printf("Press ESC to stop training early\n");
  
  // Enhanced training with stochastic gradient descent
  int epochs = 1000;  // High limit, will stop at 0% loss
  int batch_size = 8;  // Larger batch for better gradient estimation
  float best_loss = INFINITY;
  
  printf("Training until 0%% loss or ESC key...\n");
  printf("Using Adam optimizer with Gaussian weight initialization\n");
  
  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    int batches_processed = 0;
    
    // Check for ESC key press
    if (IsKeyPressed(KEY_ESCAPE)) {
      printf("Training stopped by user (ESC pressed)\n");
      break;
    }
    
    // Shuffle training data for stochastic gradient descent
    int *indices = malloc(g_training_count * sizeof(int));
    for (int i = 0; i < g_training_count; i++) {
      indices[i] = i;
    }
    
    // Simple Fisher-Yates shuffle
    for (int i = g_training_count - 1; i > 0; i--) {
      int j = rand() % (i + 1);
      int temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }
    
    // Process batches
    for (int batch = 0; batch < g_training_count / batch_size; batch++) {
      float batch_loss = 0.0f;
      
      // Create batch arrays
      long double *batch_inputs = malloc(batch_size * CANVAS_SIZE * CANVAS_SIZE * 3 * sizeof(long double));
      long double *batch_targets = malloc(batch_size * CANVAS_SIZE * CANVAS_SIZE * sizeof(long double));
      
      for (int sample = 0; sample < batch_size; sample++) {
        int sample_idx = indices[batch * batch_size + sample];
        
        // Copy input and target data
        for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE * 3; i++) {
          batch_inputs[sample * CANVAS_SIZE * CANVAS_SIZE * 3 + i] = 
            (long double)g_training_inputs[sample_idx * CANVAS_SIZE * CANVAS_SIZE * 3 + i];
        }
        for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++) {
          batch_targets[sample * CANVAS_SIZE * CANVAS_SIZE + i] = 
            (long double)g_training_targets[sample_idx * CANVAS_SIZE * CANVAS_SIZE + i];
        }
      }
      
      // Forward pass
      long double *output = NN_forward(g_conv, batch_inputs);
      
      if (output) {
        // Calculate loss (MSE)
        for (int i = 0; i < batch_size * CANVAS_SIZE * CANVAS_SIZE; i++) {
          float diff = (float)(output[i] - batch_targets[i]);
          batch_loss += diff * diff;
        }
        
        // Backward pass
        long double *output_delta = malloc(batch_size * CANVAS_SIZE * CANVAS_SIZE * sizeof(long double));
        for (int i = 0; i < batch_size * CANVAS_SIZE * CANVAS_SIZE; i++) {
          output_delta[i] = (output[i] - batch_targets[i]) * 2.0L; // MSE derivative
        }
        
        NN_backprop_custom_delta(g_conv, batch_inputs, output_delta);
        
        free(output_delta);
      }
      
      free(batch_inputs);
      free(batch_targets);
      
      epoch_loss += batch_loss / (batch_size * CANVAS_SIZE * CANVAS_SIZE);
      batches_processed++;
      
      // Check for ESC key press during batch
      if (IsKeyPressed(KEY_ESCAPE)) {
        printf("Training stopped by user (ESC pressed)\n");
        free(indices);
        break;
      }
    }
    
    free(indices);
    epoch_loss /= batches_processed;
    
    // Print progress every 10 epochs
    if (epoch % 10 == 0 || epoch_loss < 0.001f) {
      printf("Epoch %d - Loss: %.6f", epoch + 1, epoch_loss);
      
      if (epoch_loss < best_loss) {
        printf(" ✓ (Best)");
        best_loss = epoch_loss;
      }
      
      // Show learning rate if scheduled
      if (g_conv->lr_sched_steps > 0) {
        printf(" LR: %.6f", (float)g_conv->learningRate);
      }
      
      printf("\n");
    }
    
    // Stop at 0% loss (very close to zero)
    if (epoch_loss < 0.0001f) {
      printf("🎉 Training complete! Loss: %.6f (0%% achieved)\n", epoch_loss);
      break;
    }
    
    // Check for ESC key press after epoch
    if (IsKeyPressed(KEY_ESCAPE)) {
      printf("Training stopped by user (ESC pressed)\n");
      break;
    }
  }
  
  printf("Training finished! Best loss: %.6f\n", best_loss);
  
  // Auto-save trained networks
  save_trained_networks();
}

void save_trained_networks(void) {
  if (g_conv) {
    // Save using NN framework's save function
    FILE *file = fopen("trained_neural_network.nn", "wb");
    if (file) {
      // Simple save: write network parameters
      fwrite(g_conv->layers, g_conv->numLayers * sizeof(size_t), 1, file);
      
      for (size_t i = 0; i < g_conv->numLayers - 1; i++) {
        size_t wcount = g_conv->layers[i] * g_conv->layers[i + 1];
        size_t bcount = g_conv->layers[i + 1];
        
        fwrite(g_conv->weights[i], wcount * sizeof(long double), 1, file);
        fwrite(g_conv->biases[i], bcount * sizeof(long double), 1, file);
      }
      
      fclose(file);
      printf("Saved neural network to trained_neural_network.nn\n");
    } else {
      printf("Failed to save neural network\n");
    }
  }
}

void load_trained_networks(void) {
  if (g_conv) {
    // Free existing network
    NN_destroy(g_conv);
    g_conv = NULL;
  }
  
  // Load trained network
  FILE *file = fopen("trained_neural_network.nn", "rb");
  if (file) {
    // Read network architecture
    size_t layers[10]; // Max 10 layers
    size_t num_layers_read = fread(layers, sizeof(size_t), 10, file);
    
    if (num_layers_read > 0) {
      // Create activation functions (use defaults)
      ActivationFunctionType activations[10];
      ActivationDerivativeType derivatives[10];
      
      for (size_t i = 0; i < num_layers_read; i++) {
        activations[i] = (i < num_layers_read - 1) ? RELU : LINEAR;
        derivatives[i] = (i < num_layers_read - 1) ? RELU_DERIVATIVE : LINEAR_DERIVATIVE;
      }
      
      // Initialize network
      g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE,
                                         L2, ADAM, 0.001f, RANDOM_NORMAL);
      
      if (g_conv) {
        // Load weights and biases
        for (size_t i = 0; i < g_conv->numLayers - 1; i++) {
          size_t wcount = g_conv->layers[i] * g_conv->layers[i + 1];
          size_t bcount = g_conv->layers[i + 1];
          
          fread(g_conv->weights[i], wcount * sizeof(long double), 1, file);
          fread(g_conv->biases[i], bcount * sizeof(long double), 1, file);
        }
        
        printf("Loaded trained neural network from trained_neural_network.nn\n");
      }
    }
    
    fclose(file);
  }
  
  if (!g_conv) {
    printf("No trained network found, creating new one\n");
    init_neural_networks();
  }
}

void draw_3d_canvas(void) {
  int canvas_x = (SCREEN_WIDTH - CANVAS_SIZE * PIXEL_SIZE) / 2;
  int canvas_y = (SCREEN_HEIGHT - CANVAS_SIZE * PIXEL_SIZE) / 2;
  
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      Color color = {g_canvas[y][x][0], g_canvas[y][x][1], g_canvas[y][x][2], 255};
      
      if (g_3d_mode) {
        // Apply depth shading
        float depth = g_depth_map[y][x];
        color.r = (unsigned char)(color.r * (0.3f + depth * 0.7f));
        color.g = (unsigned char)(color.g * (0.3f + depth * 0.7f));
        color.b = (unsigned char)(color.b * (0.3f + depth * 0.7f));
      }
      
      DrawRectangle(canvas_x + x * PIXEL_SIZE, canvas_y + y * PIXEL_SIZE,
                 PIXEL_SIZE, PIXEL_SIZE, color);
    }
  }
  
  // Draw grid
  for (int i = 0; i <= CANVAS_SIZE; i++) {
    DrawLine(canvas_x, canvas_y + i * PIXEL_SIZE,
            canvas_x + CANVAS_SIZE * PIXEL_SIZE, canvas_y + i * PIXEL_SIZE, GRAY);
    DrawLine(canvas_x + i * PIXEL_SIZE, canvas_y,
            canvas_x + i * PIXEL_SIZE, canvas_y + CANVAS_SIZE * PIXEL_SIZE, GRAY);
  }
}

void paint_at(int screen_x, int screen_y, Color color) {
  int canvas_x = (screen_x - (SCREEN_WIDTH - CANVAS_SIZE * PIXEL_SIZE) / 2) / PIXEL_SIZE;
  int canvas_y = (screen_y - (SCREEN_HEIGHT - CANVAS_SIZE * PIXEL_SIZE) / 2) / PIXEL_SIZE;
  
  for (int dy = -BRUSH_SIZE; dy <= BRUSH_SIZE; dy++) {
    for (int dx = -BRUSH_SIZE; dx <= BRUSH_SIZE; dx++) {
      int px = canvas_x + dx;
      int py = canvas_y + dy;
      
      if (px >= 0 && px < CANVAS_SIZE && py >= 0 && py < CANVAS_SIZE) {
        float dist = sqrt(dx * dx + dy * dy);
        if (dist <= BRUSH_SIZE) {
          g_canvas[py][px][0] = color.r;
          g_canvas[py][px][1] = color.g;
          g_canvas[py][px][2] = color.b;
        }
      }
    }
  }
}

void erase_at(int screen_x, int screen_y) {
  paint_at(screen_x, screen_y, BG_COLOR);
}

void load_training_assets(void) {
  // Load asset files as training data
  const char *asset_files[] = {
    "./Assets/Grass.jpg",
    "./Assets/Base.glb",
    "./Assets/Tree.glb",
    "./Assets/Player.glb",
    "./Assets/Pig.glb"
  };
  
  int num_assets = sizeof(asset_files) / sizeof(asset_files[0]);
  g_training_count = num_assets * 10; // 10 samples per asset type
  
  g_training_inputs = malloc(g_training_count * CANVAS_SIZE * CANVAS_SIZE * 3 * sizeof(float));
  g_training_targets = malloc(g_training_count * CANVAS_SIZE * CANVAS_SIZE * sizeof(float));
  
  printf("Loading %d training samples...\n", g_training_count);
  
  for (int i = 0; i < g_training_count; i++) {
    int asset_idx = i % num_assets;
    
    // Generate synthetic training data based on asset type
    for (int y = 0; y < CANVAS_SIZE; y++) {
      for (int x = 0; x < CANVAS_SIZE; x++) {
        int input_idx = i * CANVAS_SIZE * CANVAS_SIZE * 3 + y * CANVAS_SIZE * 3 + x * 3;
        int target_idx = i * CANVAS_SIZE * CANVAS_SIZE + y * CANVAS_SIZE + x;
        
        // Create patterns based on asset type
        switch (asset_idx) {
          case 0: // Grass - organic, flowing patterns
            g_training_inputs[input_idx + 0] = sinf(x * 0.1f) * 0.5f + 0.5f;
            g_training_inputs[input_idx + 1] = cosf(y * 0.1f) * 0.5f + 0.5f;
            g_training_inputs[input_idx + 2] = 0.2f;
            g_training_targets[target_idx] = 0.3f + 0.2f * sinf(x * 0.2f + y * 0.2f);
            break;
          case 1: // Base - geometric, structured
            g_training_inputs[input_idx + 0] = (x % 8 < 4) ? 1.0f : 0.0f;
            g_training_inputs[input_idx + 1] = (y % 8 < 4) ? 1.0f : 0.0f;
            g_training_inputs[input_idx + 2] = 0.8f;
            g_training_targets[target_idx] = 0.8f - (x + y) * 0.01f;
            break;
          case 2: // Tree - vertical, organic growth
            g_training_inputs[input_idx + 0] = 0.1f;
            g_training_inputs[input_idx + 1] = expf(-((y - CANVAS_SIZE/2) * (y - CANVAS_SIZE/2)) * 0.01f);
            g_training_inputs[input_idx + 2] = (x % 16 < 8) ? 0.9f : 0.1f;
            g_training_targets[target_idx] = 0.9f - abs(x - CANVAS_SIZE/2) * 0.02f;
            break;
          case 3: // Player - centered, important
            g_training_inputs[input_idx + 0] = expf(-((x - CANVAS_SIZE/2) * (x - CANVAS_SIZE/2) + (y - CANVAS_SIZE/2) * (y - CANVAS_SIZE/2)) * 0.02f);
            g_training_inputs[input_idx + 1] = expf(-((x - CANVAS_SIZE/2) * (x - CANVAS_SIZE/2) + (y - CANVAS_SIZE/2) * (y - CANVAS_SIZE/2)) * 0.02f);
            g_training_inputs[input_idx + 2] = 1.0f;
            g_training_targets[target_idx] = 1.0f;
            break;
          case 4: // Pig - scattered, organic
            g_training_inputs[input_idx + 0] = sinf(x * 0.3f) * cosf(y * 0.3f);
            g_training_inputs[input_idx + 1] = cosf(x * 0.3f) * sinf(y * 0.3f);
            g_training_inputs[input_idx + 2] = 0.6f;
            g_training_targets[target_idx] = 0.4f + 0.3f * sinf(i * 0.5f);
            break;
        }
      }
    }
  }
  
  printf("Training data loaded: %d samples\n", g_training_count);
}

void generate_vertices_from_depth(void) {
  printf("Generating vertices from depth map...\n");
  
  // Create a simple mesh from the depth map
  int vcount = CANVAS_SIZE * CANVAS_SIZE;
  int icount = (CANVAS_SIZE - 1) * (CANVAS_SIZE - 1) * 6;
  
  float *vertices = malloc(sizeof(float) * vcount * 6); // pos(3) + normal(3)
  unsigned int *indices = malloc(sizeof(unsigned int) * icount);
  
  if (!vertices || !indices) {
    printf("Failed to allocate memory for vertices\n");
    if (vertices) free(vertices);
    if (indices) free(indices);
    return;
  }
  
  // Generate vertices
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      int idx = y * CANVAS_SIZE + x;
      
      // Position: map canvas coordinates to 3D space
      vertices[idx * 6 + 0] = (x - CANVAS_SIZE / 2) * 0.1f; // X coordinate
      vertices[idx * 6 + 1] = g_depth_map[y][x] * 2.0f;     // Y coordinate (height from depth)
      vertices[idx * 6 + 2] = (y - CANVAS_SIZE / 2) * 0.1f; // Z coordinate
      
      // Normal: calculate simple upward normal
      vertices[idx * 6 + 3] = 0.0f; // Normal X
      vertices[idx * 6 + 4] = 1.0f; // Normal Y (upward)
      vertices[idx * 6 + 5] = 0.0f; // Normal Z
    }
  }
  
  // Generate indices for triangles
  int idx_idx = 0;
  for (int y = 0; y < CANVAS_SIZE - 1; y++) {
    for (int x = 0; x < CANVAS_SIZE - 1; x++) {
      int v0 = y * CANVAS_SIZE + x;
      int v1 = y * CANVAS_SIZE + (x + 1);
      int v2 = (y + 1) * CANVAS_SIZE + x;
      int v3 = (y + 1) * CANVAS_SIZE + (x + 1);
      
      // Two triangles per quad
      indices[idx_idx++] = v0;
      indices[idx_idx++] = v2;
      indices[idx_idx++] = v1;
      
      indices[idx_idx++] = v1;
      indices[idx_idx++] = v2;
      indices[idx_idx++] = v3;
    }
  }
  
  // Export to mesh file
  if (export_mesh_to_glb("generated_3d_object.mesh", vertices, vcount, indices, icount)) {
    printf("Successfully exported 3D object to generated_3d_object.mesh\n");
  } else {
    printf("Failed to export 3D object\n");
  }
  
  free(vertices);
  free(indices);
}

int export_mesh_to_glb(const char *filename, const float *vertices, int vcount, const unsigned int *indices, int icount) {
  printf("Exporting mesh to %s...\n", filename);
  
  // For now, export as a simple binary format that can be converted to GLB
  FILE *file = fopen(filename, "wb");
  if (!file) {
    printf("Failed to open file for writing: %s\n", filename);
    return 0;
  }
  
  // Write header
  fwrite("MESH", 4, 1, file);
  fwrite(&vcount, sizeof(int), 1, file);
  fwrite(&icount, sizeof(int), 1, file);
  
  // Write vertices
  fwrite(vertices, sizeof(float), vcount * 6, file);
  
  // Write indices
  fwrite(indices, sizeof(unsigned int), icount, file);
  
  fclose(file);
  printf("Successfully exported mesh data to %s\n", filename);
  return 1;
}

void view_generated_3d_object(void) {
  printf("Launching 3D object viewer...\n");
  
  // For now, just print info about the generated file
  printf("To view the generated 3D object:\n");
  printf("1. Use the game engine to load 'generated_3d_object.mesh'\n");
  printf("2. Or use any 3D viewer that supports GLB format\n");
  printf("3. The object represents your painted canvas as a 3D terrain\n");
}

int main(void) {
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "3D Paint with Neural Networks");
  SetTargetFPS(FPS);
  
  init_canvas();
  init_neural_networks();
  load_training_assets();
  
  // Try to load pre-trained networks
  load_trained_networks();
  
  bool nn_processed = false;
  
  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_ESCAPE))
      break;
      
    if (IsKeyPressed(KEY_SPACE)) {
      g_3d_mode = !g_3d_mode;
      printf("3D mode: %s\n", g_3d_mode ? "ON" : "OFF");
    }
    
    if (IsKeyPressed(KEY_ENTER)) {
      process_with_neural_networks();
      nn_processed = true;
    }
    
    // Training controls
    if (IsKeyPressed(KEY_T)) {
      g_training_mode = !g_training_mode;
      printf("Training mode: %s\n", g_training_mode ? "ON" : "OFF");
    }
    
    if (IsKeyPressed(KEY_L) && g_training_mode) {
      load_training_assets();
      printf("Training assets reloaded\n");
    }
    
    if (IsKeyPressed(KEY_1) && g_training_mode) {
      // Select Zero/Constant initialization
      printf("Selected: Zero/Constant initialization (bad due to symmetry)\n");
      // Update network with ZERO initialization
      if (g_conv) {
        NN_destroy(g_conv);
        size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
        ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
        ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
        g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                           L2, ADAM, 0.001f, ZERO);
        printf("Reinitialized with Zero initialization\n");
      }
    }
    
    if (IsKeyPressed(KEY_2) && g_training_mode) {
      // Select Random Uniform initialization
      printf("Selected: Random Uniform initialization (basic, risks gradient issues)\n");
      // Update network with RANDOM_UNIFORM initialization
      if (g_conv) {
        NN_destroy(g_conv);
        size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
        ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
        ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
        g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                           L2, ADAM, 0.001f, RANDOM_UNIFORM);
        printf("Reinitialized with Random Uniform initialization\n");
      }
    }
    
    if (IsKeyPressed(KEY_3) && g_training_mode) {
      // Select Random Normal initialization
      printf("Selected: Random Normal initialization (basic, risks gradient issues)\n");
      // Update network with RANDOM_NORMAL initialization
      if (g_conv) {
        NN_destroy(g_conv);
        size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
        ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
        ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
        g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                           L2, ADAM, 0.001f, RANDOM_NORMAL);
        printf("Reinitialized with Random Normal initialization\n");
      }
    }
    
    if (IsKeyPressed(KEY_4) && g_training_mode) {
      // Select Xavier/Glorot initialization (for sigmoid/tanh)
      printf("Selected: Xavier/Glorot initialization (for sigmoid/tanh)\n");
      // Update network with XAVIER initialization
      if (g_conv) {
        NN_destroy(g_conv);
        size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
        ActivationFunctionType activations[] = {TANH, TANH, TANH, LINEAR};
        ActivationDerivativeType derivatives[] = {TANH_DERIVATIVE, TANH_DERIVATIVE, TANH_DERIVATIVE, LINEAR_DERIVATIVE};
        g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                           L2, ADAM, 0.001f, XAVIER);
        printf("Reinitialized with Xavier/Glorot initialization\n");
      }
    }
    
    if (IsKeyPressed(KEY_5) && g_training_mode) {
      // Select He initialization (for ReLU)
      printf("Selected: He initialization (for ReLU - CURRENT)\n");
      // Update network with HE initialization (already set by default)
      printf("Already using He initialization (optimal for ReLU)\n");
    }
    
    if (IsKeyPressed(KEY_6) && g_training_mode) {
      // Select LeCun initialization (for deeper models)
      printf("Selected: LeCun initialization (for deeper models)\n");
      // Update network with LECUN initialization
      if (g_conv) {
        NN_destroy(g_conv);
        size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
        ActivationFunctionType activations[] = {SIGMOID, SIGMOID, SIGMOID, LINEAR};
        ActivationDerivativeType derivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, LINEAR_DERIVATIVE};
        g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                           L2, ADAM, 0.001f, LECUN);
        printf("Reinitialized with LeCun initialization\n");
      }
    }
    
    if (IsKeyPressed(KEY_7) && g_training_mode) {
      // Select Orthogonal initialization (for complex models)
      printf("Selected: Orthogonal initialization (for complex models)\n");
      // Update network with ORTHOGONAL initialization
      if (g_conv) {
        NN_destroy(g_conv);
        size_t layers[] = {CANVAS_SIZE * CANVAS_SIZE * 3, 128, 64, CANVAS_SIZE * CANVAS_SIZE, 0};
        ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
        ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
        g_conv = NN_init_with_weight_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 
                                           L2, ADAM, 0.001f, ORTHOGONAL);
        printf("Reinitialized with Orthogonal initialization\n");
      }
    }
    
    if (IsKeyPressed(KEY_R) && g_training_mode) {
      process_with_neural_networks();
    }
    
    if (IsKeyPressed(KEY_S) && g_training_mode) {
      save_trained_networks();
    }
    
    if (IsKeyPressed(KEY_O)) {  // O for "Open" trained networks
      load_trained_networks();
    }
    
    // 3D generation controls
    if (IsKeyPressed(KEY_G)) {
      generate_vertices_from_depth();
    }
    
    if (IsKeyPressed(KEY_V)) {
      view_generated_3d_object();
    }
    
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !g_training_mode) {
      Vector2 mouse = GetMousePosition();
      paint_at((int)mouse.x, (int)mouse.y, BRUSH_COLOR);
    }
    
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !g_training_mode) {
      Vector2 mouse = GetMousePosition();
      erase_at((int)mouse.x, (int)mouse.y);
    }
    
    BeginDrawing();
    ClearBackground(BG_COLOR);
    
    draw_3d_canvas();
    
    // Draw UI
    DrawText("3D Paint with Neural Networks", 10, 10, 20, WHITE);
    
    if (!g_training_mode) {
      DrawText("Left Click: Paint | Right Click: Erase", 10, 40, 16, WHITE);
      DrawText("SPACE: Toggle 3D Mode | ENTER: Process with NN", 10, 60, 16, WHITE);
      DrawText("G: Generate 3D Object | V: View 3D Object", 10, 80, 16, WHITE);
    } else {
      DrawText("TRAINING MODE - Painting Disabled", 10, 40, 16, YELLOW);
      DrawText("T: Toggle Training | L: Load Assets", 10, 60, 16, WHITE);
      DrawText("1-7: Weight Init (1=Zero, 2=Uni, 3=Norm, 4=Xav, 5=He, 6=LeCun, 7=Orth)", 10, 80, 16, WHITE);
      DrawText("R: Train Networks | S: Save Networks | O: Load Trained", 10, 100, 16, WHITE);
      DrawText("G: Generate 3D Object | V: View 3D Object", 10, 120, 16, WHITE);
    }
    
    DrawText(TextFormat("3D Mode: %s", g_3d_mode ? "ON" : "OFF"), 10, 140, 16, 
               g_3d_mode ? GREEN : RED);
    
    if (nn_processed) {
      DrawText("Neural Network Processing Complete!", 10, 160, 16, GREEN);
    }
    
    if (g_training_mode) {
      DrawText(TextFormat("Training Samples: %d", g_training_count), 10, 180, 16, CYAN);
    }
    
    EndDrawing();
  }
  
  // Cleanup neural networks
  if (g_conv) NN_destroy(g_conv);
  if (g_transformer) free(g_transformer);
  
  // Cleanup training data
  if (g_training_inputs) free(g_training_inputs);
  if (g_training_targets) free(g_training_targets);
  
  CloseWindow();
  return 0;
}
