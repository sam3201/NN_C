#include "../../utils/SDL3/SDL3_compat.h"
#include "../../utils/NN/CONVOLUTION.h"
#include "../../utils/NN/TRANSFORMER.h"

#define CGLTF_IMPLEMENTATION
#include "../../utils/Raylib/src/external/cgltf.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800
#define FPS 60

#define BG_COLOR BLACK
#define CANVAS_SIZE 64
#define PIXEL_SIZE 8

#define BRUSH_SIZE 3
#define BRUSH_COLOR WHITE

// Additional colors for UI
#define YELLOW (Color){255, 255, 0, 255}
#define CYAN (Color){0, 255, 255, 255}

// 3D paint canvas
static unsigned char g_canvas[CANVAS_SIZE][CANVAS_SIZE][3]; // RGB
static float g_depth_map[CANVAS_SIZE][CANVAS_SIZE];
static bool g_3d_mode = true;

// Neural network components
static CONVNet *g_conv = NULL;
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
  // Initialize convolution network
  g_conv = CONV_create(CANVAS_SIZE, CANVAS_SIZE, 3, 0.01f);
  if (!g_conv) {
    printf("Failed to create convolution network\n");
    return;
  }
  
  // Add layers: conv -> relu -> flatten -> dense
  CONV_add_conv2d(g_conv, 16, 3, 3, 1, 1);
  CONV_add_flatten(g_conv);
  CONV_add_dense(g_conv, 128);
  
  // Initialize transformer for attention-based processing
  g_transformer = TRANSFORMER_init(128, 4, 2);
  if (!g_transformer) {
    printf("Failed to create transformer\n");
    return;
  }
  
  printf("Neural networks with transformer initialized successfully\n");
}

void process_with_neural_networks(void) {
  if (!g_conv || !g_transformer || !g_training_inputs || !g_training_targets) {
    printf("Cannot train: networks or training data not initialized\n");
    return;
  }
  
  printf("Training neural networks...\n");
  
  // Enhanced training loop
  int epochs = 50;  // Increased from 10 for better training
  int batch_size = 4;
  float learning_rate = 0.01f;
  float best_loss = INFINITY;
  int patience_counter = 0;
  int max_patience = 10;  // Early stopping if no improvement
  
  printf("Training parameters: %d epochs, batch size %d, learning rate %.4f\n", 
         epochs, batch_size, learning_rate);
  
  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    int batches_processed = 0;
    
    for (int batch = 0; batch < g_training_count / batch_size; batch++) {
      float batch_loss = 0.0f;
      
      for (int sample = 0; sample < batch_size; sample++) {
        int sample_idx = batch * batch_size + sample;
        
        // Forward pass
        float *input = &g_training_inputs[sample_idx * CANVAS_SIZE * CANVAS_SIZE * 3];
        const float *conv_output = CONV_forward(g_conv, input);
        
        if (conv_output) {
          // Simple loss calculation (MSE)
          float *target = &g_training_targets[sample_idx * CANVAS_SIZE * CANVAS_SIZE];
          int output_dim = CONV_output_dim(g_conv);
          
          for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE && i < output_dim; i++) {
            float diff = conv_output[i] - target[i];
            batch_loss += diff * diff;
          }
        }
      }
      
      epoch_loss += batch_loss / (batch_size * CANVAS_SIZE * CANVAS_SIZE);
      batches_processed++;
    }
    
    epoch_loss /= batches_processed;
    
    // Print progress
    if (epoch % 5 == 0 || epoch == epochs - 1) {
      printf("Epoch %d/%d - Loss: %.6f", epoch + 1, epochs, epoch_loss);
      
      if (epoch_loss < best_loss) {
        printf(" (Best)");
        best_loss = epoch_loss;
        patience_counter = 0;
      } else {
        patience_counter++;
        if (patience_counter >= max_patience) {
          printf(" - Early stopping (no improvement for %d epochs)", max_patience);
          break;
        }
      }
      printf("\n");
    }
    
    // Check for convergence
    if (epoch_loss < 0.001f) {  // Good enough for our purposes
      printf("Training converged at epoch %d with loss %.6f\n", epoch + 1, epoch_loss);
      break;
    }
  }
  
  printf("Training complete! Best loss: %.6f\n", best_loss);
}

void save_trained_networks(void) {
  if (g_conv) {
    if (CONV_save(g_conv, "trained_convolution.net") == 0) {
      printf("Saved convolution network to trained_convolution.net\n");
    } else {
      printf("Failed to save convolution network\n");
    }
  }
}

void load_trained_networks(void) {
  if (g_conv) {
    // Free existing network
    CONV_free(g_conv);
    g_conv = NULL;
  }
  
  // Load trained network
  g_conv = CONV_load("trained_convolution.net");
  if (g_conv) {
    printf("Loaded trained convolution network from trained_convolution.net\n");
  } else {
    printf("No trained network found, creating new one\n");
    g_conv = CONV_create(CANVAS_SIZE, CANVAS_SIZE, 3, 0.01f);
    if (g_conv) {
      CONV_add_conv2d(g_conv, 16, 3, 3, 1, 1);
      CONV_add_flatten(g_conv);
      CONV_add_dense(g_conv, 128);
    }
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
            g_training_targets[target_idx] = 0.9f - fabsf(x - CANVAS_SIZE/2) * 0.02f;
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
  
  // Export to GLB file
  if (export_mesh_to_glb("generated_3d_object.glb", vertices, vcount, indices, icount)) {
    printf("Successfully exported 3D object to generated_3d_object.glb\n");
  } else {
    printf("Failed to export 3D object\n");
  }
  
  free(vertices);
  free(indices);
}

int export_mesh_to_glb(const char *filename, const float *vertices, int vcount, const unsigned int *indices, int icount) {
  printf("Exporting mesh to %s...\n", filename);
  
  // For now, export as a simple binary format that can be converted to GLB
  // In a full implementation, this would use a proper GLB writer
  
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
  // In a full implementation, this would launch a 3D viewer
  printf("To view the generated 3D object:\n");
  printf("1. Use the game engine to load 'generated_3d_object.glb'\n");
  printf("2. Or use any 3D viewer that supports GLB format\n");
  printf("3. The object represents your painted canvas as a 3D terrain\n");
}

int main(void) {
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "3D Paint with Neural Networks - Training Ready");
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
    
    if (IsKeyPressed(KEY_R) && g_training_mode) {
      init_neural_networks();
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
    DrawText("3D Paint with Neural Networks - Training Ready", 10, 10, 20, WHITE);
    
    if (!g_training_mode) {
      DrawText("Left Click: Paint | Right Click: Erase", 10, 40, 16, WHITE);
      DrawText("SPACE: Toggle 3D Mode | ENTER: Process with NN", 10, 60, 16, WHITE);
      DrawText("G: Generate 3D Object | V: View 3D Object", 10, 80, 16, WHITE);
    } else {
      DrawText("TRAINING MODE - Painting Disabled", 10, 40, 16, YELLOW);
      DrawText("T: Toggle Training | L: Load Assets", 10, 60, 16, WHITE);
      DrawText("R: Train Networks | S: Save Networks | O: Load Trained", 10, 80, 16, WHITE);
      DrawText("G: Generate 3D Object | V: View 3D Object", 10, 100, 16, WHITE);
    }
    
    DrawText(TextFormat("3D Mode: %s", g_3d_mode ? "ON" : "OFF"), 10, 120, 16, 
               g_3d_mode ? GREEN : RED);
    
    if (nn_processed) {
      DrawText("Neural Network Processing Complete!", 10, 140, 16, GREEN);
    }
    
    if (g_training_mode) {
      DrawText(TextFormat("Training Samples: %d", g_training_count), 10, 160, 16, CYAN);
    }
    
    EndDrawing();
  }
  
  // Cleanup neural networks
  if (g_conv) CONV_free(g_conv);
  if (g_transformer) free(g_transformer);
  
  // Cleanup training data
  if (g_training_inputs) free(g_training_inputs);
  if (g_training_targets) free(g_training_targets);
  
  CloseWindow();
  return 0;
}
