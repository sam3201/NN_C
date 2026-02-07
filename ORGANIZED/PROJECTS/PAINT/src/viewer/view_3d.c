#include "../../utils/SDL3/SDL3_compat.h"
#include "../utils/mesh_loader.h"
#include <stdio.h>
#include <stdlib.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FPS 60

static float g_rotation_x = 0.0f;
static float g_rotation_y = 0.0f;
static float g_zoom = 1.0f;
static MeshData g_mesh = {0};

void draw_mesh_wireframe(void) {
  if (!g_mesh.vertices) return;
  
  // Simple wireframe rendering using 2D projection
  for (int i = 0; i < g_mesh.index_count; i += 3) {
    unsigned int i0 = g_mesh.indices[i];
    unsigned int i1 = g_mesh.indices[i + 1];
    unsigned int i2 = g_mesh.indices[i + 2];
    
    if (i0 < g_mesh.vertex_count && i1 < g_mesh.vertex_count && 
        i2 < g_mesh.vertex_count) {
      
      // Apply simple rotation and projection
      float cos_x = cosf(g_rotation_x);
      float sin_x = sinf(g_rotation_x);
      float cos_y = cosf(g_rotation_y);
      float sin_y = sinf(g_rotation_y);
      
      // Transform vertices
      float v0[3], v1[3], v2[3];
      
      // Vertex 0
      v0[0] = g_mesh.vertices[i0].x * cos_y - g_mesh.vertices[i0].z * sin_y;
      v0[1] = g_mesh.vertices[i0].y * cos_x - (g_mesh.vertices[i0].x * sin_y + g_mesh.vertices[i0].z * cos_y) * sin_x;
      v0[2] = g_mesh.vertices[i0].y * sin_x + (g_mesh.vertices[i0].x * sin_y + g_mesh.vertices[i0].z * cos_y) * cos_x;
      
      // Vertex 1
      v1[0] = g_mesh.vertices[i1].x * cos_y - g_mesh.vertices[i1].z * sin_y;
      v1[1] = g_mesh.vertices[i1].y * cos_x - (g_mesh.vertices[i1].x * sin_y + g_mesh.vertices[i1].z * cos_y) * sin_x;
      v1[2] = g_mesh.vertices[i1].y * sin_x + (g_mesh.vertices[i1].x * sin_y + g_mesh[i1].z * cos_y) * cos_x;
      
      // Vertex 2
      v2[0] = g_mesh.vertices[i2].x * cos_y - g_mesh.vertices[i2].z * sin_y;
      v2[1] = g_mesh.vertices[i2].y * cos_x - (g_mesh.vertices[i2].x * sin_y + g_mesh.vertices[i2].z * cos_y) * sin_x;
      v2[2] = g_mesh.vertices[i2].y * sin_x + (g_mesh.vertices[i2].x * sin_y + g_mesh.vertices[i2].z * cos_y) * cos_x;
      
      // Project to 2D
      float scale = 100.0f * g_zoom;
      float center_x = SCREEN_WIDTH / 2.0f;
      float center_y = SCREEN_HEIGHT / 2.0f;
      
      int x0 = (int)(center_x + v0[0] * scale);
      int y0 = (int)(center_y - v0[1] * scale);
      int x1 = (int)(center_x + v1[0] * scale);
      int y1 = (int)(center_y - v1[1] * scale);
      int x2 = (int)(center_x + v2[0] * scale);
      int y2 = (int)(center_y - v2[1] * scale);
      
      // Draw triangle edges
      DrawLine(x0, y0, x1, y1, WHITE);
      DrawLine(x1, y1, x2, y2, WHITE);
      DrawLine(x2, y2, x0, y0, WHITE);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <mesh_file.mesh>\n", argv[0]);
    return 1;
  }
  
  if (!load_custom_mesh(argv[1], &g_mesh)) {
    printf("Failed to load mesh file: %s\n", argv[1]);
    return 1;
  }
  
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "3D Mesh Viewer");
  SetTargetFPS(FPS);
  
  printf("Controls:\n");
  printf("Arrow Keys: Rotate\n");
  printf("+/-: Zoom in/out\n");
  printf("ESC: Exit\n");
  
  while (!WindowShouldClose()) {
    // Handle input
    if (IsKeyDown(KEY_LEFT)) g_rotation_y -= 0.02f;
    if (IsKeyDown(KEY_RIGHT)) g_rotation_y += 0.02f;
    if (IsKeyDown(KEY_UP)) g_rotation_x -= 0.02f;
    if (IsKeyDown(KEY_DOWN)) g_rotation_x += 0.02f;
    if (IsKeyPressed(KEY_EQUAL)) g_zoom *= 1.1f;
    if (IsKeyPressed(KEY_MINUS)) g_zoom *= 0.9f;
    
    BeginDrawing();
    ClearBackground(BLACK);
    
    draw_mesh_wireframe();
    
    // Draw UI
    DrawText("3D Mesh Viewer", 10, 10, 20, WHITE);
    DrawText("Arrow Keys: Rotate | +/-: Zoom | ESC: Exit", 10, 40, 16, WHITE);
    DrawText(TextFormat("Vertices: %d | Triangles: %d", g_mesh.vertex_count, g_mesh.index_count / 3), 
            10, 60, 16, WHITE);
    DrawText(TextFormat("Zoom: %.2f", g_zoom), 10, 80, 16, WHITE);
    
    EndDrawing();
  }
  
  free_mesh_data(&g_mesh);
  CloseWindow();
  return 0;
}
