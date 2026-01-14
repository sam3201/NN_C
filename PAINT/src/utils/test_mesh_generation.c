#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CANVAS_SIZE 64

// Simple test to generate a mesh from a sample depth map
void generate_test_mesh(void) {
    printf("Generating test mesh...\n");
    
    // Create a simple depth map (center hill)
    float depth_map[CANVAS_SIZE][CANVAS_SIZE];
    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            float dx = x - CANVAS_SIZE / 2.0f;
            float dy = y - CANVAS_SIZE / 2.0f;
            float dist = sqrtf(dx * dx + dy * dy);
            depth_map[y][x] = expf(-dist * dist / 200.0f); // Gaussian hill
        }
    }
    
    // Generate vertices
    int vcount = CANVAS_SIZE * CANVAS_SIZE;
    int icount = (CANVAS_SIZE - 1) * (CANVAS_SIZE - 1) * 6;
    
    float *vertices = malloc(sizeof(float) * vcount * 6); // pos(3) + normal(3)
    unsigned int *indices = malloc(sizeof(unsigned int) * icount);
    
    if (!vertices || !indices) {
        printf("Failed to allocate memory\n");
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
            vertices[idx * 6 + 1] = depth_map[y][x] * 2.0f;     // Y coordinate (height from depth)
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
    FILE *file = fopen("test_terrain.mesh", "wb");
    if (file) {
        fwrite("MESH", 4, 1, file);
        fwrite(&vcount, sizeof(int), 1, file);
        fwrite(&icount, sizeof(int), 1, file);
        fwrite(vertices, sizeof(float), vcount * 6, file);
        fwrite(indices, sizeof(unsigned int), icount, file);
        fclose(file);
        printf("Successfully exported test mesh to test_terrain.mesh\n");
        printf("Vertices: %d, Triangles: %d\n", vcount, icount / 3);
    } else {
        printf("Failed to create mesh file\n");
    }
    
    free(vertices);
    free(indices);
}

int main(void) {
    generate_test_mesh();
    return 0;
}
